#include "Learner.h"

#include <GigaLearnCPP/PPO/PPOLearner.h>
#include <GigaLearnCPP/PPO/ExperienceBuffer.h>

#include <torch/cuda.h>
#include <nlohmann/json.hpp>
#include <pybind11/embed.h>

#ifdef RG_CUDA_SUPPORT
#include <c10/cuda/CUDACachingAllocator.h>
#endif
#include <private/GigaLearnCPP/PPO/ExperienceBuffer.h>
#include <private/GigaLearnCPP/PPO/GAE.h>
#include <private/GigaLearnCPP/PolicyVersionManager.h>

#include "Util/KeyPressDetector.h"
#include <private/GigaLearnCPP/Util/WelfordStat.h>
#include "Util/AvgTracker.h"

using namespace RLGC;

namespace {
	inline void AppendContinuousTeacherTarget(const Action& action, std::vector<float>& outValues) {
		outValues.push_back(action.throttle);
		outValues.push_back(action.steer);
		outValues.push_back(action.pitch);
		outValues.push_back(action.yaw);
		outValues.push_back(action.roll);
		outValues.push_back(action.jump > 0 ? 1.0f : -1.0f);
		outValues.push_back(action.boost > 0 ? 1.0f : -1.0f);
		outValues.push_back(action.handbrake > 0 ? 1.0f : -1.0f);
	}

	inline void AppendContinuousTeacherActionTable(
		ActionParser& parser,
		const Player& player,
		const GameState& state,
		std::vector<float>& outValues) {

		int numActions = parser.GetActionAmount();
		for (int actionIdx = 0; actionIdx < numActions; actionIdx++)
			AppendContinuousTeacherTarget(parser.ParseAction(actionIdx, player, state), outValues);
	}
}

GGL::Learner::Learner(EnvCreateFn envCreateFn, LearnerConfig config, StepCallbackFn stepCallback) :
	envCreateFn(envCreateFn), config(config), stepCallback(stepCallback)
{
	if (config.renderMode || (config.sendMetrics && !config.renderMode)) {
		pybind11::initialize_interpreter();
		pyInterpreterInitialized = true;
	}

#ifndef NDEBUG
	RG_LOG("===========================");
	RG_LOG("WARNING: GigaLearn runs extremely slowly in debug, and there are often bizzare issues with debug-mode torch.");
	RG_LOG("It is recommended that you compile in release mode without optimization for debugging.");
	RG_SLEEP(1000);
#endif

	if (config.tsPerSave == 0)
		config.tsPerSave = config.ppo.tsPerItr;

	RG_LOG("Learner::Learner():");

	if (config.randomSeed == -1)
		config.randomSeed = RS_CUR_MS();

	RG_LOG("\tCheckpoint Save/Load Dir: " << config.checkpointFolder);

	torch::manual_seed(config.randomSeed);

	at::Device device = at::Device(at::kCPU);
	if (
		config.deviceType == LearnerDeviceType::GPU_CUDA || 
		(config.deviceType == LearnerDeviceType::AUTO && torch::cuda::is_available())
		) {
		RG_LOG("\tUsing CUDA GPU device...");

		// Test out moving a tensor to GPU and back to make sure the device is working
		torch::Tensor t;
		bool deviceTestFailed = false;
		try {
			t = torch::tensor(0);
			t = t.to(at::Device(at::kCUDA));
			t = t.cpu();
		} catch (...) {
			deviceTestFailed = true;
		}

		if (!torch::cuda::is_available() || deviceTestFailed)
			RG_ERR_CLOSE(
				"Learner::Learner(): Can't use CUDA GPU because " <<
				(torch::cuda::is_available() ? "libtorch cannot access the GPU" : "CUDA is not available to libtorch") << ".\n" <<
				"Make sure your libtorch comes with CUDA support, and that CUDA is installed properly."
			)
		device = at::Device(at::kCUDA);
	} else {
		RG_LOG("\tUsing CPU device...");
		device = at::Device(at::kCPU);
	}

	if (RocketSim::GetStage() != RocketSimStage::INITIALIZED) {
		RG_LOG("\tInitializing RocketSim...");
		RocketSim::Init("collision_meshes", true);
	}

	{
		RG_LOG("\tCreating envs...");
		if (config.physicsBackend == RLGC::EnvPhysicsBackend::ROCKETSIM_CUDA)
			RG_LOG("\tRequested RocketSimCuda env backend.");

		EnvSetConfig envSetConfig = {};
		envSetConfig.envCreateFn = envCreateFn;
		envSetConfig.numArenas = config.renderMode ? 1 : config.numGames;
		envSetConfig.tickSkip = config.tickSkip;
		envSetConfig.actionDelay = config.actionDelay;
		envSetConfig.saveRewards = config.addRewardsToMetrics;
		envSetConfig.physicsBackend = config.physicsBackend;
		envSetConfig.syncCudaWorldStateToCpu = true;
		if (config.cudaNoCpuWorldState) {
			bool canDisableCpuWorldState =
				!config.renderMode &&
				!config.addRewardsToMetrics &&
				stepCallback == NULL;

			if (canDisableCpuWorldState) {
				envSetConfig.syncCudaWorldStateToCpu = false;
				RG_LOG("\tRocketSimCuda CPU world-state sync disabled for training fast path.");
			} else {
				RG_LOG(
					"\tRocketSimCuda CPU world-state sync requested but kept enabled because "
					"render mode, reward metrics, or a step callback still needs fresh GameStates."
				);
			}
		}
		envSet = new RLGC::EnvSet(envSetConfig);
#ifdef RG_ROCKETSIMCUDA_AVAILABLE
		// With the CUDA reward fast path, lastRewards is metrics-only and the
		// metrics code only reads the first maxRewardSamples arenas.
		if (envSet->UsingCudaRewardTerminalFastPath())
			envSet->saveRewardsArenaLimit = RS_MAX(config.maxRewardSamples, 1);
#endif
		obsSize = envSet->state.obs.size[1];
		if (config.ppo.policyType == PolicyType::CONTINUOUS) {
			numActions = config.ppo.continuousActionSize;
		} else {
			numActions = envSet->actionParsers[0]->GetActionAmount();
		}
	}

	{
		if (config.standardizeReturns) {
			this->returnStat = new WelfordStat();
		} else {
			this->returnStat = NULL;
		}

		if (config.standardizeObs) {
			this->obsStat = new BatchedWelfordStat(obsSize);
		} else {
			this->obsStat = NULL;
		}
	}

	try {
		RG_LOG("\tMaking PPO learner...");
		ppo = new PPOLearner(obsSize, numActions, config.ppo, device);
	} catch (std::exception& e) {
		RG_ERR_CLOSE("Failed to create PPO learner: " << e.what());
	}

	if (config.renderMode) {
		renderSender = new RenderSender(config.renderTimeScale);
	} else {
		renderSender = NULL;
	}

	if (config.skillTracker.enabled || config.trainAgainstOldVersions)
		config.savePolicyVersions = true;

	if (config.savePolicyVersions && !config.renderMode) {
		if (config.checkpointFolder.empty())
			RG_ERR_CLOSE("Cannot save/load old policy versions with no checkpoint save folder");
		versionMgr = new PolicyVersionManager(
			config.checkpointFolder / "policy_versions", config.maxOldVersions, config.tsPerVersion,
			config.skillTracker, envSet->config
		);
	} else {
		versionMgr = NULL;
	}

	if (!config.checkpointFolder.empty())
		Load();

	if (config.savePolicyVersions && !config.renderMode) {
		if (config.checkpointFolder.empty())
			RG_ERR_CLOSE("Cannot save/load old policy versions with no checkpoint save folder");
		auto models = ppo->GetPolicyModels();
		versionMgr->LoadVersions(models, totalTimesteps);
	}

	if (config.sendMetrics && !config.renderMode) {
		if (!runID.empty())
			RG_LOG("\tRun ID: " << runID);
		metricSender = new MetricSender(config.metricsProjectName, config.metricsGroupName, config.metricsRunName, runID);
	} else {
		metricSender = NULL;
	}

	RG_LOG(RG_DIVIDER);
}

void GGL::Learner::SaveStats(std::filesystem::path path) {
	using namespace nlohmann;

	constexpr const char* ERROR_PREFIX = "Learner::SaveStats(): ";

	std::ofstream fOut(path);
	if (!fOut.good())
		RG_ERR_CLOSE(ERROR_PREFIX << "Can't open file at " << path);

	json j = {};
	j["total_timesteps"] = totalTimesteps;
	j["total_iterations"] = totalIterations;

	if (config.sendMetrics)
		j["run_id"] = metricSender->curRunID;

	if (returnStat)
		j["return_stat"] = returnStat->ToJSON();
	if (obsStat)
		j["obs_stat"] = obsStat->ToJSON();

	if (versionMgr)
		versionMgr->AddRunningStatsToJSON(j);

	std::string jStr = j.dump(4);
	fOut << jStr;
}

void GGL::Learner::LoadStats(std::filesystem::path path) {
	// TODO: Repetitive code, merge repeated code into one function called from both SaveStats() and LoadStats()

	using namespace nlohmann;
	constexpr const char* ERROR_PREFIX = "Learner::LoadStats(): ";

	std::ifstream fIn(path);
	if (!fIn.good())
		RG_ERR_CLOSE(ERROR_PREFIX << "Can't open file at " << path);

	json j = json::parse(fIn);
	totalTimesteps = j["total_timesteps"];
	totalIterations = j["total_iterations"];

	if (j.contains("run_id"))
		runID = j["run_id"];

	if (returnStat)
		returnStat->ReadFromJSON(j["return_stat"]);
	if (obsStat)
		obsStat->ReadFromJSON(j["obs_stat"]);

	if (versionMgr)
		versionMgr->LoadRunningStatsFromJSON(j);
}

// Different than RLGym-PPO to show that they are not compatible
constexpr const char* STATS_FILE_NAME = "RUNNING_STATS.json";

void GGL::Learner::Save() {
	if (config.checkpointFolder.empty())
		RG_ERR_CLOSE("Learner::Save(): Cannot save because config.checkpointSaveFolder is not set");

	std::filesystem::path saveFolder = config.checkpointFolder / std::to_string(totalTimesteps);
	std::filesystem::create_directories(saveFolder);

	RG_LOG("Saving to folder " << saveFolder << "...");
	SaveStats(saveFolder / STATS_FILE_NAME);
	ppo->SaveTo(saveFolder);

	// Remove old checkpoints
	if (config.checkpointsToKeep != -1) {
		std::set<int64_t> allSavedTimesteps = Utils::FindNumberedDirs(config.checkpointFolder);
		while (allSavedTimesteps.size() > config.checkpointsToKeep) {
			int64_t lowestCheckpointTS = INT64_MAX;
			for (int64_t savedTimesteps : allSavedTimesteps)
				lowestCheckpointTS = RS_MIN(lowestCheckpointTS, savedTimesteps);

			std::filesystem::path removePath = config.checkpointFolder / std::to_string(lowestCheckpointTS);
			try {
				std::filesystem::remove_all(removePath);
			} catch (std::exception& e) {
				RG_ERR_CLOSE("Failed to remove old checkpoint from " << removePath << ", exception: " << e.what());
			}
			allSavedTimesteps.erase(lowestCheckpointTS);
		}
	}

	if (versionMgr)
		versionMgr->SaveVersions();

	RG_LOG(" > Done.");
}

void GGL::Learner::Load() {
	if (config.checkpointFolder.empty())
		RG_ERR_CLOSE("Learner::Load(): Cannot load because config.checkpointLoadFolder is not set");

	RG_LOG("Loading most recent checkpoint in " << config.checkpointFolder << "...");

	int64_t highest = -1;
	std::set<int64_t> allSavedTimesteps = Utils::FindNumberedDirs(config.checkpointFolder);
	for (int64_t timesteps : allSavedTimesteps)
		highest = RS_MAX(timesteps, highest);

	if (highest != -1) {
		std::filesystem::path loadFolder = config.checkpointFolder / std::to_string(highest);
		RG_LOG(" > Loading checkpoint " << loadFolder << "...");
		LoadStats(loadFolder / STATS_FILE_NAME);
		ppo->LoadFrom(loadFolder);
		RG_LOG(" > Done.");
	} else {
		RG_LOG(" > No checkpoints found, starting new model.")
	}
}

void GGL::Learner::StartQuitKeyThread(bool& quitPressed, std::thread& outThread) {
	quitPressed = false;

	RG_LOG("Press 'Q' to save and quit!");
	outThread = std::thread(
		[&] {
			while (!stopRequested.load()) {
				char c = toupper(KeyPressDetector::GetPressedChar());
				if (c == 'Q') {
					RG_LOG("Save queued, will save and exit next iteration.");
					quitPressed = true;
					break;
				}
			}
		}
	);

	outThread.detach();
}
void GGL::Learner::StartTransferLearn(const TransferLearnConfig& tlConfig) {

	RG_LOG("Starting transfer learning...");
	bool isContinuous = (config.ppo.policyType == PolicyType::CONTINUOUS);
	bool oldIsContinuous = (tlConfig.oldPolicyType == PolicyType::CONTINUOUS);
	bool distillDiscreteToContinuous = isContinuous && !oldIsContinuous;
	auto fnGetStudentRolloutProb = [&]() {
		float result = tlConfig.discreteToContinuousStudentRolloutProb;
		if (isContinuous && tlConfig.discreteToContinuousStudentRolloutWarmupTimesteps > 0) {
			float warmupFrac = RS_CLAMP(
				totalTimesteps / (float)tlConfig.discreteToContinuousStudentRolloutWarmupTimesteps,
				0.0f, 1.0f
			);
			result *= warmupFrac;
		}
		return result;
	};

	if (distillDiscreteToContinuous && config.ppo.continuousActionSize != 8) {
		RG_ERR_CLOSE("StartTransferLearn: discrete-teacher -> continuous-student distillation currently expects continuousActionSize == 8");
	}
	if (tlConfig.discreteToContinuousStudentRolloutProb < 0 || tlConfig.discreteToContinuousStudentRolloutProb > 1) {
		RG_ERR_CLOSE("StartTransferLearn: tlConfig.discreteToContinuousStudentRolloutProb must be in [0, 1]");
	}
	if (distillDiscreteToContinuous && tlConfig.discreteToContinuousStudentRolloutProb > 0) {
		RG_LOG("Discrete -> continuous TL rollout mixing enabled (base student rollout probability: " << tlConfig.discreteToContinuousStudentRolloutProb << ")");
	}

	// TODO: Lots of manual obs builder stuff going on which is quite volatile
	//	Although I can't really think another way to do this

	std::vector<ObsBuilder*> oldObsBuilders = {};
	for (int i = 0; i < envSet->arenas.size(); i++)
		oldObsBuilders.push_back(tlConfig.makeOldObsFn());

	// Reset all obs builders initially
	for (int i = 0; i < envSet->arenas.size(); i++)
		oldObsBuilders[i]->Reset(envSet->state.gameStates[0]);

	std::vector<ActionParser*> oldActionParsers = {};
	if (!oldIsContinuous) {
		for (int i = 0; i < envSet->arenas.size(); i++)
			oldActionParsers.push_back(tlConfig.makeOldActFn());
	}

	int oldNumActions = oldIsContinuous ? tlConfig.oldContinuousActionSize : oldActionParsers[0]->GetActionAmount();
	bool canTeacherForceContinuousRollout =
		isContinuous && oldIsContinuous &&
		(tlConfig.oldContinuousActionSize == config.ppo.continuousActionSize);
	bool canTeacherForceDiscreteRollout =
		!isContinuous && !oldIsContinuous &&
		(oldNumActions == numActions) &&
		!tlConfig.mapActsFn;

	if (!isContinuous && !oldIsContinuous && oldNumActions != numActions) {
		if (!tlConfig.mapActsFn) {
			RG_ERR_CLOSE(
				"StartTransferLearn: Old and new action parsers have a different number of actions, but tlConfig.mapActsFn is NULL.\n" <<
				"You must implement this function to translate the action indices."
			);
		};
	}

	// Determine old obs size
	int oldObsSize;
	{
		GameState testState = envSet->state.gameStates[0];
		oldObsSize = oldObsBuilders[0]->BuildObs(testState.players[0], testState).size();
	}

	ModelSet oldModels = {};
	{
		RG_NO_GRAD;
		int oldPolicyOutputs =
			oldIsContinuous ? PPOLearner::GetContinuousPolicyOutputSize(tlConfig.oldContinuousActionSize) : oldNumActions;
		if (tlConfig.oldUseAttentionHead) {
			if (!tlConfig.oldAttentionHeadConfig)
				RG_ERR_CLOSE("StartTransferLearn: oldUseAttentionHead is true but oldAttentionHeadConfig is null");

			PPOLearner::MakeModelsWithAttention(
				false, oldObsSize, oldPolicyOutputs,
				*tlConfig.oldAttentionHeadConfig, tlConfig.oldPolicyConfig, {},
				ppo->device, oldModels,
				tlConfig.oldPolicyType, tlConfig.oldContinuousActionSize
			);
		} else {
			PPOLearner::MakeModels(
				false, oldObsSize, oldPolicyOutputs,
				tlConfig.oldSharedHeadConfig, tlConfig.oldPolicyConfig, {},
				ppo->device, oldModels,
				tlConfig.oldPolicyType, tlConfig.oldContinuousActionSize
			);
		}

		oldModels.Load(tlConfig.oldModelsPath, false, false);
	}

	if (oldIsContinuous && isContinuous && !canTeacherForceContinuousRollout)
		RG_LOG("Transfer learning rollout will fall back to student actions because old/new continuous action sizes differ.");
	if (!distillDiscreteToContinuous && !oldIsContinuous && !isContinuous && !canTeacherForceDiscreteRollout)
		RG_LOG("Transfer learning rollout will fall back to student actions because teacher discrete actions cannot be applied directly to the new action space.");
	if (isContinuous && oldIsContinuous && canTeacherForceContinuousRollout && tlConfig.discreteToContinuousStudentRolloutProb > 0)
		RG_LOG("Continuous -> continuous TL student rollout mixing enabled (base student rollout probability: " << tlConfig.discreteToContinuousStudentRolloutProb << ")");

	try {
		bool saveQueued;
		std::thread keyPressThread;
		StartQuitKeyThread(saveQueued, keyPressThread);
		int transferIterations = 0;

		while (!stopRequested.load()) {
			Report report = {};

			// Collect obs
			std::vector<float> allNewObs = {};
			std::vector<float> allOldObs = {};
			std::vector<float> allTeacherActionTable = {};
			std::vector<float> allTeacherActionTargets = {};
			std::vector<uint8_t> allNewActionMasks = {};
			std::vector<uint8_t> allOldActionMasks = {};
			std::vector<int> allActionMaps = {};
			int64_t studentRolloutActionsUsed = 0;
			int64_t totalStudentRolloutCandidates = 0;

			// Pre-reserve vectors to estimated sizes to avoid reallocation overhead
			{
				int64_t estSamples = tlConfig.batchSize;
				allNewObs.reserve(estSamples * obsSize);
				allOldObs.reserve(estSamples * oldObsSize);
				if (!oldIsContinuous)
					allOldActionMasks.reserve(estSamples * oldNumActions);
				if (distillDiscreteToContinuous) {
					allTeacherActionTable.reserve(estSamples * oldNumActions * config.ppo.continuousActionSize);
					allTeacherActionTargets.reserve(estSamples * config.ppo.continuousActionSize);
				}
			}
			int stepsCollected;
			{
				RG_NO_GRAD;
				for (stepsCollected = 0; stepsCollected < tlConfig.batchSize; stepsCollected += envSet->state.numPlayers) {
					float studentRolloutProb = fnGetStudentRolloutProb();
					struct TeacherActionCtx {
						ActionParser* parser;
						const Player* player;
						const GameState* state;
					};
					std::vector<float> stepOldObsForTeacher = {};
					std::vector<uint8_t> stepOldMasksForTeacher = {};
					std::vector<TeacherActionCtx> teacherActionCtx = {};
					std::vector<float> stepRolloutActions = {};
					
					auto terminals = envSet->state.terminals; // Backup
					envSet->Reset();
					for (int i = 0; i < envSet->arenas.size(); i++) // Manually reset old obs builders
						if (terminals[i])
							oldObsBuilders[i]->Reset(envSet->state.gameStates[i]);

					if (!config.renderMode && obsStat) {
						int numSamples = RS_MAX(envSet->state.numPlayers, config.maxObsSamples);
						for (int i = 0; i < numSamples; i++) {
							int idx = Math::RandInt(0, envSet->state.numPlayers);
							obsStat->IncrementRow(&envSet->state.obs.At(idx, 0));
						}

						std::vector<double> mean = obsStat->GetMean();
						std::vector<double> std = obsStat->GetSTD();
						for (double& f : mean)
							f = RS_CLAMP(f, -config.maxObsMeanRange, config.maxObsMeanRange);
						for (double& f : std)
							f = RS_MAX(f, config.minObsSTD);
						for (int i = 0; i < envSet->state.numPlayers; i++) {
							for (int j = 0; j < obsSize; j++) {
								float& obsVal = envSet->state.obs.At(i, j);
								obsVal = (obsVal - mean[j]) / std[j];
							}
						}
					}

					torch::Tensor tActions, tLogProbs;
					torch::Tensor tStates = DIMLIST2_TO_TENSOR<float>(envSet->state.obs);
					torch::Tensor tActionMasks = DIMLIST2_TO_TENSOR<uint8_t>(envSet->state.actionMasks);

					envSet->StepFirstHalf(true);

					allNewObs += envSet->state.obs.data;
					allNewActionMasks += envSet->state.actionMasks.data;

					// Run all old obs and old action parser on each player
					// TODO: Could be multithreaded
					for (int arenaIdx = 0; arenaIdx < envSet->arenas.size(); arenaIdx++) {
						auto& gs = envSet->state.gameStates[arenaIdx];
						for (auto& player : gs.players) {
							auto oldObs = oldObsBuilders[arenaIdx]->BuildObs(player, gs);
							allOldObs += oldObs;
							if (oldIsContinuous) {
								if (canTeacherForceContinuousRollout)
									stepOldObsForTeacher += oldObs;
							} else {
								auto oldMask = oldActionParsers[arenaIdx]->GetActionMask(player, gs);
								allOldActionMasks += oldMask;

								if (distillDiscreteToContinuous || canTeacherForceDiscreteRollout) {
									stepOldObsForTeacher += oldObs;
									stepOldMasksForTeacher += oldMask;
								}
								if (distillDiscreteToContinuous) {
									teacherActionCtx.push_back({ oldActionParsers[arenaIdx], &player, &gs });
								}

								if (!isContinuous && tlConfig.mapActsFn) {
									auto curMap = tlConfig.mapActsFn(player, gs);
									if (curMap.size() != numActions)
										RG_ERR_CLOSE("StartTransferLearn: Your action map must have the same size as the new action parser's actions");
									allActionMaps += curMap;
								}
							}
						}
					}

					if (distillDiscreteToContinuous) {
						std::vector<float> studentRolloutActions = {};
						if (studentRolloutProb > 0) {
							torch::Tensor tStudentActions;
							PPOLearner::SampleContinuousActions(
								ppo->models, tStates.to(ppo->device, true),
								true, config.ppo.useHalfPrecision,
								config.ppo.varMin, config.ppo.varMax,
								&tStudentActions, NULL
							);
							studentRolloutActions = TENSOR_TO_VEC<float>(tStudentActions.cpu().flatten());
						}

						torch::Tensor tTeacherObs = torch::tensor(stepOldObsForTeacher).reshape({ -1, oldObsSize });
						torch::Tensor tTeacherMasks = torch::tensor(stepOldMasksForTeacher).reshape({ -1, oldNumActions });
						torch::Tensor tTeacherActionIndices;
						PPOLearner::InferActionsFromModels(
							oldModels, tTeacherObs.to(ppo->device, true), tTeacherMasks.to(ppo->device, true),
							true, config.ppo.policyTemperature, config.ppo.useHalfPrecision,
							&tTeacherActionIndices, NULL
						);
						auto teacherActionIndices = TENSOR_TO_VEC<int>(tTeacherActionIndices.cpu());

						RG_ASSERT(teacherActionIndices.size() == teacherActionCtx.size());
						if (!studentRolloutActions.empty())
							RG_ASSERT(studentRolloutActions.size() == teacherActionCtx.size() * config.ppo.continuousActionSize);
						stepRolloutActions.reserve(teacherActionCtx.size() * config.ppo.continuousActionSize);
						allTeacherActionTable.reserve(allTeacherActionTable.size() + teacherActionCtx.size() * oldNumActions * config.ppo.continuousActionSize);

						for (int sampleIdx = 0; sampleIdx < teacherActionIndices.size(); sampleIdx++) {
							auto& ctx = teacherActionCtx[sampleIdx];
							RG_ASSERT(ctx.parser->GetActionAmount() == oldNumActions);
							AppendContinuousTeacherActionTable(*ctx.parser, *ctx.player, *ctx.state, allTeacherActionTable);
							Action teacherStepAction = ctx.parser->ParseAction(teacherActionIndices[sampleIdx], *ctx.player, *ctx.state);
							AppendContinuousTeacherTarget(teacherStepAction, allTeacherActionTargets);

							bool useStudentRolloutAction =
								!studentRolloutActions.empty() &&
								(RocketSim::Math::RandFloat() < studentRolloutProb);
							totalStudentRolloutCandidates++;
							if (useStudentRolloutAction) {
								studentRolloutActionsUsed++;
								int studentActionOffset = sampleIdx * config.ppo.continuousActionSize;
								for (int dim = 0; dim < config.ppo.continuousActionSize; dim++)
									stepRolloutActions.push_back(studentRolloutActions[studentActionOffset + dim]);
							} else {
								AppendContinuousTeacherTarget(teacherStepAction, stepRolloutActions);
							}
						}
					}

					envSet->Sync();
					if (distillDiscreteToContinuous) {
						envSet->StepSecondHalfContinuous(stepRolloutActions, config.ppo.continuousActionSize, false);
					} else if (canTeacherForceContinuousRollout) {
						torch::Tensor tTeacherObs = torch::tensor(stepOldObsForTeacher).reshape({ -1, oldObsSize });
						torch::Tensor tTeacherActions;
						PPOLearner::SampleContinuousActions(
							oldModels, tTeacherObs.to(ppo->device, true),
							true, config.ppo.useHalfPrecision,
							config.ppo.varMin, config.ppo.varMax,
							&tTeacherActions, NULL
						);
						auto rolloutActionsCont = TENSOR_TO_VEC<float>(tTeacherActions.cpu().flatten());
						float studentRolloutProb = fnGetStudentRolloutProb();
						if (studentRolloutProb > 0) {
							torch::Tensor tStudentActions;
							PPOLearner::SampleContinuousActions(
								ppo->models, tStates.to(ppo->device, true),
								true, config.ppo.useHalfPrecision,
								config.ppo.varMin, config.ppo.varMax,
								&tStudentActions, NULL
							);
							auto studentActionsCont = TENSOR_TO_VEC<float>(tStudentActions.cpu().flatten());
							RG_ASSERT(studentActionsCont.size() == rolloutActionsCont.size());

							int actionDim = config.ppo.continuousActionSize;
							int numActionSamples = (int)(rolloutActionsCont.size() / actionDim);
							for (int sampleIdx = 0; sampleIdx < numActionSamples; sampleIdx++) {
								totalStudentRolloutCandidates++;
								if (RocketSim::Math::RandFloat() >= studentRolloutProb)
									continue;

								studentRolloutActionsUsed++;
								int actionOffset = sampleIdx * actionDim;
								for (int dim = 0; dim < actionDim; dim++)
									rolloutActionsCont[actionOffset + dim] = studentActionsCont[actionOffset + dim];
							}
						}
						envSet->StepSecondHalfContinuous(rolloutActionsCont, config.ppo.continuousActionSize, false);
					} else if (canTeacherForceDiscreteRollout) {
						torch::Tensor tTeacherObs = torch::tensor(stepOldObsForTeacher).reshape({ -1, oldObsSize });
						torch::Tensor tTeacherMasks = torch::tensor(stepOldMasksForTeacher).reshape({ -1, oldNumActions });
						torch::Tensor tTeacherActionIndices;
						PPOLearner::InferActionsFromModels(
							oldModels, tTeacherObs.to(ppo->device, true), tTeacherMasks.to(ppo->device, true),
							true, config.ppo.policyTemperature, config.ppo.useHalfPrecision,
							&tTeacherActionIndices, NULL
						);
						auto teacherActions = TENSOR_TO_VEC<int>(tTeacherActionIndices.cpu());
						envSet->StepSecondHalf(teacherActions, false);
					} else {
						ppo->InferActions(
							tStates.to(ppo->device, true), tActionMasks.to(ppo->device, true), 
							&tActions, &tLogProbs
						);

						if (isContinuous) {
							auto curActionsCont = TENSOR_TO_VEC<float>(tActions.flatten());
							envSet->StepSecondHalfContinuous(curActionsCont, config.ppo.continuousActionSize, false);
						} else {
							auto curActions = TENSOR_TO_VEC<int>(tActions);
							envSet->StepSecondHalf(curActions, false);
						}
					}

					if (stepCallback)
						stepCallback(this, envSet->state.gameStates, report);
					if (collectionProgressCallback) {
						int64_t collected = RS_MIN((int64_t)stepsCollected + envSet->state.numPlayers, (int64_t)tlConfig.batchSize);
						collectionProgressCallback(this, collected, tlConfig.batchSize);
					}
				}
			}

			uint64_t prevTimesteps = totalTimesteps;
			totalTimesteps += stepsCollected;
			report["Total Timesteps"] = totalTimesteps;
			report["Collected Timesteps"] = stepsCollected;
			totalIterations++;
			transferIterations++;
			report["Total Iterations"] = totalIterations;

			// Make tensors
			torch::Tensor tNewObs = torch::tensor(allNewObs).reshape({ -1, obsSize }).to(ppo->device);
			torch::Tensor tOldObs = torch::tensor(allOldObs).reshape({ -1, oldObsSize }).to(ppo->device);
			torch::Tensor tNewActionMasks;
			torch::Tensor tOldActionMasks;
			if (isContinuous) {
				auto maskOptions = torch::TensorOptions().dtype(torch::kUInt8).device(ppo->device);
				tNewActionMasks = torch::ones({ tNewObs.size(0), 1 }, maskOptions);
				if (oldIsContinuous)
					tOldActionMasks = torch::ones({ tOldObs.size(0), 1 }, maskOptions);
				else
					tOldActionMasks = torch::tensor(allOldActionMasks).reshape({ -1, oldNumActions }).to(ppo->device);
			} else {
				tNewActionMasks = torch::tensor(allNewActionMasks).reshape({ -1, numActions }).to(ppo->device);
				if (oldIsContinuous) {
					auto maskOptions = torch::TensorOptions().dtype(torch::kUInt8).device(ppo->device);
					tOldActionMasks = torch::ones({ tOldObs.size(0), 1 }, maskOptions);
				} else {
					tOldActionMasks = torch::tensor(allOldActionMasks).reshape({ -1, oldNumActions }).to(ppo->device);
				}
			}

			torch::Tensor tActionMaps = {};
			if (!allActionMaps.empty())
				tActionMaps = torch::tensor(allActionMaps).reshape({ -1, numActions }).to(ppo->device);

			torch::Tensor tTeacherActionTargets = {};
			if (!allTeacherActionTargets.empty())
				tTeacherActionTargets = torch::tensor(allTeacherActionTargets).reshape({ -1, config.ppo.continuousActionSize }).to(ppo->device);

			torch::Tensor tTeacherActionTable = {};
			if (!allTeacherActionTable.empty())
				tTeacherActionTable = torch::tensor(allTeacherActionTable).reshape({ -1, oldNumActions, config.ppo.continuousActionSize }).to(ppo->device);

			// Transfer learn
			ppo->TransferLearn(
				oldModels, tNewObs, tOldObs,
				tNewActionMasks, tOldActionMasks,
				tActionMaps, tTeacherActionTable, tTeacherActionTargets,
				report, tlConfig
			);
			if (totalStudentRolloutCandidates > 0)
				report["Student Rollout Ratio"] = studentRolloutActionsUsed / (float)totalStudentRolloutCandidates;

			if (versionMgr)
				versionMgr->OnIteration(ppo, report, totalTimesteps, prevTimesteps);

			if (saveQueued || saveRequested.load()) {
				if (!config.checkpointFolder.empty())
					Save();
				saveRequested = false;
				if (saveQueued)
					break;
			}

			if (!config.checkpointFolder.empty()) {
				if (totalTimesteps / config.tsPerSave > prevTimesteps / config.tsPerSave) {
					// Auto-save
					Save();
				}
			}

			report.Finish();

			if (iterationCallback)
				iterationCallback(this, report);

			if (metricSender)
				metricSender->Send(report);

			report.Display(
				{
					"Transfer Learn Accuracy",
					"Transfer Learn Loss",
					"Transfer Learn Table Loss",
					"Transfer Learn MSE",
					"Transfer Learn MAE",
					"Transfer Learn NLL",
					"Transfer Learn Std Loss",
					"Transfer Learn Button BCE",
					"Teacher Confidence",
					"",
					"Policy Entropy",
					"Mean Policy Std",
					"Old Policy Entropy",
					"Aerial Target Ratio",
					"Student Rollout Ratio",
					"Policy Update Magnitude",
					"",
					"Collected Timesteps",
					"Total Timesteps",
					"Total Iterations"
				}
			);

			if (tlConfig.maxIterations > 0 && transferIterations >= tlConfig.maxIterations) {
				if (tlConfig.saveOnMaxIterations && !config.checkpointFolder.empty())
					Save();
				break;
			}
		}

	} catch (std::exception& e) {
		RG_ERR_CLOSE("Exception thrown during transfer learn loop: " << e.what());
	}
}

void GGL::Learner::Start() {

	bool render = config.renderMode;

	RG_LOG("Learner::Start():");
	RG_LOG("\tObs size: " << obsSize);
	RG_LOG("\tAction amount: " << numActions);

	if (render)
		RG_LOG("\t(Render mode enabled)");

	try {
		bool saveQueued;
		std::thread keyPressThread;
		StartQuitKeyThread(saveQueued, keyPressThread);

		ExperienceBuffer experience = ExperienceBuffer(config.randomSeed, torch::kCPU);

		int numPlayers = envSet->state.numPlayers;

		bool isContinuous = (config.ppo.policyType == PolicyType::CONTINUOUS);
		int contActionDim = config.ppo.continuousActionSize;

		struct Trajectory {
			FList states, nextStates, rewards, logProbs;
			std::vector<uint8_t> actionMasks;
			std::vector<int8_t> terminals;
			std::vector<int32_t> actions;        // Discrete actions
			FList continuousActions;              // Continuous actions (flat: N floats per step)
			int _contActionDim = 0;               // Continuous action dimension (for Length())

			void Clear() {
				int contActionDim = _contActionDim;
				*this = Trajectory();
				_contActionDim = contActionDim;
			}

			void Append(const Trajectory& other) {
				states += other.states;
				nextStates += other.nextStates;
				rewards += other.rewards;
				logProbs += other.logProbs;
				actionMasks += other.actionMasks;
				terminals += other.terminals;
				actions += other.actions;
				continuousActions += other.continuousActions;
				if (other._contActionDim > 0) _contActionDim = other._contActionDim;
			}

			size_t Length() const {
				if (_contActionDim > 0 && !continuousActions.empty())
					return continuousActions.size() / _contActionDim;
				return actions.size();
			}
		};

		auto trajectories = std::vector<Trajectory>(numPlayers, Trajectory{});
		for (auto& traj : trajectories)
			traj._contActionDim = contActionDim;
		int maxEpisodeLength = (int)(config.ppo.maxEpisodeDuration * (120.f / config.tickSkip));

		// Cached all-enabled action masks on the model device (used by the
		// CUDA obs fast path, where masks are constant).
		torch::Tensor devOnesMasks;

		// Master switch for the device-resident experience fast path
		// (set GGL_DEVICE_EXP=0 to fall back to the host experience path).
		const char* devExpEnv = getenv("GGL_DEVICE_EXP");
		bool deviceExpEnabled = !(devExpEnv && devExpEnv[0] == '0');
		const char* traceStartEnv = getenv("GGL_TRACE_START");
		bool traceStart = traceStartEnv && traceStartEnv[0] != '0';

		while (true) {
			if (traceStart)
				RG_LOG("[trace-start] PPO loop begin");
			Report report = {};

			bool isFirstIteration = (totalTimesteps == 0);

			// TODO: Old version switching messes up the gameplay potentially
			GGL::PolicyVersion* oldVersion = NULL;
			std::vector<bool> oldVersionPlayerMask;
			std::vector<int> newPlayerIndices = {}, oldPlayerIndices = {};
			torch::Tensor tNewPlayerIndices, tOldPlayerIndices;

			for (int i = 0; i < numPlayers; i++)
				newPlayerIndices.push_back(i);

			if (config.trainAgainstOldVersions) {
				if (traceStart)
					RG_LOG("[trace-start] checking old-version rollout");
				RG_ASSERT(config.trainAgainstOldChance >= 0 && config.trainAgainstOldChance <= 1);
				bool shouldTrainAgainstOld =
					(RocketSim::Math::RandFloat() < config.trainAgainstOldChance)
					&& !versionMgr->versions.empty()
					&& !render;

				if (shouldTrainAgainstOld) {
					if (traceStart)
						RG_LOG("[trace-start] old-version rollout selected");
					// Set up training against old versions

					int oldVersionIdx = RocketSim::Math::RandInt(0, versionMgr->versions.size());
					oldVersion = &versionMgr->versions[oldVersionIdx];

					Team oldVersionTeam = Team(RocketSim::Math::RandInt(0, 2)); 
					
					newPlayerIndices.clear();
					oldVersionPlayerMask.resize(numPlayers);
					int i = 0;
					for (auto& state : envSet->state.gameStates) {
						for (auto& player : state.players) {
							if (player.team == oldVersionTeam) {
								oldVersionPlayerMask[i] = true;
								oldPlayerIndices.push_back(i);
							} else {
								oldVersionPlayerMask[i] = false;
								newPlayerIndices.push_back(i);
							}
							i++;
						}
					}

					tNewPlayerIndices = torch::tensor(newPlayerIndices);
					tOldPlayerIndices = torch::tensor(oldPlayerIndices);
				}
			}

			int numRealPlayers = oldVersion ? newPlayerIndices.size() : envSet->state.numPlayers;

			int stepsCollected = 0;
			{ // Generate experience

				// Only contains complete episodes
				auto combinedTraj = Trajectory();
				combinedTraj._contActionDim = contActionDim;
				if (traceStart)
					RG_LOG("[trace-start] trajectory initialized");

				Timer collectionTimer = {};
				{ // Collect timesteps
					RG_NO_GRAD;

					float inferTime = 0;
					float envStepTime = 0;

					for (int step = 0; !stopRequested.load() && (combinedTraj.Length() < config.ppo.tsPerItr || render); step++, stepsCollected += numRealPlayers) {
						bool traceThisStep = traceStart && (step < 3 || (step % 25) == 0);
						if (traceThisStep)
							RG_LOG("[trace-start] step " << step << " before reset");
						Timer stepTimer = {};
						envSet->Reset();
						if (traceThisStep)
							RG_LOG("[trace-start] step " << step << " after reset");
						envStepTime += stepTimer.Elapsed();

						for (float f : envSet->state.obs.data)
							if (isnan(f) || isinf(f))
								RG_ERR_CLOSE("Obs builder produced a NaN/inf value");
						if (traceThisStep)
							RG_LOG("[trace-start] step " << step << " obs finite");

						if (!render && obsStat) {
							// TODO: This samples from old versions too
							int numSamples = RS_MAX(envSet->state.numPlayers, config.maxObsSamples);
							for (int i = 0; i < numSamples; i++) {
								int idx = Math::RandInt(0, envSet->state.numPlayers);
								obsStat->IncrementRow(&envSet->state.obs.At(idx, 0));
							}

							std::vector<double> mean = obsStat->GetMean();
							std::vector<double> std = obsStat->GetSTD();
							for (double& f : mean)
								f = RS_CLAMP(f, -config.maxObsMeanRange, config.maxObsMeanRange);
							for (double& f : std)
								f = RS_MAX(f, config.minObsSTD);
							for (int i = 0; i < envSet->state.numPlayers; i++) {
								for (int j = 0; j < obsSize; j++) {
									float& obsVal = envSet->state.obs.At(i, j);
									obsVal = (obsVal - mean[j]) / std[j];
								}
							}
						}

						torch::Tensor tActions, tLogProbs;
						torch::Tensor tStates, tActionMasks;
						torch::Tensor tdStatesDirect, tdMasksDirect;
						bool deviceObsDirect = false;
						if (traceThisStep)
							RG_LOG("[trace-start] step " << step << " before tensor state build");
#ifdef RG_ROCKETSIMCUDA_AVAILABLE
						// CUDA obs fast path + CUDA model: read obs (and masks)
						// straight from the EnvSet device buffers - no
						// GPU->CPU->GPU round trip before inference. The CPU obs
						// copy still exists for trajectories and the NaN scan.
						if (envSet->UsingCudaTrainingFastPath() && ppo->device.is_cuda() && !render && deviceExpEnabled) {
							deviceObsDirect = true;
							auto fOpts = torch::TensorOptions().dtype(torch::kFloat32).device(ppo->device);
							tdStatesDirect = torch::from_blob(
								envSet->GetCudaObsDevicePtr(),
								{ (int64_t)numPlayers, (int64_t)obsSize }, fOpts);
							auto mOpts = torch::TensorOptions().dtype(torch::kUInt8).device(ppo->device);
							tdMasksDirect = torch::from_blob(
								envSet->GetCudaActionMasksDevicePtr(),
								{ (int64_t)numPlayers, (int64_t)envSet->state.actionMasks.size[1] }, mOpts);
						}
#endif
						if (!deviceObsDirect) {
							tStates = DIMLIST2_TO_TENSOR<float>(envSet->state.obs);
							tActionMasks = DIMLIST2_TO_TENSOR<uint8_t>(envSet->state.actionMasks);
						}
						if (traceThisStep)
							RG_LOG("[trace-start] step " << step << " after tensor state build");

						if (!render) {
							for (int newPlayerIdx : newPlayerIndices) {
								trajectories[newPlayerIdx].states += envSet->state.obs.GetRow(newPlayerIdx);
								trajectories[newPlayerIdx].actionMasks += envSet->state.actionMasks.GetRow(newPlayerIdx);
							}
						}
						if (traceThisStep)
							RG_LOG("[trace-start] step " << step << " after trajectory state append");

						envSet->StepFirstHalf(true);
						if (traceThisStep)
							RG_LOG("[trace-start] step " << step << " after StepFirstHalf");

						Timer inferTimer = {};

						if (oldVersion) {
							torch::Tensor tdStates = deviceObsDirect ? tdStatesDirect : tStates.to(ppo->device, true);
							torch::Tensor tdActionMasks = deviceObsDirect ? tdMasksDirect : tActionMasks.to(ppo->device, true);

							torch::Tensor tdNewStates = tdStates.index_select(0, tNewPlayerIndices.to(ppo->device));
							torch::Tensor tdOldStates = tdStates.index_select(0, tOldPlayerIndices.to(ppo->device));
							torch::Tensor tdNewActionMasks = tdActionMasks.index_select(0, tNewPlayerIndices.to(ppo->device));
							torch::Tensor tdOldActionMasks = tdActionMasks.index_select(0, tOldPlayerIndices.to(ppo->device));

							torch::Tensor tNewActions;
							torch::Tensor tOldActions;

							ppo->InferActions(tdNewStates, tdNewActionMasks, &tNewActions, &tLogProbs);
							ppo->InferActions(tdOldStates, tdOldActionMasks, &tOldActions, NULL, &oldVersion->models);

							if (isContinuous) {
								tActions = torch::zeros({ numPlayers, contActionDim }, tNewActions.options().device(torch::kCPU));
							} else {
								tActions = torch::zeros({ numPlayers }, tNewActions.options().device(torch::kCPU));
							}
							tActions.index_copy_(0, tNewPlayerIndices, tNewActions.cpu());
							tActions.index_copy_(0, tOldPlayerIndices, tOldActions.cpu());
						} else {
							torch::Tensor tdStates = deviceObsDirect ? tdStatesDirect : tStates.to(ppo->device, true);
							torch::Tensor tdActionMasks = deviceObsDirect ? tdMasksDirect : tActionMasks.to(ppo->device, true);
							if (traceThisStep)
								RG_LOG("[trace-start] step " << step << " before InferActions");
							ppo->InferActions(tdStates, tdActionMasks, &tActions, &tLogProbs);
							if (traceThisStep)
								RG_LOG("[trace-start] step " << step << " after InferActions");
							tActions = tActions.cpu();
							if (traceThisStep)
								RG_LOG("[trace-start] step " << step << " after actions cpu");
						}
						inferTime += inferTimer.Elapsed();

						std::vector<int> curActionsDiscrete;
						FList curActionsContinuous;
						if (isContinuous) {
							curActionsContinuous = TENSOR_TO_VEC<float>(tActions.flatten());
						} else {
							curActionsDiscrete = TENSOR_TO_VEC<int>(tActions);
						}

						FList newLogProbs;
						if (tLogProbs.defined() && !render)
							newLogProbs = TENSOR_TO_VEC<float>(tLogProbs);	
						if (traceThisStep)
							RG_LOG("[trace-start] step " << step << " after tensor-to-vec");

						stepTimer.Reset();
						envSet->Sync(); // Make sure the first half is done
						if (traceThisStep)
							RG_LOG("[trace-start] step " << step << " after sync");
						if (isContinuous) {
							envSet->StepSecondHalfContinuous(curActionsContinuous, contActionDim, false);
						} else {
							envSet->StepSecondHalf(curActionsDiscrete, false);
						}
						if (traceThisStep)
							RG_LOG("[trace-start] step " << step << " after StepSecondHalf");
						envStepTime += stepTimer.Elapsed();

						if (stepCallback)
							stepCallback(this, envSet->state.gameStates, report);

						if (render) {
							renderSender->Send(envSet->state.gameStates[0]);
							continue;
						}

						// Calc average rewards
						if (config.addRewardsToMetrics && (Math::RandInt(0, config.rewardSampleRandInterval) == 0)) {
							int numSamples = RS_MIN(envSet->arenas.size(), config.maxRewardSamples);
							std::unordered_map<std::string, AvgTracker> avgRewards = {};
							for (int i = 0; i < numSamples; i++) {
								int arenaIdx = Math::RandInt(0, envSet->arenas.size());
								auto& prevRewards = envSet->state.lastRewards[i];

								for (int j = 0; j < envSet->rewards[arenaIdx].size(); j++) {
									std::string rewardName = envSet->rewards[arenaIdx][j].reward->GetName();
									avgRewards[rewardName] += prevRewards[j];
								}
							}

							for (auto& pair : avgRewards)
								report.AddAvg("Rewards/" + pair.first, pair.second.Get());
						}

						// Now that we've inferred and stepped the env, we can add that stuff to the trajectories
						int i = 0;
						for (int newPlayerIdx : newPlayerIndices) {
							if (isContinuous) {
								// Append N floats for this player
								int offset = newPlayerIdx * contActionDim;
								for (int d = 0; d < contActionDim; d++)
									trajectories[newPlayerIdx].continuousActions.push_back(curActionsContinuous[offset + d]);
							} else {
								trajectories[newPlayerIdx].actions.push_back(curActionsDiscrete[newPlayerIdx]);
							}
							trajectories[newPlayerIdx].rewards += envSet->state.rewards[newPlayerIdx];
							trajectories[newPlayerIdx].logProbs += newLogProbs[i];
							i++;
						}
						if (traceThisStep)
							RG_LOG("[trace-start] step " << step << " after reward/logprob append");

						auto curTerminals = std::vector<uint8_t>(numPlayers, 0);
						for (int idx = 0; idx < envSet->arenas.size(); idx++) {
							uint8_t terminalType = envSet->state.terminals[idx];
							if (!terminalType)
								continue;

							auto playerStartIdx = envSet->state.arenaPlayerStartIdx[idx];
							int playersInArena = envSet->state.gameStates[idx].players.size();
							for (int i = 0; i < playersInArena; i++)
								curTerminals[playerStartIdx + i] = terminalType;
						}
						if (traceThisStep)
							RG_LOG("[trace-start] step " << step << " after terminal scan");

						for (int newPlayerIdx : newPlayerIndices) {
							int8_t terminalType = curTerminals[newPlayerIdx];
							auto& traj = trajectories[newPlayerIdx];

							if (!terminalType && traj.Length() >= maxEpisodeLength) {
								// Episode is too long, truncate it here
								// This won't actually reset the env, but rather will just add it to experience buffer as truncated
								terminalType = RLGC::TerminalType::TRUNCATED;
							}

							traj.terminals.push_back(terminalType);
							if (terminalType) {

								if (terminalType == RLGC::TerminalType::TRUNCATED) {
									// Truncation requires an additional next state for the critic
									traj.nextStates += envSet->state.obs.GetRow(newPlayerIdx);
								}

								combinedTraj.Append(traj);
								traj.Clear();
							}
						}
						if (traceThisStep)
							RG_LOG("[trace-start] step " << step << " after terminal append");

						if (collectionProgressCallback && (step % 100) == 0)
							collectionProgressCallback(this, (int64_t)combinedTraj.Length(), (int64_t)config.ppo.tsPerItr);
					}

					report["Inference Time"] = inferTime;
					report["Env Step Time"] = envStepTime;
				}
				float collectionTime = collectionTimer.Elapsed();
				if (traceStart)
					RG_LOG("[trace-start] collection finished, combinedTraj.Length=" << combinedTraj.Length());

				if (stopRequested.load() && (render || combinedTraj.Length() == 0))
					break;

				Timer consumptionTimer = {};
				{ // Process timesteps
					RG_NO_GRAD;

					// Make and transpose tensors
					if (traceStart)
						RG_LOG("[trace-start] before tensor states");
					torch::Tensor tStates = torch::tensor(combinedTraj.states).reshape({ -1, obsSize });
					if (traceStart)
						RG_LOG("[trace-start] after tensor states");
					torch::Tensor tActionMasks;
					if (!isContinuous) {
						tActionMasks = torch::tensor(combinedTraj.actionMasks).reshape({ -1, numActions });
					} else {
						// For continuous, action masks are not used — create a dummy
						tActionMasks = torch::ones({ tStates.size(0), 1 }, torch::kUInt8);
					}
					if (traceStart)
						RG_LOG("[trace-start] after tensor action masks");
					torch::Tensor tActions;
					if (isContinuous) {
						if (traceStart)
							RG_LOG("[trace-start] before tensor continuous actions");
						tActions = torch::tensor(combinedTraj.continuousActions).reshape({ -1, contActionDim });
					} else {
						tActions = torch::tensor(combinedTraj.actions);
					}
					if (traceStart)
						RG_LOG("[trace-start] after tensor actions");
					if (traceStart)
						RG_LOG("[trace-start] before tensor logprobs");
					torch::Tensor tLogProbs = torch::tensor(combinedTraj.logProbs);
					if (traceStart)
						RG_LOG("[trace-start] after tensor logprobs");
					torch::Tensor tRewards = torch::tensor(combinedTraj.rewards);
					torch::Tensor tTerminals = torch::tensor(combinedTraj.terminals);
					if (traceStart)
						RG_LOG("[trace-start] after tensor rewards/terminals");

					// States we truncated at (there could be none)
					torch::Tensor tNextTruncStates;
					if (!combinedTraj.nextStates.empty())
						tNextTruncStates = torch::tensor(combinedTraj.nextStates).reshape({ -1, obsSize });
					if (traceStart)
						RG_LOG("[trace-start] after tensor next trunc states");

					if (ppo->device.is_cuda() && deviceExpEnabled) {
						if (traceStart)
							RG_LOG("[trace-start] before move experience to device");
						// Move the whole experience to the GPU once: critic
						// inference, GAE and the PPO minibatches then never
						// re-upload from host memory.
						tStates = tStates.to(ppo->device, true);
						tActionMasks = tActionMasks.to(ppo->device, true);
						tActions = tActions.to(ppo->device, true);
						tLogProbs = tLogProbs.to(ppo->device, true);
						tRewards = tRewards.to(ppo->device, true);
						tTerminals = tTerminals.to(ppo->device, true);
						if (tNextTruncStates.defined())
							tNextTruncStates = tNextTruncStates.to(ppo->device, true);
						if (traceStart)
							RG_LOG("[trace-start] after move experience to device");
					}

					report["Average Step Reward"] = tRewards.mean().item<float>();
					report["Collected Timesteps"] = stepsCollected;
					
					torch::Tensor tValPreds;
					torch::Tensor tTruncValPreds;

					if (ppo->device.is_cpu()) {
						// Predict values all at once
						tValPreds = ppo->InferCritic(tStates.to(ppo->device, true, true)).cpu();
						if (tNextTruncStates.defined())
							tTruncValPreds = ppo->InferCritic(tNextTruncStates.to(ppo->device, true, true)).cpu();
					} else {
						// Predict values using minibatching. The experience
						// already lives on the model device, so the slices and
						// outputs stay device-resident (no host round trips).
						tValPreds = torch::zeros({ (int64_t)combinedTraj.Length() },
							torch::TensorOptions().dtype(torch::kFloat32).device(tStates.device()));
						for (int i = 0; i < combinedTraj.Length(); i += ppo->config.miniBatchSize) {
							int start = i;
							int end = RS_MIN(i + ppo->config.miniBatchSize, combinedTraj.Length());
							torch::Tensor tStatesPart = tStates.slice(0, start, end);

							auto valPredsPart = ppo->InferCritic(tStatesPart.to(ppo->device, true));
							RG_ASSERT(valPredsPart.size(0) == (end - start));
							tValPreds.slice(0, start, end).copy_(valPredsPart, true);
						}

						if (tNextTruncStates.defined()) {
							// This really just should never happen
							// If this is ever actually a real problem in a legitimate use case, ping Zealan in the dead of night
							RG_ASSERT(tNextTruncStates.size(0) <= ppo->config.miniBatchSize);

							tTruncValPreds = ppo->InferCritic(tNextTruncStates.to(ppo->device, true));
							if (!tStates.is_cuda())
								tTruncValPreds = tTruncValPreds.cpu();
						}
					}

					report["Episode Length"] = 1.f / (tTerminals == 1).to(torch::kFloat32).mean().item<float>();

					Timer gaeTimer = {};
					// Run GAE
					torch::Tensor tAdvantages, tTargetVals, tReturns;
					float rewClipPortion;
					GAE::Compute(
						tRewards, tTerminals, tValPreds, tTruncValPreds,
						tAdvantages, tTargetVals, tReturns, rewClipPortion,
						config.ppo.gaeGamma, config.ppo.gaeLambda, returnStat ? returnStat->GetSTD() : 1, config.ppo.rewardClipRange
					);
					report["GAE Time"] = gaeTimer.Elapsed();
					report["Clipped Reward Portion"] = rewClipPortion;

					if (returnStat) {
						report["GAE/Returns STD"] = returnStat->GetSTD();

						int numToIncrement = RS_MIN(config.maxReturnSamples, tReturns.size(0));
						if (numToIncrement > 0) {
							auto selectedReturns = tReturns.index_select(0,
								torch::randint(tReturns.size(0), { (int64_t)numToIncrement }).to(tReturns.device()));
							returnStat->Increment(TENSOR_TO_VEC<float>(selectedReturns));
						}
					}
					report["GAE/Avg Return"] = tReturns.abs().mean().item<float>();
					report["GAE/Avg Advantage"] = tAdvantages.abs().mean().item<float>();
					report["GAE/Avg Val Target"] = tTargetVals.abs().mean().item<float>();

					// Set experience buffer
					experience.data.actions = tActions;
					experience.data.logProbs = tLogProbs;
					experience.data.actionMasks = tActionMasks;
					experience.data.states = tStates;
					experience.data.advantages = tAdvantages;
					experience.data.targetValues = tTargetVals;
				}

				// Free CUDA cache
#ifdef RG_CUDA_SUPPORT
				if (ppo->device.is_cuda())
					c10::cuda::CUDACachingAllocator::emptyCache();
#endif

				// Learn
				Timer learnTimer = {};
				ppo->Learn(experience, report, isFirstIteration);
				report["PPO Learn Time"] = learnTimer.Elapsed();

				// Set metrics
				float consumptionTime = consumptionTimer.Elapsed();
				report["Collection Time"] = collectionTime;
				report["Consumption Time"] = consumptionTime;
				report["Collection Steps/Second"] = stepsCollected / collectionTime;
				report["Consumption Steps/Second"] = stepsCollected / consumptionTime;
				report["Overall Steps/Second"] = stepsCollected / (collectionTime + consumptionTime);

				uint64_t prevTimesteps = totalTimesteps;
				totalTimesteps += stepsCollected;
				report["Total Timesteps"] = totalTimesteps;
				totalIterations++;
				report["Total Iterations"] = totalIterations;

				if (versionMgr)
					versionMgr->OnIteration(ppo, report, totalTimesteps, prevTimesteps);

				if (saveQueued || saveRequested.load()) {
					if (!config.checkpointFolder.empty())
						Save();
					saveRequested = false;
					if (saveQueued)
						break;
				}

				if (!config.checkpointFolder.empty()) {
					if (totalTimesteps / config.tsPerSave > prevTimesteps / config.tsPerSave) {
						// Auto-save
						Save();
					}
				}

				report.Finish();

				if (iterationCallback)
					iterationCallback(this, report);

				if (metricSender)
					metricSender->Send(report);

				report.Display(
					{
						"Average Step Reward",
						"Policy Entropy",
						"KL Div Loss",
						"First Accuracy",
						"",
						"Policy Update Magnitude",
						"Critic Update Magnitude",
						"Shared Head Update Magnitude",
						"",
						"Collection Steps/Second",
						"Consumption Steps/Second",
						"Overall Steps/Second",
						"",
						"Collection Time",
						"-Inference Time",
						"-Env Step Time",
						"Consumption Time",
						"-GAE Time",
						"-PPO Learn Time"
						"",
						"Collected Timesteps",
						"Total Timesteps",
						"Total Iterations"
					}
				);
			}
		}
		
	} catch (std::exception& e) {
		RG_ERR_CLOSE("Exception thrown during main learner loop: " << e.what());
	}
}

GGL::Learner::~Learner() {
	delete ppo;
	delete versionMgr;
	delete metricSender;
	delete renderSender;
	if (pyInterpreterInitialized)
		pybind11::finalize_interpreter();
}
