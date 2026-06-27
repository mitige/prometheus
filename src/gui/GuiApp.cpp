#include "GuiApp.h"

#include "imgui.h"
#include "imgui_internal.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"

#include <GLFW/glfw3.h>

#include <GigaLearnCPP/Learner.h>
#include <GigaLearnCPP/Util/Report.h>
#include <RLGymCPP/ActionParsers/DefaultAction.h>
#include <RLGymCPP/ActionParsers/DefaultContinuousAction.h>
#include <RLGymCPP/OBSBuilders/AdvancedObs.h>
#include <RLGymCPP/OBSBuilders/AdvancedObsPadded.h>
#include <RLGymCPP/OBSBuilders/DefaultObs.h>
#include <RLGymCPP/OBSBuilders/DefaultObsPadded.h>
#include <RLGymCPP/Rewards/CommonRewards.h>
#include <RLGymCPP/Rewards/ZeroSumReward.h>
#include <RLGymCPP/Rewards/mkh_rewards.h>
#include <RLGymCPP/Rewards/morerewards.h>
#include <RLGymCPP/StateSetters/CombinedState.h>
#include <RLGymCPP/StateSetters/KickoffState.h>
#include <RLGymCPP/StateSetters/RandomState.h>
#include <RLGymCPP/StateSetters/UnlimitedBoostState.h>
#include <RLGymCPP/TerminalConditions/GoalScoreCondition.h>
#include <RLGymCPP/TerminalConditions/NoTouchCondition.h>

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <unordered_map>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <Windows.h>
#define GLFW_EXPOSE_NATIVE_WIN32
#include <GLFW/glfw3native.h>
#include <windowsx.h>
#endif

using namespace GGL;
using namespace RLGC;

namespace prometheus {
namespace {

static std::mutex g_configMutex;
static GuiTrainingConfig g_trainingConfig;

static constexpr const char* kConfigPath = "prometheus_config.json";

static const char* kObsBuilders[] = {
	"AdvancedObsPadded",
	"AdvancedObs",
	"DefaultObsPadded",
	"DefaultObs",
};

static const char* kKnownRewards[] = {
	"TouchBallReward",
	"StrongTouchReward",
	"GoalReward",
	"AirReward",
	"PickupBoostReward",
	"KickoffProximityReward",
	"VelocityPlayerToBallReward",
	"VelocityBallToGoalReward",
	"GoalDirectedTouchSpeedReward",
	"Kickoff2v2SimpleReward",
	"SaveBoostReward",
	"BoostSpendPenaltyReward",
	"FreeGoalEventPenaltyReward",
	"TeamSpacingReward",
	"TeammateBumpPenaltyReward",
	"StagedFlickReward",
	"WavedashRecoveryReward",
	"AerialTouchHeightReward",
	"AirDribbleReward",
	"MKHKickoffSpeedflipReward",
	"WavedashKickoff50Reward",
	"IsBallVelocityOnTargetReward",
	"DribbleReward",
	"HyperNoStackReward",
	"MKH_Airdribble_reward"
};

static int ClampInt(int value, int minValue, int maxValue) {
	return std::max(minValue, std::min(value, maxValue));
}

static float ClampFloat(float value, float minValue, float maxValue) {
	return std::max(minValue, std::min(value, maxValue));
}

static std::string TrimCopy(const std::string& value) {
	size_t begin = 0;
	size_t end = value.size();
	while (begin < end && std::isspace((unsigned char)value[begin])) begin++;
	while (end > begin && std::isspace((unsigned char)value[end - 1])) end--;
	return value.substr(begin, end - begin);
}

static bool PathsEquivalentNoThrow(const std::filesystem::path& lhs, const std::filesystem::path& rhs) {
	try {
		return std::filesystem::weakly_canonical(lhs) == std::filesystem::weakly_canonical(rhs);
	} catch (...) {
		return lhs.lexically_normal() == rhs.lexically_normal();
	}
}

static std::vector<int> ParseLayerSizes(const std::string& text, const std::vector<int>& fallback) {
	std::vector<int> sizes;
	std::istringstream input(text);
	std::string token;
	while (std::getline(input, token, ',')) {
		token = TrimCopy(token);
		if (token.empty()) continue;
		try {
			int value = std::stoi(token);
			if (value > 0) sizes.push_back(std::max(32, value));
		} catch (...) {
		}
	}
	return sizes.empty() ? fallback : sizes;
}

static std::string LayerSizesToText(const std::vector<int>& sizes) {
	std::ostringstream output;
	for (size_t i = 0; i < sizes.size(); ++i) {
		if (i > 0) output << ", ";
		output << sizes[i];
	}
	return output.str();
}

static int MakeDivisibleMinibatch(int batchSize, int minibatchSize) {
	batchSize = std::max(1, batchSize);
	minibatchSize = ClampInt(minibatchSize, 1, batchSize);
	while (minibatchSize > 1 && (batchSize % minibatchSize) != 0)
		--minibatchSize;
	return minibatchSize;
}

static bool InputTextString(const char* label, std::string& value, ImGuiInputTextFlags flags = 0) {
	char buffer[512];
	std::snprintf(buffer, sizeof(buffer), "%s", value.c_str());
	if (ImGui::InputText(label, buffer, sizeof(buffer), flags)) {
		value = buffer;
		return true;
	}
	return false;
}

static bool ComboString(const char* label, std::string& value, const char* const* items, int itemCount) {
	int current = 0;
	for (int i = 0; i < itemCount; ++i) {
		if (value == items[i]) {
			current = i;
			break;
		}
	}
	if (ImGui::Combo(label, &current, items, itemCount)) {
		value = items[ClampInt(current, 0, itemCount - 1)];
		return true;
	}
	return false;
}

static ObsBuilder* CreateObsBuilderByName(const std::string& name, int tickSkip, int playersPerTeam) {
	if (name == "AdvancedObs") return new AdvancedObs();
	if (name == "DefaultObs") return new DefaultObs();
	if (name == "DefaultObsPadded") return new DefaultObsPadded(std::max(6, playersPerTeam * 2));
	return new AdvancedObsPadded();
}

static Reward* CreateRewardByName(const std::string& name) {
	if (name == "TouchBallReward") return new TouchBallReward();
	if (name == "StrongTouchReward") return new StrongTouchReward();
	if (name == "GoalReward") return new GoalReward(-0.8f);
	if (name == "AirReward") return new AirReward();
	if (name == "PickupBoostReward") return new PickupBoostReward();
	if (name == "KickoffProximityReward") return new KickoffProximityReward();
	if (name == "VelocityPlayerToBallReward") return new VelocityPlayerToBallReward();
	if (name == "VelocityBallToGoalReward") return new VelocityBallToGoalReward();
	if (name == "GoalDirectedTouchSpeedReward") return new GoalDirectedTouchSpeedReward();
	if (name == "Kickoff2v2SimpleReward") return new Kickoff2v2SimpleReward(1.10f, 1.0f, 200.0f, 50.0f, 80.0f, 750.0f, 1500.0f);
	if (name == "MKHKickoffSpeedflipReward") return new MKHKickoffSpeedflipReward(3.0f, 0.42f, 1250.0f, 0.32f, -0.5f, 0.28f, 1.0f);
	if (name == "WavedashKickoff50Reward") return new WavedashKickoff50Reward(3.0f, 0.6f, 1.0f, 1.2f, 0.35f, 0.75f, 0.45f);
	if (name == "IsBallVelocityOnTargetReward") return new IsBallVelocityOnTargetReward(0.3f, false);
	if (name == "SaveBoostReward") return new SaveBoostReward(0.5f);
	if (name == "BoostSpendPenaltyReward") return new BoostSpendPenaltyReward();
	if (name == "FreeGoalEventPenaltyReward") return new FreeGoalEventPenaltyReward(1.0f);
	if (name == "TeamSpacingReward") return new TeamSpacingReward(1400.0f);
	if (name == "TeammateBumpPenaltyReward") return new TeammateBumpPenaltyReward();
	if (name == "StagedFlickReward") return new StagedFlickReward();
	if (name == "WavedashRecoveryReward") return new WavedashRecoveryReward(500.0f);
	if (name == "AerialTouchHeightReward") return new AerialTouchHeightReward(140.0f, 1800.0f);
	if (name == "AirDribbleReward") return new AirDribbleReward(220.0f, 380.0f, 150.0f, 0.10f, true, false, 1.5f, 1.0f, 0.4f, 1.5f);
	if (name == "DribbleReward") return new DribbleReward();
	if (name == "HyperNoStackReward") return new HyperNoStackReward();
	if (name == "MKH_Airdribble_reward") return new MKH_Airdribble_reward();
	return nullptr;
}

static std::vector<RewardEntry> DefaultRewards() {
	return {
		{ "GoalReward", 150.0f, false, 0.0f },
		{ "TouchBallReward", 3.2f, true, 0.0f },
		{ "GoalDirectedTouchSpeedReward", 0.4f, true, 0.0f },
		{ "VelocityPlayerToBallReward", 0.28f, false, 0.0f },
		{ "VelocityBallToGoalReward", 5.0f, false, 0.0f },
		{ "Kickoff2v2SimpleReward", 0.72f, false, 0.0f },
		{ "MKHKickoffSpeedflipReward", 0.12f, false, 0.0f },
		{ "WavedashKickoff50Reward", 0.07f, false, 0.0f },
		{ "IsBallVelocityOnTargetReward", 0.28f, false, 0.0f },
		{ "PickupBoostReward", 20.0f, false, 0.0f },
		{ "SaveBoostReward", 0.2f, false, 0.0f },
		{ "BoostSpendPenaltyReward", 0.10f, false, 0.0f },
		{ "FreeGoalEventPenaltyReward", 0.42f, false, 0.0f },
		{ "TeamSpacingReward", 0.30f, false, 0.0f },
		{ "TeammateBumpPenaltyReward", 0.3f, false, 0.0f },
		{ "StagedFlickReward", 0.22f, false, 0.0f },
		{ "WavedashRecoveryReward", 0.12f, false, 0.0f },
		{ "AerialTouchHeightReward", 0.20f, true, 0.0f },
		{ "AirDribbleReward", 0.14f, false, 0.0f }
	};
}

static EnvCreateResult GuiEnvCreateFunc(int) {
	GuiTrainingConfig cfg;
	{
		std::lock_guard<std::mutex> lock(g_configMutex);
		cfg = g_trainingConfig;
	}

	std::vector<WeightedReward> rewards;
	for (const RewardEntry& entry : cfg.rewards) {
		Reward* reward = CreateRewardByName(entry.className);
		if (!reward) continue;
		if (entry.zeroSum)
			reward = new ZeroSumReward(reward, entry.teamSpirit);
		rewards.push_back({ reward, entry.weight });
	}
	if (rewards.empty()) {
		rewards = {
			{ new TouchBallReward(), 5.0f },
			{ new StrongTouchReward(), 20.0f },
			{ new GoalReward(-0.8f), 175.0f }
		};
	}

	std::vector<TerminalCondition*> terminalConditions = {
		new NoTouchCondition(10),
		new GoalScoreCondition()
	};

	auto arena = Arena::Create(GameMode::SOCCAR);
	const int playersPerTeam = ClampInt(cfg.playersPerTeam, 1, 2);
	for (int i = 0; i < playersPerTeam; ++i) {
		arena->AddCar(Team::BLUE);
		arena->AddCar(Team::ORANGE);
	}

	EnvCreateResult result = {};
	result.actionParser = new DefaultAction();
	result.continuousActionParser = new DefaultContinuousAction();
	result.obsBuilder = CreateObsBuilderByName(cfg.obsBuilder, cfg.tickSkip, cfg.playersPerTeam);
	result.stateSetter = new CombinedState({
		{ new KickoffState(), 0.85f },
		{ new RandomState(true, true, false), 0.05f },
		{ new UnlimitedBoostState(true, true, false), 0.10f }
	});
	result.terminalConditions = terminalConditions;
	result.rewards = rewards;
	result.arena = arena;
	return result;
}

static LearnerConfig BuildLearnerConfig(const GuiTrainingConfig& guiCfg) {
	LearnerConfig cfg = {};
	cfg.deviceType = guiCfg.useCuda ? LearnerDeviceType::GPU_CUDA : LearnerDeviceType::CPU;
	cfg.physicsBackend = guiCfg.useCuda ? EnvPhysicsBackend::ROCKETSIM_CUDA : EnvPhysicsBackend::ROCKETSIM_CPU;
	cfg.numGames = std::max(1, guiCfg.numArenas);
	cfg.tickSkip = ClampInt(guiCfg.tickSkip, 1, 32);
	cfg.actionDelay = cfg.tickSkip - 1;
	cfg.renderMode = guiCfg.renderMode;
	cfg.checkpointFolder = guiCfg.checkpointDir;
	cfg.tsPerSave = (int64_t)std::max(1, guiCfg.checkpointInterval) * std::max(1, guiCfg.stepsPerIteration);
	cfg.standardizeObs = guiCfg.standardizeObs;
	cfg.standardizeReturns = guiCfg.standardizeReturns;
	cfg.sendMetrics = guiCfg.sendMetrics;
	cfg.metricsProjectName = "prometheus";
	cfg.metricsGroupName = "continuous-v2-attention";
	cfg.metricsRunName = "Prometheus";
	cfg.addRewardsToMetrics = true;
	cfg.trainAgainstOldVersions = guiCfg.trainAgainstOldVersions && !guiCfg.transferLearning && !guiCfg.renderMode;
	cfg.trainAgainstOldChance = ClampFloat(guiCfg.trainAgainstOldChance, 0.0f, 1.0f);
	cfg.tsPerVersion = std::max<int64_t>(1, guiCfg.tsPerVersion);
	cfg.maxOldVersions = std::max(1, guiCfg.maxOldVersions);

	cfg.ppo.policyType = PolicyType::CONTINUOUS;
	cfg.ppo.continuousActionSize = ClampInt(guiCfg.continuousActionSize, 6, 16);
	cfg.ppo.varMin = ClampFloat(guiCfg.varMin, 0.001f, 10.0f);
	cfg.ppo.varMax = std::max(cfg.ppo.varMin, ClampFloat(guiCfg.varMax, 0.001f, 10.0f));
	cfg.ppo.tsPerItr = std::max(1, guiCfg.stepsPerIteration);
	cfg.ppo.batchSize = cfg.ppo.tsPerItr;
	cfg.ppo.miniBatchSize = MakeDivisibleMinibatch((int)cfg.ppo.batchSize, guiCfg.minibatchSize);
	cfg.ppo.epochs = ClampInt(guiCfg.ppoEpochs, 1, 16);
	cfg.ppo.entropyScale = std::max(0.0f, guiCfg.entropyScale);
	cfg.ppo.gaeGamma = ClampFloat(guiCfg.gamma, 0.0f, 0.99999f);
	cfg.ppo.gaeLambda = ClampFloat(guiCfg.lambda, 0.0f, 1.0f);
	cfg.ppo.policyLR = std::max(0.0f, guiCfg.policyLR);
	cfg.ppo.criticLR = std::max(0.0f, guiCfg.criticLR);
	cfg.ppo.clipRange = ClampFloat(guiCfg.clipRange, 0.0f, 1.0f);
	cfg.ppo.rewardClipRange = std::max(0.0f, guiCfg.rewardClipRange);
	cfg.ppo.deterministic = guiCfg.deterministic || guiCfg.renderMode;

	cfg.ppo.sharedHead.layerSizes = ParseLayerSizes(guiCfg.sharedHeadLayerSizes, { 1024, 1024, 1024 });
	cfg.ppo.policy.layerSizes = ParseLayerSizes(guiCfg.policyLayerSizes, { 1024, 1024, 1024 });
	cfg.ppo.critic.layerSizes = ParseLayerSizes(guiCfg.criticLayerSizes, { 1024, 1024, 1024 });

	auto optim = ModelOptimType::ADAM;
	cfg.ppo.policy.optimType = optim;
	cfg.ppo.critic.optimType = optim;
	cfg.ppo.sharedHead.optimType = optim;

	auto activation = ModelActivationType::LEAKY_RELU;
	cfg.ppo.policy.activationType = activation;
	cfg.ppo.critic.activationType = activation;
	cfg.ppo.sharedHead.activationType = activation;

	cfg.ppo.policy.addLayerNorm = true;
	cfg.ppo.critic.addLayerNorm = true;
	cfg.ppo.sharedHead.addLayerNorm = true;
	cfg.ppo.sharedHead.addOutputLayer = false;
	cfg.ppo.policy.addOutputLayer = true;
	cfg.ppo.critic.addOutputLayer = true;

	cfg.ppo.useAttentionHead = guiCfg.useAttentionHead;
	if (guiCfg.useAttentionHead) {
		auto* attention = new AttentionModelConfig();
		attention->numDims = ClampInt(guiCfg.attentionDims, 16, 4096);
		attention->thinkDims = ClampInt(guiCfg.attentionThinkDims, 32, 8192);
		attention->numBlocks = ClampInt(guiCfg.attentionBlocks, 1, 16);
		attention->numHeads = ClampInt(guiCfg.attentionHeads, 1, 32);
		attention->preprocessLayers = ClampInt(guiCfg.attentionPreprocessLayers, 0, 8);
		attention->postprocessLayers = ClampInt(guiCfg.attentionPostprocessLayers, 0, 8);
		attention->refinementFeedforward = std::max(0, guiCfg.attentionRefinementFeedforward);
		attention->activationType = activation;
		attention->optimType = optim;
		cfg.ppo.attentionHeadConfig = attention;
	}

	return cfg;
}

static TransferLearnConfig BuildTransferLearnConfig(const GuiTrainingConfig& guiCfg) {
	TransferLearnConfig cfg = {};
	cfg.oldPolicyType = guiCfg.transferOldContinuousPolicy ? PolicyType::CONTINUOUS : PolicyType::DISCRETE;
	cfg.oldContinuousActionSize = ClampInt(guiCfg.transferOldContinuousActionSize, 1, 32);

	std::string oldObsBuilder = guiCfg.transferOldObsBuilder;
	int tickSkip = guiCfg.tickSkip;
	int playersPerTeam = guiCfg.playersPerTeam;
	cfg.makeOldObsFn = [oldObsBuilder, tickSkip, playersPerTeam]() {
		return CreateObsBuilderByName(oldObsBuilder, tickSkip, playersPerTeam);
	};
	cfg.makeOldActFn = []() {
		return new DefaultAction();
	};

	cfg.oldModelsPath = guiCfg.transferOldModelsPath;
	cfg.lr = std::max(0.0f, guiCfg.transferLR);
	cfg.batchSize = std::max(1, guiCfg.transferBatchSize);
	cfg.epochs = ClampInt(guiCfg.transferEpochs, 1, 64);
	cfg.maxIterations = guiCfg.transferMaxIterations;
	cfg.saveOnMaxIterations = guiCfg.transferSaveOnMaxIterations;
	cfg.useKLDiv = guiCfg.transferUseKLDiv;
	cfg.lossScale = std::max(0.0f, guiCfg.transferLossScale);
	cfg.lossExponent = std::max(0.0f, guiCfg.transferLossExponent);
	cfg.continuousActionTableTopK = std::max(0, guiCfg.transferContinuousActionTableTopK);
	cfg.continuousActionTableLossWeight = std::max(0.0f, guiCfg.transferContinuousActionTableLossWeight);
	cfg.continuousActionMSEWeight = std::max(0.0f, guiCfg.transferContinuousActionMSEWeight);
	cfg.continuousActionNLLWeight = std::max(0.0f, guiCfg.transferContinuousActionNLLWeight);
	cfg.continuousStdLossWeight = std::max(0.0f, guiCfg.transferContinuousStdLossWeight);
	cfg.continuousTargetStd = std::max(0.001f, guiCfg.transferContinuousTargetStd);
	cfg.continuousButtonBCELossWeight = std::max(0.0f, guiCfg.transferContinuousButtonBCELossWeight);
	cfg.continuousAerialSampleWeight = std::max(0.0f, guiCfg.transferContinuousAerialSampleWeight);
	cfg.discreteToContinuousStudentRolloutProb = ClampFloat(guiCfg.transferStudentRolloutProb, 0.0f, 1.0f);
	cfg.discreteToContinuousStudentRolloutWarmupTimesteps =
		std::max<int64_t>(0, guiCfg.transferStudentRolloutWarmupTimesteps);

	auto optim = ModelOptimType::ADAM;
	auto activation = ModelActivationType::LEAKY_RELU;
	cfg.oldPolicyConfig.layerSizes = ParseLayerSizes(guiCfg.transferOldPolicyLayerSizes, { 1024, 1024, 1024 });
	cfg.oldPolicyConfig.activationType = activation;
	cfg.oldPolicyConfig.optimType = optim;
	cfg.oldPolicyConfig.addLayerNorm = guiCfg.transferOldPolicyLayerNorm;
	cfg.oldPolicyConfig.addOutputLayer = guiCfg.transferOldPolicyOutputLayer;

	cfg.oldSharedHeadConfig.layerSizes = ParseLayerSizes(guiCfg.transferOldSharedHeadLayerSizes, {});
	cfg.oldSharedHeadConfig.activationType = activation;
	cfg.oldSharedHeadConfig.optimType = optim;
	cfg.oldSharedHeadConfig.addLayerNorm = guiCfg.transferOldPolicyLayerNorm;
	cfg.oldSharedHeadConfig.addOutputLayer = false;

	cfg.oldUseAttentionHead = guiCfg.transferOldUseAttentionHead;
	if (guiCfg.transferOldUseAttentionHead) {
		auto* attention = new AttentionModelConfig();
		attention->numDims = ClampInt(guiCfg.transferOldAttentionDims, 16, 4096);
		attention->thinkDims = ClampInt(guiCfg.transferOldAttentionThinkDims, 32, 8192);
		attention->numBlocks = ClampInt(guiCfg.transferOldAttentionBlocks, 1, 16);
		attention->numHeads = ClampInt(guiCfg.transferOldAttentionHeads, 1, 32);
		attention->preprocessLayers = ClampInt(guiCfg.transferOldAttentionPreprocessLayers, 0, 8);
		attention->postprocessLayers = ClampInt(guiCfg.transferOldAttentionPostprocessLayers, 0, 8);
		attention->refinementFeedforward = std::max(0, guiCfg.transferOldAttentionRefinementFeedforward);
		attention->activationType = activation;
		attention->optimType = optim;
		cfg.oldAttentionHeadConfig = attention;
	}

	return cfg;
}

namespace Colors {
	static constexpr ImU32 Gold = IM_COL32(255, 215, 0, 255);
	static ImVec4 V(int r, int g, int b, int a = 255) {
		return ImVec4(r / 255.0f, g / 255.0f, b / 255.0f, a / 255.0f);
	}
}

namespace Draw {
	enum Font { BODY = 0, SEMI_BOLD = 1, HERO = 2, CAPTION = 3 };

	static bool HasFont(int idx) {
		return idx < ImGui::GetIO().Fonts->Fonts.Size;
	}

	static void PushFont(int idx) {
		if (HasFont(idx)) ImGui::PushFont(ImGui::GetIO().Fonts->Fonts[idx]);
		else ImGui::PushFont(nullptr);
	}

	static void GoldAccentBar(float width, float thickness = 2.0f, float alpha = 0.9f) {
		ImDrawList* dl = ImGui::GetWindowDrawList();
		ImVec2 p = ImGui::GetCursorScreenPos();
		ImU32 colL = IM_COL32(255, 200, 0, (int)(alpha * 255));
		ImU32 colR = IM_COL32(255, 200, 0, 0);
		dl->AddRectFilledMultiColor(p, ImVec2(p.x + width, p.y + thickness), colL, colR, colR, colL);
		ImGui::Dummy(ImVec2(width, thickness + 2.0f));
	}

	static void PulsingDot(ImVec4 color, float radius = 5.0f) {
		ImDrawList* dl = ImGui::GetWindowDrawList();
		float pulse = 0.6f + 0.4f * std::sinf((float)ImGui::GetTime() * 3.0f);
		ImVec2 p = ImGui::GetCursorScreenPos();
		p.x += radius + 1.0f;
		p.y += ImGui::GetTextLineHeight() * 0.5f;
		ImU32 glow = IM_COL32((int)(color.x * 255), (int)(color.y * 255), (int)(color.z * 255), (int)(40 * pulse));
		ImU32 core = IM_COL32((int)(color.x * 255), (int)(color.y * 255), (int)(color.z * 255), (int)(255 * pulse));
		dl->AddCircleFilled(p, radius * 2.0f * pulse, glow, 24);
		dl->AddCircleFilled(p, radius, core, 24);
		ImGui::Dummy(ImVec2(radius * 2.0f + 4.0f, ImGui::GetTextLineHeight()));
	}

	static void GradientProgressBar(float fraction, ImVec2 size, ImU32 color, const char* overlay = nullptr) {
		ImDrawList* dl = ImGui::GetWindowDrawList();
		ImVec2 p = ImGui::GetCursorScreenPos();
		ImVec2 pMax = ImVec2(p.x + size.x, p.y + size.y);
		float rounding = size.y * 0.5f;
		fraction = ClampFloat(fraction, 0.0f, 1.0f);

		dl->AddRectFilled(p, pMax, IM_COL32(30, 30, 32, 255), rounding);
		float fillW = size.x * fraction;
		if (fillW > 2.0f) {
			float drawW = std::max(fillW, rounding * 2.0f);
			dl->PushClipRect(p, ImVec2(p.x + fillW, pMax.y), true);
			dl->AddRectFilled(p, ImVec2(p.x + drawW, pMax.y), color, rounding);
			dl->AddRectFilled(p, ImVec2(p.x + drawW, p.y + size.y * 0.45f), IM_COL32(255, 255, 255, 30), rounding);
			dl->PopClipRect();
		}
		dl->AddRect(p, pMax, IM_COL32(60, 52, 22, 70), rounding, 0, 1.0f);

		if (overlay && overlay[0]) {
			ImVec2 textSize = ImGui::CalcTextSize(overlay);
			ImVec2 textPos(p.x + (size.x - textSize.x) * 0.5f, p.y + (size.y - textSize.y) * 0.5f);
			dl->AddText(ImVec2(textPos.x + 1.0f, textPos.y + 1.0f), IM_COL32(0, 0, 0, 160), overlay);
			dl->AddText(textPos, IM_COL32(240, 238, 230, 240), overlay);
		}
		ImGui::Dummy(size);
	}

	static void PanelTitle(const char* title, float availWidth = -1.0f) {
		if (availWidth < 0) availWidth = ImGui::GetContentRegionAvail().x;
		PushFont(Font::SEMI_BOLD);
		ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.94f, 0.88f, 0.65f, 1.0f));
		ImGui::TextUnformatted(title);
		ImGui::PopStyleColor();
		ImGui::PopFont();
		GoldAccentBar(availWidth * 0.4f, 2.0f, 0.7f);
	}

	static void SectionHeader(const char* title) {
		ImGui::Spacing();
		PushFont(Font::SEMI_BOLD);
		ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.82f, 0.74f, 0.45f, 1.0f));
		ImGui::TextUnformatted(title);
		ImGui::PopStyleColor();
		ImGui::PopFont();
		ImDrawList* dl = ImGui::GetWindowDrawList();
		ImVec2 p = ImGui::GetCursorScreenPos();
		float width = ImGui::GetContentRegionAvail().x;
		dl->AddRectFilled(p, ImVec2(p.x + width, p.y + 1.0f), IM_COL32(55, 48, 20, 80));
		ImGui::Dummy(ImVec2(width, 3.0f));
	}
}

struct LockedPanelLayout {
	ImVec2 pos;
	ImVec2 size;
};

struct LockedGuiLayout {
	LockedPanelLayout topBar;
	LockedPanelLayout metrics;
	LockedPanelLayout config;
	LockedPanelLayout gpu;
	LockedPanelLayout rewards;
	LockedPanelLayout logs;
};

static float ClampLayoutValue(float value, float minValue, float maxValue) {
	return std::max(minValue, std::min(value, maxValue));
}

static LockedGuiLayout BuildLockedGuiLayout(const ImGuiViewport* vp) {
	const float gap = 10.0f;
	const float availableWidth = std::max(640.0f, vp->WorkSize.x);
	const float availableHeight = std::max(520.0f, vp->WorkSize.y);
	const float originX = vp->WorkPos.x;
	const float originY = vp->WorkPos.y;

	const float topHeightMax = availableWidth < 1100.0f ? 116.0f : 108.0f;
	const float topHeight = ClampLayoutValue(availableHeight * 0.115f, 96.0f, topHeightMax);
	const float mainHeight = std::max(260.0f, availableHeight - topHeight - gap);
	float logsHeight = ClampLayoutValue(mainHeight * 0.30f, 180.0f, 320.0f);
	float bodyHeight = mainHeight - gap - logsHeight;
	if (bodyHeight < 180.0f) {
		logsHeight = std::max(140.0f, logsHeight - (180.0f - bodyHeight));
		bodyHeight = mainHeight - gap - logsHeight;
	}

	const float bodyY = originY + topHeight + gap;
	const float logsY = bodyY + bodyHeight + gap;

	LockedGuiLayout layout = {};
	layout.topBar = { ImVec2(originX, originY), ImVec2(availableWidth, topHeight) };
	layout.logs = { ImVec2(originX, logsY), ImVec2(availableWidth, logsHeight) };

	if (availableWidth >= 1320.0f) {
		float configWidth = ClampLayoutValue(availableWidth * 0.28f, 320.0f, 430.0f);
		float gpuWidth = ClampLayoutValue(availableWidth * 0.18f, 250.0f, 320.0f);
		float metricsWidth = availableWidth - configWidth - gpuWidth - gap * 2.0f;
		float gpuHeight = std::max(120.0f, (bodyHeight - gap) * 0.5f);
		float rewardsHeight = bodyHeight - gpuHeight - gap;
		float rightX = originX + configWidth + gap + metricsWidth + gap;
		layout.config = { ImVec2(originX, bodyY), ImVec2(configWidth, bodyHeight) };
		layout.metrics = { ImVec2(originX + configWidth + gap, bodyY), ImVec2(metricsWidth, bodyHeight) };
		layout.gpu = { ImVec2(rightX, bodyY), ImVec2(gpuWidth, gpuHeight) };
		layout.rewards = { ImVec2(rightX, bodyY + gpuHeight + gap), ImVec2(gpuWidth, rewardsHeight) };
	} else if (availableWidth >= 980.0f) {
		float configWidth = ClampLayoutValue(availableWidth * 0.36f, 300.0f, 420.0f);
		float rightWidth = availableWidth - configWidth - gap;
		float gpuHeight = ClampLayoutValue(bodyHeight * 0.34f, 150.0f, 220.0f);
		float metricsHeight = bodyHeight - gpuHeight - gap;
		float gpuH = std::max(80.0f, (gpuHeight - gap) * 0.5f);
		float rewardsH = gpuHeight - gpuH - gap;
		float rightX = originX + configWidth + gap;
		float gpuY = bodyY + metricsHeight + gap;
		layout.config = { ImVec2(originX, bodyY), ImVec2(configWidth, bodyHeight) };
		layout.metrics = { ImVec2(rightX, bodyY), ImVec2(rightWidth, metricsHeight) };
		layout.gpu = { ImVec2(rightX, gpuY), ImVec2(rightWidth, gpuH) };
		layout.rewards = { ImVec2(rightX, gpuY + gpuH + gap), ImVec2(rightWidth, rewardsH) };
	} else {
		float configHeight = ClampLayoutValue(bodyHeight * 0.34f, 150.0f, 230.0f);
		float gpuHeight = ClampLayoutValue(bodyHeight * 0.20f, 110.0f, 160.0f);
		float metricsHeight = bodyHeight - configHeight - gpuHeight - gap * 2.0f;
		float gpuH = std::max(60.0f, (gpuHeight - gap) * 0.5f);
		float rewardsH = gpuHeight - gpuH - gap;
		float gpuY = bodyY + metricsHeight + gap + configHeight + gap;
		layout.metrics = { ImVec2(originX, bodyY), ImVec2(availableWidth, metricsHeight) };
		layout.config = { ImVec2(originX, bodyY + metricsHeight + gap), ImVec2(availableWidth, configHeight) };
		layout.gpu = { ImVec2(originX, gpuY), ImVec2(availableWidth, gpuH) };
		layout.rewards = { ImVec2(originX, gpuY + gpuH + gap), ImVec2(availableWidth, rewardsH) };
	}

	return layout;
}

static void ApplyLockedPanelLayout(const LockedPanelLayout& panel) {
	ImGui::SetNextWindowPos(panel.pos, ImGuiCond_Always);
	ImGui::SetNextWindowSize(panel.size, ImGuiCond_Always);
}

static ImGuiWindowFlags GetLockedPanelFlags() {
	return ImGuiWindowFlags_NoCollapse |
		ImGuiWindowFlags_NoMove |
		ImGuiWindowFlags_NoResize |
		ImGuiWindowFlags_NoDocking |
		ImGuiWindowFlags_NoSavedSettings;
}

static bool BeginPanel(const char* name, const LockedPanelLayout& panel, ImGuiWindowFlags extraFlags = 0) {
	ApplyLockedPanelLayout(panel);
	return ImGui::Begin(name, nullptr, GetLockedPanelFlags() | ImGuiWindowFlags_NoTitleBar | extraFlags);
}

#ifdef _WIN32
static WNDPROC g_origWndProc = nullptr;
static constexpr int g_resizeBorderWidth = 6;
static constexpr int g_titleBarHeight = 32;

static void ToggleWindowMaximize(HWND hWnd) {
	WINDOWPLACEMENT wp = { sizeof(wp) };
	GetWindowPlacement(hWnd, &wp);
	if (wp.showCmd == SW_MAXIMIZE || IsZoomed(hWnd))
		ShowWindow(hWnd, SW_RESTORE);
	else
		ShowWindow(hWnd, SW_MAXIMIZE);
}

static void ToggleWindowMaximize(GLFWwindow* window) {
	if (window) ToggleWindowMaximize(glfwGetWin32Window(window));
}

static LRESULT CALLBACK BorderlessResizeProc(HWND hWnd, UINT msg, WPARAM wParam, LPARAM lParam) {
	if (msg == WM_NCCALCSIZE && wParam == TRUE)
		return 0;

	if (msg == WM_LBUTTONDBLCLK) {
		POINT pt = { GET_X_LPARAM(lParam), GET_Y_LPARAM(lParam) };
		if (pt.y < g_titleBarHeight) {
			ToggleWindowMaximize(hWnd);
			return 0;
		}
	}

	if (msg == WM_NCHITTEST) {
		RECT rc;
		GetWindowRect(hWnd, &rc);
		int x = GET_X_LPARAM(lParam);
		int y = GET_Y_LPARAM(lParam);
		int bw = g_resizeBorderWidth;
		bool left = (x >= rc.left && x < rc.left + bw);
		bool right = (x < rc.right && x >= rc.right - bw);
		bool top = (y >= rc.top && y < rc.top + bw);
		bool bottom = (y < rc.bottom && y >= rc.bottom - bw);
		if (top && left) return HTTOPLEFT;
		if (top && right) return HTTOPRIGHT;
		if (bottom && left) return HTBOTTOMLEFT;
		if (bottom && right) return HTBOTTOMRIGHT;
		if (left) return HTLEFT;
		if (right) return HTRIGHT;
		if (top) return HTTOP;
		if (bottom) return HTBOTTOM;
	}

	if (msg == WM_GETMINMAXINFO) {
		auto* mmi = reinterpret_cast<MINMAXINFO*>(lParam);
		HMONITOR monitor = MonitorFromWindow(hWnd, MONITOR_DEFAULTTONEAREST);
		MONITORINFO mi = { sizeof(mi) };
		if (GetMonitorInfo(monitor, &mi)) {
			const RECT& work = mi.rcWork;
			const RECT& monitorRect = mi.rcMonitor;
			mmi->ptMaxPosition.x = work.left - monitorRect.left;
			mmi->ptMaxPosition.y = work.top - monitorRect.top;
			mmi->ptMaxSize.x = work.right - work.left;
			mmi->ptMaxSize.y = work.bottom - work.top;
		}
		mmi->ptMinTrackSize.x = 800;
		mmi->ptMinTrackSize.y = 520;
		return 0;
	}

	return CallWindowProc(g_origWndProc, hWnd, msg, wParam, lParam);
}

static void InstallBorderlessResize(GLFWwindow* window) {
	HWND hWnd = glfwGetWin32Window(window);
	g_origWndProc = (WNDPROC)SetWindowLongPtr(hWnd, GWLP_WNDPROC, (LONG_PTR)BorderlessResizeProc);
	LONG style = GetWindowLong(hWnd, GWL_STYLE);
	style &= ~WS_CAPTION;
	style |= WS_THICKFRAME | WS_MINIMIZEBOX | WS_MAXIMIZEBOX;
	SetWindowLong(hWnd, GWL_STYLE, style);
	SetWindowPos(hWnd, nullptr, 0, 0, 0, 0,
		SWP_NOMOVE | SWP_NOSIZE | SWP_NOZORDER | SWP_NOACTIVATE | SWP_FRAMECHANGED);
}
#endif

static void HandleStandaloneTitleBarDrag(GLFWwindow* window, bool canDrag, bool canDoubleClick = false) {
#ifdef _WIN32
	static bool dragging = false;
	static bool pendingDrag = false;
	static POINT dragStartCursor = {};
	static int dragStartWindowX = 0;
	static int dragStartWindowY = 0;
	static GLFWwindow* dragWindow = nullptr;

	if (!window) return;
	bool isMaximized = glfwGetWindowAttrib(window, GLFW_MAXIMIZED) == GLFW_TRUE;

	if (canDoubleClick && ImGui::IsMouseDoubleClicked(ImGuiMouseButton_Left)) {
		ToggleWindowMaximize(window);
		dragging = false;
		pendingDrag = false;
		dragWindow = nullptr;
		return;
	}

	if (!ImGui::IsMouseDown(ImGuiMouseButton_Left)) {
		dragging = false;
		pendingDrag = false;
		dragWindow = nullptr;
		return;
	}

	if (canDrag && ImGui::IsMouseClicked(ImGuiMouseButton_Left)) {
		GetCursorPos(&dragStartCursor);
		glfwGetWindowPos(window, &dragStartWindowX, &dragStartWindowY);
		pendingDrag = true;
		dragging = false;
		dragWindow = window;
	}

	if (pendingDrag && dragWindow == window && ImGui::IsMouseDragging(ImGuiMouseButton_Left, 4.0f)) {
		if (isMaximized) {
			int prevW, prevH;
			glfwGetWindowSize(window, &prevW, &prevH);
			glfwRestoreWindow(window);
			int newW, newH;
			glfwGetWindowSize(window, &newW, &newH);
			POINT cursor;
			GetCursorPos(&cursor);
			float ratio = (prevW > 0) ? (float)(cursor.x) / (float)prevW : 0.5f;
			int newX = cursor.x - (int)(ratio * newW);
			int newY = cursor.y - 16;
			glfwSetWindowPos(window, newX, newY);
			dragStartCursor = cursor;
			dragStartWindowX = newX;
			dragStartWindowY = newY;
		} else {
			GetCursorPos(&dragStartCursor);
			glfwGetWindowPos(window, &dragStartWindowX, &dragStartWindowY);
		}
		dragging = true;
		pendingDrag = false;
		dragWindow = window;
	}

	if (dragging && dragWindow == window) {
		POINT cursorPos = {};
		GetCursorPos(&cursorPos);
		glfwSetWindowPos(window,
			dragStartWindowX + (cursorPos.x - dragStartCursor.x),
			dragStartWindowY + (cursorPos.y - dragStartCursor.y));
	}
#else
	(void)window;
	(void)canDrag;
	(void)canDoubleClick;
#endif
}

static void RenderStandaloneWindowControls(GLFWwindow* window, float titleBarHeight) {
	if (!window) return;

	ImGuiStyle& style = ImGui::GetStyle();
	const float buttonSize = ImGui::GetFontSize();
	const float spacing = buttonSize + 6.0f;
	const float baseX = ImGui::GetWindowPos().x + ImGui::GetWindowContentRegionMax().x - style.FramePadding.y - buttonSize;
	const float posY = ImGui::GetWindowPos().y + std::max(0.0f, std::floor((titleBarHeight - buttonSize) * 0.5f));

	if (ImGui::CloseButton(ImGui::GetID("##standalone_close"), ImVec2(baseX, posY)))
		glfwSetWindowShouldClose(window, GLFW_TRUE);

#ifdef _WIN32
	{
		float maxX = baseX - spacing;
		ImVec2 bMin(maxX, posY);
		ImVec2 bMax(maxX + buttonSize, posY + buttonSize);
		bool hovered = ImGui::IsMouseHoveringRect(bMin, bMax);
		ImU32 col = hovered ? IM_COL32(255, 255, 255, 180) : IM_COL32(180, 180, 180, 120);
		ImDrawList* dl = ImGui::GetWindowDrawList();
		bool isMaximized = glfwGetWindowAttrib(window, GLFW_MAXIMIZED) == GLFW_TRUE;
		float s = buttonSize * 0.32f;
		ImVec2 c(maxX + buttonSize * 0.5f, posY + buttonSize * 0.5f);
		if (isMaximized) {
			float off = buttonSize * 0.15f;
			dl->AddRect(ImVec2(c.x - s + off, c.y - s - off), ImVec2(c.x + s + off, c.y + s - off), col, 0, 0, 1.2f);
			dl->AddRect(ImVec2(c.x - s, c.y - s), ImVec2(c.x + s, c.y + s), col, 0, 0, 1.2f);
		} else {
			dl->AddRect(ImVec2(c.x - s, c.y - s), ImVec2(c.x + s, c.y + s), col, 0, 0, 1.2f);
		}
		if (hovered && ImGui::IsMouseClicked(ImGuiMouseButton_Left))
			ToggleWindowMaximize(window);
	}

	{
		float minX = baseX - spacing * 2.0f;
		ImVec2 bMin(minX, posY);
		ImVec2 bMax(minX + buttonSize, posY + buttonSize);
		bool hovered = ImGui::IsMouseHoveringRect(bMin, bMax);
		ImU32 col = hovered ? IM_COL32(255, 255, 255, 180) : IM_COL32(180, 180, 180, 120);
		ImDrawList* dl = ImGui::GetWindowDrawList();
		float cy = posY + buttonSize * 0.5f;
		float s = buttonSize * 0.32f;
		dl->AddLine(ImVec2(minX + buttonSize * 0.5f - s, cy), ImVec2(minX + buttonSize * 0.5f + s, cy), col, 1.2f);
		if (hovered && ImGui::IsMouseClicked(ImGuiMouseButton_Left))
			glfwIconifyWindow(window);
	}
#endif
}

static void GlfwErrorCallback(int error, const char* desc) {
	std::fprintf(stderr, "GLFW Error %d: %s\n", error, desc);
}

static void RenderMetricCard(const char* label, const char* value, ImVec4 accent) {
	ImGui::BeginChild(label, ImVec2(0, 58), true, ImGuiWindowFlags_NoScrollbar);
	Draw::PushFont(Draw::Font::CAPTION);
	ImGui::TextDisabled("%s", label);
	ImGui::PopFont();
	Draw::PushFont(Draw::Font::HERO);
	ImGui::PushStyleColor(ImGuiCol_Text, accent);
	ImGui::TextUnformatted(value);
	ImGui::PopStyleColor();
	ImGui::PopFont();
	ImGui::EndChild();
}

} // namespace

void MetricSeries::Push(float value) {
	values.push_back(value);
	if (values.size() > maxSize)
		values.pop_front();
	minVal = std::min(minVal, value);
	maxVal = std::max(maxVal, value);
}

void MetricSeries::Clear() {
	values.clear();
	minVal = 1e30f;
	maxVal = -1e30f;
}

void TrainingState::PushLog(LogEntry::Level level, const std::string& message) {
	std::lock_guard<std::mutex> lock(mtx);
	auto now = std::chrono::system_clock::now();
	std::time_t time = std::chrono::system_clock::to_time_t(now);
	char buffer[32];
	std::tm tmValue = {};
#ifdef _WIN32
	localtime_s(&tmValue, &time);
#else
	localtime_r(&time, &tmValue);
#endif
	std::strftime(buffer, sizeof(buffer), "%H:%M:%S", &tmValue);
	logs.push_back({ level, buffer, message });
	if (logs.size() > maxLogs)
		logs.pop_front();
}

void GuiTrainingConfig::SaveToFile(const std::string& path) const {
	nlohmann::json j;
	j["numArenas"] = numArenas;
	j["playersPerTeam"] = playersPerTeam;
	j["tickSkip"] = tickSkip;
	j["stepsPerIteration"] = stepsPerIteration;
	j["minibatchSize"] = minibatchSize;
	j["ppoEpochs"] = ppoEpochs;
	j["entropyScale"] = entropyScale;
	j["gamma"] = gamma;
	j["lambda"] = lambda;
	j["policyLR"] = policyLR;
	j["criticLR"] = criticLR;
	j["clipRange"] = clipRange;
	j["rewardClipRange"] = rewardClipRange;
	j["varMin"] = varMin;
	j["varMax"] = varMax;
	j["useCuda"] = useCuda;
	j["sendMetrics"] = sendMetrics;
	j["renderMode"] = renderMode;
	j["deterministic"] = deterministic;
	j["standardizeObs"] = standardizeObs;
	j["standardizeReturns"] = standardizeReturns;
	j["trainAgainstOldVersions"] = trainAgainstOldVersions;
	j["trainAgainstOldChance"] = trainAgainstOldChance;
	j["tsPerVersion"] = tsPerVersion;
	j["maxOldVersions"] = maxOldVersions;
	j["obsBuilder"] = obsBuilder;
	j["sharedHeadLayerSizes"] = sharedHeadLayerSizes;
	j["policyLayerSizes"] = policyLayerSizes;
	j["criticLayerSizes"] = criticLayerSizes;
	j["checkpointDir"] = checkpointDir;
	j["collisionMeshesDir"] = collisionMeshesDir;
	j["checkpointInterval"] = checkpointInterval;
	j["useAttentionHead"] = useAttentionHead;
	j["attentionDims"] = attentionDims;
	j["attentionThinkDims"] = attentionThinkDims;
	j["attentionBlocks"] = attentionBlocks;
	j["attentionHeads"] = attentionHeads;
	j["attentionPreprocessLayers"] = attentionPreprocessLayers;
	j["attentionPostprocessLayers"] = attentionPostprocessLayers;
	j["attentionRefinementFeedforward"] = attentionRefinementFeedforward;
	j["continuousActionSize"] = continuousActionSize;
	j["transferLearning"] = transferLearning;
	j["transferStudentCheckpointDir"] = transferStudentCheckpointDir;
	j["transferOldModelsPath"] = transferOldModelsPath;
	j["transferOldObsBuilder"] = transferOldObsBuilder;
	j["transferOldContinuousPolicy"] = transferOldContinuousPolicy;
	j["transferOldContinuousActionSize"] = transferOldContinuousActionSize;
	j["transferOldPolicyLayerSizes"] = transferOldPolicyLayerSizes;
	j["transferOldSharedHeadLayerSizes"] = transferOldSharedHeadLayerSizes;
	j["transferOldPolicyLayerNorm"] = transferOldPolicyLayerNorm;
	j["transferOldPolicyOutputLayer"] = transferOldPolicyOutputLayer;
	j["transferOldUseAttentionHead"] = transferOldUseAttentionHead;
	j["transferOldAttentionDims"] = transferOldAttentionDims;
	j["transferOldAttentionThinkDims"] = transferOldAttentionThinkDims;
	j["transferOldAttentionBlocks"] = transferOldAttentionBlocks;
	j["transferOldAttentionHeads"] = transferOldAttentionHeads;
	j["transferOldAttentionPreprocessLayers"] = transferOldAttentionPreprocessLayers;
	j["transferOldAttentionPostprocessLayers"] = transferOldAttentionPostprocessLayers;
	j["transferOldAttentionRefinementFeedforward"] = transferOldAttentionRefinementFeedforward;
	j["transferLR"] = transferLR;
	j["transferBatchSize"] = transferBatchSize;
	j["transferEpochs"] = transferEpochs;
	j["transferMaxIterations"] = transferMaxIterations;
	j["transferSaveOnMaxIterations"] = transferSaveOnMaxIterations;
	j["transferUseKLDiv"] = transferUseKLDiv;
	j["transferLossScale"] = transferLossScale;
	j["transferLossExponent"] = transferLossExponent;
	j["transferContinuousActionTableTopK"] = transferContinuousActionTableTopK;
	j["transferContinuousActionTableLossWeight"] = transferContinuousActionTableLossWeight;
	j["transferContinuousActionMSEWeight"] = transferContinuousActionMSEWeight;
	j["transferContinuousActionNLLWeight"] = transferContinuousActionNLLWeight;
	j["transferContinuousStdLossWeight"] = transferContinuousStdLossWeight;
	j["transferContinuousTargetStd"] = transferContinuousTargetStd;
	j["transferContinuousButtonBCELossWeight"] = transferContinuousButtonBCELossWeight;
	j["transferContinuousAerialSampleWeight"] = transferContinuousAerialSampleWeight;
	j["transferStudentRolloutProb"] = transferStudentRolloutProb;
	j["transferStudentRolloutWarmupTimesteps"] = transferStudentRolloutWarmupTimesteps;

	j["rewards"] = nlohmann::json::array();
	for (const RewardEntry& reward : rewards) {
		j["rewards"].push_back({
			{ "className", reward.className },
			{ "weight", reward.weight },
			{ "zeroSum", reward.zeroSum },
			{ "teamSpirit", reward.teamSpirit }
		});
	}

	std::ofstream output(path);
	if (output.good())
		output << j.dump(2);
}

GuiTrainingConfig GuiTrainingConfig::LoadFromFile(const std::string& path) {
	GuiTrainingConfig cfg;
	cfg.rewards = DefaultRewards();

	std::ifstream input(path);
	if (!input.good())
		return cfg;

	try {
		nlohmann::json j = nlohmann::json::parse(input);
		auto get = [&](const char* key, auto fallback) {
			using T = decltype(fallback);
			return j.contains(key) ? j[key].get<T>() : fallback;
		};

		cfg.numArenas = get("numArenas", cfg.numArenas);
		cfg.playersPerTeam = get("playersPerTeam", cfg.playersPerTeam);
		cfg.tickSkip = get("tickSkip", cfg.tickSkip);
		cfg.stepsPerIteration = get("stepsPerIteration", cfg.stepsPerIteration);
		cfg.minibatchSize = get("minibatchSize", cfg.minibatchSize);
		cfg.ppoEpochs = get("ppoEpochs", cfg.ppoEpochs);
		cfg.entropyScale = get("entropyScale", cfg.entropyScale);
		cfg.gamma = get("gamma", cfg.gamma);
		cfg.lambda = get("lambda", cfg.lambda);
		cfg.policyLR = get("policyLR", cfg.policyLR);
		cfg.criticLR = get("criticLR", cfg.criticLR);
		cfg.clipRange = get("clipRange", cfg.clipRange);
		cfg.rewardClipRange = get("rewardClipRange", cfg.rewardClipRange);
		cfg.varMin = get("varMin", cfg.varMin);
		cfg.varMax = get("varMax", cfg.varMax);
		cfg.useCuda = get("useCuda", cfg.useCuda);
		cfg.sendMetrics = get("sendMetrics", cfg.sendMetrics);
		cfg.renderMode = get("renderMode", cfg.renderMode);
		cfg.deterministic = get("deterministic", cfg.deterministic);
		cfg.standardizeObs = get("standardizeObs", cfg.standardizeObs);
		cfg.standardizeReturns = get("standardizeReturns", cfg.standardizeReturns);
		cfg.trainAgainstOldVersions = get("trainAgainstOldVersions", cfg.trainAgainstOldVersions);
		cfg.trainAgainstOldChance = get("trainAgainstOldChance", cfg.trainAgainstOldChance);
		cfg.tsPerVersion = get("tsPerVersion", cfg.tsPerVersion);
		cfg.maxOldVersions = get("maxOldVersions", cfg.maxOldVersions);
		cfg.obsBuilder = get("obsBuilder", cfg.obsBuilder);
		cfg.sharedHeadLayerSizes = get("sharedHeadLayerSizes", cfg.sharedHeadLayerSizes);
		cfg.policyLayerSizes = get("policyLayerSizes", cfg.policyLayerSizes);
		cfg.criticLayerSizes = get("criticLayerSizes", cfg.criticLayerSizes);
		cfg.checkpointDir = get("checkpointDir", cfg.checkpointDir);
		cfg.collisionMeshesDir = get("collisionMeshesDir", cfg.collisionMeshesDir);
		cfg.checkpointInterval = get("checkpointInterval", cfg.checkpointInterval);
		cfg.useAttentionHead = get("useAttentionHead", cfg.useAttentionHead);
		cfg.attentionDims = get("attentionDims", cfg.attentionDims);
		cfg.attentionThinkDims = get("attentionThinkDims", cfg.attentionThinkDims);
		cfg.attentionBlocks = get("attentionBlocks", cfg.attentionBlocks);
		cfg.attentionHeads = get("attentionHeads", cfg.attentionHeads);
		cfg.attentionPreprocessLayers = get("attentionPreprocessLayers", cfg.attentionPreprocessLayers);
		cfg.attentionPostprocessLayers = get("attentionPostprocessLayers", cfg.attentionPostprocessLayers);
		cfg.attentionRefinementFeedforward = get("attentionRefinementFeedforward", cfg.attentionRefinementFeedforward);
		cfg.continuousActionSize = get("continuousActionSize", cfg.continuousActionSize);
		cfg.transferLearning = get("transferLearning", cfg.transferLearning);
		cfg.transferStudentCheckpointDir = get("transferStudentCheckpointDir", cfg.transferStudentCheckpointDir);
		cfg.transferOldModelsPath = get("transferOldModelsPath", cfg.transferOldModelsPath);
		cfg.transferOldObsBuilder = get("transferOldObsBuilder", cfg.transferOldObsBuilder);
		cfg.transferOldContinuousPolicy = get("transferOldContinuousPolicy", cfg.transferOldContinuousPolicy);
		cfg.transferOldContinuousActionSize = get("transferOldContinuousActionSize", cfg.transferOldContinuousActionSize);
		cfg.transferOldPolicyLayerSizes = get("transferOldPolicyLayerSizes", cfg.transferOldPolicyLayerSizes);
		cfg.transferOldSharedHeadLayerSizes = get("transferOldSharedHeadLayerSizes", cfg.transferOldSharedHeadLayerSizes);
		cfg.transferOldPolicyLayerNorm = get("transferOldPolicyLayerNorm", cfg.transferOldPolicyLayerNorm);
		cfg.transferOldPolicyOutputLayer = get("transferOldPolicyOutputLayer", cfg.transferOldPolicyOutputLayer);
		cfg.transferOldUseAttentionHead = get("transferOldUseAttentionHead", cfg.transferOldUseAttentionHead);
		cfg.transferOldAttentionDims = get("transferOldAttentionDims", cfg.transferOldAttentionDims);
		cfg.transferOldAttentionThinkDims = get("transferOldAttentionThinkDims", cfg.transferOldAttentionThinkDims);
		cfg.transferOldAttentionBlocks = get("transferOldAttentionBlocks", cfg.transferOldAttentionBlocks);
		cfg.transferOldAttentionHeads = get("transferOldAttentionHeads", cfg.transferOldAttentionHeads);
		cfg.transferOldAttentionPreprocessLayers = get("transferOldAttentionPreprocessLayers", cfg.transferOldAttentionPreprocessLayers);
		cfg.transferOldAttentionPostprocessLayers = get("transferOldAttentionPostprocessLayers", cfg.transferOldAttentionPostprocessLayers);
		cfg.transferOldAttentionRefinementFeedforward = get("transferOldAttentionRefinementFeedforward", cfg.transferOldAttentionRefinementFeedforward);
		cfg.transferLR = get("transferLR", cfg.transferLR);
		cfg.transferBatchSize = get("transferBatchSize", cfg.transferBatchSize);
		cfg.transferEpochs = get("transferEpochs", cfg.transferEpochs);
		cfg.transferMaxIterations = get("transferMaxIterations", cfg.transferMaxIterations);
		cfg.transferSaveOnMaxIterations = get("transferSaveOnMaxIterations", cfg.transferSaveOnMaxIterations);
		cfg.transferUseKLDiv = get("transferUseKLDiv", cfg.transferUseKLDiv);
		cfg.transferLossScale = get("transferLossScale", cfg.transferLossScale);
		cfg.transferLossExponent = get("transferLossExponent", cfg.transferLossExponent);
		cfg.transferContinuousActionTableTopK = get("transferContinuousActionTableTopK", cfg.transferContinuousActionTableTopK);
		cfg.transferContinuousActionTableLossWeight = get("transferContinuousActionTableLossWeight", cfg.transferContinuousActionTableLossWeight);
		cfg.transferContinuousActionMSEWeight = get("transferContinuousActionMSEWeight", cfg.transferContinuousActionMSEWeight);
		cfg.transferContinuousActionNLLWeight = get("transferContinuousActionNLLWeight", cfg.transferContinuousActionNLLWeight);
		cfg.transferContinuousStdLossWeight = get("transferContinuousStdLossWeight", cfg.transferContinuousStdLossWeight);
		cfg.transferContinuousTargetStd = get("transferContinuousTargetStd", cfg.transferContinuousTargetStd);
		cfg.transferContinuousButtonBCELossWeight = get("transferContinuousButtonBCELossWeight", cfg.transferContinuousButtonBCELossWeight);
		cfg.transferContinuousAerialSampleWeight = get("transferContinuousAerialSampleWeight", cfg.transferContinuousAerialSampleWeight);
		cfg.transferStudentRolloutProb = get("transferStudentRolloutProb", cfg.transferStudentRolloutProb);
		cfg.transferStudentRolloutWarmupTimesteps = get("transferStudentRolloutWarmupTimesteps", cfg.transferStudentRolloutWarmupTimesteps);

		if (j.contains("rewards") && j["rewards"].is_array()) {
			cfg.rewards.clear();
			for (const auto& item : j["rewards"]) {
				RewardEntry reward;
				reward.className = item.value("className", std::string("TouchBallReward"));
				reward.weight = item.value("weight", 1.0f);
				reward.zeroSum = item.value("zeroSum", false);
				reward.teamSpirit = item.value("teamSpirit", 0.0f);
				cfg.rewards.push_back(reward);
			}
		}
	} catch (...) {
		cfg.rewards = DefaultRewards();
	}

	cfg.numArenas = std::max(1, cfg.numArenas);
	cfg.playersPerTeam = ClampInt(cfg.playersPerTeam, 1, 2);
	cfg.tickSkip = ClampInt(cfg.tickSkip, 1, 32);
	cfg.stepsPerIteration = std::max(1, cfg.stepsPerIteration);
	cfg.minibatchSize = MakeDivisibleMinibatch(cfg.stepsPerIteration, cfg.minibatchSize);
	cfg.ppoEpochs = ClampInt(cfg.ppoEpochs, 1, 16);
	cfg.gamma = ClampFloat(cfg.gamma, 0.0f, 0.99999f);
	cfg.lambda = ClampFloat(cfg.lambda, 0.0f, 1.0f);
	cfg.clipRange = ClampFloat(cfg.clipRange, 0.0f, 1.0f);
	cfg.varMin = ClampFloat(cfg.varMin, 0.001f, 10.0f);
	cfg.varMax = std::max(cfg.varMin, ClampFloat(cfg.varMax, 0.001f, 10.0f));
	cfg.trainAgainstOldChance = ClampFloat(cfg.trainAgainstOldChance, 0.0f, 1.0f);
	cfg.tsPerVersion = std::max<int64_t>(1, cfg.tsPerVersion);
	cfg.maxOldVersions = std::max(1, cfg.maxOldVersions);
	cfg.continuousActionSize = ClampInt(cfg.continuousActionSize, 6, 16);
	auto normalizeObsBuilder = [](std::string& value) {
		for (const char* item : kObsBuilders) {
			if (value == item)
				return;
		}
		value = "AdvancedObsPadded";
	};
	normalizeObsBuilder(cfg.obsBuilder);
	normalizeObsBuilder(cfg.transferOldObsBuilder);
	cfg.sharedHeadLayerSizes = LayerSizesToText(ParseLayerSizes(cfg.sharedHeadLayerSizes, { 1024, 1024, 1024 }));
	cfg.policyLayerSizes = LayerSizesToText(ParseLayerSizes(cfg.policyLayerSizes, { 1024, 1024, 1024 }));
	cfg.criticLayerSizes = LayerSizesToText(ParseLayerSizes(cfg.criticLayerSizes, { 1024, 1024, 1024 }));
	cfg.transferOldContinuousActionSize = ClampInt(cfg.transferOldContinuousActionSize, 1, 32);
	if (TrimCopy(cfg.transferStudentCheckpointDir).empty())
		cfg.transferStudentCheckpointDir = "checkpoints_transfer";
	cfg.transferOldPolicyLayerSizes = LayerSizesToText(ParseLayerSizes(cfg.transferOldPolicyLayerSizes, { 1024, 1024, 1024 }));
	cfg.transferOldSharedHeadLayerSizes = LayerSizesToText(ParseLayerSizes(cfg.transferOldSharedHeadLayerSizes, {}));
	cfg.transferOldAttentionDims = ClampInt(cfg.transferOldAttentionDims, 16, 4096);
	cfg.transferOldAttentionThinkDims = ClampInt(cfg.transferOldAttentionThinkDims, 32, 8192);
	cfg.transferOldAttentionBlocks = ClampInt(cfg.transferOldAttentionBlocks, 1, 16);
	cfg.transferOldAttentionHeads = ClampInt(cfg.transferOldAttentionHeads, 1, 32);
	cfg.transferOldAttentionPreprocessLayers = ClampInt(cfg.transferOldAttentionPreprocessLayers, 0, 8);
	cfg.transferOldAttentionPostprocessLayers = ClampInt(cfg.transferOldAttentionPostprocessLayers, 0, 8);
	cfg.attentionRefinementFeedforward = std::max(0, cfg.attentionRefinementFeedforward);
	cfg.transferOldAttentionRefinementFeedforward = std::max(0, cfg.transferOldAttentionRefinementFeedforward);
	cfg.transferLR = std::max(0.0f, cfg.transferLR);
	cfg.transferBatchSize = std::max(1, cfg.transferBatchSize);
	cfg.transferEpochs = ClampInt(cfg.transferEpochs, 1, 64);
	cfg.transferLossScale = std::max(0.0f, cfg.transferLossScale);
	cfg.transferLossExponent = std::max(0.0f, cfg.transferLossExponent);
	cfg.transferContinuousActionTableTopK = std::max(0, cfg.transferContinuousActionTableTopK);
	cfg.transferContinuousActionTableLossWeight = std::max(0.0f, cfg.transferContinuousActionTableLossWeight);
	cfg.transferContinuousActionMSEWeight = std::max(0.0f, cfg.transferContinuousActionMSEWeight);
	cfg.transferContinuousActionNLLWeight = std::max(0.0f, cfg.transferContinuousActionNLLWeight);
	cfg.transferContinuousStdLossWeight = std::max(0.0f, cfg.transferContinuousStdLossWeight);
	cfg.transferContinuousTargetStd = std::max(0.001f, cfg.transferContinuousTargetStd);
	cfg.transferContinuousButtonBCELossWeight = std::max(0.0f, cfg.transferContinuousButtonBCELossWeight);
	cfg.transferContinuousAerialSampleWeight = std::max(0.0f, cfg.transferContinuousAerialSampleWeight);
	cfg.transferStudentRolloutProb = ClampFloat(cfg.transferStudentRolloutProb, 0.0f, 1.0f);
	cfg.transferStudentRolloutWarmupTimesteps = std::max<int64_t>(0, cfg.transferStudentRolloutWarmupTimesteps);
	if (cfg.rewards.empty())
		cfg.rewards = DefaultRewards();
	return cfg;
}

GuiApp::GuiApp() = default;

GuiApp::~GuiApp() {
	Shutdown();
}

bool GuiApp::Init(int width, int height) {
	glfwSetErrorCallback(GlfwErrorCallback);
	if (!glfwInit()) {
		std::fprintf(stderr, "Failed to initialize GLFW\n");
		return false;
	}

	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
	glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
	glfwWindowHint(GLFW_DECORATED, GLFW_FALSE);
	glfwWindowHint(GLFW_VISIBLE, GLFW_FALSE);
	glfwWindowHint(GLFW_RESIZABLE, GLFW_TRUE);

	window = glfwCreateWindow(width, height, "Prometheus", nullptr, nullptr);
	if (!window) {
		std::fprintf(stderr, "Failed to create GLFW window\n");
		glfwTerminate();
		return false;
	}

	if (GLFWmonitor* monitor = glfwGetPrimaryMonitor()) {
		int workX = 0, workY = 0, workWidth = width, workHeight = height;
		glfwGetMonitorWorkarea(monitor, &workX, &workY, &workWidth, &workHeight);
		width = std::min(width, workWidth);
		const int extraTopHeight = std::max(0, (workHeight - height) / 2);
		height = std::min(height + extraTopHeight, workHeight);
		const int posX = workX + std::max(0, (workWidth - width) / 2);
		const int posY = workY;
		glfwSetWindowPos(window, posX, posY);
		glfwSetWindowSize(window, width, height);
	}

#ifdef _WIN32
	InstallBorderlessResize(window);
#endif

	glfwShowWindow(window);
	glfwMakeContextCurrent(window);
	glfwSwapInterval(1);

	IMGUI_CHECKVERSION();
	ImGui::CreateContext();
	ImGuiIO& io = ImGui::GetIO();
	io.IniFilename = nullptr;
	io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;
	io.ConfigFlags |= ImGuiConfigFlags_DockingEnable;

	ImGui::StyleColorsDark();
	ApplyGoldTheme();

	ImGui_ImplGlfw_InitForOpenGL(window, true);
	ImGui_ImplOpenGL3_Init("#version 330");

	io.Fonts->Clear();
	ImFontConfig fontCfg;
	fontCfg.OversampleH = 2;
	fontCfg.OversampleV = 1;
	fontCfg.PixelSnapH = true;

	const char* mediumPath = "src/gui/fonts/Inter-Medium.ttf";
	const char* semiBoldPath = "src/gui/fonts/Inter-SemiBold.ttf";
	if (std::filesystem::exists(mediumPath)) io.Fonts->AddFontFromFileTTF(mediumPath, 15.0f, &fontCfg);
	else io.Fonts->AddFontDefault();
	if (std::filesystem::exists(semiBoldPath)) io.Fonts->AddFontFromFileTTF(semiBoldPath, 16.0f, &fontCfg);
	if (std::filesystem::exists(semiBoldPath)) io.Fonts->AddFontFromFileTTF(semiBoldPath, 22.0f, &fontCfg);
	if (std::filesystem::exists(mediumPath)) io.Fonts->AddFontFromFileTTF(mediumPath, 12.0f, &fontCfg);

	io.Fonts->Build();
	ImGui_ImplOpenGL3_DestroyFontsTexture();
	ImGui_ImplOpenGL3_CreateFontsTexture();

	config = GuiTrainingConfig::LoadFromFile(kConfigPath);
	state.cudaAvailable = config.useCuda;
	state.PushLog(LogEntry::INFO, "Prometheus GUI initialized");
	state.PushLog(LogEntry::INFO, "Continuous V2 policy with attention head is the active learner profile.");
	return true;
}

void GuiApp::ApplyGoldTheme() {
	ImGuiStyle& s = ImGui::GetStyle();
	s.WindowRounding = 10.0f;
	s.ChildRounding = 8.0f;
	s.FrameRounding = 6.0f;
	s.GrabRounding = 6.0f;
	s.TabRounding = 6.0f;
	s.ScrollbarRounding = 8.0f;
	s.PopupRounding = 8.0f;
	s.WindowBorderSize = 1.0f;
	s.ChildBorderSize = 1.0f;
	s.FrameBorderSize = 0.0f;
	s.PopupBorderSize = 1.0f;
	s.TabBorderSize = 0.0f;
	s.FramePadding = ImVec2(10, 5);
	s.ItemSpacing = ImVec2(10, 7);
	s.ItemInnerSpacing = ImVec2(6, 4);
	s.WindowPadding = ImVec2(14, 12);
	s.ScrollbarSize = 12.0f;
	s.GrabMinSize = 10.0f;
	s.SeparatorTextBorderSize = 2.0f;
	s.WindowTitleAlign = ImVec2(0.5f, 0.5f);

	ImVec4* c = s.Colors;
	c[ImGuiCol_WindowBg] = Colors::V(12, 12, 14);
	c[ImGuiCol_ChildBg] = Colors::V(16, 16, 18);
	c[ImGuiCol_PopupBg] = Colors::V(18, 18, 20, 245);
	c[ImGuiCol_MenuBarBg] = Colors::V(18, 18, 18);
	c[ImGuiCol_Border] = Colors::V(55, 48, 20, 100);
	c[ImGuiCol_BorderShadow] = Colors::V(0, 0, 0, 0);
	c[ImGuiCol_TitleBg] = Colors::V(16, 16, 16);
	c[ImGuiCol_TitleBgActive] = Colors::V(30, 26, 8);
	c[ImGuiCol_TitleBgCollapsed] = Colors::V(12, 12, 12, 130);
	c[ImGuiCol_Tab] = Colors::V(28, 26, 14);
	c[ImGuiCol_TabHovered] = Colors::V(200, 170, 15, 200);
	c[ImGuiCol_TabSelected] = Colors::V(150, 128, 8);
	c[ImGuiCol_TabDimmed] = Colors::V(22, 22, 22);
	c[ImGuiCol_TabDimmedSelected] = Colors::V(55, 48, 12);
	c[ImGuiCol_Text] = Colors::V(240, 238, 230);
	c[ImGuiCol_TextDisabled] = Colors::V(120, 108, 70);
	c[ImGuiCol_FrameBg] = Colors::V(26, 26, 28);
	c[ImGuiCol_FrameBgHovered] = Colors::V(42, 38, 16);
	c[ImGuiCol_FrameBgActive] = Colors::V(55, 48, 12);
	c[ImGuiCol_Button] = Colors::V(38, 38, 40);
	c[ImGuiCol_ButtonHovered] = Colors::V(190, 165, 15);
	c[ImGuiCol_ButtonActive] = Colors::V(255, 215, 0);
	c[ImGuiCol_Header] = Colors::V(36, 33, 14);
	c[ImGuiCol_HeaderHovered] = Colors::V(150, 128, 8, 200);
	c[ImGuiCol_HeaderActive] = Colors::V(190, 165, 15);
	c[ImGuiCol_CheckMark] = Colors::V(255, 215, 0);
	c[ImGuiCol_SliderGrab] = Colors::V(184, 145, 18);
	c[ImGuiCol_SliderGrabActive] = Colors::V(255, 215, 0);
	c[ImGuiCol_ScrollbarBg] = Colors::V(12, 12, 14, 100);
	c[ImGuiCol_ScrollbarGrab] = Colors::V(50, 45, 22);
	c[ImGuiCol_ScrollbarGrabHovered] = Colors::V(100, 88, 20);
	c[ImGuiCol_ScrollbarGrabActive] = Colors::V(150, 128, 8);
	c[ImGuiCol_Separator] = Colors::V(60, 52, 22, 160);
	c[ImGuiCol_SeparatorHovered] = Colors::V(200, 170, 15);
	c[ImGuiCol_SeparatorActive] = Colors::V(255, 215, 0);
	c[ImGuiCol_ResizeGrip] = Colors::V(55, 48, 20, 60);
	c[ImGuiCol_ResizeGripHovered] = Colors::V(200, 170, 15, 170);
	c[ImGuiCol_ResizeGripActive] = Colors::V(255, 215, 0, 240);
	c[ImGuiCol_PlotLines] = Colors::V(255, 215, 0);
	c[ImGuiCol_PlotLinesHovered] = Colors::V(255, 240, 120);
	c[ImGuiCol_PlotHistogram] = Colors::V(200, 155, 18);
	c[ImGuiCol_PlotHistogramHovered] = Colors::V(255, 215, 0);
	c[ImGuiCol_DockingPreview] = Colors::V(255, 215, 0, 160);
	c[ImGuiCol_DockingEmptyBg] = Colors::V(8, 8, 8);
	c[ImGuiCol_TableHeaderBg] = Colors::V(28, 26, 12);
	c[ImGuiCol_TableBorderStrong] = Colors::V(55, 48, 20, 120);
	c[ImGuiCol_TableBorderLight] = Colors::V(40, 36, 16, 80);
	c[ImGuiCol_TableRowBg] = Colors::V(0, 0, 0, 0);
	c[ImGuiCol_TableRowBgAlt] = Colors::V(255, 215, 0, 8);
	c[ImGuiCol_NavHighlight] = Colors::V(255, 215, 0);
	c[ImGuiCol_DragDropTarget] = Colors::V(255, 215, 0, 200);
	c[ImGuiCol_TextSelectedBg] = Colors::V(200, 170, 15, 80);
}

void GuiApp::Run() {
	ImVec4 clearColor(0.06f, 0.06f, 0.065f, 1.0f);
	while (!glfwWindowShouldClose(window) && !shouldClose) {
		glfwPollEvents();

		ImGui_ImplOpenGL3_NewFrame();
		ImGui_ImplGlfw_NewFrame();
		ImGui::NewFrame();

#ifdef _WIN32
		if (ImGui::IsKeyPressed(ImGuiKey_F11, false))
			ToggleWindowMaximize(window);
#endif

		ImGuiViewport* vp = ImGui::GetMainViewport();
		ImGui::SetNextWindowPos(vp->WorkPos);
		ImGui::SetNextWindowSize(vp->WorkSize);
		ImGui::SetNextWindowViewport(vp->ID);
		ImGuiWindowFlags dockFlags =
			ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoCollapse |
			ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoMove |
			ImGuiWindowFlags_NoBringToFrontOnFocus | ImGuiWindowFlags_NoNavFocus |
			ImGuiWindowFlags_NoDocking | ImGuiWindowFlags_NoSavedSettings |
			ImGuiWindowFlags_NoBackground;

		ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, 0.0f);
		ImGui::PushStyleVar(ImGuiStyleVar_WindowBorderSize, 0.0f);
		ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0, 0));
		ImGui::Begin("##DockSpace", nullptr, dockFlags);
		ImGui::PopStyleVar(3);
		ImGuiID dockId = ImGui::GetID("PrometheusDock");
		ImGui::DockSpace(dockId, ImVec2(0, 0), ImGuiDockNodeFlags_PassthruCentralNode);
		ImGui::End();

		if (!state.running.load() && trainingThread.joinable()) {
			trainingThread.join();
			state.stopRequested = false;
		}

		RenderTopBar();
		RenderMetricsPanel();
		RenderConfigPanel();
		RenderGPUStatusPanel();
		RenderRewardsPanel();
		RenderLogPanel();

		ImGui::Render();
		int fw, fh;
		glfwGetFramebufferSize(window, &fw, &fh);
		glViewport(0, 0, fw, fh);
		glClearColor(clearColor.x, clearColor.y, clearColor.z, clearColor.w);
		glClear(GL_COLOR_BUFFER_BIT);
		ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
		glfwSwapBuffers(window);
	}
}

void GuiApp::Shutdown() {
	config.SaveToFile(kConfigPath);

	if (state.running.load()) {
		state.stopRequested = true;
		std::lock_guard<std::mutex> lock(learnerMutex);
		if (learner)
			learner->stopRequested = true;
	}

	if (trainingThread.joinable())
		trainingThread.join();

	if (window) {
		ImGui_ImplOpenGL3_Shutdown();
		ImGui_ImplGlfw_Shutdown();
		ImGui::DestroyContext();
		glfwDestroyWindow(window);
		glfwTerminate();
		window = nullptr;
	}
}

void GuiApp::RenderTopBar() {
	const LockedGuiLayout layout = BuildLockedGuiLayout(ImGui::GetMainViewport());
	constexpr float kTitleBarHeight = 32.0f;
	ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0, 0));
	BeginPanel("Training Controls", layout.topBar);
	ImGui::PopStyleVar();

	ImGui::PushStyleColor(ImGuiCol_ChildBg, Colors::V(22, 22, 24));
	ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(12, 6));
	ImGui::BeginChild("##titlebar", ImVec2(0, kTitleBarHeight), false, ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoScrollWithMouse);
	ImGui::PopStyleVar();
	ImGui::PopStyleColor();

	bool titleBarHovered = ImGui::IsWindowHovered(ImGuiHoveredFlags_AllowWhenBlockedByActiveItem);
	float titleTextY = std::max(0.0f, std::floor((kTitleBarHeight - ImGui::GetTextLineHeight()) * 0.5f));
	ImGui::SetCursorPosX(12.0f);
	ImGui::SetCursorPosY(titleTextY);
	Draw::PushFont(Draw::Font::SEMI_BOLD);
	ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(1.0f, 0.84f, 0.0f, 1.0f));
	ImGui::TextUnformatted("Prometheus");
	ImGui::PopStyleColor();
	ImGui::PopFont();
	ImGui::SameLine();
	ImGui::TextDisabled("GGL Continuous V2 Attention");
	ImGui::SameLine();
	RenderStandaloneWindowControls(window, kTitleBarHeight);
	bool allowDrag = titleBarHovered && !ImGui::IsAnyItemHovered() && !ImGui::IsAnyItemActive();
	HandleStandaloneTitleBarDrag(window, allowDrag, titleBarHovered);
	ImGui::EndChild();

	ImGui::SetCursorPosX(12.0f);
	ImGui::SetCursorPosY(ImGui::GetCursorPosY() + 8.0f);

	bool isRunning = state.running.load();
	ImVec4 statusCol = isRunning ? ImVec4(0.2f, 1.0f, 0.3f, 1.0f) : ImVec4(0.5f, 0.5f, 0.5f, 1.0f);
	if (isRunning)
		Draw::PulsingDot(statusCol, 4.0f);
	else {
		ImDrawList* dl = ImGui::GetWindowDrawList();
		ImVec2 dotP = ImGui::GetCursorScreenPos();
		dotP.x += 5.0f;
		dotP.y += ImGui::GetTextLineHeight() * 0.5f;
		dl->AddCircleFilled(dotP, 4.0f, ImGui::ColorConvertFloat4ToU32(statusCol), 16);
		ImGui::Dummy(ImVec2(12.0f, ImGui::GetTextLineHeight()));
	}
	ImGui::SameLine();
	Draw::PushFont(Draw::Font::SEMI_BOLD);
	ImGui::PushStyleColor(ImGuiCol_Text, statusCol);
	ImGui::TextUnformatted(isRunning ? "TRAINING" : "IDLE");
	ImGui::PopStyleColor();
	ImGui::PopFont();
	ImGui::SameLine();

	if (!isRunning) {
		ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.15f, 0.5f, 0.15f, 1.0f));
		ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(0.2f, 0.7f, 0.2f, 1.0f));
		if (ImGui::Button("Start Training")) {
			StartTraining(GuiRunMode::Train);
		}
		ImGui::PopStyleColor(2);

		ImGui::SameLine();
		ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.5f, 0.35f, 0.12f, 1.0f));
		ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(0.72f, 0.48f, 0.14f, 1.0f));
		if (ImGui::Button("Transfer Learn")) {
			StartTraining(GuiRunMode::TransferLearn);
		}
		ImGui::PopStyleColor(2);

		ImGui::SameLine();
		ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.15f, 0.4f, 0.6f, 1.0f));
		ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(0.2f, 0.5f, 0.8f, 1.0f));
		if (ImGui::Button("Run Render")) {
			StartTraining(GuiRunMode::Render);
		}
		ImGui::PopStyleColor(2);
	} else {
		ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.6f, 0.1f, 0.1f, 1.0f));
		ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(0.8f, 0.15f, 0.15f, 1.0f));
		if (ImGui::Button("Stop")) StopTraining();
		ImGui::PopStyleColor(2);
	}

	ImGui::SameLine();
	if (ImGui::Button("Save Checkpoint")) SaveCheckpoint();

	ImGui::SameLine();
	ImGui::Spacing();
	ImGui::SameLine();
	int64_t ts = state.totalTimesteps.load();
	int iter = state.iteration.load();
	float sps = state.stepsPerSec.load();
	ImGui::Text("Steps: %s", FormatTimesteps(ts).c_str());
	ImGui::SameLine();
	ImGui::Text("Iter: %d", iter);
	ImGui::SameLine();
	ImGui::Text("SPS: %.0f", sps);
	ImGui::SameLine();
	ImGui::Text("Policy: continuous x%d", config.continuousActionSize);
	ImGui::SameLine();
	ImGui::Text("Attention: %s", config.useAttentionHead ? "on" : "off");
	ImGui::SameLine();
	GuiRunMode mode = (GuiRunMode)state.activeMode.load();
	ImGui::Text("Mode: %s", mode == GuiRunMode::TransferLearn ? "transfer" : (mode == GuiRunMode::Render ? "render" : "train"));

	int64_t progress = state.collectionProgress.load();
	int64_t target = state.collectionTarget.load();
	if (target > 0 && isRunning) {
		float fraction = ClampFloat((float)((double)progress / (double)target), 0.0f, 1.0f);
		ImGui::SetCursorPosX(12.0f);
		char overlay[96];
		std::snprintf(overlay, sizeof(overlay), "Collecting %s / %s", FormatTimesteps(progress).c_str(), FormatTimesteps(target).c_str());
		Draw::GradientProgressBar(fraction, ImVec2(std::max(200.0f, ImGui::GetContentRegionAvail().x), 14.0f), Colors::Gold, overlay);
	}

	ImGui::End();
}

void GuiApp::RenderMetricsPanel() {
	const LockedGuiLayout layout = BuildLockedGuiLayout(ImGui::GetMainViewport());
	BeginPanel("Metrics", layout.metrics);
	Draw::PanelTitle("Metrics");

	float columnWidth = (ImGui::GetContentRegionAvail().x - 20.0f) / 3.0f;
	columnWidth = std::max(130.0f, columnWidth);
	char value[64];
	{
		std::lock_guard<std::mutex> lock(state.mtx);
		bool transferView = (GuiRunMode)state.activeMode.load() == GuiRunMode::TransferLearn;
		std::snprintf(value, sizeof(value), "%.4f", transferView ? state.transferLoss : state.meanReward);
		ImGui::PushItemWidth(columnWidth);
		RenderMetricCard(transferView ? "TL Loss" : "Mean Reward", value, ImVec4(1.0f, 0.84f, 0.0f, 1.0f));
		ImGui::SameLine();
		std::snprintf(value, sizeof(value), "%.4f", transferView ? state.transferAccuracy : state.entropy);
		RenderMetricCard(transferView ? "TL Accuracy" : "Entropy", value, ImVec4(0.75f, 0.92f, 1.0f, 1.0f));
		ImGui::SameLine();
		std::snprintf(value, sizeof(value), transferView ? "%.4f" : "%.0f",
			transferView ? state.teacherConfidence : state.stepsPerSec.load());
		RenderMetricCard(transferView ? "Teacher Conf" : "Steps/sec", value, ImVec4(0.5f, 1.0f, 0.55f, 1.0f));
		ImGui::PopItemWidth();

		const char* plotNames[] = {
			"Reward", "Policy Loss", "Critic Loss", "Entropy", "Steps/sec", "Touch Ratio", "Boost",
			"TL Loss", "TL Accuracy", "Teacher Confidence"
		};
		MetricSeries* series[] = {
			&state.rewardSeries,
			&state.policyLossSeries,
			&state.criticLossSeries,
			&state.entropySeries,
			&state.stepsPerSecSeries,
			&state.touchSeries,
			&state.boostSeries,
			&state.transferLossSeries,
			&state.transferAccuracySeries,
			&state.teacherConfidenceSeries
		};

		ImGui::Spacing();
		ImGui::Combo("Plot", &selectedPlot, plotNames, IM_ARRAYSIZE(plotNames));
		selectedPlot = ClampInt(selectedPlot, 0, IM_ARRAYSIZE(plotNames) - 1);
		RenderPlot(plotNames[selectedPlot], *series[selectedPlot], std::max(120.0f, ImGui::GetContentRegionAvail().y * 0.35f));

		Draw::SectionHeader(transferView ? "Transfer Learning" : "Training");
		ImGui::Columns(2, nullptr, false);
		if (transferView) {
			ImGui::Text("TL loss"); ImGui::NextColumn(); ImGui::Text("%.6f", state.transferLoss); ImGui::NextColumn();
			ImGui::Text("Accuracy"); ImGui::NextColumn(); ImGui::Text("%.4f", state.transferAccuracy); ImGui::NextColumn();
			ImGui::Text("Table loss"); ImGui::NextColumn(); ImGui::Text("%.6f", state.transferTableLoss); ImGui::NextColumn();
			ImGui::Text("MSE / MAE"); ImGui::NextColumn(); ImGui::Text("%.5f / %.5f", state.transferMSE, state.transferMAE); ImGui::NextColumn();
			ImGui::Text("NLL"); ImGui::NextColumn(); ImGui::Text("%.6f", state.transferNLL); ImGui::NextColumn();
			ImGui::Text("Std loss"); ImGui::NextColumn(); ImGui::Text("%.6f", state.transferStdLoss); ImGui::NextColumn();
			ImGui::Text("Button BCE"); ImGui::NextColumn(); ImGui::Text("%.6f", state.transferButtonBCE); ImGui::NextColumn();
			ImGui::Text("Old entropy"); ImGui::NextColumn(); ImGui::Text("%.4f", state.oldPolicyEntropy); ImGui::NextColumn();
			ImGui::Text("Student rollout"); ImGui::NextColumn(); ImGui::Text("%.4f", state.studentRolloutRatio); ImGui::NextColumn();
			ImGui::Text("Aerial targets"); ImGui::NextColumn(); ImGui::Text("%.4f", state.aerialTargetRatio); ImGui::NextColumn();
		} else {
			ImGui::Text("Policy loss"); ImGui::NextColumn(); ImGui::Text("%.6f", state.policyLoss); ImGui::NextColumn();
			ImGui::Text("Critic loss"); ImGui::NextColumn(); ImGui::Text("%.6f", state.criticLoss); ImGui::NextColumn();
			ImGui::Text("Mean policy std"); ImGui::NextColumn(); ImGui::Text("%.4f", state.meanPolicyStd); ImGui::NextColumn();
			ImGui::Text("Clip fraction"); ImGui::NextColumn(); ImGui::Text("%.4f", state.clipFraction); ImGui::NextColumn();
			ImGui::Text("PPO learn time"); ImGui::NextColumn(); ImGui::Text("%.2fs", state.ppoLearnTime); ImGui::NextColumn();
			ImGui::Text("Collection"); ImGui::NextColumn(); ImGui::Text("%.2fs", state.collectionTime); ImGui::NextColumn();
		}
		ImGui::Columns(1);

		Draw::SectionHeader("Reward Signals");
		if (state.rewardMetrics.empty()) {
			ImGui::TextDisabled("Reward metrics will appear after the first sampled report.");
		} else {
			for (auto& pair : state.rewardMetrics) {
				if (pair.second.values.empty()) continue;
				ImGui::Text("%s", pair.first.c_str());
				ImGui::SameLine();
				ImGui::TextColored(ImVec4(1.0f, 0.84f, 0.0f, 1.0f), "%.4f", pair.second.values.back());
			}
		}
	}

	ImGui::End();
}

void GuiApp::RenderConfigPanel() {
	const LockedGuiLayout layout = BuildLockedGuiLayout(ImGui::GetMainViewport());
	BeginPanel("Configuration", layout.config);
	Draw::PanelTitle("Configuration");

	bool locked = state.running.load();
	if (locked) ImGui::BeginDisabled();

	Draw::SectionHeader("Runtime");
	ImGui::InputInt("Arenas", &config.numArenas);
	ImGui::InputInt("Players/team", &config.playersPerTeam);
	ImGui::InputInt("Tick skip", &config.tickSkip);
	ImGui::InputInt("Steps/iteration", &config.stepsPerIteration);
	ImGui::InputInt("Minibatch", &config.minibatchSize);
	ImGui::InputInt("Epochs", &config.ppoEpochs);
	ImGui::Checkbox("Use CUDA", &config.useCuda);
	ImGui::Checkbox("Send metrics", &config.sendMetrics);
	ImGui::Checkbox("Deterministic policy", &config.deterministic);
	ComboString("Student obs", config.obsBuilder, kObsBuilders, IM_ARRAYSIZE(kObsBuilders));

	Draw::SectionHeader("PPO");
	ImGui::InputFloat("Policy LR", &config.policyLR, 0.0f, 0.0f, "%.6g");
	ImGui::InputFloat("Critic LR", &config.criticLR, 0.0f, 0.0f, "%.6g");
	ImGui::InputFloat("Entropy", &config.entropyScale, 0.0f, 0.0f, "%.6g");
	ImGui::SliderFloat("Gamma", &config.gamma, 0.90f, 0.99999f, "%.5f");
	ImGui::SliderFloat("Lambda", &config.lambda, 0.80f, 1.0f, "%.3f");
	ImGui::SliderFloat("Clip range", &config.clipRange, 0.0f, 0.5f, "%.3f");
	ImGui::InputFloat("Reward clip", &config.rewardClipRange, 0.0f, 0.0f, "%.3f");

	Draw::SectionHeader("Continuous Actions");
	ImGui::InputInt("Action dims", &config.continuousActionSize);
	ImGui::InputFloat("Std min", &config.varMin, 0.0f, 0.0f, "%.4f");
	ImGui::InputFloat("Std max", &config.varMax, 0.0f, 0.0f, "%.4f");

	Draw::SectionHeader("Attention Head");
	ImGui::Checkbox("Use attention", &config.useAttentionHead);
	ImGui::InputInt("Dims", &config.attentionDims);
	ImGui::InputInt("Think dims", &config.attentionThinkDims);
	ImGui::InputInt("Blocks", &config.attentionBlocks);
	ImGui::InputInt("Heads", &config.attentionHeads);
	ImGui::InputInt("Pre layers", &config.attentionPreprocessLayers);
	ImGui::InputInt("Post layers", &config.attentionPostprocessLayers);

	Draw::SectionHeader("Networks");
	InputTextString("Shared head", config.sharedHeadLayerSizes);
	InputTextString("Policy", config.policyLayerSizes);
	InputTextString("Critic", config.criticLayerSizes);

	Draw::SectionHeader("Transfer Learning");
	ImGui::Checkbox("Enable TL profile", &config.transferLearning);
	InputTextString("Student checkpoints", config.transferStudentCheckpointDir);
	InputTextString("Teacher checkpoint", config.transferOldModelsPath);
	ComboString("Teacher obs", config.transferOldObsBuilder, kObsBuilders, IM_ARRAYSIZE(kObsBuilders));
	ImGui::Checkbox("Teacher continuous", &config.transferOldContinuousPolicy);
	ImGui::InputInt("Teacher action dims", &config.transferOldContinuousActionSize);
	InputTextString("Teacher policy", config.transferOldPolicyLayerSizes);
	InputTextString("Teacher shared", config.transferOldSharedHeadLayerSizes);
	ImGui::Checkbox("Teacher layer norm", &config.transferOldPolicyLayerNorm);
	ImGui::Checkbox("Teacher output layer", &config.transferOldPolicyOutputLayer);
	ImGui::Checkbox("Teacher attention", &config.transferOldUseAttentionHead);
	ImGui::InputInt("Teacher att dims", &config.transferOldAttentionDims);
	ImGui::InputInt("Teacher think", &config.transferOldAttentionThinkDims);
	ImGui::InputInt("Teacher blocks", &config.transferOldAttentionBlocks);
	ImGui::InputInt("Teacher heads", &config.transferOldAttentionHeads);
	ImGui::InputInt("Teacher pre", &config.transferOldAttentionPreprocessLayers);
	ImGui::InputInt("Teacher post", &config.transferOldAttentionPostprocessLayers);
	ImGui::InputFloat("TL LR", &config.transferLR, 0.0f, 0.0f, "%.6g");
	ImGui::InputInt("TL batch", &config.transferBatchSize);
	ImGui::InputInt("TL epochs", &config.transferEpochs);
	ImGui::InputInt("TL max iters", &config.transferMaxIterations);
	ImGui::Checkbox("Save at max iters", &config.transferSaveOnMaxIterations);
	ImGui::Checkbox("Use KL div", &config.transferUseKLDiv);
	ImGui::InputFloat("Loss scale", &config.transferLossScale, 0.0f, 0.0f, "%.3g");
	ImGui::InputFloat("Loss exponent", &config.transferLossExponent, 0.0f, 0.0f, "%.3g");
	ImGui::InputInt("Action table top K", &config.transferContinuousActionTableTopK);
	ImGui::InputFloat("Table loss w", &config.transferContinuousActionTableLossWeight, 0.0f, 0.0f, "%.3g");
	ImGui::InputFloat("MSE w", &config.transferContinuousActionMSEWeight, 0.0f, 0.0f, "%.3g");
	ImGui::InputFloat("NLL w", &config.transferContinuousActionNLLWeight, 0.0f, 0.0f, "%.3g");
	ImGui::InputFloat("Std loss w", &config.transferContinuousStdLossWeight, 0.0f, 0.0f, "%.3g");
	ImGui::InputFloat("Target std", &config.transferContinuousTargetStd, 0.0f, 0.0f, "%.3g");
	ImGui::InputFloat("Button BCE w", &config.transferContinuousButtonBCELossWeight, 0.0f, 0.0f, "%.3g");
	ImGui::InputFloat("Aerial sample w", &config.transferContinuousAerialSampleWeight, 0.0f, 0.0f, "%.3g");
	ImGui::SliderFloat("Student rollout", &config.transferStudentRolloutProb, 0.0f, 1.0f, "%.2f");
	ImGui::InputScalar("Rollout warmup", ImGuiDataType_S64, &config.transferStudentRolloutWarmupTimesteps);

	Draw::SectionHeader("Paths");
	InputTextString("Checkpoints", config.checkpointDir);
	InputTextString("Collision meshes", config.collisionMeshesDir);
	ImGui::InputInt("Save interval", &config.checkpointInterval);

	if (locked) ImGui::EndDisabled();

	if (ImGui::Button("Save Config")) {
		config.SaveToFile(kConfigPath);
		state.PushLog(LogEntry::INFO, "Saved prometheus_config.json");
	}
	ImGui::SameLine();
	if (ImGui::Button("Reload")) {
		config = GuiTrainingConfig::LoadFromFile(kConfigPath);
		state.PushLog(LogEntry::INFO, "Reloaded prometheus_config.json");
	}

	config.playersPerTeam = ClampInt(config.playersPerTeam, 1, 2);
	config.tickSkip = ClampInt(config.tickSkip, 1, 32);
	config.stepsPerIteration = std::max(1, config.stepsPerIteration);
	config.minibatchSize = MakeDivisibleMinibatch(config.stepsPerIteration, config.minibatchSize);
	config.ppoEpochs = ClampInt(config.ppoEpochs, 1, 16);
	config.continuousActionSize = ClampInt(config.continuousActionSize, 6, 16);
	config.varMin = ClampFloat(config.varMin, 0.001f, 10.0f);
	config.varMax = std::max(config.varMin, ClampFloat(config.varMax, 0.001f, 10.0f));
	if (TrimCopy(config.transferStudentCheckpointDir).empty())
		config.transferStudentCheckpointDir = "checkpoints_transfer";
	config.transferOldContinuousActionSize = ClampInt(config.transferOldContinuousActionSize, 1, 32);
	config.transferBatchSize = std::max(1, config.transferBatchSize);
	config.transferEpochs = ClampInt(config.transferEpochs, 1, 64);
	config.transferLR = std::max(0.0f, config.transferLR);
	config.transferLossScale = std::max(0.0f, config.transferLossScale);
	config.transferLossExponent = std::max(0.0f, config.transferLossExponent);
	config.transferContinuousActionTableTopK = std::max(0, config.transferContinuousActionTableTopK);
	config.transferContinuousActionTableLossWeight = std::max(0.0f, config.transferContinuousActionTableLossWeight);
	config.transferContinuousActionMSEWeight = std::max(0.0f, config.transferContinuousActionMSEWeight);
	config.transferContinuousActionNLLWeight = std::max(0.0f, config.transferContinuousActionNLLWeight);
	config.transferContinuousStdLossWeight = std::max(0.0f, config.transferContinuousStdLossWeight);
	config.transferContinuousTargetStd = std::max(0.001f, config.transferContinuousTargetStd);
	config.transferContinuousButtonBCELossWeight = std::max(0.0f, config.transferContinuousButtonBCELossWeight);
	config.transferContinuousAerialSampleWeight = std::max(0.0f, config.transferContinuousAerialSampleWeight);
	config.transferStudentRolloutProb = ClampFloat(config.transferStudentRolloutProb, 0.0f, 1.0f);
	config.transferStudentRolloutWarmupTimesteps = std::max<int64_t>(0, config.transferStudentRolloutWarmupTimesteps);

	ImGui::End();
}

void GuiApp::RenderGPUStatusPanel() {
	const LockedGuiLayout layout = BuildLockedGuiLayout(ImGui::GetMainViewport());
	BeginPanel("GPU", layout.gpu);
	Draw::PanelTitle("System");

	bool cuda = state.cudaAvailable;
	ImGui::Text("CUDA: %s", cuda ? "available" : "not available");
	ImGui::Text("Device mode: %s", config.useCuda ? "GPU_CUDA" : "CPU");
	ImGui::Text("Physics: %s", config.useCuda ? "RocketSimCuda" : "RocketSim CPU");
	ImGui::Text("Checkpoint dir: %s", config.checkpointDir.c_str());
	ImGui::Text("Meshes: %s", std::filesystem::is_directory(config.collisionMeshesDir) ? "found" : "missing");

	int64_t progress = state.collectionProgress.load();
	int64_t target = state.collectionTarget.load();
	float fraction = target > 0 ? ClampFloat((float)((double)progress / (double)target), 0.0f, 1.0f) : 0.0f;
	char overlay[64];
	std::snprintf(overlay, sizeof(overlay), "%d%%", (int)(fraction * 100.0f));
	Draw::GradientProgressBar(fraction, ImVec2(ImGui::GetContentRegionAvail().x, 16.0f), Colors::Gold, overlay);

	if (state.running.load()) {
		float elapsed = std::chrono::duration<float>(std::chrono::steady_clock::now() - trainStartTime).count();
		ImGui::Text("Runtime: %s", FormatDuration(elapsed).c_str());
	}

	ImGui::End();
}

void GuiApp::RenderRewardsPanel() {
	const LockedGuiLayout layout = BuildLockedGuiLayout(ImGui::GetMainViewport());
	BeginPanel("Rewards", layout.rewards);
	Draw::PanelTitle("Rewards");

	bool locked = state.running.load();
	if (locked) ImGui::BeginDisabled();

	static int rewardToAdd = 0;
	ImGui::SetNextItemWidth(std::max(120.0f, ImGui::GetContentRegionAvail().x - 58.0f));
	ImGui::Combo("##rewardAdd", &rewardToAdd, kKnownRewards, IM_ARRAYSIZE(kKnownRewards));
	ImGui::SameLine();
	if (ImGui::Button("Add")) {
		config.rewards.push_back({ kKnownRewards[ClampInt(rewardToAdd, 0, IM_ARRAYSIZE(kKnownRewards) - 1)], 1.0f, false, 0.0f });
	}

	ImGui::Separator();
	if (ImGui::BeginTable("RewardsTable", 4, ImGuiTableFlags_RowBg | ImGuiTableFlags_BordersInnerV | ImGuiTableFlags_SizingStretchProp)) {
		ImGui::TableSetupColumn("Reward", ImGuiTableColumnFlags_WidthStretch);
		ImGui::TableSetupColumn("Weight", ImGuiTableColumnFlags_WidthFixed, 72.0f);
		ImGui::TableSetupColumn("ZS", ImGuiTableColumnFlags_WidthFixed, 34.0f);
		ImGui::TableSetupColumn("", ImGuiTableColumnFlags_WidthFixed, 28.0f);
		ImGui::TableHeadersRow();
		for (int i = 0; i < (int)config.rewards.size(); ++i) {
			RewardEntry& reward = config.rewards[i];
			ImGui::PushID(i);
			ImGui::TableNextRow();
			ImGui::TableSetColumnIndex(0);
			ImGui::TextUnformatted(reward.className.c_str());
			ImGui::TableSetColumnIndex(1);
			ImGui::SetNextItemWidth(-FLT_MIN);
			ImGui::InputFloat("##w", &reward.weight, 0.0f, 0.0f, "%.3g");
			ImGui::TableSetColumnIndex(2);
			ImGui::Checkbox("##zs", &reward.zeroSum);
			ImGui::TableSetColumnIndex(3);
			if (ImGui::SmallButton("x")) {
				config.rewards.erase(config.rewards.begin() + i);
				--i;
			}
			ImGui::PopID();
		}
		ImGui::EndTable();
	}

	if (locked) ImGui::EndDisabled();
	ImGui::End();
}

void GuiApp::RenderLogPanel() {
	const LockedGuiLayout layout = BuildLockedGuiLayout(ImGui::GetMainViewport());
	BeginPanel("Logs", layout.logs);
	Draw::PanelTitle("Logs");
	ImGui::Checkbox("Auto-scroll", &autoScrollLogs);
	ImGui::SameLine();
	if (ImGui::Button("Clear")) {
		std::lock_guard<std::mutex> lock(state.mtx);
		state.logs.clear();
	}
	ImGui::Separator();

	ImGui::BeginChild("##logscroll", ImVec2(0, 0), false);
	{
		std::lock_guard<std::mutex> lock(state.mtx);
		for (const LogEntry& log : state.logs) {
			ImVec4 col = log.level == LogEntry::ERR ? ImVec4(1.0f, 0.35f, 0.3f, 1.0f) :
				(log.level == LogEntry::WARN ? ImVec4(1.0f, 0.84f, 0.2f, 1.0f) : ImVec4(0.8f, 0.8f, 0.78f, 1.0f));
			ImGui::TextDisabled("[%s]", log.timestamp.c_str());
			ImGui::SameLine();
			ImGui::PushStyleColor(ImGuiCol_Text, col);
			ImGui::TextWrapped("%s", log.message.c_str());
			ImGui::PopStyleColor();
		}
	}
	if (autoScrollLogs && ImGui::GetScrollY() >= ImGui::GetScrollMaxY() - 8.0f)
		ImGui::SetScrollHereY(1.0f);
	ImGui::EndChild();
	ImGui::End();
}

void GuiApp::RenderPlot(const char* label, const MetricSeries& series, float height) {
	if (series.values.empty()) {
		ImGui::BeginChild(label, ImVec2(0, height), true);
		ImGui::TextDisabled("No samples yet");
		ImGui::EndChild();
		return;
	}

	std::vector<float> values(series.values.begin(), series.values.end());
	float minVal = series.minVal;
	float maxVal = series.maxVal;
	if (std::abs(maxVal - minVal) < 1e-6f) {
		minVal -= 1.0f;
		maxVal += 1.0f;
	}
	ImGui::PlotLines(label, values.data(), (int)values.size(), 0, nullptr, minVal, maxVal, ImVec2(0, height));
}

void GuiApp::StartTraining(GuiRunMode mode) {
	if (state.running.load()) return;
	if (trainingThread.joinable())
		trainingThread.join();

	if (!std::filesystem::is_directory(config.collisionMeshesDir)) {
		state.PushLog(LogEntry::ERR, "Collision meshes not found: " + config.collisionMeshesDir);
		return;
	}
	if (mode == GuiRunMode::TransferLearn && !std::filesystem::is_directory(config.transferOldModelsPath)) {
		state.PushLog(LogEntry::ERR, "Teacher checkpoint folder not found: " + config.transferOldModelsPath);
		return;
	}
	if (mode == GuiRunMode::TransferLearn) {
		if (TrimCopy(config.transferStudentCheckpointDir).empty())
			config.transferStudentCheckpointDir = "checkpoints_transfer";

		std::filesystem::path studentDir = config.transferStudentCheckpointDir;
		std::filesystem::path teacherDir = config.transferOldModelsPath;
		if (PathsEquivalentNoThrow(studentDir, teacherDir) ||
			PathsEquivalentNoThrow(studentDir, teacherDir.parent_path())) {
			state.PushLog(
				LogEntry::ERR,
				"Transfer student checkpoint dir must be separate from the teacher checkpoint folder: " +
				config.transferStudentCheckpointDir
			);
			return;
		}
	}

	config.renderMode = mode == GuiRunMode::Render;
	config.transferLearning = mode == GuiRunMode::TransferLearn;
	config.SaveToFile(kConfigPath);
	{
		std::lock_guard<std::mutex> lock(g_configMutex);
		g_trainingConfig = config;
	}

	{
		std::lock_guard<std::mutex> lock(state.mtx);
		state.rewardSeries.Clear();
		state.policyLossSeries.Clear();
		state.criticLossSeries.Clear();
		state.entropySeries.Clear();
		state.stepsPerSecSeries.Clear();
		state.touchSeries.Clear();
		state.boostSeries.Clear();
		state.transferLossSeries.Clear();
		state.transferAccuracySeries.Clear();
		state.teacherConfidenceSeries.Clear();
		state.rewardMetrics.clear();
		state.activeMode.store((int)mode);
	}
	state.collectionProgress = 0;
	state.collectionTarget = mode == GuiRunMode::TransferLearn ? config.transferBatchSize : config.stepsPerIteration;
	state.running = true;
	state.stopRequested = false;
	trainStartTime = std::chrono::steady_clock::now();
	const char* modeName = mode == GuiRunMode::TransferLearn ? "transfer learning" :
		(mode == GuiRunMode::Render ? "render" : "training");
	state.PushLog(LogEntry::INFO, std::string("Starting Prometheus ") + modeName +
		" with continuous actions and attention head.");

	trainingThread = std::thread([this, mode]() {
		Learner* localLearner = nullptr;
		GuiTrainingConfig localConfig = config;
		try {
			if (mode == GuiRunMode::TransferLearn)
				localConfig.checkpointDir = localConfig.transferStudentCheckpointDir;

			RocketSim::Init(localConfig.collisionMeshesDir, true);
			LearnerConfig learnerCfg = BuildLearnerConfig(localConfig);

			auto stepCb = [this, localConfig](Learner* learner, const std::vector<GameState>& states, Report& report) {
				(void)learner;
				bool doExpensiveMetrics = (std::rand() % 4) == 0;
				for (const GameState& stateValue : states) {
					if (doExpensiveMetrics) {
						for (const Player& player : stateValue.players) {
							report.AddAvg("Player/In Air Ratio", !player.isOnGround);
							report.AddAvg("Player/Ball Touch Ratio", player.ballTouchedStep);
							report.AddAvg("Player/Speed", player.vel.Length());
							Vec dirToBall = (stateValue.ball.pos - player.pos).Normalized();
							report.AddAvg("Player/Speed Towards Ball", RS_MAX(0, player.vel.Dot(dirToBall)));
							report.AddAvg("Player/Boost", player.boost);
							if (player.ballTouchedStep)
								report.AddAvg("Player/Touch Height", stateValue.ball.pos.z);
						}
					}
					if (stateValue.goalScored)
						report.AddAvg("Game/Goal Speed", stateValue.ball.vel.Length());
				}
			};

			localLearner = new Learner(GuiEnvCreateFunc, learnerCfg, stepCb);
			localLearner->collectionProgressCallback = [this](Learner*, int64_t collected, int64_t target) {
				state.collectionProgress = collected;
				state.collectionTarget = target;
			};
			localLearner->iterationCallback = [this, localConfig, mode](Learner* learner, const Report& report) {
				auto get = [&report](const std::string& key) -> float {
					return report.Has(key) ? (float)report.data.at(key) : 0.0f;
				};

				std::lock_guard<std::mutex> lock(state.mtx);
				state.totalTimesteps = learner->totalTimesteps;
				state.iteration = (int)learner->totalIterations;
				state.meanReward = get("Average Step Reward");
				state.policyLoss = get("Policy Loss");
				state.criticLoss = get("Critic Loss");
				state.entropy = get("Policy Entropy");
				state.meanPolicyStd = get("Mean Policy Std");
				state.clipFraction = get("SB3 Clip Fraction");
				state.ppoLearnTime = report.Has("PPO Learn Time") ? get("PPO Learn Time") : get(" - PPO Learn Time");
				state.collectionTime = get("Collection Time");
				state.consumptionTime = get("Consumption Time");
				state.touchRatio = get("Player/Ball Touch Ratio");
				state.inAirRatio = get("Player/In Air Ratio");
				state.avgBoost = get("Player/Boost");
				float sps = get("Overall Steps/Second");
				state.stepsPerSec = sps;

				state.rewardSeries.Push(state.meanReward);
				state.policyLossSeries.Push(state.policyLoss);
				state.criticLossSeries.Push(state.criticLoss);
				state.entropySeries.Push(state.entropy);
				state.stepsPerSecSeries.Push(sps);
				state.touchSeries.Push(state.touchRatio);
				state.boostSeries.Push(state.avgBoost);
				state.transferLoss = get("Transfer Learn Loss");
				state.transferAccuracy = get("Transfer Learn Accuracy");
				state.transferTableLoss = get("Transfer Learn Table Loss");
				state.transferMSE = get("Transfer Learn MSE");
				state.transferMAE = get("Transfer Learn MAE");
				state.transferNLL = get("Transfer Learn NLL");
				state.transferStdLoss = get("Transfer Learn Std Loss");
				state.transferButtonBCE = get("Transfer Learn Button BCE");
				state.teacherConfidence = get("Teacher Confidence");
				state.oldPolicyEntropy = get("Old Policy Entropy");
				state.studentRolloutRatio = get("Student Rollout Ratio");
				state.aerialTargetRatio = get("Aerial Target Ratio");
				state.transferLossSeries.Push(state.transferLoss);
				state.transferAccuracySeries.Push(state.transferAccuracy);
				state.teacherConfidenceSeries.Push(state.teacherConfidence);

				for (const auto& pair : report.data) {
					if (pair.first.rfind("Rewards/", 0) == 0) {
						std::string name = pair.first.substr(8);
						auto& series = state.rewardMetrics[name];
						if (series.name.empty()) series.name = name;
						series.Push((float)pair.second);
					}
				}
				state.collectionProgress = 0;
				state.collectionTarget = mode == GuiRunMode::TransferLearn ? localConfig.transferBatchSize : localConfig.stepsPerIteration;
			};

			{
				std::lock_guard<std::mutex> lock(learnerMutex);
				learner = localLearner;
				if (state.stopRequested.load())
					learner->stopRequested = true;
			}

			if (mode == GuiRunMode::TransferLearn) {
				state.PushLog(LogEntry::INFO, "Learner ready. Entering transfer learning loop.");
				TransferLearnConfig tlCfg = BuildTransferLearnConfig(localConfig);
				localLearner->StartTransferLearn(tlCfg);
			} else {
				state.PushLog(LogEntry::INFO, "Learner ready. Entering PPO loop.");
				localLearner->Start();
			}
			state.PushLog(LogEntry::INFO, "Learner loop finished.");
		} catch (const std::exception& e) {
			state.PushLog(LogEntry::ERR, std::string("Training error: ") + e.what());
		}

		{
			std::lock_guard<std::mutex> lock(learnerMutex);
			if (learner == localLearner)
				learner = nullptr;
		}
		delete localLearner;
		state.running = false;
	});
}

void GuiApp::StopTraining() {
	if (!state.running.load())
		return;
	state.stopRequested = true;
	state.PushLog(LogEntry::WARN, "Stop requested. Waiting for the active iteration to finish.");
	std::lock_guard<std::mutex> lock(learnerMutex);
	if (learner)
		learner->stopRequested = true;
}

void GuiApp::SaveCheckpoint() {
	bool queued = false;
	{
		std::lock_guard<std::mutex> lock(learnerMutex);
		if (learner) {
			learner->saveRequested = true;
			queued = true;
		}
	}
	state.PushLog(queued ? LogEntry::INFO : LogEntry::WARN,
		queued ? "Checkpoint save queued for the next iteration." : "No active learner to save.");
}

std::string GuiApp::FormatTimesteps(int64_t timesteps) {
	char buffer[32];
	if (timesteps >= 1000000000LL)
		std::snprintf(buffer, sizeof(buffer), "%.2fB", timesteps / 1e9);
	else if (timesteps >= 1000000LL)
		std::snprintf(buffer, sizeof(buffer), "%.2fM", timesteps / 1e6);
	else if (timesteps >= 1000LL)
		std::snprintf(buffer, sizeof(buffer), "%.1fK", timesteps / 1e3);
	else
		std::snprintf(buffer, sizeof(buffer), "%lld", (long long)timesteps);
	return buffer;
}

std::string GuiApp::FormatDuration(float seconds) {
	int totalH = (int)(seconds / 3600);
	int d = totalH / 24;
	int h = totalH % 24;
	int m = (int)(std::fmod(seconds, 3600.0f) / 60);
	int s = (int)(std::fmod(seconds, 60.0f));
	char buffer[64];
	if (d > 0) std::snprintf(buffer, sizeof(buffer), "%dj %02dh %02dm", d, h, m);
	else if (totalH > 0) std::snprintf(buffer, sizeof(buffer), "%dh%02dm%02ds", totalH, m, s);
	else if (m > 0) std::snprintf(buffer, sizeof(buffer), "%dm%02ds", m, s);
	else std::snprintf(buffer, sizeof(buffer), "%ds", s);
	return buffer;
}

} // namespace prometheus
