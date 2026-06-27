#pragma once
#include "../Gamestates/GameState.h"
#include "../BasicTypes/Action.h"
#include "../TerminalConditions/TerminalCondition.h"
#include "../Rewards/Reward.h"
#include "../ObsBuilders/ObsBuilder.h"
#include "../ActionParsers/ActionParser.h"
#include "../StateSetters/StateSetter.h"
#include "../ThreadPool.h"
#include <RLGymCPP/Rewards/Reward.h>
#include <memory>

#ifdef RG_ROCKETSIMCUDA_AVAILABLE
#include <RocketSimCuda.h>
#endif

namespace RLGC {

	struct EnvCreateResult {
		Arena* arena;
		std::vector<WeightedReward> rewards;
		std::vector<TerminalCondition*> terminalConditions;
		ObsBuilder* obsBuilder;
		ActionParser* actionParser;
		ContinuousActionParser* continuousActionParser = nullptr; // Optional, for continuous policy
		StateSetter* stateSetter;

		void* userInfo; // Optional userinfo pointer if you want to track some data per-env
	};
	typedef std::function<EnvCreateResult(int index)> EnvCreateFn;

	enum class EnvPhysicsBackend : uint8_t {
		ROCKETSIM_CPU = 0,
		ROCKETSIM_CUDA = 1
	};

	struct EnvSetConfig {
		EnvCreateFn envCreateFn;
		int numArenas;
		int tickSkip;
		int actionDelay;
		bool saveRewards;
		bool shuffleRewardSampling = true;
		EnvPhysicsBackend physicsBackend = EnvPhysicsBackend::ROCKETSIM_CPU;
		bool syncCudaWorldStateToCpu = true;
	};

	struct EnvState {
		int numPlayers;
		std::vector<GameState> gameStates;
		std::vector<GameState> prevGameStates;
		DimList2<float> obs;
		DimList2<uint8_t> actionMasks;
		std::vector<float> rewards;
		std::vector<std::vector<float>> lastRewards; // Only from the first arena
		std::vector<uint8_t> terminals;

		std::vector<int> arenaPlayerStartIdx = {};

		void Resize(std::vector<Arena*>& arenas) {
			numPlayers = 0;
			for (int i = 0; i < arenas.size(); i++) {
				arenaPlayerStartIdx.push_back(numPlayers);
				numPlayers += arenas[i]->_cars.size();
			}

			gameStates.resize(arenas.size());
			prevGameStates.resize(arenas.size());
			rewards.resize(numPlayers);
			lastRewards.resize(arenas.size());
			terminals.resize(arenas.size());
		}
	};

	struct EnvSet {

		struct CallbackUserInfo {
			RLGC::EnvSet* envSet;
			Arena* arena;
			int arenaIdx;
		};

		std::vector<Arena*> arenas;
		std::vector<GameEventTracker*> eventTrackers;

		std::vector<CallbackUserInfo*> eventCallbackInfos;
		std::vector<void*> userInfos;

		EnvSetConfig config;

		int obsSize;
		int numActions;

		std::vector<std::vector<WeightedReward>> rewards;
		std::vector<std::vector<TerminalCondition*>> terminalConditions;
		std::vector<ObsBuilder*> obsBuilders;
		std::vector<ActionParser*> actionParsers;
		std::vector<ContinuousActionParser*> continuousActionParsers; // Optional, parallel to actionParsers
		std::vector<StateSetter*> stateSetters;

		EnvState state = {};

		// When the CUDA reward fast path is active, lastRewards is metrics-only
		// (training rewards come from the GPU). Consumers (Learner) only read
		// the first maxRewardSamples arenas, so the expensive CPU reward eval
		// can be limited to those. INT32_MAX = evaluate all (default behavior).
		int saveRewardsArenaLimit = INT32_MAX;

#ifdef RG_ROCKETSIMCUDA_AVAILABLE
		std::unique_ptr<rsc::RocketSimCudaBatch> cudaBatch;
		int cudaMaxCarsPerArena = 0;
		bool cudaAdvancedObsDefaultActionFastPath = false;
		bool cudaRewardsTerminalFastPath = false;
		std::vector<std::vector<Car*>> cudaArenaCars;
		std::vector<rsc::CarControls> cudaControlBuffer;
		std::vector<rsc::CarState> cudaAllCarStates;
		std::vector<rsc::BallState> cudaAllBallStates;
		std::vector<rsc::BoostPadState> cudaAllBoostPadStates;
		std::vector<rsc::ArenaInfo> cudaAllArenaInfos;
		std::vector<float> cudaTrainingRewards;
		std::vector<uint8_t> cudaTrainingTerminals;
#endif

		EnvSet(const EnvSetConfig& config);

		RG_NO_COPY(EnvSet);

		~EnvSet() {
			for (Arena* arena : arenas)
				delete arena;

			for (auto& eventTracker : eventTrackers)
				delete eventTracker;
			for (auto& eventCallbackInfo : eventCallbackInfos)
				delete eventCallbackInfo;
		}

		////////////////////
		
		void StepFirstHalf(bool async);
		void StepSecondHalf(const IList& actionIndices, bool async);
		void StepSecondHalfContinuous(const FList& continuousActions, int actionDim, bool async);
		void Sync() { g_ThreadPool.WaitUntilDone(); }
		void ResetArena(int index);
		void Reset();

#ifdef RG_ROCKETSIMCUDA_AVAILABLE
		bool UsingCudaBackend() const {
			return config.physicsBackend == EnvPhysicsBackend::ROCKETSIM_CUDA && cudaBatch != NULL;
		}

		bool UsingCudaTrainingFastPath() const {
			return UsingCudaBackend() && cudaAdvancedObsDefaultActionFastPath;
		}

		bool UsingCudaRewardTerminalFastPath() const {
			return UsingCudaBackend() && cudaRewardsTerminalFastPath;
		}

		bool UsingCudaNoCpuWorldStateFastPath() const {
			return UsingCudaTrainingFastPath() &&
				UsingCudaRewardTerminalFastPath() &&
				!config.syncCudaWorldStateToCpu &&
				!config.saveRewards;
		}

		void RefreshCudaTrainingBuffers();
		void SyncCudaDevice() const;
		void* GetCudaObsDevicePtr() const;
		void* GetCudaActionMasksDevicePtr() const;
		void* GetCudaRewardsDevicePtr() const;
#endif
	};
}
