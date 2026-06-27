#pragma once
#include <RLGymCPP/BasicTypes/Lists.h>

#include "../Util/ModelConfig.h"

namespace GGL {

	struct AttentionModelConfig {
		int numInputs = -1;

		int numDims = 256;
		int thinkDims = 1024;
		int numBlocks = 2;
		int numHeads = 4;
		int preprocessLayers = 1;
		int postprocessLayers = 0;

		int refinementFeedforward = 0;

		ModelActivationType activationType = ModelActivationType::RELU;
		ModelOptimType optimType = ModelOptimType::ADAM;

		bool IsValid() const {
			return numInputs > 0 && numDims > 0 && thinkDims > 0 && numBlocks > 0 && numHeads > 0;
		}

		int GetOutputDims() const { return numDims; }
	};

	// Policy type: discrete softmax policy or hybrid analog+button policy.
	enum class PolicyType { DISCRETE, CONTINUOUS };

	// https://github.com/AechPro/rlgym-ppo/blob/main/rlgym_ppo/ppo/ppo_learner.py
	struct PPOLearnerConfig {

		// === Policy Type ===
		PolicyType policyType = PolicyType::DISCRETE; // Default: discrete actions

		// === Hybrid Action Config (only used when policyType == CONTINUOUS) ===
		// External action layout remains 8 floats:
		// throttle, steer, pitch, yaw, roll, jump, boost, handbrake.
		// Internally, the first 5 controls use a diagonal Gaussian and the last 3 use Bernoulli logits.
		int continuousActionSize = 8;
		float varMin = 0.1f;            // Minimum std for the analog Gaussian controls
		float varMax = 1.0f;            // Maximum std for the analog Gaussian controls

		int64_t tsPerItr = 50'000;
		int64_t batchSize = 50'000;
		int64_t miniBatchSize = 0; // Set to 0 to just use batchSize

		// On the last batch of the iteration, 
		//	if the amount of remaining experience exceeds the batch size, 
		//	all remaining experience is used as a larger batch.
		// This prevents experience loss due to batch size rounding.
		// This will only happen if the amount of remaining experience is < batchSize*2.
		bool overbatching = true;

		double maxEpisodeDuration = 120; // In seconds

		// Actions with the highest probability are always chosen, instead of being more likely
		// This will make your bot play better (usually), but is horrible for learning
		// Trying to run a PPO learn iteration with deterministic mode will throw an exception
		bool deterministic = false;

		// Use half-precision models for inference
		// This is much faster on GPU, not so much for CPU
		bool useHalfPrecision = false;

		PartialModelConfig policy, critic, sharedHead;

		// === Attention Head Config ===
		// When true, the shared head uses an AttentionModel (Refinement->Think blocks)
		// instead of a plain MLP Model. Ignores sharedHead config when enabled.
		bool useAttentionHead = false;
		// Config for the attention shared head (only used when useAttentionHead = true)
		// Set numInputs to -1 to auto-detect from obsSize
		AttentionModelConfig* attentionHeadConfig = nullptr;

		int epochs = 2;
		float policyLR = 3e-4f; // Policy learning rate
		float criticLR = 3e-4f; // Critic learning rate

		float entropyScale = 0.018f; // The scale of the normalized entropy loss
		// Whether to ignore invalid actions in the entropy calculation.
		// True means that entropy will be determined only from available actions.
		// False means that entropy for unavailable actions will be zero, 
		//	meaning the entropy of the state is limited to the fraction of available actions in that state.
		bool maskEntropy = false; 

		float clipRange = 0.2f;
		float vfCoef = 0.5f; // Value function loss coefficient (balances critic vs policy gradient in shared head)
		
		// Temperature of the policy's softmax distribution
		float policyTemperature = 1;

		float gaeLambda = 0.95f;
		float gaeGamma = 0.99f;
		float rewardClipRange = 10; // Clip range for normalized rewards, set 0 to disable

		bool useGuidingPolicy = false;
		std::filesystem::path guidingPolicyPath = "guiding_policy/"; // Path of the guiding policy model(s)
		float guidingStrength = 0.03f;

		PPOLearnerConfig() {
			policy = {};
			policy.layerSizes = { 256, 256, 256 };
			critic = {};
			critic.layerSizes = { 256, 256, 256 };
			sharedHead = {};
			sharedHead.layerSizes = { 256 };
			sharedHead.addOutputLayer = false;
		}
	};
}
