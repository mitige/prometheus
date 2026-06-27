#pragma once
#include "ExperienceBuffer.h"
#include <GigaLearnCPP/Util/Report.h>
#include <GigaLearnCPP/Util/Timer.h>
#include <GigaLearnCPP/PPO/PPOLearnerConfig.h>
#include <GigaLearnCPP/PPO/TransferLearnConfig.h>

#include "../Util/Models.h"
#include "../Util/AttentionModel.h"

#include <torch/optim/adam.h>
#include <torch/nn/modules/loss.h>
#include <torch/nn/modules/container/sequential.h>
namespace GGL {

	// https://github.com/AechPro/rlgym-ppo/blob/main/rlgym_ppo/ppo/ppo_learner.py
	class PPOLearner {
	public:
		ModelSet models = {};
		ModelSet guidingPolicyModels = {};

		PPOLearnerConfig config;
		torch::Device device;

		PPOLearner(
			int obsSize, int numActions,
			PPOLearnerConfig config, torch::Device device
		);

		static int GetContinuousPolicyOutputSize(int continuousActionSize);

		static void MakeModels(
			bool makeCritic, 
			int obsSize, int numActions, 
			PartialModelConfig sharedHeadConfig, PartialModelConfig policyConfig, PartialModelConfig criticConfig,
			torch::Device device,
			ModelSet& outModels,
			PolicyType policyType = PolicyType::DISCRETE,
			int continuousActionSize = 8
		);

		// Create models with attention-based shared head
		static void MakeModelsWithAttention(
			bool makeCritic,
			int obsSize, int numActions,
			AttentionModelConfig attentionConfig, PartialModelConfig policyConfig, PartialModelConfig criticConfig,
			torch::Device device,
			ModelSet& outModels,
			PolicyType policyType = PolicyType::DISCRETE,
			int continuousActionSize = 8
		);
		
		// If models is null, this->models will be used
		void InferActions(torch::Tensor obs, torch::Tensor actionMasks, torch::Tensor* outActions, torch::Tensor* outLogProbs, ModelSet* models = NULL);
		torch::Tensor InferCritic(torch::Tensor obs);

		// === Discrete Policy Inference ===
		static torch::Tensor InferPolicyProbsFromModels(
			ModelSet& models, 
			torch::Tensor obs, torch::Tensor actionMasks, 
			float temperature,
			bool halfPrec
		);
		static void InferActionsFromModels(
			ModelSet& models, 
			torch::Tensor obs, torch::Tensor actionMasks, 
			bool deterministic, float temperature, bool halfPrec,
			torch::Tensor* outActions, torch::Tensor* outLogProbs
		);

		// === Hybrid Policy Inference ===
		// Manual Gaussian log-pdf for the analog controls.
		static torch::Tensor GaussianLogPdf(torch::Tensor x, torch::Tensor mean, torch::Tensor stddev);

		// Run shared head + policy model and split into analog Gaussian params + button logits.
		static void InferContinuousPolicyFromModels(
			ModelSet& models, torch::Tensor obs, bool halfPrec,
			torch::Tensor& outMean, torch::Tensor& outStd, torch::Tensor& outButtonLogits,
			float varMin, float varMax
		);

		// Sample hybrid actions and compute log probs for the analog + button factors.
		static void SampleContinuousActions(
			ModelSet& models, torch::Tensor obs,
			bool deterministic, bool halfPrec,
			float varMin, float varMax,
			torch::Tensor* outActions, torch::Tensor* outLogProbs
		);

		// Compute normalized entropy for the hybrid policy.
		static torch::Tensor ComputeContinuousEntropy(torch::Tensor mean, torch::Tensor stddev, torch::Tensor buttonLogits, float varMax);

		void Learn(ExperienceBuffer& experience, Report& report, bool isFirstIteration);

		void TransferLearn(
			ModelSet& oldModels, 
			torch::Tensor newObs, torch::Tensor oldObs, 
			torch::Tensor newActionMasks, torch::Tensor oldActionMasks, 
			torch::Tensor actionMaps,
			torch::Tensor teacherActionTable,
			torch::Tensor teacherActionTargets,
			Report& report, 
			const TransferLearnConfig& transferLearnConfig
		);

		void SaveTo(std::filesystem::path folderPath);
		void LoadFrom(std::filesystem::path folderPath);
		void SetLearningRates(float policyLR, float criticLR);

		ModelSet GetPolicyModels();
	};
}
