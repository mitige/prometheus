#pragma once
#include <RLGymCPP/Framework.h>
#include "../FrameworkTorch.h"
#include "Models.h"

#include <torch/nn/modules/activation.h>
#include <torch/nn/modules/container/modulelist.h>
#include <torch/nn/modules/container/sequential.h>
#include <torch/nn/modules/normalization.h>
#include <torch/nn/modules/linear.h>

#include <GigaLearnCPP/Util/ModelConfig.h>
#include <GigaLearnCPP/PPO/PPOLearnerConfig.h>

namespace GGL {

	/////////////////////////////
	// Refinement Block
	/////////////////////////////

	class RefinementBlockImpl : public torch::nn::Module {
	public:
		torch::nn::MultiheadAttention attention;
		torch::nn::Linear linear1, linear2;
		torch::nn::LayerNorm norm_q, norm_kv, norm_ff;

		RefinementBlockImpl(int n_dims, int n_heads, int dim_feedforward)
			: attention(torch::nn::MultiheadAttentionOptions(n_dims, n_heads)),
			  linear1(n_dims, dim_feedforward),
			  linear2(dim_feedforward, n_dims),
			  norm_q(torch::nn::LayerNormOptions({(int64_t)n_dims})),
			  norm_kv(torch::nn::LayerNormOptions({(int64_t)n_dims})),
			  norm_ff(torch::nn::LayerNormOptions({(int64_t)n_dims}))
		{
			register_module("attention", attention);
			register_module("linear1", linear1);
			register_module("linear2", linear2);
			register_module("norm_q", norm_q);
			register_module("norm_kv", norm_kv);
			register_module("norm_ff", norm_ff);
		}

		torch::Tensor forward(torch::Tensor query, torch::Tensor key_value,
							  torch::Tensor mask = {}) {
			// Pre-LN cross-attention
			auto q_norm = norm_q->forward(query);
			auto kv_norm = norm_kv->forward(key_value);

			// MultiheadAttention without batch_first expects (seq, batch, dim)
			q_norm = q_norm.transpose(0, 1);
			kv_norm = kv_norm.transpose(0, 1);

			auto attn_result = attention->forward(q_norm, kv_norm, kv_norm,
				/*key_padding_mask=*/mask);
			auto attn_out = std::get<0>(attn_result);
			// Transpose back to (batch, seq, dim)
			attn_out = attn_out.transpose(0, 1);
			query = query + attn_out;

			// Pre-LN feedforward
			auto ff_norm = norm_ff->forward(query);
			auto ff_out = linear2->forward(torch::relu(linear1->forward(ff_norm)));
			query = query + ff_out;

			return query;
		}
	};
	TORCH_MODULE(RefinementBlock);

	/////////////////////////////
	// Think Block
	/////////////////////////////

	class ThinkBlockImpl : public torch::nn::Module {
	public:
		torch::nn::Linear expand, contract;
		torch::nn::LayerNorm norm;

		ThinkBlockImpl(int n_dims, int think_dims)
			: expand(n_dims, think_dims),
			  contract(think_dims, n_dims),
			  norm(torch::nn::LayerNormOptions({(int64_t)n_dims}))
		{
			register_module("expand", expand);
			register_module("contract", contract);
			register_module("norm", norm);
		}

		torch::Tensor forward(torch::Tensor x) {
			// Pre-LN think expansion: n_dims -> think_dims -> n_dims
			auto x_norm = norm->forward(x);
			auto out = contract->forward(torch::relu(expand->forward(x_norm)));
			return x + out;
		}
	};
	TORCH_MODULE(ThinkBlock);

	/////////////////////////////
	// Attention Model
	/////////////////////////////

	class AttentionModel : public Model {
	public:
		AttentionModelConfig attnConfig;

		// Preprocessing MLPs
		torch::nn::Sequential queryPreprocess, kvPreprocess;

		// Alternating blocks
		torch::nn::ModuleList refinementBlocks, thinkBlocks;

		// Postprocessing
		torch::nn::Sequential postprocess;

		// Half-precision copies
		torch::nn::Sequential queryPreprocessHalf, kvPreprocessHalf, postprocessHalf;
		torch::nn::ModuleList refinementBlocksHalf, thinkBlocksHalf;
		bool _attnHalfOutdated = true;

		AttentionModel() : Model() {} // Uninitialized

		AttentionModel(
			const char* modelName,
			AttentionModelConfig config,
			torch::Device device
		);

		// Forward: takes batched observations, internally splits for perceiver-style attention
		// For shared-head usage: obs is (batch, obs_size), treated as single query attending to itself
		torch::Tensor Forward(torch::Tensor input, bool halfPrec) override;

		// Perceiver-style forward with explicit query and key-value inputs
		torch::Tensor ForwardPerceiver(torch::Tensor query, torch::Tensor keyValue,
									   bool halfPrec, torch::Tensor mask = {});

		void Save(std::filesystem::path folder, bool saveOptim = true) override;
		void Load(std::filesystem::path folder, bool allowNotExist, bool loadOptim = true) override;
		torch::Tensor CopyParams() const override;

		void StepOptim() override {
			Model::StepOptim();
			_attnHalfOutdated = true;
		}

		Model* MakeEmptyClone() override {
			return new AttentionModel(modelName, attnConfig, device);
		}

		Model* MakeClone() override {
			RG_NO_GRAD;
			auto* clone = static_cast<AttentionModel*>(MakeEmptyClone());
			auto fromParams = this->parameters();
			auto toParams = clone->parameters();
			for (size_t i = 0; i < fromParams.size(); i++)
				toParams[i].copy_(fromParams[i], true);
			return clone;
		}

		uint64_t GetParamCount();
	};

} // namespace GGL
