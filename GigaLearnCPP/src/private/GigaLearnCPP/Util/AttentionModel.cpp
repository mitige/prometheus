#include "AttentionModel.h"

#include <torch/csrc/api/include/torch/serialize.h>
#include <torch/csrc/api/include/torch/nn/utils/convert_parameters.h>
#include <torch/nn/modules/normalization.h>
#include <torch/nn/init.h>

GGL::AttentionModel::AttentionModel(
	const char* modelName,
	AttentionModelConfig config,
	torch::Device device)
	: Model(), attnConfig(config),
	  queryPreprocess({}), kvPreprocess({}),
	  postprocess({}),
	  queryPreprocessHalf({}), kvPreprocessHalf({}),
	  postprocessHalf({})
{
	this->modelName = modelName;
	this->device = device;

	int dimFF = config.refinementFeedforward > 0 ? config.refinementFeedforward : 2 * config.numDims;

	// Build preprocessing MLP: input features -> n_dims
	{
		int lastSize = config.numInputs;
		for (int i = 0; i < config.preprocessLayers; i++) {
			queryPreprocess->push_back(torch::nn::Linear(lastSize, config.numDims));
			queryPreprocess->push_back(torch::nn::LayerNorm(
				torch::nn::LayerNormOptions({(int64_t)config.numDims})));
			AddActivationFunc(queryPreprocess, config.activationType);

			kvPreprocess->push_back(torch::nn::Linear(lastSize, config.numDims));
			kvPreprocess->push_back(torch::nn::LayerNorm(
				torch::nn::LayerNormOptions({(int64_t)config.numDims})));
			AddActivationFunc(kvPreprocess, config.activationType);

			lastSize = config.numDims;
		}
		if (config.preprocessLayers == 0) {
			// Identity: just a linear projection if dimensions don't match
			if (config.numInputs != config.numDims) {
				queryPreprocess->push_back(torch::nn::Linear(config.numInputs, config.numDims));
				kvPreprocess->push_back(torch::nn::Linear(config.numInputs, config.numDims));
			}
		}
	}

	// Build alternating Refinement -> Think blocks
	for (int i = 0; i < config.numBlocks; i++) {
		auto refine = RefinementBlock(config.numDims, config.numHeads, dimFF);
		auto think = ThinkBlock(config.numDims, config.thinkDims);

		refinementBlocks->push_back(refine);
		thinkBlocks->push_back(think);
	}

	// Build postprocessing MLP
	if (config.postprocessLayers > 0) {
		int lastSize = config.numDims;
		for (int i = 0; i < config.postprocessLayers; i++) {
			postprocess->push_back(torch::nn::Linear(lastSize, config.numDims));
			if (i < config.postprocessLayers - 1) {
				postprocess->push_back(torch::nn::LayerNorm(
					torch::nn::LayerNormOptions({(int64_t)config.numDims})));
				AddActivationFunc(postprocess, config.activationType);
			}
			lastSize = config.numDims;
		}
	}

	// Register all modules
	register_module("queryPreprocess", queryPreprocess);
	register_module("kvPreprocess", kvPreprocess);
	register_module("refinementBlocks", refinementBlocks);
	register_module("thinkBlocks", thinkBlocks);
	register_module("postprocess", postprocess);

	// Move to device
	this->to(device);

	// Create optimizer over all parameters
	optim = MakeOptimizer(config.optimType, this->parameters(), 0);

	// Xavier init for all params
	for (auto& p : this->parameters()) {
		if (p.dim() > 1)
			torch::nn::init::xavier_uniform_(p);
	}
}

torch::Tensor GGL::AttentionModel::Forward(torch::Tensor input, bool halfPrec) {
	// For shared-head usage: obs is (batch, obs_size)
	// Reshape to (batch, 1, obs_size) as single query and key-value
	bool wasFlatInput = (input.dim() == 2);
	if (wasFlatInput) {
		input = input.unsqueeze(1); // (batch, 1, features)
	}

	auto result = ForwardPerceiver(input, input, halfPrec);

	if (wasFlatInput) {
		result = result.squeeze(1); // Back to (batch, n_dims)
	}
	return result;
}

torch::Tensor GGL::AttentionModel::ForwardPerceiver(
	torch::Tensor query, torch::Tensor keyValue,
	bool halfPrec, torch::Tensor mask) {

	if (torch::GradMode::is_enabled())
		halfPrec = false;

	if (halfPrec) {
		// Sync half-precision copies when parameters are outdated
		if (_attnHalfOutdated) {
			_attnHalfOutdated = false;

			auto syncSeq = [](torch::nn::Sequential& src, torch::nn::Sequential& dst) {
				if (dst->size() == 0) {
					for (auto& mod : *src)
						dst->push_back(mod.clone());
					dst->to(RG_HALFPERC_TYPE, true);
				} else {
					auto from = src->parameters();
					auto to = dst->parameters();
					for (size_t i = 0; i < from.size(); i++)
						to[i].copy_(from[i].to(RG_HALFPERC_TYPE, true), true);
				}
			};

			auto syncModList = [](torch::nn::ModuleList& src, torch::nn::ModuleList& dst) {
				if (dst->size() == 0) {
					for (size_t i = 0; i < src->size(); i++)
						dst->push_back(src->ptr(i)->clone());
					dst->to(RG_HALFPERC_TYPE, true);
				} else {
					auto from = src->parameters();
					auto to = dst->parameters();
					for (size_t i = 0; i < from.size(); i++)
						to[i].copy_(from[i].to(RG_HALFPERC_TYPE, true), true);
				}
			};

			syncSeq(queryPreprocess, queryPreprocessHalf);
			syncSeq(kvPreprocess, kvPreprocessHalf);
			syncModList(refinementBlocks, refinementBlocksHalf);
			syncModList(thinkBlocks, thinkBlocksHalf);
			syncSeq(postprocess, postprocessHalf);
		}

		// Run in half precision
		auto hQuery = query.to(RG_HALFPERC_TYPE);
		auto hKeyValue = keyValue.to(RG_HALFPERC_TYPE);

		auto q_emb = queryPreprocessHalf->size() > 0 ? queryPreprocessHalf->forward(hQuery) : hQuery;
		auto kv_emb = kvPreprocessHalf->size() > 0 ? kvPreprocessHalf->forward(hKeyValue) : hKeyValue;

		for (int i = 0; i < attnConfig.numBlocks; i++) {
			auto refine = std::dynamic_pointer_cast<RefinementBlockImpl>(refinementBlocksHalf->ptr(i));
			q_emb = refine->forward(q_emb, kv_emb, mask);

			auto think = std::dynamic_pointer_cast<ThinkBlockImpl>(thinkBlocksHalf->ptr(i));
			q_emb = think->forward(q_emb);
		}

		if (postprocessHalf->size() > 0)
			q_emb = postprocessHalf->forward(q_emb);

		return q_emb.to(torch::kFloat);
	}

	// Full precision path
	auto q_emb = queryPreprocess->size() > 0 ? queryPreprocess->forward(query) : query;
	auto kv_emb = kvPreprocess->size() > 0 ? kvPreprocess->forward(keyValue) : keyValue;

	for (int i = 0; i < attnConfig.numBlocks; i++) {
		auto refine = std::dynamic_pointer_cast<RefinementBlockImpl>(refinementBlocks->ptr(i));
		q_emb = refine->forward(q_emb, kv_emb, mask);

		auto think = std::dynamic_pointer_cast<ThinkBlockImpl>(thinkBlocks->ptr(i));
		q_emb = think->forward(q_emb);
	}

	if (postprocess->size() > 0)
		q_emb = postprocess->forward(q_emb);

	return q_emb;
}

void GGL::AttentionModel::Save(std::filesystem::path folder, bool saveOptim) {
	std::filesystem::path path = GetSavePath(folder);

	// Save all submodules as a single archive
	torch::serialize::OutputArchive archive;

	auto saveMod = [&](const std::string& name, torch::nn::Module& mod) {
		torch::serialize::OutputArchive subArchive;
		mod.save(subArchive);
		archive.write(name, subArchive);
	};

	saveMod("queryPreprocess", *queryPreprocess);
	saveMod("kvPreprocess", *kvPreprocess);
	saveMod("refinementBlocks", *refinementBlocks);
	saveMod("thinkBlocks", *thinkBlocks);
	saveMod("postprocess", *postprocess);

	archive.save_to(path.string());

	if (saveOptim) {
		torch::serialize::OutputArchive optimArchive;
		optim->save(optimArchive);
		optimArchive.save_to(GetOptimSavePath(folder).string());
	}
}

void GGL::AttentionModel::Load(std::filesystem::path folder, bool allowNotExist, bool loadOptim) {
	std::filesystem::path path = GetSavePath(folder);

	if (!std::filesystem::exists(path)) {
		if (allowNotExist) {
			RG_LOG("Warning: AttentionModel \"" << modelName << "\" does not exist in " << folder << " and will be reset");
			return;
		} else {
			RG_ERR_CLOSE("AttentionModel \"" << modelName << "\" does not exist in " << folder);
		}
	}

	try {
		torch::serialize::InputArchive archive;
		archive.load_from(path.string(), device);

		auto loadMod = [&](const std::string& name, torch::nn::Module& mod) {
			torch::serialize::InputArchive subArchive;
			archive.read(name, subArchive);
			mod.load(subArchive);
		};

		loadMod("queryPreprocess", *queryPreprocess);
		loadMod("kvPreprocess", *kvPreprocess);
		loadMod("refinementBlocks", *refinementBlocks);
		loadMod("thinkBlocks", *thinkBlocks);
		loadMod("postprocess", *postprocess);
	}
	catch (std::exception& e) {
		RG_ERR_CLOSE(
			"Failed to load AttentionModel \"" << modelName << "\", checkpoint may be corrupt or wrong arch.\n" <<
			"Exception: " << e.what()
		);
	}

	if (loadOptim) {
		std::filesystem::path optimPath = GetOptimSavePath(folder);
		if (std::filesystem::exists(optimPath)) {
			std::ifstream testStream = std::ifstream(optimPath, std::istream::ate | std::ios::binary);
			if (testStream.tellg() > 0) {
				torch::serialize::InputArchive optimArchive;
				optimArchive.load_from(optimPath.string(), device);
				optim->load(optimArchive);
			} else {
				RG_LOG("WARNING: Saved optimizer at " << optimPath << " is empty, optimizer will be reset");
			}
		} else {
			RG_LOG("WARNING: No optimizer found at " << optimPath << ", optimizer will be reset");
		}
	}

	_attnHalfOutdated = true;
}

torch::Tensor GGL::AttentionModel::CopyParams() const {
	return torch::nn::utils::parameters_to_vector(parameters()).cpu();
}

uint64_t GGL::AttentionModel::GetParamCount() {
	uint64_t total = 0;
	for (auto& param : this->parameters()) {
		if (!param.requires_grad())
			continue;
		total += param.numel();
	}
	return total;
}
