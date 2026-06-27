#include "GAE.h"

#ifdef RG_CUDA_SUPPORT
#include <ATen/cuda/CUDAContext.h>
#endif

#ifdef RG_ROCKETSIMCUDA_AVAILABLE
#include <RocketSimCudaTraining.h>
#endif

void GGL::GAE::Compute(
	torch::Tensor rews, torch::Tensor terminals, torch::Tensor valPreds, torch::Tensor truncValPreds,
	torch::Tensor& outAdvantages, torch::Tensor& outTargetValues, torch::Tensor& outReturns, float& outRewClipPortion,
	float gamma, float lambda, float returnStd, float clipRange
) {

	bool hasTruncValPreds = truncValPreds.defined();
	int numReturns = rews.size(0);

#ifdef RG_CUDA_SUPPORT
#ifdef RG_ROCKETSIMCUDA_AVAILABLE
	if (rews.is_cuda()) {
		rews = rews.contiguous();
		terminals = terminals.contiguous();
		valPreds = valPreds.contiguous();
		if (hasTruncValPreds)
			truncValPreds = truncValPreds.contiguous();

		auto outOpts = torch::TensorOptions().dtype(torch::kFloat32).device(rews.device());
		outAdvantages = torch::zeros({ numReturns }, outOpts);
		outReturns = torch::zeros({ numReturns }, outOpts);
		outTargetValues = torch::zeros({ numReturns }, outOpts);
		torch::Tensor tClipPortion = torch::zeros({ 1 }, outOpts);

		rsc::DeviceGAEParams params = {};
		params.gamma = gamma;
		params.lambda = lambda;
		params.returnStd = returnStd;
		params.clipRange = clipRange;

		void* streamHandle = at::cuda::getCurrentCUDAStream().stream();
		rsc::ComputeGAEOnDevice(
			rews.data_ptr<float>(),
			terminals.data_ptr<int8_t>(),
			valPreds.data_ptr<float>(),
			hasTruncValPreds ? truncValPreds.data_ptr<float>() : nullptr,
			hasTruncValPreds ? (int)truncValPreds.size(0) : 0,
			outAdvantages.data_ptr<float>(),
			outTargetValues.data_ptr<float>(),
			outReturns.data_ptr<float>(),
			tClipPortion.data_ptr<float>(),
			numReturns,
			params,
			streamHandle
		);

		outRewClipPortion = tClipPortion.cpu().item<float>();
		return;
	}
#endif
#endif

	float prevLambda = 0;
	outAdvantages = torch::zeros(numReturns);
	outReturns = torch::zeros(numReturns);
	float prevRet = 0;
	int truncCount = 0;

	float totalRew = 0, totalClippedRew = 0;

	// Make sure all tensors are contiguous first
	rews = rews.contiguous();
	terminals = terminals.contiguous();
	valPreds = valPreds.contiguous();
	if (hasTruncValPreds)
		truncValPreds = truncValPreds.contiguous();

	// Accessing the raw pointers makes this all like 10x faster
	auto _terminals = terminals.const_data_ptr<int8_t>();
	auto _rews = rews.const_data_ptr<float>();
	auto _valPreds = valPreds.const_data_ptr<float>();

	const float* _truncValPreds;
	int numTruncs;
	if (hasTruncValPreds) {
		_truncValPreds = truncValPreds.const_data_ptr<float>();
		numTruncs = truncValPreds.size(0);
	} else {
		_truncValPreds = NULL;
		numTruncs = 0;
	}

	auto _outReturns = std::vector<float>(numReturns, 0);
	auto _outAdvantages = std::vector<float>(numReturns, 0);

	for (int step = numReturns - 1; step >= 0; step--) {
		uint8_t terminal = _terminals[step];
		float done = terminal == RLGC::TerminalType::NORMAL;
		float trunc = terminal == RLGC::TerminalType::TRUNCATED;

		float curReward;
		if (returnStd != 0) {
			curReward = _rews[step] / returnStd;

			totalRew += abs(curReward);

			// We only clip if returns are standardized
			if (clipRange > 0)
				curReward = RS_CLAMP(curReward, -clipRange, clipRange);

			totalClippedRew += abs(curReward);
		} else {
			curReward = _rews[step];
			totalRew += abs(curReward);
		}

		float nextValPred;
		if (terminal == RLGC::TerminalType::TRUNCATED) {
			// We've encountered a truncation
			// Pull the next truncated value

			if (!hasTruncValPreds)
				RG_ERR_CLOSE("GAE encountered a truncated terminal, but has no truncated val pred");

			if (truncCount >= numTruncs)
				RG_ERR_CLOSE("GAE encountered too many truncated terminals, not enough val preds (max: " << numTruncs << ")")

			nextValPred = _truncValPreds[truncCount];
			truncCount++;
		} else if (terminal == RLGC::TerminalType::NORMAL) {
			nextValPred = 0;
		} else {
			nextValPred = _valPreds[step + 1];
		}

		float predReturn = curReward + gamma * nextValPred * (1 - done);
		float delta = predReturn - _valPreds[step];
		float curReturn = _rews[step] + prevRet * gamma * (1 - done) * (1 - trunc);
		_outReturns[step] = curReturn;
		
		prevLambda = delta + gamma * lambda * (1 - done) * (1 - trunc) * prevLambda;
		_outAdvantages[step] = prevLambda;

		prevRet = curReturn;
	}
	
	if (hasTruncValPreds)
		if (truncCount != truncValPreds.size(0))
			RG_ERR_CLOSE("GAE didn't receive expected truncation count (only " << truncCount << "/" << truncValPreds.size(0) << ")");

	outReturns = torch::tensor(_outReturns);
	outAdvantages = torch::tensor(_outAdvantages);
	outTargetValues = valPreds.slice(0, 0, numReturns) + outAdvantages;
	outRewClipPortion = (totalRew - totalClippedRew) / RS_MAX(totalRew, 1e-7f);
}
