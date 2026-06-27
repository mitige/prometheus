#pragma once

#include <atomic>
#include <chrono>
#include <deque>
#include <filesystem>
#include <map>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

struct GLFWwindow;
namespace GGL { class Learner; }

namespace prometheus {

enum class GuiRunMode {
	Train = 0,
	TransferLearn = 1,
	Render = 2
};

struct RewardEntry {
	std::string className;
	float weight = 1.0f;
	bool zeroSum = false;
	float teamSpirit = 0.0f;
};

struct MetricSeries {
	std::string name;
	std::deque<float> values;
	size_t maxSize = 600;
	float minVal = 1e30f;
	float maxVal = -1e30f;

	void Push(float value);
	void Clear();
};

struct LogEntry {
	enum Level { INFO, WARN, ERR };
	Level level = INFO;
	std::string timestamp;
	std::string message;
};

struct TrainingState {
	std::mutex mtx;

	std::atomic<bool> running{ false };
	std::atomic<bool> stopRequested{ false };
	std::atomic<int64_t> totalTimesteps{ 0 };
	std::atomic<int64_t> collectionProgress{ 0 };
	std::atomic<int64_t> collectionTarget{ 0 };
	std::atomic<int> iteration{ 0 };
	std::atomic<float> stepsPerSec{ 0.0f };

	float meanReward = 0.0f;
	float policyLoss = 0.0f;
	float criticLoss = 0.0f;
	float entropy = 0.0f;
	float meanPolicyStd = 0.0f;
	float clipFraction = 0.0f;
	float ppoLearnTime = 0.0f;
	float collectionTime = 0.0f;
	float consumptionTime = 0.0f;
	float touchRatio = 0.0f;
	float inAirRatio = 0.0f;
	float avgBoost = 0.0f;
	float transferLoss = 0.0f;
	float transferAccuracy = 0.0f;
	float transferTableLoss = 0.0f;
	float transferMSE = 0.0f;
	float transferMAE = 0.0f;
	float transferNLL = 0.0f;
	float transferStdLoss = 0.0f;
	float transferButtonBCE = 0.0f;
	float teacherConfidence = 0.0f;
	float oldPolicyEntropy = 0.0f;
	float studentRolloutRatio = 0.0f;
	float aerialTargetRatio = 0.0f;
	bool cudaAvailable = false;

	std::atomic<int> activeMode{ (int)GuiRunMode::Train };
	MetricSeries rewardSeries{ "Mean Reward" };
	MetricSeries policyLossSeries{ "Policy Loss" };
	MetricSeries criticLossSeries{ "Critic Loss" };
	MetricSeries entropySeries{ "Entropy" };
	MetricSeries stepsPerSecSeries{ "Steps/sec" };
	MetricSeries touchSeries{ "Touch Ratio" };
	MetricSeries boostSeries{ "Boost" };
	MetricSeries transferLossSeries{ "Transfer Loss" };
	MetricSeries transferAccuracySeries{ "Transfer Accuracy" };
	MetricSeries teacherConfidenceSeries{ "Teacher Confidence" };

	std::map<std::string, MetricSeries> rewardMetrics;
	std::deque<LogEntry> logs;
	size_t maxLogs = 1000;

	void PushLog(LogEntry::Level level, const std::string& message);
};

struct GuiTrainingConfig {
	int numArenas = 1024;
	int playersPerTeam = 2;
	int tickSkip = 8;
	int stepsPerIteration = 200000;
	int minibatchSize = 20000;
	int ppoEpochs = 2;

	float entropyScale = 0.025f;
	float gamma = 0.993f;
	float lambda = 0.975f;
	float policyLR = 1.0e-4f;
	float criticLR = 1.0e-4f;
	float clipRange = 0.2f;
	float rewardClipRange = 10.0f;
	float varMin = 0.1f;
	float varMax = 1.0f;

	bool useCuda = true;
	bool sendMetrics = false;
	bool renderMode = false;
	bool deterministic = false;
	bool standardizeObs = false;
	bool standardizeReturns = true;

	bool trainAgainstOldVersions = true;
	float trainAgainstOldChance = 0.20f;
	int64_t tsPerVersion = 25000000;
	int maxOldVersions = 32;

	std::string obsBuilder = "AdvancedObsPadded";
	std::string sharedHeadLayerSizes = "1024, 1024, 1024";
	std::string policyLayerSizes = "1024, 1024, 1024";
	std::string criticLayerSizes = "1024, 1024, 1024";
	std::string checkpointDir = "checkpoints";
	std::string collisionMeshesDir = "collision_meshes";
	int checkpointInterval = 5;

	bool useAttentionHead = true;
	int attentionDims = 256;
	int attentionThinkDims = 1024;
	int attentionBlocks = 2;
	int attentionHeads = 4;
	int attentionPreprocessLayers = 1;
	int attentionPostprocessLayers = 1;
	int attentionRefinementFeedforward = 512;
	int continuousActionSize = 8;

	bool transferLearning = false;
	std::string transferStudentCheckpointDir = "checkpoints_transfer";
	std::string transferOldModelsPath = "checkpoints/28188091392";
	std::string transferOldObsBuilder = "AdvancedObsPadded";
	bool transferOldContinuousPolicy = true;
	int transferOldContinuousActionSize = 8;
	std::string transferOldPolicyLayerSizes = "1024, 1024, 1024";
	std::string transferOldSharedHeadLayerSizes;
	bool transferOldPolicyLayerNorm = true;
	bool transferOldPolicyOutputLayer = true;
	bool transferOldUseAttentionHead = true;
	int transferOldAttentionDims = 256;
	int transferOldAttentionThinkDims = 1024;
	int transferOldAttentionBlocks = 2;
	int transferOldAttentionHeads = 4;
	int transferOldAttentionPreprocessLayers = 1;
	int transferOldAttentionPostprocessLayers = 1;
	int transferOldAttentionRefinementFeedforward = 0;
	float transferLR = 1.0e-4f;
	int transferBatchSize = 32768;
	int transferEpochs = 5;
	int transferMaxIterations = -1;
	bool transferSaveOnMaxIterations = true;
	bool transferUseKLDiv = true;
	float transferLossScale = 500.0f;
	float transferLossExponent = 1.0f;
	int transferContinuousActionTableTopK = 0;
	float transferContinuousActionTableLossWeight = 0.0f;
	float transferContinuousActionMSEWeight = 5.0f;
	float transferContinuousActionNLLWeight = 1.0f;
	float transferContinuousStdLossWeight = 0.5f;
	float transferContinuousTargetStd = 0.1f;
	float transferContinuousButtonBCELossWeight = 5.0f;
	float transferContinuousAerialSampleWeight = 8.0f;
	float transferStudentRolloutProb = 0.5f;
	int64_t transferStudentRolloutWarmupTimesteps = 5000000;

	std::vector<RewardEntry> rewards;

	void SaveToFile(const std::string& path = "prometheus_config.json") const;
	static GuiTrainingConfig LoadFromFile(const std::string& path = "prometheus_config.json");
};

class GuiApp {
public:
	GuiApp();
	~GuiApp();

	bool Init(int width = 1440, int height = 900);
	void Run();
	void Shutdown();

private:
	void ApplyGoldTheme();

	void RenderTopBar();
	void RenderMetricsPanel();
	void RenderConfigPanel();
	void RenderGPUStatusPanel();
	void RenderRewardsPanel();
	void RenderLogPanel();
	void RenderPlot(const char* label, const MetricSeries& series, float height = 92.0f);

	void StartTraining(GuiRunMode mode);
	void StopTraining();
	void SaveCheckpoint();

	static std::string FormatTimesteps(int64_t timesteps);
	static std::string FormatDuration(float seconds);

	GLFWwindow* window = nullptr;
	bool shouldClose = false;

	TrainingState state;
	GuiTrainingConfig config;

	std::mutex learnerMutex;
	std::thread trainingThread;
	GGL::Learner* learner = nullptr;
	std::chrono::steady_clock::time_point trainStartTime;

	int selectedPlot = 0;
	bool autoScrollLogs = true;
};

} // namespace prometheus
