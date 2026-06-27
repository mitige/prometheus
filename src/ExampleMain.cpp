#include "RLBotClient.h"
#include <GigaLearnCPP/Learner.h>
#include <GigaLearnCPP/Util/InferUnit.h>

#include <fstream>
#include <string>

#include <RLGymCPP/ActionParsers/DefaultAction.h>
#include <RLGymCPP/ActionParsers/DefaultContinuousAction.h>
#include <RLGymCPP/OBSBuilders/AdvancedObsPadded.h>
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

using namespace GGL;  // GigaLearn
using namespace RLGC; // RLGymCPP

static constexpr int kAdvancedObsPaddedSize = 225;

static AdvancedObsPadded *MakeAdvancedObsPadded() {
  return new AdvancedObsPadded();
}

// Create the RLGymCPP environment for each of our games
EnvCreateResult EnvCreateFunc(int index) {
  // These are ok rewards that will produce a scoring bot in ~100m steps
  std::vector<WeightedReward> rewards = {
      // Core game rewards: a compact public shaping set for scoring,
      // possession, boost management, recoveries, and aerial play.
      {new GoalReward(-0.80f), 150.0f},
      {new ZeroSumReward(new TouchBallReward(), 0.0f), 3.2f},
      {new VelocityBallToGoalReward(), 5.0f},
      {new KickoffProximityReward(), 0.45f},
      {new PickupBoostReward(), 20.0f},
      {new SaveBoostReward(0.5f), 0.30f},
      {new TeammateBumpPenaltyReward(), 0.3f},
      {new AirReward(), 0.06f},
      {new AerialTouchHeightReward(140.0f, 1800.0f), 0.20f},
      {new WavedashRecoveryReward(500.0f), 0.12f},
      {new AirDribbleReward(220.0f, 380.0f, 150.0f, 0.10f, true, false,
                            1.5f, 1.0f, 0.4f, 1.5f),
       0.14f},
  };

  std::vector<TerminalCondition *> terminalConditions = {
      new NoTouchCondition(10), new GoalScoreCondition()};

  // Make the arena
  int playersPerTeam = 2;
  auto arena = Arena::Create(GameMode::SOCCAR);
  for (int i = 0; i < playersPerTeam; i++) {
    arena->AddCar(Team::BLUE);
    arena->AddCar(Team::ORANGE);
  }

  EnvCreateResult result = {};
  result.actionParser = new DefaultAction();
  result.continuousActionParser = new DefaultContinuousAction();
  result.obsBuilder = MakeAdvancedObsPadded();
  result.stateSetter =
      new CombinedState({{new KickoffState(), 0.85f},
                         {new RandomState(true, true, false), 0.05f},
                         {new UnlimitedBoostState(true, true, false), 0.10f}});
  result.terminalConditions = terminalConditions;
  result.rewards = rewards;

  result.arena = arena;

  return result;
}

void StepCallback(Learner *learner, const std::vector<GameState> &states,
                  Report &report) {
  // To prevent expensive metrics from eating at performance, we will only run
  // them on 1/4th of steps This doesn't really matter unless you have expensive
  // metrics (which this example doesn't)
  bool doExpensiveMetrics = (rand() % 4) == 0;

  // Add our metrics
  for (auto &state : states) {
    if (doExpensiveMetrics) {
      for (auto &player : state.players) {
        report.AddAvg("Player/In Air Ratio", !player.isOnGround);
        report.AddAvg("Player/Ball Touch Ratio", player.ballTouchedStep);
        report.AddAvg("Player/Demoed Ratio", player.isDemoed);

        report.AddAvg("Player/Speed", player.vel.Length());
        Vec dirToBall = (state.ball.pos - player.pos).Normalized();
        report.AddAvg("Player/Speed Towards Ball",
                      RS_MAX(0, player.vel.Dot(dirToBall)));

        report.AddAvg("Player/Boost", player.boost);

        if (player.ballTouchedStep)
          report.AddAvg("Player/Touch Height", state.ball.pos.z);
      }
    }

    if (state.goalScored)
      report.AddAvg("Game/Goal Speed", state.ball.vel.Length());
  }
}

TransferLearnConfig BuildTransferLearnConfig(bool oneIteration) {
  TransferLearnConfig cfg = {};
  cfg.oldPolicyType = PolicyType::CONTINUOUS;
  cfg.oldContinuousActionSize = 8;
  cfg.makeOldObsFn = []() { return new AdvancedObsPadded(); };
  cfg.makeOldActFn = []() { return new DefaultAction(); };
  cfg.oldModelsPath = "checkpoints/28188091392";

  cfg.oldPolicyConfig.layerSizes = {1024, 1024, 1024};
  cfg.oldPolicyConfig.activationType = ModelActivationType::LEAKY_RELU;
  cfg.oldPolicyConfig.optimType = ModelOptimType::ADAM;
  cfg.oldPolicyConfig.addLayerNorm = true;
  cfg.oldPolicyConfig.addOutputLayer = true;

  cfg.oldSharedHeadConfig.layerSizes = {};
  cfg.oldSharedHeadConfig.activationType = ModelActivationType::LEAKY_RELU;
  cfg.oldSharedHeadConfig.optimType = ModelOptimType::ADAM;
  cfg.oldSharedHeadConfig.addLayerNorm = true;
  cfg.oldSharedHeadConfig.addOutputLayer = false;

  cfg.oldUseAttentionHead = true;
  cfg.oldAttentionHeadConfig = new AttentionModelConfig();
  cfg.oldAttentionHeadConfig->numDims = 256;
  cfg.oldAttentionHeadConfig->thinkDims = 1024;
  cfg.oldAttentionHeadConfig->numBlocks = 2;
  cfg.oldAttentionHeadConfig->numHeads = 4;
  cfg.oldAttentionHeadConfig->preprocessLayers = 1;
  cfg.oldAttentionHeadConfig->postprocessLayers = 1;
  cfg.oldAttentionHeadConfig->activationType = ModelActivationType::LEAKY_RELU;
  cfg.oldAttentionHeadConfig->optimType = ModelOptimType::ADAM;

  cfg.lr = 1.0e-4f;
  cfg.batchSize = 32'768;
  cfg.epochs = 5;
  cfg.maxIterations = oneIteration ? 1 : -1;
  cfg.saveOnMaxIterations = true;
  cfg.useKLDiv = true;
  cfg.lossScale = 500.0f;
  cfg.lossExponent = 1.0f;
  cfg.continuousTargetStd = 0.1f;

  return cfg;
}

int main(int argc, char *argv[]) {
  // Initialize RocketSim with collision meshes
  // Change this path to point to your meshes!
  RocketSim::Init("./collision_meshes");

  bool agentMode = (argc > 1 && std::string(argv[1]) == "agent");
  if (agentMode) {
    const int tickSkip = 8;
    const int actionDelay = tickSkip - 1;
    const int continuousActionSize = 8;

    PartialModelConfig sharedHeadCfg;
    sharedHeadCfg.layerSizes = {1024, 1024,
                                1024}; // ignored with the attention head
    sharedHeadCfg.activationType = ModelActivationType::LEAKY_RELU;
    sharedHeadCfg.optimType = ModelOptimType::ADAM;
    sharedHeadCfg.addLayerNorm = true;
    sharedHeadCfg.addOutputLayer = false;

    // Must match the training architecture below
    PartialModelConfig policyCfg;
    policyCfg.layerSizes = {1024, 1024, 1024};
    policyCfg.activationType = ModelActivationType::LEAKY_RELU;
    policyCfg.optimType = ModelOptimType::ADAM;
    policyCfg.addLayerNorm = true;
    policyCfg.addOutputLayer = true;

    auto *obs = MakeAdvancedObsPadded();
    int obsSize = kAdvancedObsPaddedSize;

    // Match the attention config used during training
    auto *attnCfg = new AttentionModelConfig();
    attnCfg->numDims = 256;
    attnCfg->thinkDims = 1024;
    attnCfg->numBlocks = 2;
    attnCfg->numHeads = 4;
    attnCfg->preprocessLayers = 1;
    attnCfg->postprocessLayers = 1;
    attnCfg->refinementFeedforward = 512;

    auto *inferUnit = new InferUnit(
        obs, obsSize, new DefaultAction(), sharedHeadCfg, policyCfg,
        "checkpoints", /*useGPU=*/true, PolicyType::CONTINUOUS, 0.1f, 1.0f,
        continuousActionSize, new DefaultContinuousAction(), attnCfg);

    std::ifstream portFile("./rlbot/port.cfg");
    int port = 23233;
    if (portFile.good())
      portFile >> port;

    RLBotParams params = {};
    params.port = port;
    params.tickSkip = tickSkip;
    params.actionDelay = actionDelay;
    params.inferUnit = inferUnit;

    RG_LOG("Starting RLBot agent on port " << port << "...");
    RLBotClient::Run(params);
    return EXIT_SUCCESS;
  }

  auto hasArg = [&](const char *value) {
    for (int i = 1; i < argc; i++) {
      if (std::string(argv[i]) == value)
        return true;
    }
    return false;
  };
  auto getIntArg = [&](const char *prefix, int fallback) {
    std::string prefixStr(prefix);
    for (int i = 1; i < argc; i++) {
      std::string arg(argv[i]);
      if (arg.rfind(prefixStr, 0) == 0)
        return std::stoi(arg.substr(prefixStr.size()));
    }
    return fallback;
  };

  bool transferOnceMode = hasArg("transfer-once");
  bool transferMode = hasArg("transfer");
  bool renderMode = hasArg("render");
  // "train" skips the transfer-learn phase and runs normal PPO training
  // (use once the teacher transfer has converged)
  bool directTrainMode = hasArg("train");
  bool ppoTrainMode = !(transferOnceMode || transferMode || renderMode);
  bool cpuPhysicsMode = hasArg("cpu") || hasArg("--cpu") ||
                        hasArg("rocketsim-cpu") || hasArg("--rocketsim-cpu");
  bool metricsMode = hasArg("metrics") || hasArg("--metrics");
  bool noOldVersionsMode = hasArg("no-old") || hasArg("--no-old");
  int numGamesOverride = getIntArg("games=", 0);

  // Make configuration for the learner
  LearnerConfig cfg = {};

  cfg.deviceType = LearnerDeviceType::GPU_CUDA;
  cfg.physicsBackend = cpuPhysicsMode ? EnvPhysicsBackend::ROCKETSIM_CPU
                                      : EnvPhysicsBackend::ROCKETSIM_CUDA;
  // Skip the per-step GPU->CPU world-state sync when nothing CPU-side needs it
  // (auto-kept-on while a step callback / reward metrics / render mode is
  // active).
  cfg.cudaNoCpuWorldState = ppoTrainMode && !cpuPhysicsMode;
  cfg.addRewardsToMetrics = !ppoTrainMode;
  // continuous attention: fresh folder — the architecture below is bigger than the
  // earlier legacy transfer model, old checkpoints are incompatible.
  cfg.checkpointFolder = "checkpoints";
  if (transferOnceMode || transferMode)
    cfg.checkpointFolder = "checkpoints_transfer";

  cfg.tickSkip = 8;
  cfg.actionDelay = cfg.tickSkip - 1; // Normal value in other RLGym frameworks

  // Play around with this to see what the optimal is for your machine, more
  // games will consume more RAM
  cfg.numGames = numGamesOverride > 0 ? numGamesOverride : 512;
  cfg.trainAgainstOldVersions =
      !(transferOnceMode || transferMode || renderMode || noOldVersionsMode);
  cfg.trainAgainstOldChance = 0.20f;
  cfg.tsPerVersion = 25'000'000;
  cfg.maxOldVersions = 32;

  int tsPerItr = 200'000;
  cfg.ppo.tsPerItr = tsPerItr;
  cfg.ppo.batchSize = tsPerItr;
  cfg.ppo.miniBatchSize = 20'000;

  // Using 2 epochs seems pretty optimal when comparing time training to skill
  // Perhaps 1 or 3 is better for you, test and find out!
  cfg.ppo.epochs = 1;

  // This scales differently than "ent_coef" in other frameworks.
  // For the hybrid analog+button policy, start a bit lower than the discrete
  // example to avoid over-noisy exploration.
  cfg.ppo.policyType = PolicyType::CONTINUOUS;
  cfg.ppo.entropyScale = 0.025f;
  cfg.ppo.varMin = 0.1f;
  cfg.ppo.varMax = 1.0f;

  // Rate of reward decay
  // Starting low tends to work out
  cfg.ppo.gaeGamma = 0.993f;
  cfg.ppo.gaeLambda = 0.975f;

  // Good learning rate to start
  cfg.ppo.policyLR = 1.0e-4f;
  cfg.ppo.criticLR = 1.0e-4f;

  // NOTE: with useAttentionHead=true the sharedHead MLP config is IGNORED
  // (the attention model below IS the shared head). Policy/critic take the
  // attention output (numDims) as input.
  // Continuous attention policy network.
  cfg.ppo.policy.layerSizes = {1024, 1024, 1024};
  cfg.ppo.critic.layerSizes = {1024, 1024, 1024};

  auto optim = ModelOptimType::ADAM;
  cfg.ppo.policy.optimType = optim;
  cfg.ppo.critic.optimType = optim;
  cfg.ppo.sharedHead.optimType = optim;

  auto activation = ModelActivationType::LEAKY_RELU;
  cfg.ppo.policy.activationType = activation;
  cfg.ppo.critic.activationType = activation;
  cfg.ppo.sharedHead.activationType = activation;

  bool addLayerNorm = true;
  cfg.ppo.policy.addLayerNorm = addLayerNorm;
  cfg.ppo.critic.addLayerNorm = addLayerNorm;
  cfg.ppo.sharedHead.addLayerNorm = addLayerNorm;

  // Attention head (bigger than the Kue example config:
  // 512 dims / 8 heads / 3 blocks vs 256/4/2)
  cfg.ppo.useAttentionHead = true;
  cfg.ppo.attentionHeadConfig = new GGL::AttentionModelConfig();
  cfg.ppo.attentionHeadConfig->numDims = 256; // must stay divisible by numHeads
  cfg.ppo.attentionHeadConfig->thinkDims = 1024;
  cfg.ppo.attentionHeadConfig->numBlocks = 2;
  cfg.ppo.attentionHeadConfig->numHeads = 4;
  cfg.ppo.attentionHeadConfig->preprocessLayers = 1;
  cfg.ppo.attentionHeadConfig->postprocessLayers = 1;
  cfg.ppo.attentionHeadConfig->refinementFeedforward = 512;

  cfg.sendMetrics = metricsMode && !(transferOnceMode || transferMode);
  cfg.renderMode = renderMode;
  cfg.ppo.deterministic = cfg.renderMode;

  RG_LOG("Physics backend: " << (cpuPhysicsMode ? "RocketSim CPU"
                                                : "RocketSimCuda"));
  RG_LOG("Python metrics: " << (cfg.sendMetrics ? "enabled" : "disabled"));
  RG_LOG("Num games: " << cfg.numGames);
  RG_LOG("Old versions: " << (cfg.trainAgainstOldVersions ? "enabled"
                                                          : "disabled"));
  RG_LOG("Obs builder: AdvancedObsPadded (" << kAdvancedObsPaddedSize << ")");
  if (directTrainMode)
    RG_LOG("Direct PPO train mode selected.");

  // Make the learner with the environment creation function and the config we
  // just made
  Learner *learner =
      new Learner(EnvCreateFunc, cfg, ppoTrainMode ? nullptr : StepCallback);

  if (transferOnceMode || transferMode) {
    TransferLearnConfig transferCfg =
        BuildTransferLearnConfig(transferOnceMode);
    learner->StartTransferLearn(transferCfg);
  } else {
    learner->Start();
  }

  return EXIT_SUCCESS;
}

/*
Information

How to use Continuous
1. Set cfg.ppo.policyType = GGL::PolicyType::CONTINUOUS;
2. The bot will output 8 hybrid continuous actions
3. When using continuous actions use a slightly lower entropyScale (e.g. 0.02 or
0.01) compared to discrete

How to use the Attention (Kue example config)
cfg.ppo.useAttentionHead = true;
cfg.ppo.attentionHeadConfig = new GGL::AttentionModelConfig();
cfg.ppo.attentionHeadConfig->numDims = 256;
cfg.ppo.attentionHeadConfig->thinkDims = 1024;
cfg.ppo.attentionHeadConfig->numBlocks = 2;
cfg.ppo.attentionHeadConfig->numHeads = 4;

Skilltracking should be fully supported, but I haven't tested it
*/
