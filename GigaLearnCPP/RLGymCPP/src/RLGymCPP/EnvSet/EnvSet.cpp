#include "EnvSet.h"
#include  "../Rewards/ZeroSumReward.h"
#include "../Rewards/CommonRewards.h"
#include "../Rewards/mkh_rewards.h"
#include "../ObsBuilders/AdvancedObs.h"
#include "../ActionParsers/DefaultAction.h"
#include "../TerminalConditions/GoalScoreCondition.h"
#include "../TerminalConditions/NoTouchCondition.h"

#include <algorithm>

#ifdef RG_ROCKETSIMCUDA_AVAILABLE
namespace {
	bool NearlyEqual(float a, float b, float eps = 1e-3f) {
		return abs(a - b) <= eps;
	}

	bool NearlyEqualVec(const Vec& a, const Vec& b, float eps = 1e-3f) {
		return NearlyEqual(a.x, b.x, eps) && NearlyEqual(a.y, b.y, eps) && NearlyEqual(a.z, b.z, eps);
	}

	rsc::Vec3 ToCudaVec(const Vec& v) {
		return { v.x, v.y, v.z };
	}

	rsc::RotMat ToCudaRot(const RotMat& m) {
		return {
			ToCudaVec(m.forward),
			ToCudaVec(m.right),
			ToCudaVec(m.up),
		};
	}

	rsc::BallState ToCudaBallState(const BallState& src) {
		rsc::BallState dst = {};
		dst.pos = ToCudaVec(src.pos);
		dst.rotMat = ToCudaRot(src.rotMat);
		dst.vel = ToCudaVec(src.vel);
		dst.angVel = ToCudaVec(src.angVel);
		return dst;
	}

	rsc::CarControls ToCudaControls(const CarControls& controls) {
		rsc::CarControls out = {};
		out.throttle = controls.throttle;
		out.steer = controls.steer;
		out.pitch = controls.pitch;
		out.yaw = controls.yaw;
		out.roll = controls.roll;
		out.jump = controls.jump;
		out.boost = controls.boost;
		out.handbrake = controls.handbrake;
		return out;
	}

	rsc::CarControls ToCudaControls(const RLGC::Action& action) {
		return ToCudaControls((CarControls)action);
	}

	CarControls ToCpuControls(const rsc::CarControls& controls) {
		CarControls out = {};
		out.throttle = controls.throttle;
		out.steer = controls.steer;
		out.pitch = controls.pitch;
		out.yaw = controls.yaw;
		out.roll = controls.roll;
		out.jump = controls.jump;
		out.boost = controls.boost;
		out.handbrake = controls.handbrake;
		return out;
	}

	rsc::CarPreset MapCarPreset(const CarConfig& config) {
		struct PresetMatch {
			const CarConfig* config;
			rsc::CarPreset preset;
		};

		const PresetMatch presets[] = {
			{ &CAR_CONFIG_OCTANE, rsc::CarPreset::OCTANE },
			{ &CAR_CONFIG_DOMINUS, rsc::CarPreset::DOMINUS },
			{ &CAR_CONFIG_PLANK, rsc::CarPreset::PLANK },
			{ &CAR_CONFIG_BREAKOUT, rsc::CarPreset::BREAKOUT },
			{ &CAR_CONFIG_HYBRID, rsc::CarPreset::HYBRID },
			{ &CAR_CONFIG_MERC, rsc::CarPreset::MERC },
		};

		for (const auto& preset : presets) {
			const CarConfig& ref = *preset.config;
			if (
				NearlyEqualVec(config.hitboxSize, ref.hitboxSize) &&
				NearlyEqualVec(config.hitboxPosOffset, ref.hitboxPosOffset) &&
				NearlyEqual(config.frontWheels.wheelRadius, ref.frontWheels.wheelRadius) &&
				NearlyEqual(config.frontWheels.suspensionRestLength, ref.frontWheels.suspensionRestLength) &&
				NearlyEqualVec(config.frontWheels.connectionPointOffset, ref.frontWheels.connectionPointOffset) &&
				NearlyEqual(config.backWheels.wheelRadius, ref.backWheels.wheelRadius) &&
				NearlyEqual(config.backWheels.suspensionRestLength, ref.backWheels.suspensionRestLength) &&
				NearlyEqualVec(config.backWheels.connectionPointOffset, ref.backWheels.connectionPointOffset) &&
				NearlyEqual(config.dodgeDeadzone, ref.dodgeDeadzone)
			) {
				return preset.preset;
			}
		}

		return rsc::CarPreset::OCTANE;
	}

	rsc::Team ToCudaTeam(Team team) {
		return team == Team::BLUE ? rsc::Team::BLUE : rsc::Team::ORANGE;
	}

	rsc::MutatorConfig ToCudaMutatorConfig(const MutatorConfig& src) {
		rsc::MutatorConfig dst = {};
		dst.gravity = ToCudaVec(src.gravity);
		dst.carMass = src.carMass;
		dst.ballMass = src.ballMass;
		dst.ballMaxSpeed = src.ballMaxSpeed;
		dst.ballDrag = src.ballDrag;
		dst.ballWorldFriction = src.ballWorldFriction;
		dst.ballWorldRestitution = src.ballWorldRestitution;
		dst.ballRadius = src.ballRadius;
		dst.jumpAccel = src.jumpAccel;
		dst.jumpImmediateForce = src.jumpImmediateForce;
		dst.boostAccelGround = src.boostAccelGround;
		dst.boostAccelAir = src.boostAccelAir;
		dst.boostUsedPerSecond = src.boostUsedPerSecond;
		dst.carSpawnBoostAmount = src.carSpawnBoostAmount;
		dst.respawnDelay = src.respawnDelay;
		dst.bumpCooldownTime = src.bumpCooldownTime;
		dst.boostPadCooldown_Big = src.boostPadCooldown_Big;
		dst.boostPadCooldown_Small = src.boostPadCooldown_Small;
		dst.ballHitExtraForceScale = src.ballHitExtraForceScale;
		dst.bumpForceScale = src.bumpForceScale;
		dst.goalBaseThresholdY = src.goalBaseThresholdY;
		return dst;
	}

	rsc::TrainingRewardEntry MapCudaTrainingReward(RLGC::Reward* reward, float weight) {
		rsc::TrainingRewardEntry entry = {};
		entry.weight = weight;

		if (dynamic_cast<RLGC::GoalReward*>(reward)) {
			auto* goalReward = dynamic_cast<RLGC::GoalReward*>(reward);
			entry.id = rsc::TrainingRewardID::GOAL_REWARD;
			entry.params[0] = goalReward->concedeScale;
			return entry;
		}

		if (dynamic_cast<RLGC::VelocityBallToGoalReward*>(reward)) {
			auto* velBall = dynamic_cast<RLGC::VelocityBallToGoalReward*>(reward);
			entry.id = rsc::TrainingRewardID::VELOCITY_BALL_TO_GOAL;
			entry.params[0] = velBall->ownGoal ? 1.f : 0.f;
			return entry;
		}

		if (dynamic_cast<RLGC::VelocityPlayerToBallReward*>(reward)) {
			entry.id = rsc::TrainingRewardID::VELOCITY_PLAYER_TO_BALL;
			return entry;
		}

		if (dynamic_cast<RLGC::FaceBallReward*>(reward)) {
			entry.id = rsc::TrainingRewardID::FACE_BALL;
			return entry;
		}

		if (dynamic_cast<RLGC::TouchBallReward*>(reward)) {
			entry.id = rsc::TrainingRewardID::TOUCH_BALL;
			return entry;
		}

		if (dynamic_cast<RLGC::TouchAccelReward*>(reward)) {
			entry.id = rsc::TrainingRewardID::TOUCH_ACCEL;
			return entry;
		}

		if (dynamic_cast<RLGC::StrongTouchReward*>(reward)) {
			auto* strongTouch = dynamic_cast<RLGC::StrongTouchReward*>(reward);
			entry.id = rsc::TrainingRewardID::STRONG_TOUCH;
			entry.params[0] = strongTouch->minRewardedVel;
			entry.params[1] = strongTouch->maxRewardedVel;
			return entry;
		}

		if (dynamic_cast<RLGC::SpeedReward*>(reward)) {
			entry.id = rsc::TrainingRewardID::SPEED;
			return entry;
		}

		if (dynamic_cast<RLGC::VelocityReward*>(reward)) {
			auto* velocityReward = dynamic_cast<RLGC::VelocityReward*>(reward);
			entry.id = rsc::TrainingRewardID::VELOCITY;
			entry.params[0] = velocityReward->isNegative ? -1.f : 1.f;
			return entry;
		}

		if (dynamic_cast<RLGC::AirReward*>(reward)) {
			entry.id = rsc::TrainingRewardID::AIR;
			return entry;
		}

		if (dynamic_cast<RLGC::WavedashReward*>(reward)) {
			entry.id = rsc::TrainingRewardID::WAVEDASH;
			return entry;
		}

		if (dynamic_cast<RLGC::PickupBoostReward*>(reward)) {
			entry.id = rsc::TrainingRewardID::PICKUP_BOOST;
			return entry;
		}

		if (dynamic_cast<RLGC::SaveBoostReward*>(reward)) {
			auto* saveBoost = dynamic_cast<RLGC::SaveBoostReward*>(reward);
			entry.id = rsc::TrainingRewardID::SAVE_BOOST;
			entry.params[0] = saveBoost->exponent;
			return entry;
		}

		if (dynamic_cast<RLGC::BumpReward*>(reward)) {
			entry.id = rsc::TrainingRewardID::BUMP;
			return entry;
		}

		if (dynamic_cast<RLGC::BumpedPenalty*>(reward)) {
			entry.id = rsc::TrainingRewardID::BUMPED_PENALTY;
			return entry;
		}

		if (dynamic_cast<RLGC::DemoReward*>(reward)) {
			entry.id = rsc::TrainingRewardID::DEMO;
			return entry;
		}

		if (dynamic_cast<RLGC::DemoedPenalty*>(reward)) {
			entry.id = rsc::TrainingRewardID::DEMOED_PENALTY;
			return entry;
		}

		if (dynamic_cast<RLGC::KickoffProximityReward*>(reward)) {
			entry.id = rsc::TrainingRewardID::KICKOFF_PROXIMITY;
			return entry;
		}

		if (dynamic_cast<RLGC::TeammateBumpPenaltyReward*>(reward)) {
			entry.id = rsc::TrainingRewardID::TEAMMATE_BUMP_PENALTY;
			return entry;
		}

		entry.id = rsc::TrainingRewardID::UNKNOWN;
		return entry;
	}

	bool BuildCudaTrainingRewardConfig(
		const std::vector<RLGC::WeightedReward>& rewards,
		rsc::TrainingRewardConfig& outConfig,
		std::string& outReason
	) {
		outConfig = {};
		if (rewards.size() > rsc::MAX_TRAINING_REWARD_ENTRIES) {
			outReason = "too many rewards for RocketSimCuda training fast path";
			return false;
		}

		for (int rewardIdx = 0; rewardIdx < rewards.size(); rewardIdx++) {
			RLGC::Reward* innerReward = rewards[rewardIdx].reward;
			auto* zeroSum = dynamic_cast<RLGC::ZeroSumReward*>(innerReward);
			if (zeroSum)
				innerReward = zeroSum->child;

			rsc::TrainingRewardEntry entry = MapCudaTrainingReward(innerReward, rewards[rewardIdx].weight);
			if (entry.id == rsc::TrainingRewardID::UNKNOWN) {
				outReason = "unsupported reward: " + innerReward->GetName();
				return false;
			}

			if (zeroSum) {
				entry.isZeroSum = 1;
				entry.teamSpirit = zeroSum->teamSpirit;
				entry.opponentScale = zeroSum->opponentScale;
			}

			outConfig.entries[outConfig.numEntries++] = entry;
		}

		return true;
	}

	bool BuildCudaTrainingTerminalConfig(
		const std::vector<RLGC::TerminalCondition*>& terminalConditions,
		rsc::TrainingTerminalConfig& outConfig,
		std::string& outReason
	) {
		outConfig = {};

		for (auto* condition : terminalConditions) {
			if (dynamic_cast<RLGC::GoalScoreCondition*>(condition)) {
				outConfig.useGoalScore = 1;
				continue;
			}

			if (dynamic_cast<RLGC::NoTouchCondition*>(condition)) {
				auto* noTouch = dynamic_cast<RLGC::NoTouchCondition*>(condition);
				if (outConfig.noTouchTimeoutSeconds <= 0.f || noTouch->maxTime < outConfig.noTouchTimeoutSeconds)
					outConfig.noTouchTimeoutSeconds = noTouch->maxTime;
				continue;
			}

			outReason = "unsupported terminal condition in RocketSimCuda fast path";
			return false;
		}

		return true;
	}

	rsc::CarState ToCudaCarState(Car* car) {
		CarState src = car->GetState();

		rsc::CarState dst = {};
		dst.pos = ToCudaVec(src.pos);
		dst.rotMat = ToCudaRot(src.rotMat);
		dst.vel = ToCudaVec(src.vel);
		dst.angVel = ToCudaVec(src.angVel);
		dst.boost = src.boost;
		dst.timeSpentBoosting = src.timeSpentBoosting;
		dst.isOnGround = src.isOnGround;
		dst.hasJumped = src.hasJumped;
		dst.hasDoubleJumped = src.hasDoubleJumped;
		dst.hasFlipped = src.hasFlipped;
		dst.isFlipping = src.isFlipping;
		dst.isJumping = src.isJumping;
		dst.flipRelTorque = ToCudaVec(src.flipRelTorque);
		dst.jumpTime = src.jumpTime;
		dst.flipTime = src.flipTime;
		dst.airTime = src.airTime;
		dst.airTimeSinceJump = src.airTimeSinceJump;
		dst.isSupersonic = src.isSupersonic;
		dst.supersonicTime = src.supersonicTime;
		dst.handbrakeVal = src.handbrakeVal;
		dst.isAutoFlipping = src.isAutoFlipping;
		dst.autoFlipTimer = src.autoFlipTimer;
		dst.autoFlipTorqueScale = src.autoFlipTorqueScale;
		dst.isDemoed = src.isDemoed;
		dst.demoRespawnTimer = src.demoRespawnTimer;
		dst.worldContactHasContact = src.worldContact.hasContact;
		dst.worldContactNormal = ToCudaVec(src.worldContact.contactNormal);
		dst.carContactOtherID = src.carContact.otherCarID;
		dst.carContactCooldownTimer = src.carContact.cooldownTimer;
		dst.ballHitValid = src.ballHitInfo.isValid;
		dst.ballHitTickCount = src.ballHitInfo.tickCountWhenHit;
		dst.team = ToCudaTeam(car->team);
		dst.preset = MapCarPreset(car->config);
		dst.id = car->id;
		dst.lastControls = ToCudaControls(src.lastControls);
		return dst;
	}

	void SyncShadowArenaFromCuda(RLGC::EnvSet* envSet, int arenaIdx) {
		auto& batch = *envSet->cudaBatch;
		auto* arena = envSet->arenas[arenaIdx];
		rsc::ArenaInfo arenaInfo = batch.GetArenaInfo(arenaIdx);

		arena->tickCount = arenaInfo.tickCount;

		BallState ballState = arena->ball->GetState();
		rsc::BallState cudaBall = batch.GetBallState(arenaIdx);
		ballState.pos = Vec(cudaBall.pos.x, cudaBall.pos.y, cudaBall.pos.z);
		ballState.rotMat = RotMat(
			Vec(cudaBall.rotMat.forward.x, cudaBall.rotMat.forward.y, cudaBall.rotMat.forward.z),
			Vec(cudaBall.rotMat.right.x, cudaBall.rotMat.right.y, cudaBall.rotMat.right.z),
			Vec(cudaBall.rotMat.up.x, cudaBall.rotMat.up.y, cudaBall.rotMat.up.z)
		);
		ballState.vel = Vec(cudaBall.vel.x, cudaBall.vel.y, cudaBall.vel.z);
		ballState.angVel = Vec(cudaBall.angVel.x, cudaBall.angVel.y, cudaBall.angVel.z);
		arena->ball->SetState(ballState);
		// Stamp the absolute cuda tick onto the shadow ball/car AFTER SetState (which zeroes
		// updateCounter). The shadow arena is never Step()'d, so these stamps are the
		// only writers, and GameEventTracker::Update reads updateCounter to derive
		// deltaTicks for goal/shot/save attribution. A relative delta (or 0) would stall the
		// tracker's "game continuing" gate. Not a dead stamp â€” do not drop.
		arena->ball->_internalState.updateCounter = arenaInfo.tickCount;

		for (int carIdx = 0; carIdx < envSet->cudaArenaCars[arenaIdx].size(); carIdx++) {
			Car* car = envSet->cudaArenaCars[arenaIdx][carIdx];
			rsc::CarState cudaCar = batch.GetCarState(arenaIdx, carIdx);
			CarState carState = car->GetState();
			carState.pos = Vec(cudaCar.pos.x, cudaCar.pos.y, cudaCar.pos.z);
			carState.rotMat = RotMat(
				Vec(cudaCar.rotMat.forward.x, cudaCar.rotMat.forward.y, cudaCar.rotMat.forward.z),
				Vec(cudaCar.rotMat.right.x, cudaCar.rotMat.right.y, cudaCar.rotMat.right.z),
				Vec(cudaCar.rotMat.up.x, cudaCar.rotMat.up.y, cudaCar.rotMat.up.z)
			);
			carState.vel = Vec(cudaCar.vel.x, cudaCar.vel.y, cudaCar.vel.z);
			carState.angVel = Vec(cudaCar.angVel.x, cudaCar.angVel.y, cudaCar.angVel.z);
			carState.boost = cudaCar.boost;
			carState.timeSpentBoosting = cudaCar.timeSpentBoosting;
			carState.isOnGround = cudaCar.isOnGround;
			carState.hasJumped = cudaCar.hasJumped;
			carState.hasDoubleJumped = cudaCar.hasDoubleJumped;
			carState.hasFlipped = cudaCar.hasFlipped;
			carState.isFlipping = cudaCar.isFlipping;
			carState.isJumping = cudaCar.isJumping;
			carState.flipRelTorque = Vec(cudaCar.flipRelTorque.x, cudaCar.flipRelTorque.y, cudaCar.flipRelTorque.z);
			carState.jumpTime = cudaCar.jumpTime;
			carState.flipTime = cudaCar.flipTime;
			carState.airTime = cudaCar.airTime;
			carState.airTimeSinceJump = cudaCar.airTimeSinceJump;
			carState.isSupersonic = cudaCar.isSupersonic;
			carState.supersonicTime = cudaCar.supersonicTime;
			carState.handbrakeVal = cudaCar.handbrakeVal;
			carState.isAutoFlipping = cudaCar.isAutoFlipping;
			carState.autoFlipTimer = cudaCar.autoFlipTimer;
			carState.autoFlipTorqueScale = cudaCar.autoFlipTorqueScale;
			carState.isDemoed = cudaCar.isDemoed;
			carState.demoRespawnTimer = cudaCar.demoRespawnTimer;
			carState.worldContact.hasContact = cudaCar.worldContactHasContact;
			carState.worldContact.contactNormal = Vec(cudaCar.worldContactNormal.x, cudaCar.worldContactNormal.y, cudaCar.worldContactNormal.z);
			carState.carContact.otherCarID = cudaCar.carContactOtherID;
			carState.carContact.cooldownTimer = cudaCar.carContactCooldownTimer;
			carState.ballHitInfo.isValid = cudaCar.ballHitValid;
			carState.ballHitInfo.tickCountWhenHit = cudaCar.ballHitTickCount;
			carState.lastControls = ToCpuControls(cudaCar.lastControls);
			car->team = cudaCar.team == rsc::Team::BLUE ? Team::BLUE : Team::ORANGE;
			car->id = cudaCar.id;
			car->SetState(carState);
			car->_internalState.updateCounter = arenaInfo.tickCount;
		}
	}

	void SyncShadowArenaFromCudaSnapshot(
		RLGC::EnvSet* envSet,
		int arenaIdx,
		const rsc::ArenaInfo& arenaInfo,
		const rsc::BallState& cudaBall,
		const rsc::CarState* cudaCars
	) {
		auto* arena = envSet->arenas[arenaIdx];
		arena->tickCount = arenaInfo.tickCount;

		BallState ballState = arena->ball->GetState();
		ballState.pos = Vec(cudaBall.pos.x, cudaBall.pos.y, cudaBall.pos.z);
		ballState.rotMat = RotMat(
			Vec(cudaBall.rotMat.forward.x, cudaBall.rotMat.forward.y, cudaBall.rotMat.forward.z),
			Vec(cudaBall.rotMat.right.x, cudaBall.rotMat.right.y, cudaBall.rotMat.right.z),
			Vec(cudaBall.rotMat.up.x, cudaBall.rotMat.up.y, cudaBall.rotMat.up.z)
		);
		ballState.vel = Vec(cudaBall.vel.x, cudaBall.vel.y, cudaBall.vel.z);
		ballState.angVel = Vec(cudaBall.angVel.x, cudaBall.angVel.y, cudaBall.angVel.z);
		arena->ball->SetState(ballState);
		arena->ball->_internalState.updateCounter = arenaInfo.tickCount;

		for (int carIdx = 0; carIdx < envSet->cudaArenaCars[arenaIdx].size(); carIdx++) {
			Car* car = envSet->cudaArenaCars[arenaIdx][carIdx];
			const rsc::CarState& cudaCar = cudaCars[carIdx];
			CarState carState = car->GetState();
			carState.pos = Vec(cudaCar.pos.x, cudaCar.pos.y, cudaCar.pos.z);
			carState.rotMat = RotMat(
				Vec(cudaCar.rotMat.forward.x, cudaCar.rotMat.forward.y, cudaCar.rotMat.forward.z),
				Vec(cudaCar.rotMat.right.x, cudaCar.rotMat.right.y, cudaCar.rotMat.right.z),
				Vec(cudaCar.rotMat.up.x, cudaCar.rotMat.up.y, cudaCar.rotMat.up.z)
			);
			carState.vel = Vec(cudaCar.vel.x, cudaCar.vel.y, cudaCar.vel.z);
			carState.angVel = Vec(cudaCar.angVel.x, cudaCar.angVel.y, cudaCar.angVel.z);
			carState.boost = cudaCar.boost;
			carState.timeSpentBoosting = cudaCar.timeSpentBoosting;
			carState.isOnGround = cudaCar.isOnGround;
			carState.hasJumped = cudaCar.hasJumped;
			carState.hasDoubleJumped = cudaCar.hasDoubleJumped;
			carState.hasFlipped = cudaCar.hasFlipped;
			carState.isFlipping = cudaCar.isFlipping;
			carState.isJumping = cudaCar.isJumping;
			carState.flipRelTorque = Vec(cudaCar.flipRelTorque.x, cudaCar.flipRelTorque.y, cudaCar.flipRelTorque.z);
			carState.jumpTime = cudaCar.jumpTime;
			carState.flipTime = cudaCar.flipTime;
			carState.airTime = cudaCar.airTime;
			carState.airTimeSinceJump = cudaCar.airTimeSinceJump;
			carState.isSupersonic = cudaCar.isSupersonic;
			carState.supersonicTime = cudaCar.supersonicTime;
			carState.handbrakeVal = cudaCar.handbrakeVal;
			carState.isAutoFlipping = cudaCar.isAutoFlipping;
			carState.autoFlipTimer = cudaCar.autoFlipTimer;
			carState.autoFlipTorqueScale = cudaCar.autoFlipTorqueScale;
			carState.isDemoed = cudaCar.isDemoed;
			carState.demoRespawnTimer = cudaCar.demoRespawnTimer;
			carState.worldContact.hasContact = cudaCar.worldContactHasContact;
			carState.worldContact.contactNormal = Vec(cudaCar.worldContactNormal.x, cudaCar.worldContactNormal.y, cudaCar.worldContactNormal.z);
			carState.carContact.otherCarID = cudaCar.carContactOtherID;
			carState.carContact.cooldownTimer = cudaCar.carContactCooldownTimer;
			carState.ballHitInfo.isValid = cudaCar.ballHitValid;
			carState.ballHitInfo.tickCountWhenHit = cudaCar.ballHitTickCount;
			carState.lastControls = ToCpuControls(cudaCar.lastControls);
			car->team = cudaCar.team == rsc::Team::BLUE ? Team::BLUE : Team::ORANGE;
			car->id = cudaCar.id;
			car->SetState(carState);
			car->_internalState.updateCounter = arenaInfo.tickCount;
		}
	}

	void ApplyCudaBumpDemoEvents(RLGC::GameState& gs, RLGC::GameState* prev) {
		for (auto& player : gs.players) {
			if (player.isDemoed && (!player.prev || !player.prev->isDemoed))
				player.eventState.demoed = true;
		}

		for (auto& player : gs.players) {
			if (!player.carContact.otherCarID || player.carContact.cooldownTimer <= 0)
				continue;

			bool isNewContact =
				!player.prev ||
				player.prev->carContact.otherCarID != player.carContact.otherCarID ||
				player.prev->carContact.cooldownTimer <= 0 ||
				player.prev->carContact.cooldownTimer < player.carContact.cooldownTimer;

			if (!isNewContact)
				continue;

			RLGC::Player* otherPlayer = NULL;
			for (auto& other : gs.players) {
				if (other.carId == player.carContact.otherCarID) {
					otherPlayer = &other;
					break;
				}
			}

			if (!otherPlayer || otherPlayer->team == player.team)
				continue;

			player.eventState.bump = true;
			otherPlayer->eventState.bumped = true;

			bool otherNewlyDemoed = otherPlayer->isDemoed && (!otherPlayer->prev || !otherPlayer->prev->isDemoed);
			if (otherNewlyDemoed) {
				player.eventState.demo = true;
				otherPlayer->eventState.demoed = true;
			}
		}
	}
}
#endif

template<bool RLGC::PlayerEventState::* DATA_VAR>
void IncPlayerCounter(Car* car, void* userInfoPtr) {
	if (!car)
		return;

	auto userInfo = (RLGC::EnvSet::CallbackUserInfo*)userInfoPtr;

	auto& gs = userInfo->envSet->state.gameStates[userInfo->arenaIdx];
	for (auto& player : gs.players)
		if (player.carId == car->id)
			(player.eventState.*DATA_VAR) = true;
}

void _ShotEventCallback(Arena* arena, Car* shooter, Car* passer, void* userInfo) {
	IncPlayerCounter<&RLGC::PlayerEventState::shot>(shooter, userInfo);
	IncPlayerCounter<&RLGC::PlayerEventState::shotPass>(passer, userInfo);
}

void _GoalEventCallback(Arena* arena, Car* scorer, Car* passer, void* userInfo) {
	IncPlayerCounter<&RLGC::PlayerEventState::goal>(scorer, userInfo);
	IncPlayerCounter<&RLGC::PlayerEventState::assist>(passer, userInfo);
}

void _SaveEventCallback(Arena* arena, Car* saver, void* userInfo) {
	IncPlayerCounter<&RLGC::PlayerEventState::save>(saver, userInfo);
}

void _BumpCallback(Arena* arena, Car* bumper, Car* victim, bool isDemo, void* userInfo) {
	if (bumper->team == victim->team)
		return;

	IncPlayerCounter<&RLGC::PlayerEventState::bump>(bumper, userInfo);
	IncPlayerCounter<&RLGC::PlayerEventState::bumped>(victim, userInfo);

	if (isDemo) {
		IncPlayerCounter<&RLGC::PlayerEventState::demo>(bumper, userInfo);
		IncPlayerCounter<&RLGC::PlayerEventState::demoed>(victim, userInfo);
	}
}

/////////////////////////////

RLGC::EnvSet::EnvSet(const EnvSetConfig& config) : config(config) {

	RG_ASSERT(config.tickSkip > 0);
	RG_ASSERT(config.actionDelay >= 0 && config.actionDelay <= config.tickSkip);

#ifndef RG_ROCKETSIMCUDA_AVAILABLE
	if (config.physicsBackend == EnvPhysicsBackend::ROCKETSIM_CUDA) {
		RG_LOG("EnvSet: RocketSimCuda backend requested but RocketSimCuda is not compiled in, falling back to RocketSim CPU.");
	}
#endif

	std::mutex appendMutex = {};
	auto fnCreateArenas = [&](int idx) {
		auto createResult = config.envCreateFn(idx);
		auto arena = createResult.arena;

		appendMutex.lock();
		{
			arenas.push_back(arena);

			auto userInfo = new CallbackUserInfo();
			userInfo->arena = arena;
			userInfo->arenaIdx = idx;
			userInfo->envSet = this;
			eventCallbackInfos.push_back(userInfo);
			arena->SetCarBumpCallback(_BumpCallback, userInfo);

			if (arena->gameMode != GameMode::HEATSEEKER) {
				GameEventTracker* tracker = new GameEventTracker({});
				eventTrackers.push_back(tracker);

				tracker->SetShotCallback(_ShotEventCallback, userInfo);
				tracker->SetGoalCallback(_GoalEventCallback, userInfo);
				tracker->SetSaveCallback(_SaveEventCallback, userInfo);
			} else {
				eventTrackers.push_back(NULL);
				eventCallbackInfos.push_back(NULL);
			}

			userInfos.push_back(createResult.userInfo);

			rewards.push_back(createResult.rewards);
			terminalConditions.push_back(createResult.terminalConditions);
			obsBuilders.push_back(createResult.obsBuilder);
			actionParsers.push_back(createResult.actionParser);
			continuousActionParsers.push_back(createResult.continuousActionParser);
			stateSetters.push_back(createResult.stateSetter);
		}
		appendMutex.unlock();
	};
	g_ThreadPool.StartBatchedJobs(fnCreateArenas, config.numArenas, false);

	state.Resize(arenas);

#ifdef RG_ROCKETSIMCUDA_AVAILABLE
	if (config.physicsBackend == EnvPhysicsBackend::ROCKETSIM_CUDA) {
		RG_ASSERT(!arenas.empty());

		cudaMaxCarsPerArena = 0;
		for (auto* arena : arenas)
			cudaMaxCarsPerArena = RS_MAX(cudaMaxCarsPerArena, (int)arena->_cars.size());

		rsc::BatchConfig batchConfig = {};
		batchConfig.numArenas = (int)arenas.size();
		batchConfig.maxCarsPerArena = cudaMaxCarsPerArena;
		batchConfig.tickRate = arenas[0]->GetTickRate();
		batchConfig.gameMode = (arenas[0]->gameMode == GameMode::THE_VOID) ? rsc::GameMode::THE_VOID : rsc::GameMode::SOCCAR;
		batchConfig.mutatorConfig = ToCudaMutatorConfig(arenas[0]->GetMutatorConfig());

		// Use the real arena collision meshes (exact reference geometry) â€”
		// same folder RocketSim::Init was called with.
		static std::string collisionMeshesPath;
		collisionMeshesPath = RocketSim::_collisionMeshesFolder.string();
		batchConfig.collisionMeshesPath = collisionMeshesPath.empty() ? nullptr : collisionMeshesPath.c_str();

		cudaBatch = std::make_unique<rsc::RocketSimCudaBatch>();
		cudaBatch->Init(batchConfig);

		cudaArenaCars.resize(arenas.size());
		cudaControlBuffer.resize(arenas.size() * cudaMaxCarsPerArena);
		cudaAllCarStates.resize(arenas.size() * cudaMaxCarsPerArena);
		cudaAllBallStates.resize(arenas.size());
		cudaAllBoostPadStates.resize(arenas.size() * rsc::NUM_BOOST_PADS);
		cudaAllArenaInfos.resize(arenas.size());

		for (int arenaIdx = 0; arenaIdx < arenas.size(); arenaIdx++) {
			auto* arena = arenas[arenaIdx];
			auto& cars = cudaArenaCars[arenaIdx];
			cars.assign(arena->_cars.begin(), arena->_cars.end());
			std::sort(cars.begin(), cars.end(), [](const Car* a, const Car* b) { return a->id < b->id; });

			for (int carIdx = 0; carIdx < cars.size(); carIdx++) {
				Car* car = cars[carIdx];
				cudaBatch->AddCar(arenaIdx, ToCudaTeam(car->team), MapCarPreset(car->config));

				rsc::CarState cudaState = cudaBatch->GetCarState(arenaIdx, carIdx);
				uint32_t oldID = car->id;
				arena->_carIDMap.erase(oldID);
				car->id = cudaState.id;
				arena->_carIDMap[car->id] = car;
				arena->_lastCarID = RS_MAX(arena->_lastCarID, car->id);
			}
		}

		RG_LOG("EnvSet: RocketSimCuda backend enabled for env stepping.");
	}
#endif
	
	// Determine obs size and action amount, initialize arrays accordingly
	{
		stateSetters[0]->ResetArena(arenas[0]);
		GameState testState = GameState(arenas[0]);
		testState.userInfo = userInfos[0];
		obsBuilders[0]->Reset(testState);
		obsSize = obsBuilders[0]->BuildObs(testState.players[0], testState).size();
		state.obs = DimList2<float>(state.numPlayers, obsSize);

		state.actionMasks = DimList2<uint8_t>(state.numPlayers, actionParsers[0]->GetActionAmount());
	}

#ifdef RG_ROCKETSIMCUDA_AVAILABLE
	if (UsingCudaBackend()) {
		bool allArenasFull = true;
		bool allAdvancedObs = true;
		bool allDefaultAction = true;
		for (int arenaIdx = 0; arenaIdx < arenas.size(); arenaIdx++) {
			allArenasFull &= (int)arenas[arenaIdx]->_cars.size() == cudaMaxCarsPerArena;
			allAdvancedObs &= dynamic_cast<AdvancedObs*>(obsBuilders[arenaIdx]) != NULL;
			allDefaultAction &= dynamic_cast<DefaultAction*>(actionParsers[arenaIdx]) != NULL;
		}

		bool advancedFastPath =
			allArenasFull &&
			allAdvancedObs &&
			allDefaultAction &&
			(obsSize == cudaBatch->GetAdvancedObsSize()) &&
			(state.actionMasks.size[1] == rsc::DEFAULT_ACTION_COUNT);

		cudaAdvancedObsDefaultActionFastPath = advancedFastPath;

		if (advancedFastPath) {
			RG_LOG("EnvSet: CUDA AdvancedObs/DefaultAction fast path enabled.");
		} else {
			RG_LOG("EnvSet: CUDA obs fast path unavailable, keeping CPU obs/action mask path.");
		}
		// OPT-118 guard: disabling CPU world-state sync (cudaNoCpuWorldState) only stops
		// the device->CPU GameState copy. The obs/action-mask buffers must then come from
		// the device via RefreshCudaTrainingBuffers(), which only runs when the
		// AdvancedObs/DefaultAction fast path is active. Without that fast path, ResetArena
		// / StepSecondHalf would rebuild obs from the un-synced (stale) CPU GameStates and
		// silently feed garbage observations to the learner. Refuse the unsafe combo rather
		// than train on corrupt data.
		if (!config.syncCudaWorldStateToCpu && !cudaAdvancedObsDefaultActionFastPath) {
			RG_ERR_CLOSE(
				"EnvSet: syncCudaWorldStateToCpu=false (cudaNoCpuWorldState) requires the CUDA "
				"AdvancedObs/DefaultAction fast path as the device obs source, but it is unavailable. "
				"Likely causes: not all arenas are full, the obs builder is not AdvancedObs, the "
				"action parser is not DefaultAction, obs size != cudaBatch AdvancedObs size, or the "
				"action-mask count != DEFAULT_ACTION_COUNT. Re-enable CPU world-state sync or fix the "
				"env so the fast path engages."
			);
		}

		rsc::TrainingRewardConfig cudaRewardConfig = {};
		rsc::TrainingTerminalConfig cudaTerminalConfig = {};
		std::string cudaTrainingReason;
		bool rewardsSupported = BuildCudaTrainingRewardConfig(rewards[0], cudaRewardConfig, cudaTrainingReason);
		bool terminalsSupported = rewardsSupported && BuildCudaTrainingTerminalConfig(terminalConditions[0], cudaTerminalConfig, cudaTrainingReason);

		if (rewardsSupported && terminalsSupported) {
			cudaBatch->ConfigureTrainingRewards(cudaRewardConfig);
			cudaBatch->ConfigureTrainingTerminals(cudaTerminalConfig);
			cudaTrainingRewards.resize(state.numPlayers);
			cudaTrainingTerminals.resize(arenas.size());
			cudaRewardsTerminalFastPath = true;
			RG_LOG("EnvSet: CUDA reward/terminal fast path enabled.");
		} else {
			cudaRewardsTerminalFastPath = false;
			RG_LOG("EnvSet: CUDA reward/terminal fast path unavailable, keeping CPU reward/terminal path. Reason: " << cudaTrainingReason);
		}

		if (UsingCudaNoCpuWorldStateFastPath())
			RG_LOG("EnvSet: CUDA no-CPU-world-state fast path enabled.");
	}
#endif

	// Reset all arenas initially
	if (config.physicsBackend == EnvPhysicsBackend::ROCKETSIM_CPU) {
		for (int arenaIdx = 0; arenaIdx < config.numArenas; arenaIdx++)
			ResetArena(arenaIdx);
	} else {
		g_ThreadPool.StartBatchedJobs(
			std::bind(&RLGC::EnvSet::ResetArena, this, std::placeholders::_1),
			config.numArenas, false
		);
	}

#ifdef RG_ROCKETSIMCUDA_AVAILABLE
	if (UsingCudaTrainingFastPath())
		RefreshCudaTrainingBuffers();
#endif
	
}

void RLGC::EnvSet::StepFirstHalf(bool async) {

#ifdef RG_ROCKETSIMCUDA_AVAILABLE
	if (UsingCudaBackend()) {
		auto fnStepBatch = [&](int) {
			if (!UsingCudaNoCpuWorldStateFastPath()) {
				for (int arenaIdx = 0; arenaIdx < state.gameStates.size(); arenaIdx++) {
					auto& gs = state.gameStates[arenaIdx];
					state.prevGameStates[arenaIdx] = gs;
					gs.ResetBeforeStep();
				}
			}

			if (UsingCudaRewardTerminalFastPath())
				cudaBatch->SnapshotTrainingState();

			if (config.actionDelay > 0)
				cudaBatch->Step(config.actionDelay);
		};

		g_ThreadPool.StartBatchedJobs(fnStepBatch, 1, async);
		return;
	}
#endif

	auto fnStepArena = [&](int arenaIdx) {
		Arena* arena = arenas[arenaIdx];
		auto& gs = state.gameStates[arenaIdx];

		{
			// Set previous gamestates
			state.prevGameStates[arenaIdx] = gs;
		}

		gs.ResetBeforeStep();

		// Step arena with old actions
		arena->Step(config.actionDelay);
	};

	const char* serialCpuEnv = getenv("GGL_SERIAL_CPU_ENV");
	if (serialCpuEnv && serialCpuEnv[0] != '0') {
		for (int arenaIdx = 0; arenaIdx < arenas.size(); arenaIdx++)
			fnStepArena(arenaIdx);
	} else {
		g_ThreadPool.StartBatchedJobs(fnStepArena, arenas.size(), async);
	}
}

void RLGC::EnvSet::StepSecondHalf(const IList& actionIndices, bool async) {

#ifdef RG_ROCKETSIMCUDA_AVAILABLE
	if (UsingCudaBackend()) {
		// Every SecondHalf call site passes async=false: run the GPU work serially
		// on this thread, then fan the per-arena CPU bookkeeping out across the
		// thread pool (it previously ran single-threaded inside one pool job).
		{
			std::fill(cudaControlBuffer.begin(), cudaControlBuffer.end(), rsc::CarControls());
			bool skipCpuWorldState = UsingCudaNoCpuWorldStateFastPath();
			std::vector<std::vector<Action>> allActions(skipCpuWorldState ? 0 : arenas.size());

			for (int arenaIdx = 0; arenaIdx < arenas.size(); arenaIdx++) {
				auto& gs = state.gameStates[arenaIdx];
				int playerStartIdx = state.arenaPlayerStartIdx[arenaIdx];
				std::vector<Action>* actions = skipCpuWorldState ? NULL : &allActions[arenaIdx];
				if (actions)
					actions->resize(gs.players.size());

				for (int i = 0; i < gs.players.size(); i++) {
					Action action = actionParsers[arenaIdx]->ParseAction(actionIndices[playerStartIdx + i], gs.players[i], gs);
					if (actions)
						(*actions)[i] = action;
					cudaControlBuffer[arenaIdx * cudaMaxCarsPerArena + i] = ToCudaControls(action);
				}
			}

			cudaBatch->SetAllCarControls(cudaControlBuffer.data());
			if (config.tickSkip - config.actionDelay > 0)
				cudaBatch->Step(config.tickSkip - config.actionDelay);
			if (UsingCudaRewardTerminalFastPath()) {
				cudaBatch->BuildRewardsAndTerminals(config.tickSkip);
				cudaBatch->CopyBuiltRewards(cudaTrainingRewards.data());
				cudaBatch->CopyBuiltTerminals(cudaTrainingTerminals.data());
			}

			if (skipCpuWorldState) {
				for (int arenaIdx = 0; arenaIdx < arenas.size(); arenaIdx++) {
					auto& gs = state.gameStates[arenaIdx];
					int playerStartIdx = state.arenaPlayerStartIdx[arenaIdx];
					state.terminals[arenaIdx] = cudaTrainingTerminals[arenaIdx];
					for (int i = 0; i < gs.players.size(); i++)
						state.rewards[playerStartIdx + i] = cudaTrainingRewards[playerStartIdx + i];
				}
			} else {
				cudaBatch->GetAllArenaInfos(cudaAllArenaInfos.data());
				cudaBatch->GetAllBallStates(cudaAllBallStates.data());
				cudaBatch->GetAllCarStates(cudaAllCarStates.data());
				cudaBatch->GetAllBoostPadStates(cudaAllBoostPadStates.data());

				auto fnArenaUpdate = [&](int arenaIdx) {
					auto& gs = state.gameStates[arenaIdx];
					int playerStartIdx = state.arenaPlayerStartIdx[arenaIdx];
					const rsc::ArenaInfo& arenaInfo = cudaAllArenaInfos[arenaIdx];
					const rsc::BallState& ballState = cudaAllBallStates[arenaIdx];
					const rsc::CarState* carStates = cudaAllCarStates.data() + arenaIdx * cudaMaxCarsPerArena;
					const rsc::BoostPadState* padStates = cudaAllBoostPadStates.data() + arenaIdx * rsc::NUM_BOOST_PADS;

					SyncShadowArenaFromCudaSnapshot(this, arenaIdx, arenaInfo, ballState, carStates);

					if (eventTrackers[arenaIdx])
						eventTrackers[arenaIdx]->Update(arenas[arenaIdx]);

					GameState* gsPrev = &state.prevGameStates[arenaIdx];
					if (gsPrev->IsEmpty())
						gsPrev = NULL;

					gs.UpdateFromCudaSnapshot(arenaInfo, ballState, carStates, padStates, allActions[arenaIdx], gsPrev);
					gs.lastArena = arenas[arenaIdx];
					gs.userInfo = userInfos[arenaIdx];
					ApplyCudaBumpDemoEvents(gs, gsPrev);

					uint8_t terminalType = TerminalType::NOT_TERMINAL;
					if (UsingCudaRewardTerminalFastPath()) {
						terminalType = cudaTrainingTerminals[arenaIdx];
					} else {
						for (auto cond : terminalConditions[arenaIdx]) {
							if (cond->IsTerminal(gs)) {
								bool isTrunc = cond->IsTruncation();
								uint8_t curTerminalType = isTrunc ? TerminalType::TRUNCATED : TerminalType::NORMAL;
								if (terminalType == TerminalType::NOT_TERMINAL || curTerminalType == TerminalType::NORMAL)
									terminalType = curTerminalType;
							}
						}
					}
					state.terminals[arenaIdx] = terminalType;

					if (UsingCudaRewardTerminalFastPath()) {
						for (int i = 0; i < gs.players.size(); i++)
							state.rewards[playerStartIdx + i] = cudaTrainingRewards[playerStartIdx + i];
					}

					if (!UsingCudaTrainingFastPath()) {
						for (int i = 0; i < gs.players.size(); i++)
							state.obs.Set(playerStartIdx + i, obsBuilders[arenaIdx]->BuildObs(gs.players[i], gs));

						for (int i = 0; i < gs.players.size(); i++)
							state.actionMasks.Set(playerStartIdx + i, actionParsers[arenaIdx]->GetActionMask(gs.players[i], gs));
					}
				};
				g_ThreadPool.StartBatchedJobs(fnArenaUpdate, (int)arenas.size(), false);

				// Reward objects can share process-global state across arenas
				// (e.g. stateful reward internals), so reward
				// evaluation must stay on a single thread.
				for (int arenaIdx = 0; arenaIdx < arenas.size(); arenaIdx++) {
					auto& gs = state.gameStates[arenaIdx];
					int playerStartIdx = state.arenaPlayerStartIdx[arenaIdx];
					uint8_t terminalType = state.terminals[arenaIdx];

					if (UsingCudaRewardTerminalFastPath()) {
						if (config.saveRewards && arenaIdx < saveRewardsArenaLimit) {
							for (auto& weighted : rewards[arenaIdx])
								weighted.reward->PreStep(gs);

							if (state.lastRewards[arenaIdx].empty())
								state.lastRewards[arenaIdx].resize(rewards[arenaIdx].size());

							for (int rewardIdx = 0; rewardIdx < rewards[arenaIdx].size(); rewardIdx++) {
								auto& weightedReward = rewards[arenaIdx][rewardIdx];
								FList output = weightedReward.reward->GetAllRewards(gs, terminalType);

								int playerSampleIndex;
								if (config.shuffleRewardSampling) {
									playerSampleIndex = RocketSim::Math::RandInt(0, output.size());
								} else {
									playerSampleIndex = 0;
									int lowestID = gs.players[0].carId;
									for (int i = 1; i < gs.players.size(); i++) {
										auto id = gs.players[i].carId;
										if (id < lowestID) {
											lowestID = id;
											playerSampleIndex = i;
										}
									}
								}

								float rewardToSave = output[playerSampleIndex];
								if (ZeroSumReward* zeroSum = dynamic_cast<ZeroSumReward*>(weightedReward.reward))
									rewardToSave = zeroSum->_lastRewards[playerSampleIndex];
								state.lastRewards[arenaIdx][rewardIdx] = rewardToSave;
							}
						}
					} else {
						for (auto& weighted : rewards[arenaIdx])
							weighted.reward->PreStep(gs);

						FList allRewards = FList(gs.players.size(), 0);
						for (int rewardIdx = 0; rewardIdx < rewards[arenaIdx].size(); rewardIdx++) {
							auto& weightedReward = rewards[arenaIdx][rewardIdx];
							FList output = weightedReward.reward->GetAllRewards(gs, terminalType);
							for (int i = 0; i < gs.players.size(); i++)
								allRewards[i] += output[i] * weightedReward.weight;

							if (config.saveRewards) {
								int playerSampleIndex;
								if (config.shuffleRewardSampling) {
									playerSampleIndex = RocketSim::Math::RandInt(0, output.size());
								} else {
									playerSampleIndex = 0;
									int lowestID = gs.players[0].carId;
									for (int i = 1; i < gs.players.size(); i++) {
										auto id = gs.players[i].carId;
										if (id < lowestID) {
											lowestID = id;
											playerSampleIndex = i;
										}
									}
								}

								float rewardToSave = output[playerSampleIndex];
								if (ZeroSumReward* zeroSum = dynamic_cast<ZeroSumReward*>(weightedReward.reward))
									rewardToSave = zeroSum->_lastRewards[playerSampleIndex];

								if (state.lastRewards[arenaIdx].empty())
									state.lastRewards[arenaIdx].resize(rewards[arenaIdx].size());

								state.lastRewards[arenaIdx][rewardIdx] = rewardToSave;
							}
						}

						for (int i = 0; i < gs.players.size(); i++)
							state.rewards[playerStartIdx + i] = allRewards[i];
					}
				}
			}

			if (UsingCudaTrainingFastPath())
				RefreshCudaTrainingBuffers();
		}

		return;
	}
#endif

	auto fnStepArenas = [&](int arenaIdx) {

		Arena* arena = arenas[arenaIdx];
		auto& gs = state.gameStates[arenaIdx];
		int playerStartIdx = state.arenaPlayerStartIdx[arenaIdx];
			
		// Parse and set actions
		auto actions = std::vector<Action>(gs.players.size());
		auto carItr = arena->_cars.begin();
		for (int i = 0; i < gs.players.size(); i++, carItr++) {
			auto& player = gs.players[i];
			Car* car = *carItr;
			Action action = actionParsers[arenaIdx]->ParseAction(actionIndices[playerStartIdx + i], player, gs);
			car->controls = (CarControls)action;
			actions[i] = action;
		}

		// Step arena with new actions we got from observing the last state
		// Update the gamestate after
		{
			arena->Step(config.tickSkip - config.actionDelay);

			if (eventTrackers[arenaIdx])
				eventTrackers[arenaIdx]->Update(arena);

			GameState* gsPrev = &state.prevGameStates[arenaIdx];
			if (gsPrev->IsEmpty())
				gsPrev = NULL;

			gs.UpdateFromArena(arena, actions, gsPrev);
		}

		// Update terminal
		uint8_t terminalType = TerminalType::NOT_TERMINAL;
		{
			for (auto cond : terminalConditions[arenaIdx]) {
				if (cond->IsTerminal(gs)) {
					bool isTrunc = cond->IsTruncation();
					uint8_t curTerminalType = isTrunc ? TerminalType::TRUNCATED : TerminalType::NORMAL;
					if (terminalType == TerminalType::NOT_TERMINAL) {
						terminalType = curTerminalType;
					} else {
						// We already know this state is terminal
						// However, if we only know it is a truncated terminal, we should let normal terminals take priority
						// (Normal terminals are better information than truncations)
						if (curTerminalType == TerminalType::NORMAL)
							terminalType = curTerminalType;
					}

					// NOTE: We can't break since terminal conditions are guaranteed to be called once per step
				}
			}
			state.terminals[arenaIdx] = terminalType;
		}
		
		// Pre-step rewards
		{
			for (auto& weighted : rewards[arenaIdx])
				weighted.reward->PreStep(gs);
		}

		// Update rewards
		{
			FList allRewards = FList(gs.players.size(), 0);
			for (int rewardIdx = 0; rewardIdx < rewards[arenaIdx].size(); rewardIdx++) {
				auto& weightedReward = rewards[arenaIdx][rewardIdx];
				FList output = weightedReward.reward->GetAllRewards(gs, terminalType);
				for (int i = 0; i < gs.players.size(); i++)
					allRewards[i] += output[i] * weightedReward.weight;

				// Save the reward
				if (config.saveRewards) {
					int playerSampleIndex;
					if (config.shuffleRewardSampling) {
						playerSampleIndex = RocketSim::Math::RandInt(0, output.size());
					} else {
						// Find player with the lowest id
						playerSampleIndex = 0;
						int lowestID = gs.players[0].carId;
						for (int i = 1; i < gs.players.size(); i++) {
							auto id = gs.players[i].carId;
							if (id < lowestID) {
								lowestID = id;
								playerSampleIndex = i;
							}
						}
					}
					// We will only take the reward from a random player
					float rewardToSave = output[playerSampleIndex];
						
					// If zero-sum, use the inner reward
					if (ZeroSumReward* zeroSum = dynamic_cast<ZeroSumReward*>(weightedReward.reward))
						rewardToSave = zeroSum->_lastRewards[playerSampleIndex];

					// If needed, initialize last rewards
					if (state.lastRewards[arenaIdx].empty())
						state.lastRewards[arenaIdx].resize(rewards[arenaIdx].size());

					state.lastRewards[arenaIdx][rewardIdx] = rewardToSave;
				}
			}

			for (int i = 0; i < gs.players.size(); i++)
				state.rewards[playerStartIdx + i] = allRewards[i];
		}

		// Update observations
		{
			for (int i = 0; i < gs.players.size(); i++)
				state.obs.Set(playerStartIdx + i, obsBuilders[arenaIdx]->BuildObs(gs.players[i], gs));
		}

		// Update action masks
		{
			for (int i = 0; i < gs.players.size(); i++)
				state.actionMasks.Set(playerStartIdx + i, actionParsers[arenaIdx]->GetActionMask(gs.players[i], gs));
		}
	};

	g_ThreadPool.StartBatchedJobs(fnStepArenas, arenas.size(), async);
}

void RLGC::EnvSet::StepSecondHalfContinuous(const FList& continuousActions, int actionDim, bool async) {

#ifdef RG_ROCKETSIMCUDA_AVAILABLE
	if (UsingCudaBackend()) {
		// Same structure as StepSecondHalf: serial GPU work on this thread, then
		// per-arena CPU bookkeeping fanned out across the thread pool.
		{
			std::fill(cudaControlBuffer.begin(), cudaControlBuffer.end(), rsc::CarControls());
			bool skipCpuWorldState = UsingCudaNoCpuWorldStateFastPath();
			std::vector<std::vector<Action>> allActions(skipCpuWorldState ? 0 : arenas.size());

			for (int arenaIdx = 0; arenaIdx < arenas.size(); arenaIdx++) {
				auto& gs = state.gameStates[arenaIdx];
				int playerStartIdx = state.arenaPlayerStartIdx[arenaIdx];
				std::vector<Action>* actions = skipCpuWorldState ? NULL : &allActions[arenaIdx];
				if (actions)
					actions->resize(gs.players.size());

				for (int i = 0; i < gs.players.size(); i++) {
					auto& player = gs.players[i];
					const float* actionPtr = &continuousActions[(playerStartIdx + i) * actionDim];

					Action action;
					if (arenaIdx < continuousActionParsers.size() && continuousActionParsers[arenaIdx]) {
						action = continuousActionParsers[arenaIdx]->ParseContinuousAction(actionPtr, actionDim, player, gs);
					} else {
						RG_ASSERT(actionDim >= (int)Action::ELEM_AMOUNT);
						action.throttle = actionPtr[0];
						action.steer = actionPtr[1];
						action.pitch = actionPtr[2];
						action.yaw = actionPtr[3];
						action.roll = actionPtr[4];
						action.jump = actionPtr[5] > 0.0f ? 1.0f : 0.0f;
						action.boost = actionPtr[6] > 0.0f ? 1.0f : 0.0f;
						action.handbrake = actionPtr[7] > 0.0f ? 1.0f : 0.0f;
					}

					if (actions)
						(*actions)[i] = action;
					cudaControlBuffer[arenaIdx * cudaMaxCarsPerArena + i] = ToCudaControls(action);
				}
			}

			cudaBatch->SetAllCarControls(cudaControlBuffer.data());
			if (config.tickSkip - config.actionDelay > 0)
				cudaBatch->Step(config.tickSkip - config.actionDelay);
			if (UsingCudaRewardTerminalFastPath()) {
				cudaBatch->BuildRewardsAndTerminals(config.tickSkip);
				cudaBatch->CopyBuiltRewards(cudaTrainingRewards.data());
				cudaBatch->CopyBuiltTerminals(cudaTrainingTerminals.data());
			}

			if (skipCpuWorldState) {
				for (int arenaIdx = 0; arenaIdx < arenas.size(); arenaIdx++) {
					auto& gs = state.gameStates[arenaIdx];
					int playerStartIdx = state.arenaPlayerStartIdx[arenaIdx];
					state.terminals[arenaIdx] = cudaTrainingTerminals[arenaIdx];
					for (int i = 0; i < gs.players.size(); i++)
						state.rewards[playerStartIdx + i] = cudaTrainingRewards[playerStartIdx + i];
				}
			} else {
				cudaBatch->GetAllArenaInfos(cudaAllArenaInfos.data());
				cudaBatch->GetAllBallStates(cudaAllBallStates.data());
				cudaBatch->GetAllCarStates(cudaAllCarStates.data());
				cudaBatch->GetAllBoostPadStates(cudaAllBoostPadStates.data());

				auto fnArenaUpdate = [&](int arenaIdx) {
				auto& gs = state.gameStates[arenaIdx];
				int playerStartIdx = state.arenaPlayerStartIdx[arenaIdx];
				const rsc::ArenaInfo& arenaInfo = cudaAllArenaInfos[arenaIdx];
				const rsc::BallState& ballState = cudaAllBallStates[arenaIdx];
				const rsc::CarState* carStates = cudaAllCarStates.data() + arenaIdx * cudaMaxCarsPerArena;
				const rsc::BoostPadState* padStates = cudaAllBoostPadStates.data() + arenaIdx * rsc::NUM_BOOST_PADS;

				SyncShadowArenaFromCudaSnapshot(this, arenaIdx, arenaInfo, ballState, carStates);

				if (eventTrackers[arenaIdx])
					eventTrackers[arenaIdx]->Update(arenas[arenaIdx]);

				GameState* gsPrev = &state.prevGameStates[arenaIdx];
				if (gsPrev->IsEmpty())
					gsPrev = NULL;

				gs.UpdateFromCudaSnapshot(arenaInfo, ballState, carStates, padStates, allActions[arenaIdx], gsPrev);
				gs.lastArena = arenas[arenaIdx];
				gs.userInfo = userInfos[arenaIdx];
				ApplyCudaBumpDemoEvents(gs, gsPrev);

				uint8_t terminalType = TerminalType::NOT_TERMINAL;
				if (UsingCudaRewardTerminalFastPath()) {
					terminalType = cudaTrainingTerminals[arenaIdx];
				} else {
					for (auto cond : terminalConditions[arenaIdx]) {
						if (cond->IsTerminal(gs)) {
							bool isTrunc = cond->IsTruncation();
							uint8_t curTerminalType = isTrunc ? TerminalType::TRUNCATED : TerminalType::NORMAL;
							if (terminalType == TerminalType::NOT_TERMINAL || curTerminalType == TerminalType::NORMAL)
								terminalType = curTerminalType;
						}
					}
				}
				state.terminals[arenaIdx] = terminalType;

				if (UsingCudaRewardTerminalFastPath()) {
					for (int i = 0; i < gs.players.size(); i++)
						state.rewards[playerStartIdx + i] = cudaTrainingRewards[playerStartIdx + i];
				}

				if (!UsingCudaTrainingFastPath()) {
					for (int i = 0; i < gs.players.size(); i++)
						state.obs.Set(playerStartIdx + i, obsBuilders[arenaIdx]->BuildObs(gs.players[i], gs));

					for (int i = 0; i < gs.players.size(); i++)
						state.actionMasks.Set(playerStartIdx + i, actionParsers[arenaIdx]->GetActionMask(gs.players[i], gs));
				}
				};
				g_ThreadPool.StartBatchedJobs(fnArenaUpdate, (int)arenas.size(), false);

				// Reward objects can share process-global state across arenas
				// (e.g. stateful reward internals), so reward
				// evaluation must stay on a single thread.
				for (int arenaIdx = 0; arenaIdx < arenas.size(); arenaIdx++) {
					auto& gs = state.gameStates[arenaIdx];
					int playerStartIdx = state.arenaPlayerStartIdx[arenaIdx];
					uint8_t terminalType = state.terminals[arenaIdx];

					if (UsingCudaRewardTerminalFastPath()) {
						if (config.saveRewards && arenaIdx < saveRewardsArenaLimit) {
							for (auto& weighted : rewards[arenaIdx])
								weighted.reward->PreStep(gs);

							if (state.lastRewards[arenaIdx].empty())
								state.lastRewards[arenaIdx].resize(rewards[arenaIdx].size());

							for (int rewardIdx = 0; rewardIdx < rewards[arenaIdx].size(); rewardIdx++) {
								auto& weightedReward = rewards[arenaIdx][rewardIdx];
								FList output = weightedReward.reward->GetAllRewards(gs, terminalType);

								int playerSampleIndex;
								if (config.shuffleRewardSampling) {
									playerSampleIndex = RocketSim::Math::RandInt(0, output.size());
								} else {
									playerSampleIndex = 0;
									int lowestID = gs.players[0].carId;
									for (int i = 1; i < gs.players.size(); i++) {
										auto id = gs.players[i].carId;
										if (id < lowestID) {
											lowestID = id;
											playerSampleIndex = i;
										}
									}
								}

								float rewardToSave = output[playerSampleIndex];
								if (ZeroSumReward* zeroSum = dynamic_cast<ZeroSumReward*>(weightedReward.reward))
									rewardToSave = zeroSum->_lastRewards[playerSampleIndex];
								state.lastRewards[arenaIdx][rewardIdx] = rewardToSave;
							}
						}
					} else {
						for (auto& weighted : rewards[arenaIdx])
							weighted.reward->PreStep(gs);

						FList allRewards = FList(gs.players.size(), 0);
						for (int rewardIdx = 0; rewardIdx < rewards[arenaIdx].size(); rewardIdx++) {
							auto& weightedReward = rewards[arenaIdx][rewardIdx];
							FList output = weightedReward.reward->GetAllRewards(gs, terminalType);
							for (int i = 0; i < gs.players.size(); i++)
								allRewards[i] += output[i] * weightedReward.weight;

							if (config.saveRewards) {
								int playerSampleIndex;
								if (config.shuffleRewardSampling) {
									playerSampleIndex = RocketSim::Math::RandInt(0, output.size());
								} else {
									playerSampleIndex = 0;
									int lowestID = gs.players[0].carId;
									for (int i = 1; i < gs.players.size(); i++) {
										auto id = gs.players[i].carId;
										if (id < lowestID) {
											lowestID = id;
											playerSampleIndex = i;
										}
									}
								}

								float rewardToSave = output[playerSampleIndex];
								if (ZeroSumReward* zeroSum = dynamic_cast<ZeroSumReward*>(weightedReward.reward))
									rewardToSave = zeroSum->_lastRewards[playerSampleIndex];

								if (state.lastRewards[arenaIdx].empty())
									state.lastRewards[arenaIdx].resize(rewards[arenaIdx].size());

								state.lastRewards[arenaIdx][rewardIdx] = rewardToSave;
							}
						}

						for (int i = 0; i < gs.players.size(); i++)
							state.rewards[playerStartIdx + i] = allRewards[i];
					}
				}
			}

			if (UsingCudaTrainingFastPath())
				RefreshCudaTrainingBuffers();
		}

		return;
	}
#endif

	auto fnStepArenas = [&](int arenaIdx) {

		Arena* arena = arenas[arenaIdx];
		auto& gs = state.gameStates[arenaIdx];
		int playerStartIdx = state.arenaPlayerStartIdx[arenaIdx];

		// Parse and set continuous actions
		auto actions = std::vector<Action>(gs.players.size());
		auto carItr = arena->_cars.begin();
		for (int i = 0; i < gs.players.size(); i++, carItr++) {
			auto& player = gs.players[i];
			Car* car = *carItr;

			const float* actionPtr = &continuousActions[(playerStartIdx + i) * actionDim];

			Action action;
			if (arenaIdx < continuousActionParsers.size() && continuousActionParsers[arenaIdx]) {
				action = continuousActionParsers[arenaIdx]->ParseContinuousAction(actionPtr, actionDim, player, gs);
			} else {
				// Fallback: direct mapping (throttle, steer, pitch, yaw, roll, jump, boost, handbrake)
				action.throttle = actionPtr[0];
				action.steer    = actionPtr[1];
				action.pitch    = actionPtr[2];
				action.yaw      = actionPtr[3];
				action.roll     = actionPtr[4];
				action.jump      = actionPtr[5] > 0.0f ? 1.0f : 0.0f;
				action.boost     = actionPtr[6] > 0.0f ? 1.0f : 0.0f;
				action.handbrake = actionPtr[7] > 0.0f ? 1.0f : 0.0f;
			}

			car->controls = (CarControls)action;
			actions[i] = action;
		}

		// Step arena with new actions we got from observing the last state
		// Update the gamestate after
		{
			arena->Step(config.tickSkip - config.actionDelay);

			if (eventTrackers[arenaIdx])
				eventTrackers[arenaIdx]->Update(arena);

			GameState* gsPrev = &state.prevGameStates[arenaIdx];
			if (gsPrev->IsEmpty())
				gsPrev = NULL;

			gs.UpdateFromArena(arena, actions, gsPrev);
		}

		// Update terminal
		uint8_t terminalType = TerminalType::NOT_TERMINAL;
		{
			for (auto cond : terminalConditions[arenaIdx]) {
				if (cond->IsTerminal(gs)) {
					bool isTrunc = cond->IsTruncation();
					uint8_t curTerminalType = isTrunc ? TerminalType::TRUNCATED : TerminalType::NORMAL;
					if (terminalType == TerminalType::NOT_TERMINAL) {
						terminalType = curTerminalType;
					} else {
						if (curTerminalType == TerminalType::NORMAL)
							terminalType = curTerminalType;
					}
				}
			}
			state.terminals[arenaIdx] = terminalType;
		}

		// Update observations
		{
			for (int i = 0; i < gs.players.size(); i++)
				state.obs.Set(playerStartIdx + i, obsBuilders[arenaIdx]->BuildObs(gs.players[i], gs));
		}

		// Update action masks (still useful for discrete fallback, or just skip for continuous)
		{
			for (int i = 0; i < gs.players.size(); i++)
				state.actionMasks.Set(playerStartIdx + i, actionParsers[arenaIdx]->GetActionMask(gs.players[i], gs));
		}
	};

	const char* serialCpuEnv = getenv("GGL_SERIAL_CPU_ENV");
	if (serialCpuEnv && serialCpuEnv[0] != '0') {
		for (int arenaIdx = 0; arenaIdx < arenas.size(); arenaIdx++)
			fnStepArenas(arenaIdx);
	} else {
		g_ThreadPool.StartBatchedJobs(fnStepArenas, arenas.size(), async);
	}
	if (async && !(serialCpuEnv && serialCpuEnv[0] != '0'))
		g_ThreadPool.WaitUntilDone();

	// Reward objects can share process-global state across arenas
	// (e.g. stateful reward internals), so reward evaluation must
	// stay on a single thread. The CUDA path already follows this rule; the
	// CPU continuous path needs the same protection.
	for (int arenaIdx = 0; arenaIdx < arenas.size(); arenaIdx++) {
		auto& gs = state.gameStates[arenaIdx];
		int playerStartIdx = state.arenaPlayerStartIdx[arenaIdx];
		uint8_t terminalType = state.terminals[arenaIdx];

		for (auto& weighted : rewards[arenaIdx])
			weighted.reward->PreStep(gs);

		FList allRewards = FList(gs.players.size(), 0);
		for (int rewardIdx = 0; rewardIdx < rewards[arenaIdx].size(); rewardIdx++) {
			auto& weightedReward = rewards[arenaIdx][rewardIdx];
			FList output = weightedReward.reward->GetAllRewards(gs, terminalType);
			for (int i = 0; i < gs.players.size(); i++)
				allRewards[i] += output[i] * weightedReward.weight;

			if (config.saveRewards) {
				int playerSampleIndex;
				if (config.shuffleRewardSampling) {
					playerSampleIndex = RocketSim::Math::RandInt(0, output.size());
				} else {
					playerSampleIndex = 0;
					int lowestID = gs.players[0].carId;
					for (int i = 1; i < gs.players.size(); i++) {
						auto id = gs.players[i].carId;
						if (id < lowestID) {
							lowestID = id;
							playerSampleIndex = i;
						}
					}
				}
				float rewardToSave = output[playerSampleIndex];

				if (ZeroSumReward* zeroSum = dynamic_cast<ZeroSumReward*>(weightedReward.reward))
					rewardToSave = zeroSum->_lastRewards[playerSampleIndex];

				if (state.lastRewards[arenaIdx].empty())
					state.lastRewards[arenaIdx].resize(rewards[arenaIdx].size());

				state.lastRewards[arenaIdx][rewardIdx] = rewardToSave;
			}
		}

		for (int i = 0; i < gs.players.size(); i++)
			state.rewards[playerStartIdx + i] = allRewards[i];
	}
}

void RLGC::EnvSet::ResetArena(int index) {
	stateSetters[index]->ResetArena(arenas[index]);

	GameState newState;

#ifdef RG_ROCKETSIMCUDA_AVAILABLE
	if (UsingCudaBackend()) {
		auto* arena = arenas[index];
		arena->tickCount = 0;

		cudaBatch->ResetArena(index);
		for (int carIdx = 0; carIdx < cudaArenaCars[index].size(); carIdx++)
			cudaBatch->SetCarState(index, carIdx, ToCudaCarState(cudaArenaCars[index][carIdx]));
		cudaBatch->SetBallState(index, ToCudaBallState(arena->ball->GetState()));

		SyncShadowArenaFromCuda(this, index);
		newState.UpdateFromCudaBatch(*cudaBatch, index, std::vector<Action>(cudaArenaCars[index].size()), NULL);
		newState.lastArena = arena;
	} else
#endif
	{
		newState = GameState(arenas[index]);
	}

	newState.userInfo = userInfos[index];
	state.gameStates[index] = newState;

	// Update event tracker
	if (eventTrackers[index])
		eventTrackers[index]->ResetPersistentInfo();

	// Reset all the other stuff
	obsBuilders[index]->Reset(newState);
	for (auto& cond : terminalConditions[index])
		cond->Reset(newState);
	for (auto& weightedReward : rewards[index])
		weightedReward.reward->Reset(newState);

	int playerStartIdx = state.arenaPlayerStartIdx[index];
	bool buildObsAndMasksCpu = true;
#ifdef RG_ROCKETSIMCUDA_AVAILABLE
	buildObsAndMasksCpu = !UsingCudaTrainingFastPath();
#endif
	if (buildObsAndMasksCpu) {
		for (int i = 0; i < newState.players.size(); i++) {

			// Update obs
			auto obs = obsBuilders[index]->BuildObs(newState.players[i], newState);
			state.obs.Set(playerStartIdx + i, obs);

			// Update action mask
			auto actionMask = actionParsers[index]->GetActionMask(newState.players[i], newState);
			state.actionMasks.Set(playerStartIdx + i, actionMask);
		}
	}

	// Remove previous state
	state.prevGameStates[index].MakeEmpty();
}

void RLGC::EnvSet::Reset() {
	if (config.physicsBackend == EnvPhysicsBackend::ROCKETSIM_CPU) {
		for (int i = 0; i < arenas.size(); i++) {
			if (state.terminals[i])
				ResetArena(i);
		}
		std::fill(state.terminals.begin(), state.terminals.end(), 0);
	} else {
		for (int i = 0; i < arenas.size(); i++)
			if (state.terminals[i])
				g_ThreadPool.StartJobAsync(std::bind(&EnvSet::ResetArena, this, std::placeholders::_1), i);
		std::fill(state.terminals.begin(), state.terminals.end(), 0);
		g_ThreadPool.WaitUntilDone();
	}

#ifdef RG_ROCKETSIMCUDA_AVAILABLE
	if (UsingCudaTrainingFastPath())
		RefreshCudaTrainingBuffers();
#endif
}

#ifdef RG_ROCKETSIMCUDA_AVAILABLE
void RLGC::EnvSet::RefreshCudaTrainingBuffers() {
	if (!UsingCudaTrainingFastPath())
		return;

	cudaBatch->BuildAdvancedObsAndDefaultMasks();
	 cudaBatch->CopyBuiltAdvancedObs(state.obs.data.data());
	 cudaBatch->CopyBuiltDefaultActionMasks(state.actionMasks.data.data());
}

void RLGC::EnvSet::SyncCudaDevice() const {
	if (!UsingCudaBackend())
		return;
	cudaBatch->Synchronize();
}

void* RLGC::EnvSet::GetCudaObsDevicePtr() const {
	if (!UsingCudaTrainingFastPath())
		return NULL;
	return cudaBatch->GetBuiltAdvancedObsDevicePtr();
}

void* RLGC::EnvSet::GetCudaActionMasksDevicePtr() const {
	if (!UsingCudaTrainingFastPath())
		return NULL;
	return cudaBatch->GetBuiltDefaultActionMasksDevicePtr();
}

void* RLGC::EnvSet::GetCudaRewardsDevicePtr() const {
	if (!UsingCudaRewardTerminalFastPath())
		return NULL;
	return cudaBatch->GetBuiltRewardsDevicePtr();
}
#endif
