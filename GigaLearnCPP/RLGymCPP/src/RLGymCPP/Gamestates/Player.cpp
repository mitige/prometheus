#include "Player.h"

#ifdef RG_ROCKETSIMCUDA_AVAILABLE
namespace {
	Vec ToRsVec(const rsc::Vec3& v) {
		return Vec(v.x, v.y, v.z);
	}

	RotMat ToRsRot(const rsc::RotMat& m) {
		return RotMat(
			ToRsVec(m.forward),
			ToRsVec(m.right),
			ToRsVec(m.up)
		);
	}

	CarControls ToRsControls(const rsc::CarControls& controls) {
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

	CarState ToRsCarState(const rsc::CarState& src) {
		CarState dst = {};
		dst.pos = ToRsVec(src.pos);
		dst.rotMat = ToRsRot(src.rotMat);
		dst.vel = ToRsVec(src.vel);
		dst.angVel = ToRsVec(src.angVel);
		dst.boost = src.boost;
		dst.timeSpentBoosting = src.timeSpentBoosting;
		dst.isOnGround = src.isOnGround;
		dst.hasJumped = src.hasJumped;
		dst.hasDoubleJumped = src.hasDoubleJumped;
		dst.hasFlipped = src.hasFlipped;
		dst.isFlipping = src.isFlipping;
		dst.isJumping = src.isJumping;
		dst.flipRelTorque = ToRsVec(src.flipRelTorque);
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
		dst.worldContact.hasContact = src.worldContactHasContact;
		dst.worldContact.contactNormal = ToRsVec(src.worldContactNormal);
		dst.carContact.otherCarID = src.carContactOtherID;
		dst.carContact.cooldownTimer = src.carContactCooldownTimer;
		dst.ballHitInfo.isValid = src.ballHitValid;
		dst.ballHitInfo.tickCountWhenHit = src.ballHitTickCount;
		dst.lastControls = ToRsControls(src.lastControls);
		return dst;
	}
}
#endif

namespace RLGC {
	void Player::ResetBeforeStep() {
		this->eventState = {};
	}

	void Player::UpdateFromCar(Car* car, uint64_t tickCount, int tickSkip, const Action& prevAction, Player* prev) {

		this->prev = prev;
		if (prev)
			prev->prev = NULL;

		carId = car->id;
		team = car->team;
		*(CarState*)this = car->GetState();

		if (ballHitInfo.isValid) {
			ballTouchedStep = ballHitInfo.tickCountWhenHit >= (tickCount - tickSkip);
			ballTouchedTick = ballHitInfo.tickCountWhenHit == (tickCount - 1);
		} else {
			ballTouchedStep = ballTouchedTick = false;
		}

		this->prevAction = prevAction;
	}

#ifdef RG_ROCKETSIMCUDA_AVAILABLE
	void Player::UpdateFromCudaCarState(const rsc::CarState& carState, uint64_t tickCount, int tickSkip, const Action& prevAction, Player* prev) {

		this->prev = prev;
		if (prev)
			prev->prev = NULL;

		carId = carState.id;
		team = (carState.team == rsc::Team::BLUE) ? Team::BLUE : Team::ORANGE;
		*(CarState*)this = ToRsCarState(carState);

		if (ballHitInfo.isValid) {
			ballTouchedStep = ballHitInfo.tickCountWhenHit >= (tickCount - tickSkip);
			ballTouchedTick = ballHitInfo.tickCountWhenHit == (tickCount - 1);
		} else {
			ballTouchedStep = ballTouchedTick = false;
		}

		this->prevAction = prevAction;
	}
#endif
}
