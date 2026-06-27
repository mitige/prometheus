#include "GameState.h"

#include "../Math.h"

using namespace RLGC;

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

	BallState ToRsBallState(const rsc::BallState& src) {
		BallState dst;
		dst.pos = ToRsVec(src.pos);
		dst.rotMat = ToRsRot(src.rotMat);
		dst.vel = ToRsVec(src.vel);
		dst.angVel = ToRsVec(src.angVel);
		return dst;
	}
}
#endif

static int boostPadIndexMap[CommonValues::BOOST_LOCATIONS_AMOUNT] = {};
static bool boostPadIndexMapBuilt = false;
static std::mutex boostPadIndexMapMutex = {};

#ifdef RG_ROCKETSIMCUDA_AVAILABLE
static int boostPadCudaIndexMap[CommonValues::BOOST_LOCATIONS_AMOUNT] = {};
static bool boostPadCudaIndexMapBuilt = false;

static constexpr Vec CUDA_BOOST_PAD_LOCATIONS[CommonValues::BOOST_LOCATIONS_AMOUNT] = {
	{0.f, -4240.f, 70.f},
	{-1792.f, -4184.f, 70.f},
	{1792.f, -4184.f, 70.f},
	{-940.f, -3308.f, 70.f},
	{940.f, -3308.f, 70.f},
	{0.f, -2816.f, 70.f},
	{-3584.f, -2484.f, 70.f},
	{3584.f, -2484.f, 70.f},
	{-1788.f, -2300.f, 70.f},
	{1788.f, -2300.f, 70.f},
	{-2048.f, -1036.f, 70.f},
	{0.f, -1024.f, 70.f},
	{2048.f, -1036.f, 70.f},
	{-1024.f, 0.f, 70.f},
	{1024.f, 0.f, 70.f},
	{-2048.f, 1036.f, 70.f},
	{0.f, 1024.f, 70.f},
	{2048.f, 1036.f, 70.f},
	{-1788.f, 2300.f, 70.f},
	{1788.f, 2300.f, 70.f},
	{-3584.f, 2484.f, 70.f},
	{3584.f, 2484.f, 70.f},
	{0.f, 2816.f, 70.f},
	{-940.f, 3308.f, 70.f},
	{940.f, 3308.f, 70.f},
	{-1792.f, 4184.f, 70.f},
	{1792.f, 4184.f, 70.f},
	{0.f, 4240.f, 70.f},
	{-3584.f, 0.f, 73.f},
	{3584.f, 0.f, 73.f},
	{-3072.f, 4096.f, 73.f},
	{3072.f, 4096.f, 73.f},
	{-3072.f, -4096.f, 73.f},
	{3072.f, -4096.f, 73.f},
};

void _BuildCudaBoostPadIndexMap() {
	constexpr const char* ERROR_PREFIX = "_BuildCudaBoostPadIndexMap(): ";
#ifdef RG_VERBOSE
	RG_LOG("Building CUDA boost pad index map...");
#endif

	bool found[CommonValues::BOOST_LOCATIONS_AMOUNT] = {};
	for (int i = 0; i < CommonValues::BOOST_LOCATIONS_AMOUNT; i++) {
		Vec targetPos = CommonValues::BOOST_LOCATIONS[i];
		for (int j = 0; j < CommonValues::BOOST_LOCATIONS_AMOUNT; j++) {
			Vec padPos = CUDA_BOOST_PAD_LOCATIONS[j];

			if (padPos.DistSq2D(targetPos) < 10) {
				if (!found[i]) {
					found[i] = true;
					boostPadCudaIndexMap[i] = j;
				} else {
					RG_ERR_CLOSE(
						ERROR_PREFIX << "Matched duplicate CUDA boost pad at " << targetPos << "=" << padPos
					);
				}
				break;
			}
		}

		if (!found[i])
			RG_ERR_CLOSE(ERROR_PREFIX << "Failed to find matching CUDA pad at " << targetPos);
	}

#ifdef RG_VERBOSE
	RG_LOG(" > Done");
#endif
	boostPadCudaIndexMapBuilt = true;
}
#endif

void _BuildBoostPadIndexMap(Arena* arena) {
	constexpr const char* ERROR_PREFIX = "_BuildBoostPadIndexMap(): ";
#ifdef RG_VERBOSE
	RG_LOG("Building boost pad index map...");
#endif

	if (arena->_boostPads.size() != CommonValues::BOOST_LOCATIONS_AMOUNT) {
		RG_ERR_CLOSE(
			ERROR_PREFIX << "Arena boost pad count does not match CommonValues::BOOST_LOCATIONS_AMOUNT " <<
			"(" << arena->_boostPads.size() << "/" << CommonValues::BOOST_LOCATIONS_AMOUNT << ")"
		);
	}
	
	bool found[CommonValues::BOOST_LOCATIONS_AMOUNT] = {};
	for (int i = 0; i < CommonValues::BOOST_LOCATIONS_AMOUNT; i++) {
		Vec targetPos = CommonValues::BOOST_LOCATIONS[i];
		for (int j = 0; j < arena->_boostPads.size(); j++) {
			Vec padPos = arena->_boostPads[j]->config.pos;

			if (padPos.DistSq2D(targetPos) < 10) {
				if (!found[i]) {
					found[i] = true;
					boostPadIndexMap[i] = j;
				} else {
					RG_ERR_CLOSE(
						ERROR_PREFIX << "Matched duplicate boost pad at " << targetPos << "=" << padPos
					);
				}
				break;
			}
		}

		if (!found[i])
			RS_ERR_CLOSE(ERROR_PREFIX << "Failed to find matching pad at " << targetPos);
	}

#ifdef RG_VERBOSE
	RG_LOG(" > Done");
#endif
	boostPadIndexMapBuilt = true;
}

void RLGC::GameState::ResetBeforeStep() {
	for (auto& player : players)
		player.ResetBeforeStep();
}

void RLGC::GameState::UpdateFromArena(Arena* arena, const std::vector<Action>& actions, GameState* prev) {
	this->prev = prev;
	if (prev)
		prev->prev = NULL;

	lastArena = arena;
	int tickSkip = RS_MAX(arena->tickCount - lastTickCount, 0);
	deltaTime = tickSkip * (1 / 120.f);

	ball = arena->ball->GetState();

	players.resize(arena->_cars.size());

	auto carItr = arena->_cars.begin();
	for (int i = 0; i < players.size(); i++) {
		auto& player = players[i];
		player.index = i;
		player.UpdateFromCar(*carItr, arena->tickCount, tickSkip, actions[i], prev ? &prev->players[i] : NULL);
		if (player.ballTouchedStep)
			lastTouchCarID = player.carId;

		carItr++;
	}

	if (!boostPadIndexMapBuilt) {
		boostPadIndexMapMutex.lock();
		// Check again? This seems stupid but also makes sense to me
		//	Without this, we could lock as the index map is building, then go build again
		//	I would like to keep the mutex inside the if statement so it is only checked a few times
		if (!boostPadIndexMapBuilt) 
			_BuildBoostPadIndexMap(arena);
		boostPadIndexMapMutex.unlock();
	}

	int numBoostPads = arena->_boostPads.size();
	boostPads.resize(numBoostPads);
	boostPadsInv.resize(numBoostPads);
	boostPadTimers.resize(numBoostPads);
	boostPadTimersInv.resize(numBoostPads);
	for (int i = 0; i < arena->_boostPads.size(); i++) {
		int idx = boostPadIndexMap[i];
		int invIdx = boostPadIndexMap[CommonValues::BOOST_LOCATIONS_AMOUNT - i - 1];

		auto state = arena->_boostPads[idx]->GetState();
		auto stateInv = arena->_boostPads[invIdx]->GetState();

		boostPads[i] = state.isActive;
		boostPadsInv[i] = stateInv.isActive;

		boostPadTimers[i] = state.cooldown;
		boostPadTimersInv[i] = stateInv.cooldown;
	}

	// Update goal scoring
	// If you don't have a GoalScoreCondition then that's not my problem lmao
	goalScored = arena->IsBallScored();

	lastTickCount = arena->tickCount;
}

#ifdef RG_ROCKETSIMCUDA_AVAILABLE
void RLGC::GameState::UpdateFromCudaBatch(const rsc::RocketSimCudaBatch& batch, int arenaIdx, const std::vector<Action>& actions, GameState* prev) {
	std::vector<rsc::CarState> carStates(batch.GetMaxCarsPerArena());
	std::vector<rsc::BoostPadState> padStates(rsc::NUM_BOOST_PADS);
	rsc::ArenaInfo arenaInfo = batch.GetArenaInfo(arenaIdx);
	rsc::BallState ballState = batch.GetBallState(arenaIdx);
	for (int i = 0; i < arenaInfo.numCars; i++)
		carStates[i] = batch.GetCarState(arenaIdx, i);
	batch.GetBoostPadStates(arenaIdx, padStates.data());

	UpdateFromCudaSnapshot(arenaInfo, ballState, carStates.data(), padStates.data(), actions, prev);
}

void RLGC::GameState::UpdateFromCudaSnapshot(const rsc::ArenaInfo& arenaInfo, const rsc::BallState& ballState, const rsc::CarState* carStates, const rsc::BoostPadState* padStates, const std::vector<Action>& actions, GameState* prev) {
	this->prev = prev;
	if (prev)
		prev->prev = NULL;

	lastArena = NULL;

	int tickSkip = RS_MAX((int)(arenaInfo.tickCount - lastTickCount), 0);
	deltaTime = tickSkip * (1 / 120.f);

	ball = ToRsBallState(ballState);

	players.resize(arenaInfo.numCars);
	for (int i = 0; i < arenaInfo.numCars; i++) {
		auto& player = players[i];
		player.index = i;
		Player* prevPlayer = (prev && i < prev->players.size()) ? &prev->players[i] : NULL;
		Action prevAction = (i < actions.size()) ? actions[i] : Action();
		player.UpdateFromCudaCarState(carStates[i], arenaInfo.tickCount, tickSkip, prevAction, prevPlayer);
		if (player.ballTouchedStep)
			lastTouchCarID = player.carId;
	}

	RG_ASSERT(rsc::NUM_BOOST_PADS == CommonValues::BOOST_LOCATIONS_AMOUNT);

	if (!boostPadCudaIndexMapBuilt) {
		boostPadIndexMapMutex.lock();
		if (!boostPadCudaIndexMapBuilt)
			_BuildCudaBoostPadIndexMap();
		boostPadIndexMapMutex.unlock();
	}

	boostPads.resize(CommonValues::BOOST_LOCATIONS_AMOUNT);
	boostPadsInv.resize(CommonValues::BOOST_LOCATIONS_AMOUNT);
	boostPadTimers.resize(CommonValues::BOOST_LOCATIONS_AMOUNT);
	boostPadTimersInv.resize(CommonValues::BOOST_LOCATIONS_AMOUNT);
	for (int i = 0; i < CommonValues::BOOST_LOCATIONS_AMOUNT; i++) {
		int idx = boostPadCudaIndexMap[i];
		int invIdx = boostPadCudaIndexMap[CommonValues::BOOST_LOCATIONS_AMOUNT - i - 1];
		boostPads[i] = padStates[idx].isActive;
		boostPadsInv[i] = padStates[invIdx].isActive;
		boostPadTimers[i] = padStates[idx].cooldown;
		boostPadTimersInv[i] = padStates[invIdx].cooldown;
	}

	goalScored = arenaInfo.goalScored;
	lastTickCount = arenaInfo.tickCount;
}
#endif
