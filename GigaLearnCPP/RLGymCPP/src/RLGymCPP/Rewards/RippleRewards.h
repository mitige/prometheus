class RippleDemoReward : public Reward {
public:
    float demo_attacker;
    float demo_victim;

    RippleDemoReward(float demo_attacker = 5.f, float demo_victim = 1.f)
        : demo_attacker(demo_attacker), demo_victim(demo_victim) {}

    virtual float GetReward(const Player& player, const GameState& state, bool isFinal) override {
        if (!state.prev) return 0.f;

        const Player* prev = nullptr;
        for (const auto& p : state.prev->players) {
            if (p.carId == player.carId) {
                prev = &p;
                break;
            }
        }
        if (!prev) return 0.f;

        float reward = 0.f;

        if (player.isDemoed && !prev->isDemoed) {
            reward -= demo_victim;
        }

        if (player.eventState.demo && !prev->eventState.demo) {
            reward += demo_attacker;
        }

        return reward;
    }
};




class RipplesFlipResetFreestyleChain : public Reward {
public:
	// Dense approach reward
	float approach_weight = 1.0f;
	float min_height = 300.0f;
	float max_approach_dist = 500.0f;
	float min_roof_up = 0.6f;
	float min_approach_speed = 150.0f;

	// Reset rewards (escalating)
	float base_reset_reward = 10.0f;
	float chain_multiplier = 1.5f;
	float max_chain_multiplier = 5.0f;
	int   max_chain_count = 4;              // hard cap on resets per aerial
	uint64_t min_ticks_between_resets = 90;  // cooldown — ignore resets too close together

	// Air dribble / sustained control between resets
	float air_control_reward = 0.3f;
	float max_air_control_dist = 250.0f;

	// Flip usage reward
	float flip_shot_reward = 20.0f;
	float chain_flip_bonus = 10.0f;
	float goal_align_bonus = 15.0f;
	float max_ticks_to_use = 300.f;

	// Direction scaling
	bool use_direction_scaling = true;

	// State tracking
	struct PlayerState {
		int   chain_count = 0;
		bool  has_flip_available = false;
		uint64_t last_reset_tick = 0;
		float last_dist_to_ball = 99999.f;
		bool  was_near_ball = false;
	};
	std::unordered_map<uint32_t, PlayerState> player_states;

	std::string GetName() override { return "RipplesFlipResetFreestyleChain"; }

	void Reset(const GameState& initialState) override {
		player_states.clear();
	}

	bool HasFlip(const Player& p) {
		return !p.hasDoubleJumped && !p.hasFlipped;
	}

	bool HasJump(const Player& p) {
		return !p.hasJumped;
	}

	bool HasFlipOrJump(const Player& p) {
		return HasFlip(p) || HasJump(p);
	}

	float ChainScale(int chain_count) {
		float scale = std::pow(chain_multiplier, static_cast<float>(chain_count));
		return std::min(scale, max_chain_multiplier);
	}

	float GetReward(const Player& player, const GameState& state, bool isFinal) override {
		if (!state.lastArena) return 0.f;

		uint64_t currentTick = state.lastArena->tickCount;
		auto& ps = player_states[player.carId];
		float reward = 0.0f;

		float car_height = player.pos.z;
		Vec down = -player.rotMat.up;
		Vec ball_pos = state.ball.pos;
		Vec player_pos = player.pos;

		bool isBlue = (player.team == Team::BLUE);
		float targetGoalY = isBlue ? CommonValues::BACK_WALL_Y : -CommonValues::BACK_WALL_Y;

		float dist = (player_pos - ball_pos).Length();
		dist = std::max(0.0f, dist - CommonValues::BALL_RADIUS);
		float dt = state.deltaTime > 1e-6f ? state.deltaTime : (1.0f / 120.0f);
		float approach_speed = (ps.last_dist_to_ball - dist) / dt;

		// =================================================================
		// LANDED = FULL CHAIN RESET
		// =================================================================
		if (player.isOnGround && car_height < min_height) {
			ps.chain_count = 0;
			ps.has_flip_available = false;
			ps.was_near_ball = false;
			ps.last_dist_to_ball = dist;
			return 0.0f;
		}

		if (car_height < min_height) {
			ps.last_dist_to_ball = dist;
			return 0.0f;
		}

		bool below_ball = player_pos.z < ball_pos.z;

		// Direction scaling
		float direction_scale = 1.0f;
		if (use_direction_scaling) {
			Vec opp_goal = (player.team == Team::BLUE)
				? CommonValues::ORANGE_GOAL_CENTER
				: CommonValues::BLUE_GOAL_CENTER;
			Vec own_goal = (player.team == Team::BLUE)
				? CommonValues::BLUE_GOAL_CENTER
				: CommonValues::ORANGE_GOAL_CENTER;

			Vec ball_to_opp = (opp_goal - ball_pos).Normalized();
			Vec ball_to_own = (own_goal - ball_pos).Normalized();
			Vec player_vel_norm = player.vel.Normalized();

			float opp_dot = ball_to_opp.Dot(player_vel_norm);
			float own_dot = ball_to_own.Dot(player_vel_norm);
			direction_scale = 0.5f + 0.25f * (opp_dot - own_dot);
			direction_scale = std::max(0.2f, direction_scale);
		}

		// =================================================================
		// AIR CONTROL REWARD (between resets)
		// =================================================================
		if (car_height > min_height && ball_pos.z > min_height && dist < max_air_control_dist) {
			float proximity = 1.0f - (dist / max_air_control_dist);
			float height_bonus = std::min((car_height - min_height) / 500.0f, 1.0f);
			float chain_bonus = 1.0f + 0.3f * ps.chain_count;
			reward += air_control_reward * proximity * height_bonus * chain_bonus * direction_scale;
			ps.was_near_ball = true;
		} else {
			ps.was_near_ball = false;
		}

		// =================================================================
		// CONTINUOUS APPROACH
		// =================================================================
		if (!ps.has_flip_available &&
			below_ball &&
			ball_pos.z > min_height &&
			dist < max_approach_dist &&
			down.z > min_roof_up &&
			approach_speed >= min_approach_speed) {

			Vec player_to_ball = (ball_pos - player_pos).Normalized();
			float cossim = player_to_ball.Dot(down);
			float dist_factor = 1.0f - (dist / max_approach_dist);
			float speed_factor = std::min(approach_speed / (min_approach_speed * 3.0f), 1.5f);
			float chain_approach = 1.0f + 0.2f * ps.chain_count;

			float approach_reward = approach_weight * cossim * dist_factor * direction_scale * speed_factor * chain_approach;
			reward += std::max(0.0f, approach_reward);
		}

		// =================================================================
		// RESET DETECTION (capped at 4, 90-tick cooldown)
		// =================================================================
		if (state.prev && !ps.has_flip_available && ps.chain_count < max_chain_count) {
			const Player* prev = nullptr;
			for (const auto& p : state.prev->players) {
				if (p.carId == player.carId) {
					prev = &p;
					break;
				}
			}

			if (prev) {
				bool was_in_air = !prev->isOnGround;
				bool still_in_air = !player.isOnGround;
				bool got_flip_back = (HasFlip(player) && !HasFlip(*prev)) ||
				                     (HasJump(player) && !HasJump(*prev));

				// Cooldown check (skip on first reset where last_reset_tick == 0)
				bool cooldown_ok = (ps.last_reset_tick == 0) ||
				                   (currentTick - ps.last_reset_tick >= min_ticks_between_resets);

				if (was_in_air && still_in_air && got_flip_back && cooldown_ok) {
					if (ball_pos.z > min_height &&
						car_height > min_height &&
						dist < 200.0f &&
						down.z > 0.5f) {

						float scale = ChainScale(ps.chain_count);
						reward += base_reset_reward * scale * direction_scale;

						ps.chain_count++;
						ps.has_flip_available = true;
						ps.last_reset_tick = currentTick;
					}
				}
			}
		}

		// =================================================================
		// FLIP SHOT / FLIP USAGE
		// =================================================================
		if (ps.has_flip_available && state.prev) {
			const Player* prev = nullptr;
			for (const auto& p : state.prev->players) {
				if (p.carId == player.carId) {
					prev = &p;
					break;
				}
			}

			if (prev) {
				bool used_flip = HasFlipOrJump(*prev) && !HasFlipOrJump(player);
				bool touched_ball = player.ballTouchedStep;
				bool in_air = !player.isOnGround;
				uint64_t ticks_since_reset = currentTick - ps.last_reset_tick;

				if (used_flip && in_air && ticks_since_reset <= max_ticks_to_use) {
					if (touched_ball) {
						Vec goalCenter(0.f, targetGoalY, CommonValues::GOAL_HEIGHT / 2.f);
						Vec ballToGoal = (goalCenter - ball_pos).Normalized();
						float goalAlign = std::max(0.f, state.ball.vel.Normalized().Dot(ballToGoal));

						float chain_bonus_val = chain_flip_bonus * static_cast<float>(ps.chain_count);
						reward += flip_shot_reward + chain_bonus_val + goal_align_bonus * goalAlign;
					}
					ps.has_flip_available = false;
				}
				else if (ticks_since_reset > max_ticks_to_use) {
					ps.has_flip_available = false;
				}
			}
		}

		ps.last_dist_to_ball = dist;
		return reward;
	}
};



class RipplesFlipResetRewardSpeedV2 : public Reward {
	public:
		// Dense approach reward (only fires when NO reset yet)
		float approach_weight = 1.0f;
		float min_height = 300.0f;
		float max_approach_dist = 500.0f;
		float min_roof_up = 0.7f;
		float min_approach_speed = 200.0f;

		// Event reward
		float reset_reward = 10.0f;
		bool  use_direction_scaling = true;

		// Flip shot reward
		float flip_shot_reward = 30.0f;
		float goal_align_bonus = 15.0f;
		float max_ticks_to_use = 300.f;

		// State tracking
		struct PlayerState {
			bool has_flipreset = false;
			uint64_t reset_tick = 0;
			float last_dist_to_ball = 99999.f;
		};
		std::unordered_map<uint32_t, PlayerState> player_states;

		std::string GetName() override { return "RipplesFlipResetRewardSpeedV2"; }

		void Reset(const GameState& initialState) override {
			player_states.clear();
		}

		bool HasFlip(const Player& p) {
			return !p.hasDoubleJumped && !p.hasFlipped;
		}

		bool HasJump(const Player& p) {
			return !p.hasJumped;
		}

		bool HasFlipOrJump(const Player& p) {
			return HasFlip(p) || HasJump(p);
		}

		float GetReward(const Player& player, const GameState& state, bool isFinal) override {
			if (!state.lastArena) return 0.f;

			uint64_t currentTick = state.lastArena->tickCount;
			auto& ps = player_states[player.carId];
			float reward = 0.0f;

			float car_height = player.pos.z;
			Vec down = -player.rotMat.up;
			Vec ball_pos = state.ball.pos;
			Vec player_pos = player.pos;

			bool isBlue = (player.team == Team::BLUE);
			float targetGoalY = isBlue ? CommonValues::BACK_WALL_Y : -CommonValues::BACK_WALL_Y;

			float dist = (player_pos - ball_pos).Length();
			dist = std::max(0.0f, dist - CommonValues::BALL_RADIUS);
			float dt = state.deltaTime > 1e-6f ? state.deltaTime : (1.0f / 120.0f);
			float approach_speed = (ps.last_dist_to_ball - dist) / dt;

			// =================================================================
			// LANDED = RESET CYCLE
			// =================================================================
			if (player.isOnGround && car_height < min_height) {
				ps.has_flipreset = false;
				ps.last_dist_to_ball = dist;
				return 0.0f;
			}

			// Must be in air above min height
			if (car_height < min_height) {
				ps.last_dist_to_ball = dist;
				return 0.0f;
			}

			// Must be below ball for approach/reset
			bool below_ball = player_pos.z < ball_pos.z;

			// Direction scaling
			float direction_scale = 1.0f;
			if (use_direction_scaling) {
				Vec opp_goal = (player.team == Team::BLUE)
					? CommonValues::ORANGE_GOAL_CENTER
					: CommonValues::BLUE_GOAL_CENTER;
				Vec own_goal = (player.team == Team::BLUE)
					? CommonValues::BLUE_GOAL_CENTER
					: CommonValues::ORANGE_GOAL_CENTER;

				Vec ball_to_opp = (opp_goal - ball_pos).Normalized();
				Vec ball_to_own = (own_goal - ball_pos).Normalized();
				Vec player_vel_norm = player.vel.Normalized();

				float opp_dot = ball_to_opp.Dot(player_vel_norm);
				float own_dot = ball_to_own.Dot(player_vel_norm);
				direction_scale = 0.5f + 0.25f * (opp_dot - own_dot);
				direction_scale = std::max(0.2f, direction_scale);
			}

			// =================================================================
			// CONTINUOUS APPROACH (ONLY IF NO RESET YET)
			// =================================================================
			if (!ps.has_flipreset &&
				below_ball &&
				ball_pos.z > min_height &&
				dist < max_approach_dist &&
				down.z > min_roof_up &&
				approach_speed >= min_approach_speed) {

				Vec player_to_ball = (ball_pos - player_pos).Normalized();
				float cossim = player_to_ball.Dot(down);
				float dist_factor = 1.0f - (dist / max_approach_dist);
				float speed_factor = std::min(approach_speed / (min_approach_speed * 3.0f), 1.5f);

				float approach_reward = approach_weight * cossim * dist_factor * direction_scale * speed_factor;
				reward += std::max(0.0f, approach_reward);
			}

			// =================================================================
			// RESET DETECTION (ONLY IF NO RESET YET)
			// =================================================================
			if (state.prev && !ps.has_flipreset) {
				const Player* prev = nullptr;
				for (const auto& p : state.prev->players) {
					if (p.carId == player.carId) {
						prev = &p;
						break;
					}
				}

				if (prev) {
					bool was_in_air = !prev->isOnGround;
					bool still_in_air = !player.isOnGround;
					bool got_flip_back = (HasFlip(player) && !HasFlip(*prev)) ||
					                     (HasJump(player) && !HasJump(*prev));

					if (was_in_air && still_in_air && got_flip_back) {
						if (ball_pos.z > min_height &&
							car_height > min_height &&
							dist < 200.0f &&
							down.z > 0.5f) {

							reward += reset_reward * direction_scale;
							ps.has_flipreset = true;
							ps.reset_tick = currentTick;
						}
					}
				}
			}

			// =================================================================
			// FLIP SHOT (ONLY IF HAS RESET)
			// =================================================================
			if (ps.has_flipreset && state.prev) {
				const Player* prev = nullptr;
				for (const auto& p : state.prev->players) {
					if (p.carId == player.carId) {
						prev = &p;
						break;
					}
				}

				if (prev) {
					bool used_flip = HasFlipOrJump(*prev) && !HasFlipOrJump(player);
					bool touched_ball = player.ballTouchedStep;
					bool in_air = !player.isOnGround;
					uint64_t ticks_since_reset = currentTick - ps.reset_tick;

					if (used_flip && touched_ball && in_air && ticks_since_reset <= max_ticks_to_use) {
						Vec goalCenter(0.f, targetGoalY, CommonValues::GOAL_HEIGHT / 2.f);
						Vec ballToGoal = (goalCenter - ball_pos).Normalized();
						float goalAlign = std::max(0.f, state.ball.vel.Normalized().Dot(ballToGoal));

						reward += flip_shot_reward + goal_align_bonus * goalAlign;
						ps.has_flipreset = false;
					}
					// Timeout
					else if (ticks_since_reset > max_ticks_to_use) {
						ps.has_flipreset = false;
					}
				}
			}

			ps.last_dist_to_ball = dist;
			return reward;
		}
	};
