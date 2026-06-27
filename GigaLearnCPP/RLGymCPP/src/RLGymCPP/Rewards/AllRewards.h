// =============================================================================
// AllRewards.h  -  Combined reward library
// Sources:
//   CommonRewards(1).h       - primary reward collection
//   Monkey (1).cpp           - HoldAirRollWhileApproachingBallXYReward
//   New Text Document.txt.h  - additional rewards (see NOTE below)
//
// NOTE: New Text Document.txt.h has corrupted syntax (missing *, ->, ::, <>
//       template<> etc.). The unique class names from that file are listed
//       at the bottom in a comment block for reference.
// =============================================================================

#pragma once
#include "../Math.h"
#include "Reward.h"
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <optional>
#include <random>
#include <string>
#include <unordered_map>
#include <vector>

namespace RLGC {

using namespace CommonValues;

inline float distance2D(float x1, float z1, float x2, float z2) {
  float dx = x2 - x1;
  float dz = z2 - z1;
  return std::sqrt(dx * dx + dz * dz);
}

inline std::vector<float> linspace(float start_in, float end_in, int num_in) {
  std::vector<float> linspaced;

  if (num_in <= 0) {
    return linspaced;
  }
  if (num_in == 1) {
    linspaced.push_back(start_in);
    return linspaced;
  }

  float delta = (end_in - start_in) / (num_in - 1);
  for (int i = 0; i < num_in; ++i) {
    linspaced.push_back(start_in + delta * i);
  }
  return linspaced;
}

template <bool PlayerEventState::*VAR, bool NEGATIVE>
class PlayerDataEventReward : public Reward {
  virtual float GetReward(const Player &player, const GameState &state,
                          bool isFinal) {
    bool val = player.eventState.*VAR;

    if (NEGATIVE) {
      return -(float)val;
    } else {
      return (float)val;
    }
  }
};

typedef PlayerDataEventReward<&PlayerEventState::goal, false>
    PlayerGoalReward; // NOTE: Given only to the player who last touched the
                      // ball on the opposing team
typedef PlayerDataEventReward<&PlayerEventState::assist, false> AssistReward;
typedef PlayerDataEventReward<&PlayerEventState::shot, false> ShotReward;
typedef PlayerDataEventReward<&PlayerEventState::shotPass, false>
    ShotPassReward;
typedef PlayerDataEventReward<&PlayerEventState::save, false> SaveReward;
typedef PlayerDataEventReward<&PlayerEventState::bump, false> BumpReward;
typedef PlayerDataEventReward<&PlayerEventState::bumped, true> BumpedPenalty;
typedef PlayerDataEventReward<&PlayerEventState::demo, false> DemoReward;
typedef PlayerDataEventReward<&PlayerEventState::demoed, true> DemoedPenalty;

// Rewards a goal by anyone on the team
// NOTE: Already zero-sum
class GoalReward : public Reward {
public:
  float concedeScale;
  GoalReward(float concedeScale = -.8) : concedeScale(concedeScale) {}

  virtual float GetReward(const Player &player, const GameState &state,
                          bool isFinal) {
    if (!state.goalScored)
      return 0;

    bool scored = (player.team != RS_TEAM_FROM_Y(state.ball.pos.y));
    return scored ? 1 : concedeScale;
  }
};
class FlipResetReward : public Reward {
private:
  // Configurable reward weights
  float flipResetGainReward = 5.0f;
  float flipResetUsageReward = 10.0f;
  float positioningReward = 2.0f;
  float ballControlReward = 3.0f;
  float followUpReward = 4.0f;
  float wastedFlipPenalty = -2.0f;

public:
  FlipResetReward(float gainWeight = 5.0f, float usageWeight = 10.0f,
                  float posWeight = 2.0f, float controlWeight = 3.0f,
                  float followWeight = 4.0f, float wastePenalty = -2.0f)
      : flipResetGainReward(gainWeight), flipResetUsageReward(usageWeight),
        positioningReward(posWeight), ballControlReward(controlWeight),
        followUpReward(followWeight), wastedFlipPenalty(wastePenalty) {}

  virtual float GetReward(const Player &player, const GameState &state,
                          bool isFinal) override {
    if (!player.prev)
      return 0.0f;

    float totalReward = 0.0f;

    Vec ballPos = state.ball.pos;
    Vec playerPos = player.pos;
    Vec ballToCar = playerPos - ballPos;
    float distToBall = ballToCar.Length();

    bool hasFlipResetNow = player.HasFlipReset();
    bool hadFlipResetBefore = player.prev->HasFlipReset();
    bool gotFlipReset = player.GotFlipReset();

    // === 1. FLIP RESET GAIN REWARDS ===
    if (hasFlipResetNow && !hadFlipResetBefore) {
      // Base reward for gaining flip reset
      float gainReward = flipResetGainReward;

      // Bonus for gaining flip reset when close to ball
      if (distToBall < 300.0f) {
        float proximityBonus = (300.0f - distToBall) / 300.0f;
        gainReward += proximityBonus * 2.0f;
      }

      // Bonus for gaining flip reset at good height
      if (playerPos.z > 150.0f && playerPos.z < 600.0f) {
        gainReward += 1.0f;
      }

      totalReward += gainReward;
    }

    // === 2. FLIP RESET USAGE REWARDS ===
    if (hasFlipResetNow) {
      // Continuous reward for having flip reset and being airborne
      if (!player.isOnGround) {
        totalReward += 0.1f * flipResetUsageReward;

        // Major bonus for using flip reset with ball contact
        if (player.ballTouchedStep > 0) {
          float usageReward = 1.0f * flipResetUsageReward;

          // Extra bonus for powerful contact during flip
          if (player.isFlipping) {
            float ballSpeed = state.ball.vel.Length();
            float speedBonus =
                std::min(ballSpeed / CommonValues::BALL_MAX_SPEED, 1.0f);
            usageReward += speedBonus * 3.0f;
          }

          totalReward += usageReward;
        }

        // Bonus for using flip reset when ball is close
        if (distToBall < 200.0f && player.isFlipping) {
          totalReward += 0.6f * flipResetUsageReward;
        }
      }
    }

    // === 3. POSITIONING REWARDS ===
    if ((hasFlipResetNow || gotFlipReset) && !player.isOnGround) {
      float posReward = 0.0f;

      // Reward for being positioned below the ball
      if (playerPos.z < ballPos.z && ballPos.z > 200.0f) {
        float heightDiff = ballPos.z - playerPos.z;
        if (heightDiff > 50.0f && heightDiff < 200.0f) {
          posReward += 0.5f;
        }
      }

      // Reward for approaching ball from good angles
      Vec playerVel = player.vel;
      if (playerVel.Length() > 100.0f) {
        Vec dirToBall = -ballToCar.Normalized();
        float approachAlignment = dirToBall.Dot(playerVel.Normalized());
        if (approachAlignment > 0.5f) {
          posReward += approachAlignment * 0.4f;
        }
      }

      // Reward for proper orientation (car upright-ish)
      Vec upVector = Vec(0, 0, 1);
      float uprightness = std::abs(player.rotMat.up.Dot(upVector));
      posReward += uprightness * 0.2f;

      totalReward += posReward * positioningReward;
    }

    // === 4. BALL CONTROL REWARDS ===
    if (hasFlipResetNow || gotFlipReset) {
      float controlReward = 0.0f;

      // Reward for keeping ball close during flip reset sequence
      if (distToBall < 300.0f) {
        controlReward += (300.0f - distToBall) / 300.0f * 0.3f;
      }

      // Reward for controlled ball touches (not too powerful)
      if (player.ballTouchedStep > 0) {
        float ballSpeed = state.ball.vel.Length();
        // FIX: Use previous player state to estimate previous ball speed
        // Since prevBall doesn't exist, we'll use an alternative approach
        float prevBallSpeed = ballSpeed; // Default fallback
        if (player.prev && player.prev->ballTouchedStep == 0) {
          // If previous step had no ball contact, assume similar speed
          prevBallSpeed = ballSpeed * 0.9f; // Rough estimate
        }

        float speedChange = ballSpeed - prevBallSpeed;

        // Prefer controlled touches
        if (speedChange > 0 && speedChange < 1000.0f) {
          controlReward += 0.6f;
        } else if (speedChange > 1500.0f) {
          controlReward -= 0.3f; // Slight penalty for overly aggressive touches
        }

        // Bonus for touches that direct ball toward goal
        bool targetOrangeGoal = player.team == Team::BLUE;
        Vec targetGoal = targetOrangeGoal ? CommonValues::ORANGE_GOAL_BACK
                                          : CommonValues::BLUE_GOAL_BACK;
        Vec ballToGoal = (targetGoal - ballPos).Normalized();
        Vec ballVelDir = state.ball.vel.Normalized();
        float goalDirection = ballToGoal.Dot(ballVelDir);

        if (goalDirection > 0.3f) {
          controlReward += goalDirection * 0.5f;
        }
      }

      totalReward += controlReward * ballControlReward;
    }

    // === 5. FOLLOW-UP REWARDS ===
    // Apply when player recently used a flip reset
    if (!hasFlipResetNow && hadFlipResetBefore) {
      float followReward = 0.0f;

      // Reward for maintaining momentum after flip reset
      float currentSpeed = player.vel.Length();
      float prevSpeed = player.prev->vel.Length();

      if (currentSpeed > prevSpeed && currentSpeed > 1000.0f) {
        followReward += 0.4f;
      }

      // Reward for staying close to ball after flip reset
      if (distToBall < 250.0f) {
        followReward += 0.3f;
      }

      // Major bonus for scoring opportunities after flip reset
      if (player.ballTouchedStep > 0) {
        bool targetOrangeGoal = player.team == Team::BLUE;
        Vec targetGoal = targetOrangeGoal ? CommonValues::ORANGE_GOAL_BACK
                                          : CommonValues::BLUE_GOAL_BACK;
        Vec ballToGoal = (targetGoal - ballPos);
        float distBallToGoal = ballToGoal.Length();

        if (distBallToGoal < 2000.0f) {
          // FIX: ballVelToGoal should be float, not Vec
          float ballVelToGoal = state.ball.vel.Dot(ballToGoal.Normalized());
          if (ballVelToGoal > 0) {
            followReward += std::min(ballVelToGoal / 1000.0f, 1.0f);
          }
        }
      }

      totalReward += followReward * followUpReward;
    }

    // === 6. PENALTIES FOR BAD HABITS ===
    // Penalty for wasting flip reset (using it when far from ball)
    if (hadFlipResetBefore && !hasFlipResetNow && player.ballTouchedStep == 0) {
      if (distToBall > 400.0f) {
        totalReward += wastedFlipPenalty;
      }
    }

    // Penalty for getting flip reset but immediately landing
    if (hasFlipResetNow && player.isOnGround) {
      totalReward += wastedFlipPenalty * 0.3f;
    }

    // Penalty for having flip reset but moving away from ball
    if (hasFlipResetNow && !player.isOnGround) {
      Vec playerVel = player.vel;
      if (playerVel.Length() > 200.0f && distToBall < 500.0f) {
        Vec dirToBall = -ballToCar.Normalized();
        float movingAway = dirToBall.Dot(playerVel.Normalized());
        if (movingAway < -0.5f) {
          totalReward += wastedFlipPenalty * 0.2f;
        }
      }
    }

    return totalReward;
  }
};
class FlipPenaltyReward : public Reward {
public:
  float minSpeedForFlip;
  float minDistanceToBall;

  FlipPenaltyReward(float minSpeedForFlip = 800.0f,
                    float minDistanceToBall = 300.0f)
      : minSpeedForFlip(minSpeedForFlip), minDistanceToBall(minDistanceToBall) {
  }

  virtual float GetReward(const Player &player, const GameState &state,
                          bool isFinal) override {
    if (!player.prev)
      return 0;

    bool justFlipped = player.isFlipping && !player.prev->isFlipping;

    if (justFlipped) {
      float speed = player.vel.Length();
      float distToBall = (state.ball.pos - player.pos).Length();

      bool isGoodFlip =
          (speed > minSpeedForFlip) || (distToBall < minDistanceToBall);

      if (isGoodFlip) {
        return 0;
      } else {
        return -1.0f;
      }
    }

    return 0;
  }
};
class ControlReward : public Reward {
public:
  virtual float GetReward(const Player &player, const GameState &state,
                          bool isFinal) override {
    if (!player.ballTouchedStep || !player.prev) {
      return 0;
    }

    // Calculate distance to ball
    float dist = player.pos.Dist(state.ball.pos);

    // Reward is higher the closer the player is to the ball, max reward is 1.
    // The max distance is set based on a rough estimate of dribble distance.
    constexpr float MAX_DIST = 500.f;
    return exp(-0.5 * (dist / MAX_DIST) * (dist / MAX_DIST));
  }
};
class AirRollReward : public Reward {
public:
  virtual float GetReward(const Player &player, const GameState &state,
                          bool isFinal) override {
    if (!player.isOnGround && player.prevAction.roll != 0) {
      return 1.0f;
    }
    return 0;
  }
};
class BoostSeekingReward : public Reward {
public:
  virtual float GetReward(const Player &player, const GameState &state,
                          bool isFinal) override {
    if (player.boost >= 100)
      return 0; // Don't seek boost if full

    Vec closest_pad_pos;
    float closest_dist_sq = -1;

    auto &boost_pads = state.GetBoostPads(player.team == Team::ORANGE);
    for (int i = 0; i < CommonValues::BOOST_LOCATIONS_AMOUNT; ++i) {
      if (boost_pads[i]) { // If boost pad is active
        Vec pad_pos = CommonValues::BOOST_LOCATIONS[i];
        float dist_sq = player.pos.DistSq(pad_pos);

        if (closest_dist_sq < 0 || dist_sq < closest_dist_sq) {
          closest_dist_sq = dist_sq;
          closest_pad_pos = pad_pos;
        }
      }
    }

    if (closest_dist_sq < 0)
      return 0; // No active boost pads

    Vec dir_to_boost = (closest_pad_pos - player.pos).Normalized();
    return player.vel.Dot(dir_to_boost) / CommonValues::CAR_MAX_SPEED;
  }
};
class WaveDashReward2 : public Reward {
public:
  virtual float GetReward(const Player &player, const GameState &state,
                          bool isFinal) override {
    if (!player.prev)
      return 0;

    // Check for transition from flipping in air to being on ground
    if (player.isOnGround && player.prev->isFlipping &&
        !player.prev->isOnGround) {
      // Reward the speed increase from the wavedash
      float speed_increase = player.vel.Length() - player.prev->vel.Length();
      if (speed_increase > 100) { // Must provide a meaningful speed boost
        return 1.0f;
      }
    }
    return 0;
  }
};
class NoBoostPenalty : public Reward {
public:
  virtual float GetReward(const Player &player, const GameState &state,
                          bool isFinal) override {
    // We need to look at the state *before* the action was taken to know the
    // boost level.
    if (player.prev == NULL) {
      return 0;
    }

    // Condition: The bot tried to boost in the last action AND had 0 boost
    // before that action.
    bool triedToBoost = player.prevAction.boost > 0;
    bool hadNoBoost = player.prev->boost == 0;

    if (triedToBoost && hadNoBoost) {
      return -1.0f; // Apply penalty
    }

    return 0.0f;
  }
};
class HalfFlipReward2 : public Reward {
public:
  virtual float GetReward(const Player &player, const GameState &state,
                          bool isFinal) override {
    if (!player.isOnGround || !player.prev || !player.prev->prev)
      return 0;

    // Check if the player was recently flipping backwards.
    if (player.prev->hasFlipped && player.prev->prevAction.pitch > 0.5) {
      // Check if the car is now facing the opposite direction.
      float dot = player.rotMat.forward.Dot(player.prev->prev->rotMat.forward);
      if (dot < -0.8) { // Facing nearly the opposite direction
        return 1.0f;
      }
    }
    return 0;
  }
};
class AdvancedSpeedFlipReward : public Reward {
public:
  virtual float GetReward(const Player &player, const GameState &state,
                          bool isFinal) override {
    if (!player.prev || player.isSupersonic == player.prev->isSupersonic)
      return 0;

    // If we just became supersonic and our previous speed was low
    if (player.isSupersonic && player.prev->vel.Length() < 500) {
      return 1.0f;
    }

    return 0;
  }
};
class KickoffSpeedFlipRewardV2 : public Reward {
public:
  virtual float GetReward(const Player &player, const GameState &state,
                          bool isFinal) override {
    // Check if it's a kickoff scenario (ball is at center)
    if (state.ball.pos.Length2D() < 10) {
      if (player.ballTouchedStep) {
        // Reward is inversely proportional to the time it took to touch the
        // ball
        return 2.5f - state.ball.pos.z / (CommonValues::BALL_RADIUS * 2);
      }
    }
    return 0;
  }
};
class convertedFlipResetReward : public Reward {
private:
  // Internal state to track for each player
  struct PlayerState {
    bool last_has_flip_reset = false;
    bool reset_detected = false;
    bool setup_active = false;
  };

  std::map<uint32_t, PlayerState> player_states;

  // --- Configuration parameters ---
  float reset_reward;
  float setup_bonus;
  float proximity_bonus;
  float inverted_bonus;
  float min_air_height;
  float max_ball_distance;
  float inverted_threshold_deg; // Converted from radians for easier comparison
  bool debug;

  // --- Helper methods ---

  bool _is_in_air(const Player &player) const {
    return player.pos.z > min_air_height && !player.isOnGround;
  }

  bool _is_car_inverted(const Player &player) const {
    Angle car_angle = Angle::FromRotMat(player.rotMat);
    float roll_deg = abs(car_angle.roll * 180 / M_PI);
    return abs(roll_deg - 180) < inverted_threshold_deg;
  }

  bool _is_close_to_ball(const Player &player, const GameState &state) const {
    return player.pos.Dist(state.ball.pos) <= max_ball_distance;
  }

  float _calculate_setup_quality(const Player &player,
                                 const GameState &state) const {
    float quality = 0.0f;

    if (_is_in_air(player)) {
      quality += 0.3f;
    }

    if (_is_car_inverted(player)) {
      quality += 0.3f;
    }

    float distance = player.pos.Dist(state.ball.pos);
    if (distance <= max_ball_distance) {
      float proximity_score = 1.0f - (distance / max_ball_distance);
      quality += 0.4f * proximity_score;
    }

    return quality;
  }

public:
  convertedFlipResetReward(float reset_reward = 15.0f, float setup_bonus = 2.0f,
                           float proximity_bonus = 1.0f,
                           float inverted_bonus = 1.0f,
                           float min_air_height = 50.0f,
                           float max_ball_distance = 150.0f,
                           float inverted_threshold = 150.0f,
                           bool debug = false)
      : reset_reward(reset_reward), setup_bonus(setup_bonus),
        proximity_bonus(proximity_bonus), inverted_bonus(inverted_bonus),
        min_air_height(min_air_height), max_ball_distance(max_ball_distance),
        inverted_threshold_deg(inverted_threshold), debug(debug) {}

  virtual void Reset(const GameState &initialState) override {
    player_states.clear();
    for (const auto &player : initialState.players) {
      player_states[player.carId] = PlayerState();
    }
  }

  virtual float GetReward(const Player &player, const GameState &state,
                          bool isFinal) override {
    // Ensure player state exists
    if (player_states.find(player.carId) == player_states.end()) {
      player_states[player.carId] = PlayerState();
    }
    PlayerState &p_state = player_states[player.carId];

    float reward = 0.0f;

    bool current_has_flip_reset = player.HasFlipReset();
    bool flip_reset_detected_this_tick =
        current_has_flip_reset && !p_state.last_has_flip_reset;

    bool in_air = _is_in_air(player);
    bool inverted = _is_car_inverted(player);
    bool close_to_ball = _is_close_to_ball(player, state);

    if (in_air && player.ballTouchedStep && flip_reset_detected_this_tick) {
      if (!p_state.reset_detected) {
        reward += reset_reward;
        p_state.reset_detected = true;
        if (debug) {
          std::cout << "FLIP RESET DETECTED! Player " << player.carId
                    << " - Reward: " << reset_reward << std::endl;
        }
      }
    } else if (in_air && inverted && close_to_ball) {
      float setup_quality = _calculate_setup_quality(player, state);
      float setup_reward = setup_bonus * setup_quality;
      reward += setup_reward;

      if (debug && setup_reward > 0) {
        std::cout << "Player " << player.carId
                  << " setup bonus: " << setup_reward << std::endl;
      }

      if (!p_state.setup_active) {
        p_state.setup_active = true;
      }
    } else {
      if (p_state.setup_active) {
        p_state.setup_active = false;
      }
      if (p_state.reset_detected) {
        p_state.reset_detected = false;
      }
    }

    p_state.last_has_flip_reset = current_has_flip_reset;

    return reward;
  }

  virtual std::string GetName() override { return "FlipResetReward"; }
};
class KaiyoEnergyReward : public Reward {
public:
  const double GRAVITY = 650;
  const double MASS = 180;
  virtual float GetReward(const Player &player, const GameState &state,
                          bool isFinal) override {
    auto max_energy = (MASS * GRAVITY * (CEILING_Z - 17.)) +
                      (0.5 * MASS * (CAR_MAX_SPEED * CAR_MAX_SPEED));
    double energy = 0;

    if (player.HasFlipOrJump()) {
      energy += 0.35 * MASS * 292. * 292.;
    }

    if (player.HasFlipOrJump() && !player.isOnGround) {
      energy += 0.35 * MASS * 550. * 550.;
    }

    energy += MASS * GRAVITY * (player.pos.z - 17.) * 0.75;

    double velocity = player.vel.Length();
    energy += 0.5 * MASS * velocity * velocity;
    energy += 7.97e6 * player.boost;

    double norm_energy = player.isDemoed ? 0.0f : (energy / max_energy);

    return static_cast<float>(norm_energy);
  }
};
class KickoffFirstTouchReward : public Reward {
private:
  bool _kickoff_active;
  int _first_touch_player_id_this_tick;

public:
  KickoffFirstTouchReward()
      : _kickoff_active(false), _first_touch_player_id_this_tick(-1) {}
  void Reset(const GameState &initial_state) override {
    _kickoff_active = initial_state.ball.vel.LengthSq() < 1.f;
    _first_touch_player_id_this_tick = -1;
  }

  void PreStep(const GameState &state) override {
    _first_touch_player_id_this_tick = -1;

    if (!_kickoff_active)
      return;
    bool kickoff_ended = false;

    for (const auto &p : state.players) {
      if (p.ballTouchedStep) {
        _first_touch_player_id_this_tick = p.carId;
        kickoff_ended = true;
        break;
      }
    }

    if (!kickoff_ended && state.ball.vel.LengthSq() > 1.f) {
      kickoff_ended = true;
    }

    if (kickoff_ended) {
      _kickoff_active = false;
    }
  }

  float GetReward(const Player &player, const GameState &state,
                  bool isFinal) override {
    if (_first_touch_player_id_this_tick != -1) {
      if (player.carId == _first_touch_player_id_this_tick) {
        return 1.0f;
      } else {
        return -1.0f;
      }
    }

    return 0.0f;
  }
};
class CeilingShotReward : public Reward {
public:
  virtual float GetReward(const Player &player, const GameState &state,
                          bool isFinal) override {
    if (player.eventState.goal && player.prev) {
      // Crude check: was the player on the ceiling recently?
      // A proper implementation would need stateful tracking per player.
      const Player *p = player.prev;
      for (int i = 0; i < 120 * 3 && p != nullptr;
           ++i, p = p->prev) { // Check last 3 seconds
        if (p->worldContact.hasContact &&
            p->worldContact.contactNormal.z < -0.9f) {
          return 1.0f;
        }
      }
    }
    return 0;
  }
};
class GenericFlickReward : public Reward {
public:
  virtual float GetReward(const Player &player, const GameState &state,
                          bool isFinal) override {
    if (player.ballTouchedStep && player.isFlipping && player.prev &&
        player.prev->isOnGround) {
      float z_vel_change = state.ball.vel.z - state.prev->ball.vel.z;
      if (z_vel_change > 300) {
        return z_vel_change / (CommonValues::CAR_MAX_SPEED / 2);
      }
    }
    return 0;
  }
};
class MawkyzFlickReward : public GenericFlickReward {};
class EnhancedDiagonalFlickReward : public GenericFlickReward {};
class RefinedDemoReward : public Reward {
private:
  float max_demo_angle;
  float optimal_demo_angle;
  float base_reward;
  float flip_bonus;
  float max_reward;
  int flip_window_ticks;

  struct OpponentState {
    bool was_demoed = false;
  };
  struct PlayerFlipState {
    int last_flip_tick = -999;
    bool prev_has_flip = true;
    bool prev_jump_pressed = false;
  };

  std::unordered_map<uint32_t, OpponentState> opponent_states;
  std::unordered_map<uint32_t, PlayerFlipState> player_flip_states;
  int current_tick;

public:
  // Constructor Implementation
  RefinedDemoReward(float max_demo_angle_deg = 70.0f,
                    float optimal_demo_angle_deg = 45.0f,
                    float base_reward_ = 1.0f, float flip_bonus_ = 1.5f,
                    float max_reward_ = 3.0f, int flip_window_ticks_ = 12)
      : max_demo_angle(max_demo_angle_deg * (M_PI / 180.0f)),
        optimal_demo_angle(optimal_demo_angle_deg * (M_PI / 180.0f)),
        base_reward(base_reward_), flip_bonus(flip_bonus_),
        max_reward(max_reward_), flip_window_ticks(flip_window_ticks_),
        current_tick(0) {}

  // Reset Implementation
  void Reset(const GameState &initialState) override {
    opponent_states.clear();
    player_flip_states.clear();
    current_tick = 0;

    for (const auto &p : initialState.players) {
      opponent_states[p.carId].was_demoed = p.isDemoed;
      player_flip_states[p.carId] = PlayerFlipState();
    }
  }

  // GetReward Implementation with LESS SPAMMY Logging (Only prints on
  // Demo/Missed Demo)
  float GetReward(const Player &player, const GameState &state,
                  bool isFinal) override {
    float reward = 0.0f;
    uint32_t carId = player.carId;
    current_tick++;

    auto &flipState = player_flip_states[carId];
    const Action &action = player.prevAction;
    bool jump_pressed = (action.jump > 0.5f);
    bool has_flip =
        !player.hasFlipped && !player.isOnGround && player.hasJumped;

    // Flip detection is still needed for reward calculation, but we won't log
    // it every time.
    if (jump_pressed && !flipState.prev_jump_pressed && has_flip) {
      flipState.last_flip_tick = current_tick;
    }

    flipState.prev_jump_pressed = jump_pressed;
    flipState.prev_has_flip = has_flip;

    Vec car_pos = player.pos;
    Vec car_forward = player.rotMat.forward;
    float car_speed = player.vel.Length();
    bool is_supersonic = (car_speed >= 2200.0f);

    // Check all opponents for demos
    for (const auto &opponent : state.players) {
      if (opponent.team == player.team)
        continue;

      uint32_t opp_id = opponent.carId;
      auto &opp_state = opponent_states[opp_id];
      bool was_demoed = opp_state.was_demoed;
      bool now_demoed = opponent.isDemoed;

      // *** CORE LOGIC & LOGGING TRIGGER: Only triggers when a DEMO state
      // change happens. ***
      if (!was_demoed && now_demoed) {
        Vec to_enemy = opponent.pos - car_pos;
        Vec to_enemy_flat(to_enemy.x, to_enemy.y, 0.0f);
        float to_enemy_dist = to_enemy_flat.Length();

        if (to_enemy_dist > 1e-6f) {
          Vec to_enemy_dir = to_enemy_flat / to_enemy_dist;
          Vec car_fwd_flat(car_forward.x, car_forward.y, 0.0f);
          float car_fwd_dist = car_fwd_flat.Length();
          if (car_fwd_dist < 1e-6f)
            car_fwd_dist = 1e-6f;
          Vec car_fwd_dir = car_fwd_flat / car_fwd_dist;

          float dot_product = car_fwd_dir.Dot(to_enemy_dir);
          dot_product = std::max(-1.0f, std::min(1.0f, dot_product));
          float angle = std::acos(dot_product);
          float angle_deg = angle * 180.0f / M_PI;

          if (angle <= max_demo_angle) {
            float angle_score = std::max(0.0f, 1.0f - (angle / max_demo_angle));
            int ticks_since_flip = current_tick - flipState.last_flip_tick;
            bool flipped_recently = (ticks_since_flip <= flip_window_ticks);

            if (flipped_recently && !is_supersonic) {
              reward =
                  std::min(max_reward, base_reward + flip_bonus + angle_score);
              // LOGGING: HIGH-VALUE EVENT
              std::cout << "[DemoReward] 🚀 FLIP DEMO: P" << carId
                        << " demoed O" << opp_id << " | Angle: " << angle_deg
                        << "° | Reward: " << reward << std::endl;
            } else {
              reward = base_reward + angle_score;
              // LOGGING: HIGH-VALUE EVENT
              std::cout << "[DemoReward] 💥 REGULAR DEMO: P" << carId
                        << " demoed O" << opp_id
                        << " | Supersonic: " << (is_supersonic ? "Y" : "N")
                        << " | Reward: " << reward << std::endl;
            }
          } else {
            // LOGGING: Important failure case
            std::cout << "[DemoReward] ❌ MISSED REWARD: P" << carId
                      << " demoed O" << opp_id << " but angle was " << angle_deg
                      << "° (Max: " << (max_demo_angle * 180.0f / M_PI) << "°)."
                      << std::endl;
          }
        }
      }

      // Update opponent demo status for next tick
      opp_state.was_demoed = now_demoed;
    }

    return reward;
  }
};
class ApproachForDemoHelperReward2 : public Reward {
private:
  float angleWeight;
  float distanceWeight;

public:
  // Constructor Implementation
  ApproachForDemoHelperReward2(float angle_weight = 0.5f,
                               float distance_weight = 0.5f)
      : angleWeight(angle_weight), distanceWeight(distance_weight) {}

  // Reset Implementation
  virtual void Reset(const GameState &initialState) override {}

  // GetReward Implementation with LESS SPAMMY Logging (Only prints significant
  // rewards)
  virtual float GetReward(const Player &player, const GameState &state,
                          bool isFinal) override {
    float reward = 0.0f;

    Vec myPos = player.pos;
    Vec myVel = player.vel;
    Vec myFwd = player.rotMat.forward;
    Team myTeam = player.team;

    const Player *closestOpponent = nullptr;
    float closestDist = FLT_MAX;

    // Find nearest opponent
    for (const auto &opponent : state.players) {
      if (opponent.team == myTeam || opponent.carId == player.carId)
        continue;
      float dist = (opponent.pos - myPos).Length();
      if (dist < closestDist) {
        closestDist = dist;
        closestOpponent = &opponent;
      }
    }

    if (closestOpponent) {
      Vec toEnemy = closestOpponent->pos - myPos;
      float distance = toEnemy.Length();

      if (distance < 0.001f)
        return 0.0f;

      Vec toEnemyDir = toEnemy / distance;
      float myFwdLength = myFwd.Length();
      if (myFwdLength < 0.001f)
        return 0.0f;
      Vec myFwdDir = myFwd / myFwdLength;

      float mySpeed = myVel.Length();

      float alignment = myFwdDir.Dot(toEnemyDir);
      float angleReward = std::max(0.0f, alignment);

      float distanceReward = std::max(0.0f, 1.0f - (distance / 5000.0f));

      float speed_scale = mySpeed / 2300.0f;
      reward = (angleWeight * angleReward + distanceWeight * distanceReward) *
               speed_scale;

      // *** LOGGING CONDITION CHANGE: Only print if reward is above a
      // significant threshold (e.g., 0.5) *** This filters out tiny rewards the
      // agent gets every tick for vaguely facing the right way.
      if (reward >= 0.5f) {
        std::cout << "[ApproachReward] 🎯 HIGH APPROACH: P" << player.carId
                  << " towards O" << closestOpponent->carId
                  << " | Dist: " << distance << " | Speed: " << mySpeed
                  << " | Reward: " << reward << std::endl;
      }
    }

    return reward;
  }
};
class UpsideDownAirdribbleReward : public Reward {
private:
  float rewardScale;
  float minBallHeight;
  float minCarHeight;
  float maxDistanceToBall;
  float minUpsideDownAngle; // Minimum angle to consider "upside down" (in
                            // radians)

public:
  UpsideDownAirdribbleReward(
      float rewardScale = 1.0f,
      float minBallHeight = 200.0f, // Ball must be at least this high
      float minCarHeight = 150.0f,  // Car must be at least this high
      float maxDistanceToBall =
          300.0f, // Max distance from ball to count as airdribble
      float minUpsideDownAngle =
          2.8f // ~160 degrees in radians (close to upside down)
      )
      : rewardScale(rewardScale), minBallHeight(minBallHeight),
        minCarHeight(minCarHeight), maxDistanceToBall(maxDistanceToBall),
        minUpsideDownAngle(minUpsideDownAngle) {}

  virtual float GetReward(const Player &player, const GameState &state,
                          bool isFinal) override {
    // Get ball position and velocity
    const Vec &ballPos = state.ball.pos;
    const Vec &carPos = player.pos;
    const RotMat &carRot = player.rotMat;

    // Check if ball is airborne
    if (ballPos.z < minBallHeight) {
      return 0.0f;
    }

    // Check if car is airborne
    if (carPos.z < minCarHeight) {
      return 0.0f;
    }

    // Check distance to ball (airdribble condition)
    float distanceToBall = (ballPos - carPos).Length();
    if (distanceToBall > maxDistanceToBall) {
      return 0.0f;
    }

    // Check if car is upside down
    // The car's up vector is directly accessible from rotMat
    Vec carUpVector = carRot.up;

    // Calculate angle between car's up vector and world up vector (0, 0, 1)
    Vec worldUp = Vec(0, 0, 1);
    float dotProduct = carUpVector.Dot(worldUp);
    float angle = std::acos(std::abs(dotProduct)); // Angle between vectors

    // If angle is greater than minUpsideDownAngle, car is considered upside
    // down
    if (angle < minUpsideDownAngle) {
      return 0.0f;
    }

    // Calculate reward based on how well the airdribble is being performed
    float baseReward = 1.0f;

    // Bonus for being closer to the ball while upside down
    float distanceBonus =
        std::max(0.0f, 1.0f - (distanceToBall / maxDistanceToBall));

    // Bonus for being more upside down (closer to 180 degrees)
    float upsideDownBonus =
        (angle - minUpsideDownAngle) / (M_PI - minUpsideDownAngle);

    // Bonus for height (higher airdribbles are more impressive)
    float heightBonus = std::min(1.0f, (carPos.z - minCarHeight) / 500.0f);

    // Combine all factors
    float totalReward = baseReward + (distanceBonus * 0.5f) +
                        (upsideDownBonus * 0.7f) + (heightBonus * 0.3f);

    return totalReward * rewardScale;
  }
};
inline std::shared_ptr<UpsideDownAirdribbleReward>
MakeUpsideDownAirdribbleReward(float rewardScale = 1.0f,
                               float minBallHeight = 200.0f,
                               float minCarHeight = 150.0f,
                               float maxDistanceToBall = 300.0f,
                               float minUpsideDownAngle = 2.8f) {
  return std::make_shared<UpsideDownAirdribbleReward>(
      rewardScale, minBallHeight, minCarHeight, maxDistanceToBall,
      minUpsideDownAngle);
}
class BackboardDefenseReward : public Reward {
public:
  virtual float GetReward(const Player &player, const GameState &state,
                          bool isFinal) override {
    if (!player.ballTouchedStep)
      return 0;

    bool is_on_own_backboard =
        (abs(player.pos.y) > CommonValues::BACK_WALL_Y - 400) &&
        (player.pos.z > CommonValues::GOAL_HEIGHT) &&
        (RS_SGN(player.pos.y) == (player.team == Team::ORANGE ? 1 : -1));

    if (is_on_own_backboard) {
      Vec own_goal_pos = player.team == Team::BLUE
                             ? CommonValues::BLUE_GOAL_BACK
                             : CommonValues::ORANGE_GOAL_BACK;
      Vec dir_from_goal = (state.ball.pos - own_goal_pos).Normalized();

      // Reward clearing the ball away from goal
      return state.ball.vel.Dot(dir_from_goal) / CommonValues::BALL_MAX_SPEED;
    }

    return 0;
  }
};
class kuesresetreward : public Reward {
public:
  float minHeight;
  float maxDist;

  kuesresetreward(float minHeight = 150.f, float maxDist = 300.f)
      : minHeight(minHeight), maxDist(maxDist) {}

  virtual float GetReward(const Player &player, const GameState &state,
                          bool isFinal) override {
    if (player.isOnGround || player.HasFlipOrJump()) {
      return 0.f;
    }

    if (player.pos.z < minHeight) {
      return 0.f;
    }

    if (player.rotMat.up.z >= 0) {
      return 0.f;
    }

    float distToBall = (player.pos - state.ball.pos).Length();
    if (distToBall > maxDist) {
      return 0.f;
    }

    Vec dirToBall = (state.ball.pos - player.pos).Normalized();
    Vec relVel = player.vel - state.ball.vel;
    float approachSpeed = relVel.Dot(dirToBall);
    if (approachSpeed <= 0) {
      return 0.f;
    }

    float normSpeed =
        RS_CLAMP(approachSpeed / CommonValues::CAR_MAX_SPEED, 0.f, 1.f);
    float normAlign = ((-player.rotMat.up).Dot(dirToBall) + 1.f) / 2.f;
    float normDist = 1.f - RS_CLAMP(distToBall / maxDist, 0.f, 1.f);

    return std::min({normSpeed, normAlign, normDist});
  }
};
class BouncyAirDribbleReward : public Reward {
public:
  float bouncyDistThreshold;
  float bouncyVelThreshold;
  float minHeightForDribble;
  float minSpeedTowardGoal;
  float maxDistFromBall;
  float minPlayerToBallUpwards;
  float nearGoalShutoffTime;
  BouncyAirDribbleReward(float bouncyDist = 15.f, float bouncyVel = 50.f,
                         float minHeight = 250.f, float minSpeed = 800.f,
                         float maxDist = 250.f, float minPlayerToBallZ = 0.1f,
                         float shutoffTime = 2.f)
      : bouncyDistThreshold(bouncyDist), bouncyVelThreshold(bouncyVel),
        minHeightForDribble(minHeight), minSpeedTowardGoal(minSpeed),
        maxDistFromBall(maxDist), minPlayerToBallUpwards(minPlayerToBallZ),
        nearGoalShutoffTime(shutoffTime) {}
  virtual float GetReward(const Player &player, const GameState &state,
                          bool isFinal) override {
    Vec opponentGoal = (player.team == Team::BLUE)
                           ? CommonValues::ORANGE_GOAL_CENTER
                           : CommonValues::BLUE_GOAL_CENTER;
    bool isBouncy = false;
    float distToBall = (player.pos - state.ball.pos).Length();
    if (distToBall > bouncyDistThreshold) {
      isBouncy = true;
    } else {
      float relVelMag = (player.vel - state.ball.vel).Length();
      if (relVelMag > bouncyVelThreshold) {
        isBouncy = true;
      }
    }
    if (!isBouncy) {
      return 0.f;
    }
    bool heightConditionMet = false;
    if (state.ball.pos.z > minHeightForDribble) {
      heightConditionMet = true;
    } else if (player.vel.z > 0 && state.ball.vel.z > 0) {
      heightConditionMet = true;
    }
    if (!heightConditionMet)
      return 0.f;
    if (distToBall > maxDistFromBall)
      return 0.f;
    float distToGoal2D = (player.pos - opponentGoal).Length2D();
    float minBoostRequired = 15.f * std::pow(distToGoal2D / 5000.f, 1.75f);
    if (player.boost < minBoostRequired)
      return 0.f;
    Vec playerToBallDir = (state.ball.pos - player.pos).Normalized();
    if (playerToBallDir.z < minPlayerToBallUpwards)
      return 0.f;
    Vec ballToGoalDir = (opponentGoal - state.ball.pos).Normalized();
    if (player.vel.Dot(ballToGoalDir) < minSpeedTowardGoal ||
        state.ball.vel.Dot(ballToGoalDir) < minSpeedTowardGoal) {
      return 0.f;
    }
    Vec playerToBallDir2D =
        Vec(playerToBallDir.x, playerToBallDir.y, 0).Normalized();
    Vec ballToGoalDir2D = Vec(ballToGoalDir.x, ballToGoalDir.y, 0).Normalized();
    if (playerToBallDir2D.Dot(ballToGoalDir2D) < 0.75)
      return 0.f; // near the goal value
    float yDistToGoal = std::abs(state.ball.pos.y - opponentGoal.y);
    float timeToGoal = yDistToGoal / std::max(1.f, std::abs(state.ball.vel.y));
    if (timeToGoal > 0) {
      float gravityZ = state.lastArena
                           ? state.lastArena->GetMutatorConfig().gravity.z
                           : -650.f;
      float heightAtGoal = state.ball.pos.z + state.ball.vel.z * timeToGoal +
                           0.5f * gravityZ * timeToGoal * timeToGoal;
      if (heightAtGoal > CommonValues::GOAL_HEIGHT + 100)
        return 0.f;
    }

    if (timeToGoal < nearGoalShutoffTime) {
      return 0.f;
    }
    return 1.f;
  }
};
class AirDribbleBumpRewardV2 : public Reward {
public:
  AirDribbleBumpRewardV2(float minHeight = 300.0f,
                         float maxHeight = 643.0f, // new max height vro 💔
                         float maxBallDistance = 250.0f,
                         float proximityWeight = 0.0f,
                         float velocityWeight = 2.0f, float heightWeight = 0.0f,
                         float bumpBonus = 10.0f, float demoBonus = 25.0f,
                         float opponentProximityWeight = 3.0f,
                         float opponentDisplacementWeight = 0.01f,
                         float goalProximityPenaltyWeight = 4.0f)
      : minHeight(minHeight), maxHeight(maxHeight),
        maxBallDistance(maxBallDistance), proximityWeight(proximityWeight),
        velocityWeight(velocityWeight), heightWeight(heightWeight),
        bumpBonus(bumpBonus), demoBonus(demoBonus),
        opponentProximityWeight(opponentProximityWeight),
        opponentDisplacementWeight(opponentDisplacementWeight),
        goalProximityPenaltyWeight(goalProximityPenaltyWeight) {}

  inline virtual void Reset(const GameState &initialState) override {
    playerStates.clear();
    for (const auto &player : initialState.players) {
      playerStates[player.carId] = {false};
    }
  }

  inline virtual float GetReward(const Player &player, const GameState &state,
                                 bool isFinal) override {
    if (playerStates.find(player.carId) == playerStates.end()) {
      playerStates[player.carId] = {false};
    }
    auto &pState = playerStates[player.carId];

    // only reward if bot between minHeight n maxHeight 💔😭
    bool isAirborneAndNearBall =
        !player.isOnGround && player.pos.z >= minHeight &&
        player.pos.z <= maxHeight && state.ball.pos.z >= minHeight &&
        state.ball.pos.z <= maxHeight &&
        player.pos.Dist(state.ball.pos) < maxBallDistance;

    pState.isAirDribbling = isAirborneAndNearBall;

    if (!pState.isAirDribbling) {
      return 0.0f;
    }

    float reward = 0.0f;

    // --- Air Dribble Maintenance Rewards ---
    Vec distVec = state.ball.pos - player.pos;
    float dist = distVec.Length();

    float proximityFactor = std::max(
        0.0f, 1.0f - (dist - CommonValues::BALL_RADIUS) /
                         std::max(1e-6f,
                                  maxBallDistance - CommonValues::BALL_RADIUS));
    float proxReward = proximityFactor * proximityWeight;
    reward += proxReward;
    if (proxReward > 0)
      std::cout << "[Reward] Proximity reward: " << proxReward
                << " vro ong ts bussin\n";

    if (dist > 1e-6f) {
      Vec dirToBall = distVec / dist;
      float speedTowardsBall = player.vel.Dot(dirToBall);
      if (speedTowardsBall > 0) {
        float velReward =
            (speedTowardsBall / CommonValues::CAR_MAX_SPEED) * velocityWeight;
        reward += velReward;
        if (velReward > 0)
          std::cout << "[Reward] Velocity reward: " << velReward
                    << " vro ong ts bussin\n";
      }
    }

    float heightFactor =
        std::max(0.0f, (player.pos.z - minHeight) /
                           std::max(1e-6f, maxHeight - minHeight));
    float heightReward = heightFactor * heightWeight;
    reward += heightReward;
    if (heightReward > 0)
      std::cout << "[Reward] Height reward: " << heightReward
                << " vro ong ts bussin\n";

    // --- Opponent Interaction Rewards ---
    Player *closestOpponent = nullptr;
    float minOpponentDistSq = FLT_MAX;
    for (const auto &opponent : state.players) {
      if (opponent.team != player.team && !opponent.isDemoed) {
        float distSq = player.pos.DistSq(opponent.pos);
        if (distSq < minOpponentDistSq) {
          minOpponentDistSq = distSq;
          closestOpponent = const_cast<Player *>(&opponent);
        }
      }
    }

    if (closestOpponent) {
      float opponentProximityFactor =
          std::max(0.0f, 1.0f - (sqrtf(minOpponentDistSq) / 2000.0f));
      float oppReward = opponentProximityFactor * opponentProximityWeight;
      reward += oppReward;
      if (oppReward > 0)
        std::cout << "[Reward] Opponent proximity reward: " << oppReward
                  << " vro 😭💔\n";
    }

    // Reward for bumps and demos
    if (player.eventState.bump) {
      reward += bumpBonus;
      std::cout << "[Reward] Bump bonus: " << bumpBonus << " vro 💔😭\n";

      for (const auto &victim : state.players) {
        if (victim.team != player.team && victim.eventState.bumped &&
            victim.prev) {
          float displacement = victim.pos.Dist(victim.prev->pos);
          float dispReward = displacement * opponentDisplacementWeight;
          reward += dispReward;
          std::cout << "[Reward] Opponent displacement reward: " << dispReward
                    << " vro 😭🙏\n";
          break;
        }
      }
    }

    if (player.eventState.demo) {
      reward += demoBonus;
      std::cout << "[Reward] Demo bonus: " << demoBonus << " vro 💔😭\n";
    }

    // --- Penalties ---
    Vec ownGoalPos = (player.team == Team::BLUE)
                         ? CommonValues::BLUE_GOAL_BACK
                         : CommonValues::ORANGE_GOAL_BACK;
    float distToOwnGoal = player.pos.Dist(ownGoalPos);
    if (distToOwnGoal < 2000.0f) {
      float penaltyFactor = 1.0f - (distToOwnGoal / 2000.0f);
      float penalty = penaltyFactor * goalProximityPenaltyWeight;
      reward -= penalty;
      std::cout << "[Penalty] Too close to own goal: -" << penalty
                << " vro 😭💔\n";
    }

    if (reward > 0)
      std::cout << "[Reward] Total reward this tick: " << reward
                << " vro 💔🙏\n";

    return reward;
  }

private:
  struct PlayerState {
    bool isAirDribbling;
  };

  std::map<uint32_t, PlayerState> playerStates;

  float minHeight;
  float maxHeight; // new max height vro 💔
  float maxBallDistance;
  float proximityWeight;
  float velocityWeight;
  float heightWeight;
  float bumpBonus;
  float demoBonus;
  float opponentProximityWeight;
  float opponentDisplacementWeight;
  float goalProximityPenaltyWeight;
};
class GoodGoalPlacementReward : public Reward { // pretty good
public:
  virtual float GetReward(const Player &player, const GameState &state,
                          bool isFinal) {
    Vec goalPos = (player.team == Team::BLUE) ? CommonValues::ORANGE_GOAL_CENTER
                                              : CommonValues::BLUE_GOAL_CENTER;

    float closestDist = FLT_MAX;
    Vec closestEnemyPos = goalPos;

    for (const auto &other : state.players) {
      if (other.team != player.team) {
        float dist = other.pos.Dist(goalPos);
        if (dist < closestDist) {
          closestDist = dist;
          closestEnemyPos = other.pos;
        }
      }
    }

    std::vector<float> xs = linspace(-801.505f, 801.505f, 10);
    std::vector<float> zs = linspace(91.25f, 551.525f, 10);
    float y = goalPos.y;

    float maxMinDist = -1.0f;
    Vec bestPoint = {0, y, 0};

    for (float x : xs) {
      for (float z : zs) {
        float minDistToAnyEnemy = FLT_MAX;
        for (const auto &other : state.players) {
          if (other.team != player.team) {
            float dist = distance2D(x, z, other.pos.x, other.pos.z);
            minDistToAnyEnemy = std::min(minDistToAnyEnemy, dist);
          }
        }
        if (minDistToAnyEnemy > maxMinDist) {
          maxMinDist = minDistToAnyEnemy;
          bestPoint = {x, y, z};
        }
      }
    }

    Vec bPos = state.ball.pos;

    float reward = 0.0f;

    if (state.goalScored && state.ball.pos.Dist(goalPos) < 2000) {
      float dstToTarget = (bPos - bestPoint).Length();
      reward = std::max(0.0f, 1.0f - (dstToTarget / 1200.0f));

      if (reward > 0.0f) {
        float velScale = std::clamp(
            state.ball.vel.Length() / CommonValues::BALL_MAX_SPEED, 0.0f, 1.0f);
        reward += velScale;
        reward = std::clamp(reward, 0.0f, 2.0f);
      }
    }
    return reward;
  }
};
inline Vec FindBestGoalPoint(const Player &player, const GameState &state) {
  const Vec goalPos = (player.team == Team::BLUE)
                          ? CommonValues::ORANGE_GOAL_CENTER
                          : CommonValues::BLUE_GOAL_CENTER;

  auto isOpponent = [&](const Player &me, const Player &other) {
    return me.team != other.team;
  };

  // search grid inside the goal
  const std::vector<float> xs = linspace(-801.505f, 801.505f, 10);
  const std::vector<float> zs = linspace(91.25f, 551.525f, 10);
  const float y = goalPos.y;

  float bestMinDst = -1.f;
  Vec bestPoint = {0, y, 0};

  for (float x : xs)
    for (float z : zs) {
      float minToEnemy = FLT_MAX;
      for (const auto &other : state.players)
        if (isOpponent(player, other))
          minToEnemy =
              std::min(minToEnemy, distance2D(x, z, other.pos.x, other.pos.z));

      if (minToEnemy > bestMinDst) {
        bestMinDst = minToEnemy;
        bestPoint = {x, y, z};
      }
    }
  return bestPoint;
}
class WeightedRandomGoalReward : public Reward {
private:
  mutable std::mt19937 rng;
  mutable std::bernoulli_distribution distribution;
  float probability_of_two_points;

public:
  /**
   * Constructor
   * @param prob_two_points - Probability of getting 2 points (0.0 to 1.0)
   *                         Default 0.5 means equal chance of 1 or 2 points
   */
  WeightedRandomGoalReward(float prob_two_points = 0.5f)
      : rng(std::chrono::steady_clock::now().time_since_epoch().count()),
        distribution(prob_two_points),
        probability_of_two_points(prob_two_points) {}

  virtual float GetReward(const Player &player, const GameState &state,
                          bool isFinal) override {
    if (player.eventState.goal) {
      // Return 2 points with specified probability, otherwise 1 point
      return distribution(rng) ? 2.0f : 1.0f;
    }

    return 0.0f;
  }

  void SetProbability(float prob_two_points) {
    probability_of_two_points = prob_two_points;
    distribution = std::bernoulli_distribution(prob_two_points);
  }

  float GetProbability() const { return probability_of_two_points; }
};
class AirDribbleWithRollRewardud : public Reward {
public:
  AirDribbleWithRollRewardud(
      float minHeight = 326.0f, float maxDistance = 250.0f,
      float proximityWeight = 3.0f, float velocityWeight = 3.0f,
      float heightWeight = 1.0f, float touchBonus = 3.0f,
      float boostThreshold = 0.2f, float heightTarget = 1000.0f,
      float heightTargetWeight = .0f, float upwardBonus = 0.4f,
      float orientationWeight = 10.f, // Reward for being upside down
      float speedWeight = 1.0f, float boostEfficiencyWeight = 1.2f,
      float goalDirectionWeight =
          0.5f, // Bonus for moving towards opponent goal
      float awayFromGoalPenaltyWeight =
          0.f // Penalty for moving away from opponent goal
      )
      : minHeight(minHeight), maxDistance(maxDistance),
        proximityWeight(proximityWeight), velocityWeight(velocityWeight),
        heightWeight(heightWeight), touchBonus(touchBonus),
        boostThreshold(boostThreshold), heightTarget(heightTarget),
        heightTargetWeight(heightTargetWeight), upwardBonus(upwardBonus),
        orientationWeight(orientationWeight), speedWeight(speedWeight),
        boostEfficiencyWeight(boostEfficiencyWeight),
        goalDirectionWeight(goalDirectionWeight),
        awayFromGoalPenaltyWeight(awayFromGoalPenaltyWeight) {}

  inline virtual void Reset(const GameState &initialState) override {
    playerStates.clear();
    for (const auto &player : initialState.players) {
      playerStates[player.carId] = {player.ballTouchedTick, 0};
    }
  }

  inline virtual float GetReward(const Player &player, const GameState &state,
                                 bool isFinal) override {
    if (playerStates.find(player.carId) == playerStates.end()) {
      playerStates[player.carId] = {player.ballTouchedTick, 0};
    }
    auto &pState = playerStates[player.carId];

    auto resetState = [&]() {
      pState.airTouchStreak = 0;
      pState.lastBallTouch = player.ballTouchedStep;
      return 0.0f;
    };

    bool isAirborne = !player.isOnGround && player.pos.z >= minHeight &&
                      state.ball.pos.z >= minHeight;

    if (!isAirborne)
      return resetState();

    Vec distVec = state.ball.pos - player.pos;
    float dist = distVec.Length();
    if (dist > maxDistance)
      return resetState();

    float proximityFactor = std::max(
        0.0f,
        1.0f - (dist - CommonValues::BALL_RADIUS) /
                   std::max(1e-6f, maxDistance - CommonValues::BALL_RADIUS));
    float proxReward = proximityFactor * proximityWeight;

    float velReward = 0.0f;
    if (dist > 1e-6f) {
      Vec dirToBall = distVec / dist;
      float speedTowardsBall = player.vel.Dot(dirToBall);
      if (speedTowardsBall > 0) {
        velReward =
            (speedTowardsBall / CommonValues::CAR_MAX_SPEED) * velocityWeight;
      }
    }

    float heightFactor = std::max(
        0.0f, (player.pos.z - minHeight) /
                  std::max(1e-6f, CommonValues::CEILING_Z - minHeight));
    float heightReward = heightFactor * heightWeight;

    float heightDiff = abs(player.pos.z - heightTarget);
    float heightScale = std::max(0.0f, 1.0f - heightDiff / heightTarget);
    float heightTargetReward = heightScale * heightTargetWeight;

    float controlledSpeedReward = 0.0f;
    float playerSpeed = player.vel.Length();
    constexpr float OPTIMAL_MIN_SPEED = 800.0f;
    constexpr float OPTIMAL_MAX_SPEED = 1400.0f;
    if (playerSpeed >= OPTIMAL_MIN_SPEED && playerSpeed <= OPTIMAL_MAX_SPEED) {
      controlledSpeedReward = speedWeight;
    } else if (playerSpeed > OPTIMAL_MAX_SPEED) {
      float excessSpeedPenalty =
          std::min(0.5f, (playerSpeed - OPTIMAL_MAX_SPEED) / 1000.0f);
      controlledSpeedReward = -excessSpeedPenalty * speedWeight;
    }

    float boostConservationReward = 0.0f;
    if (player.boost > 30.f) {
      float boostFactor = powf(player.boost / 100.f, 0.7f);
      boostConservationReward = boostFactor * boostEfficiencyWeight;
    }

    Vec targetPos = (player.team == Team::BLUE) ? CommonValues::ORANGE_GOAL_BACK
                                                : CommonValues::BLUE_GOAL_BACK;
    Vec dirToGoal = (targetPos - player.pos).Normalized();
    float goalAlignment = dirToGoal.Dot(player.vel.Normalized());

    // **Bonus for dribbling towards the opponent's goal**
    float goalDirectionReward =
        std::max(0.f, goalAlignment) * goalDirectionWeight;

    // **Penalty for dribbling away from the opponent's goal**
    float awayPenalty = std::min(0.f, goalAlignment) *
                        awayFromGoalPenaltyWeight; // awayFromGoalPenaltyWeight
                                                   // should be positive

    Vec carUp = player.rotMat.up;
    // Reward for being upside down (carUp.z is negative), penalize for being
    // right side up (carUp.z is positive)
    float orientationReward = -carUp.z * orientationWeight;

    float baseReward = proxReward + velReward + heightReward +
                       heightTargetReward + controlledSpeedReward +
                       boostConservationReward + goalDirectionReward +
                       awayPenalty + orientationReward;
    float totalWeight = proximityWeight + velocityWeight + heightWeight +
                        heightTargetWeight + speedWeight +
                        boostEfficiencyWeight + goalDirectionWeight +
                        awayFromGoalPenaltyWeight + orientationWeight;

    float normalizedBaseReward = baseReward / std::max(1e-6f, totalWeight);
    float finalReward = normalizedBaseReward;

    if (player.vel.z > 0)
      finalReward += upwardBonus;

    bool belowBall = player.pos.z < state.ball.pos.z - 30;
    // Check if the car is upside down and under the ball, with its bottom
    // facing the ball
    bool upsideDownUnderBall =
        dist > 1e-6f && distVec.Dot(carUp) / dist < -0.7f;
    if (belowBall && upsideDownUnderBall)
      finalReward += 0.4f;

    if (!pState.lastBallTouch && player.ballTouchedStep) {
      pState.airTouchStreak++;
      float touchMultiplier = 1.0f;

      if (playerSpeed >= OPTIMAL_MIN_SPEED &&
          playerSpeed <= OPTIMAL_MAX_SPEED) {
        touchMultiplier *= 2.5f;
      } else if (playerSpeed > OPTIMAL_MAX_SPEED) {
        float speedPenalty =
            std::min(0.7f, (playerSpeed - OPTIMAL_MAX_SPEED) / 1000.0f);
        touchMultiplier *= std::max(0.1f, 1.5f - speedPenalty);
      } else {
        touchMultiplier *= 1.2f;
      }

      if (player.boost > (boostThreshold * 100))
        touchMultiplier *= 1.8f;
      if (dist < CommonValues::BALL_RADIUS + 150.0f)
        touchMultiplier *= 1.4f;

      float touchReward = touchBonus * pState.airTouchStreak * touchMultiplier;
      finalReward += touchReward;
    }

    pState.lastBallTouch = player.ballTouchedStep;

    float heightScaling = state.ball.pos.z / 1000.0f;
    finalReward *= std::max(0.1f, std::min(2.0f, heightScaling));

    return finalReward;
  }

private:
  struct PlayerState {
    bool lastBallTouch;
    int airTouchStreak;
  };

  std::map<uint32_t, PlayerState> playerStates;

  // Reward parameters
  float minHeight;
  float maxDistance;
  float proximityWeight;
  float velocityWeight;
  float heightWeight;
  float touchBonus;
  float boostThreshold;
  float heightTarget;
  float heightTargetWeight;
  float upwardBonus;
  float orientationWeight;
  float speedWeight;
  float boostEfficiencyWeight;
  float goalDirectionWeight;
  float awayFromGoalPenaltyWeight;
};
class AirDribbleWithRollReward : public Reward {
public:
  AirDribbleWithRollReward(
      float minHeight = 326.0f, float maxDistance = 250.0f,
      float proximityWeight = 3.0f, float velocityWeight = 6.0f,
      float heightWeight = .01f, float touchBonus = 3.0f,
      float boostThreshold = 0.2f, float heightTarget = 1000.0f,
      float heightTargetWeight = .0f, float upwardBonus = 0.4f,
      float airRollWeight = 5.f, float speedWeight = 1.0f,
      float boostEfficiencyWeight = 1.2f,
      float goalDirectionWeight =
          10.5f, // Bonus for moving towards opponent goal
      float awayFromGoalPenaltyWeight =
          21.f // Penalty for moving away from opponent goal
      )
      : minHeight(minHeight), maxDistance(maxDistance),
        proximityWeight(proximityWeight), velocityWeight(velocityWeight),
        heightWeight(heightWeight), touchBonus(touchBonus),
        boostThreshold(boostThreshold), heightTarget(heightTarget),
        heightTargetWeight(heightTargetWeight), upwardBonus(upwardBonus),
        airRollWeight(airRollWeight), speedWeight(speedWeight),
        boostEfficiencyWeight(boostEfficiencyWeight),
        goalDirectionWeight(goalDirectionWeight),
        awayFromGoalPenaltyWeight(awayFromGoalPenaltyWeight) {}

  inline virtual void Reset(const GameState &initialState) override {
    playerStates.clear();
    for (const auto &player : initialState.players) {
      playerStates[player.carId] = {player.ballTouchedTick, 0, 0.0f, 0.0f};
    }
  }

  inline virtual float GetReward(const Player &player, const GameState &state,
                                 bool isFinal) override {
    if (playerStates.find(player.carId) == playerStates.end()) {
      playerStates[player.carId] = {player.ballTouchedTick, 0, 0.0f, 0.0f};
    }
    auto &pState = playerStates[player.carId];

    auto resetState = [&]() {
      pState.airTouchStreak = 0;
      pState.lastBallTouch = player.ballTouchedStep;
      pState.lastRollInput = 0.0f;
      pState.directionChanges = 0.0f;
      return 0.0f;
    };

    bool isAirborne = !player.isOnGround && player.pos.z >= minHeight &&
                      state.ball.pos.z >= minHeight;

    if (!isAirborne)
      return resetState();

    Vec distVec = state.ball.pos - player.pos;
    float dist = distVec.Length();
    if (dist > maxDistance)
      return resetState();

    float proximityFactor = std::max(
        0.0f,
        1.0f - (dist - CommonValues::BALL_RADIUS) /
                   std::max(1e-6f, maxDistance - CommonValues::BALL_RADIUS));
    float proxReward = proximityFactor * proximityWeight;

    float velReward = 0.0f;
    if (dist > 1e-6f) {
      Vec dirToBall = distVec / dist;
      float speedTowardsBall = player.vel.Dot(dirToBall);
      if (speedTowardsBall > 0) {
        velReward =
            (speedTowardsBall / CommonValues::CAR_MAX_SPEED) * velocityWeight;
      }
    }

    float heightFactor = std::max(
        0.0f, (player.pos.z - minHeight) /
                  std::max(1e-6f, CommonValues::CEILING_Z - minHeight));
    float heightReward = heightFactor * heightWeight;

    float heightDiff = abs(player.pos.z - heightTarget);
    float heightScale = std::max(0.0f, 1.0f - heightDiff / heightTarget);
    float heightTargetReward = heightScale * heightTargetWeight;

    float controlledSpeedReward = 0.0f;
    float playerSpeed = player.vel.Length();
    constexpr float OPTIMAL_MIN_SPEED = 800.0f;
    constexpr float OPTIMAL_MAX_SPEED = 1400.0f;
    if (playerSpeed >= OPTIMAL_MIN_SPEED && playerSpeed <= OPTIMAL_MAX_SPEED) {
      controlledSpeedReward = speedWeight;
    } else if (playerSpeed > OPTIMAL_MAX_SPEED) {
      float excessSpeedPenalty =
          std::min(0.5f, (playerSpeed - OPTIMAL_MAX_SPEED) / 1000.0f);
      controlledSpeedReward = -excessSpeedPenalty * speedWeight;
    }

    float boostConservationReward = 0.0f;
    if (player.boost > 30.f) {
      float boostFactor = powf(player.boost / 100.f, 0.7f);
      boostConservationReward = boostFactor * boostEfficiencyWeight;
    }

    Vec targetPos = (player.team == Team::BLUE) ? CommonValues::ORANGE_GOAL_BACK
                                                : CommonValues::BLUE_GOAL_BACK;
    Vec dirToGoal = (targetPos - player.pos).Normalized();
    float goalAlignment = dirToGoal.Dot(player.vel.Normalized());

    // **Bonus for dribbling towards the opponent's goal**
    float goalDirectionReward =
        std::max(0.f, goalAlignment) * goalDirectionWeight;

    // **Penalty for dribbling away from the opponent's goal**
    float awayPenalty = std::min(0.f, goalAlignment) *
                        awayFromGoalPenaltyWeight; // awayFromGoalPenaltyWeight
                                                   // should be positive

    float baseReward = proxReward + velReward + heightReward +
                       heightTargetReward + controlledSpeedReward +
                       boostConservationReward + goalDirectionReward +
                       awayPenalty;
    float totalWeight = proximityWeight + velocityWeight + heightWeight +
                        heightTargetWeight + speedWeight +
                        boostEfficiencyWeight + goalDirectionWeight +
                        awayFromGoalPenaltyWeight;

    float airRollReward = 0.0f;
    float currentRollInput = player.prevAction.roll;

    if (abs(currentRollInput) > 0.1f && abs(pState.lastRollInput) > 0.1f) {
      if (currentRollInput * pState.lastRollInput < 0) {
        pState.directionChanges += 1.0f;
      }
    }

    if (pState.directionChanges > 0) {
      pState.directionChanges = std::max(0.0f, pState.directionChanges - 0.05f);
    }

    pState.lastRollInput = currentRollInput;
    bool isWiggling = pState.directionChanges > 2.0f;

    if (!isWiggling && abs(currentRollInput) > 0.1f) {
      float baseRollReward = airRollWeight;
      if (currentRollInput > 0.1f)
        baseRollReward *= 1.5f;
      else
        baseRollReward *= 0.3f;

      if (player.prevAction.boost > 0 &&
          player.boost > (boostThreshold * 100)) {
        baseRollReward *= 1.4f;
      }
      airRollReward = baseRollReward;
    }

    baseReward += airRollReward;
    totalWeight += 1.0f;

    float normalizedBaseReward = baseReward / std::max(1e-6f, totalWeight);
    float finalReward = normalizedBaseReward;

    if (player.vel.z > 0)
      finalReward += upwardBonus;

    Vec carUp = player.rotMat.up;
    bool belowBall = player.pos.z < state.ball.pos.z - 30;
    bool facingBall = dist > 1e-6f && distVec.Dot(carUp) / dist > 0.7f;
    if (belowBall && facingBall)
      finalReward += 0.2f;

    if (!pState.lastBallTouch && player.ballTouchedStep) {
      pState.airTouchStreak++;
      float touchMultiplier = 1.0f;

      if (playerSpeed >= OPTIMAL_MIN_SPEED &&
          playerSpeed <= OPTIMAL_MAX_SPEED) {
        touchMultiplier *= 2.5f;
      } else if (playerSpeed > OPTIMAL_MAX_SPEED) {
        float speedPenalty =
            std::min(0.7f, (playerSpeed - OPTIMAL_MAX_SPEED) / 1000.0f);
        touchMultiplier *= std::max(0.1f, 1.5f - speedPenalty);
      } else {
        touchMultiplier *= 1.2f;
      }

      if (player.boost > (boostThreshold * 100))
        touchMultiplier *= 1.8f;
      if (dist < CommonValues::BALL_RADIUS + 150.0f)
        touchMultiplier *= 1.4f;

      float touchReward = touchBonus * pState.airTouchStreak * touchMultiplier;
      finalReward += touchReward;
    }

    pState.lastBallTouch = player.ballTouchedStep;

    float heightScaling = state.ball.pos.z / 1000.0f;
    finalReward *= std::max(0.1f, std::min(2.0f, heightScaling));

    return finalReward;
  }

private:
  struct PlayerState {
    bool lastBallTouch;
    int airTouchStreak;
    float lastRollInput;
    float directionChanges;
  };

  std::map<uint32_t, PlayerState> playerStates;

  // Reward parameters
  float minHeight;
  float maxDistance;
  float proximityWeight;
  float velocityWeight;
  float heightWeight;
  float touchBonus;
  float boostThreshold;
  float heightTarget;
  float heightTargetWeight;
  float upwardBonus;
  float airRollWeight;
  float speedWeight;
  float boostEfficiencyWeight;
  float goalDirectionWeight;
  float awayFromGoalPenaltyWeight;
};
class ContinuousFlipResetReward : public Reward {
public:
  float minHeight;
  float maxDist;

  ContinuousFlipResetReward(float minHeight = 150.f, float maxDist = 300.f)
      : minHeight(minHeight), maxDist(maxDist) {}

  virtual float GetReward(const Player &player, const GameState &state,
                          bool isFinal) override {
    if (player.isOnGround || player.HasFlipOrJump()) {
      return 0.f;
    }

    if (player.pos.z < minHeight) {
      return 0.f;
    }

    if (player.rotMat.up.z >= 0) {
      return 0.f;
    }

    float distToBall = (player.pos - state.ball.pos).Length();
    if (distToBall > maxDist) {
      return 0.f;
    }

    Vec dirToBall = (state.ball.pos - player.pos).Normalized();
    Vec relVel = player.vel - state.ball.vel;
    float approachSpeed = relVel.Dot(dirToBall);
    if (approachSpeed <= 0) {
      return 0.f;
    }

    float normSpeed =
        RS_CLAMP(approachSpeed / CommonValues::CAR_MAX_SPEED, 0.f, 1.f);
    float normAlign = ((-player.rotMat.up).Dot(dirToBall) + 1.f) / 2.f;
    float normDist = 1.f - RS_CLAMP(distToBall / maxDist, 0.f, 1.f);

    // Calculate the reward
    float reward = std::min({normSpeed, normAlign, normDist});
    return reward;
  }
};
class FlipResetRewardGiga : public Reward {
public:
  FlipResetRewardGiga(float flipResetR = 1.0f, float holdFlipResetR = 0.0f)
      : flipResetR(flipResetR), holdFlipResetR(holdFlipResetR) {}
  void Reset(const GameState &initialState) override;
  float GetReward(const Player &player, const GameState &state,
                  bool isFinal) override;

private:
  std::unordered_map<uint32_t, bool> prevCanJump; // keyed by carId
  std::unordered_map<uint32_t, bool> hasReset;    // keyed by carId
  float flipResetR;
  float holdFlipResetR;
};
void FlipResetRewardGiga::Reset(const GameState & /*initialState*/) {
  prevCanJump.clear();
  hasReset.clear();
}
float FlipResetRewardGiga::GetReward(const Player &player,
                                     const GameState &state, bool /*isFinal*/) {
  const float BALL_RADIUS = CommonValues::BALL_RADIUS;
  const float CEILING_Z = CommonValues::CEILING_Z;
  const float SIDE_WALL_X = CommonValues::SIDE_WALL_X;
  const float BACK_WALL_Y = CommonValues::BACK_WALL_Y;

  uint32_t carId = player.carId;
  float reward = 0.0f;

  // distance to ball
  bool nearBall = (player.pos - state.ball.pos).Length() < 170.0f;

  // height and wall checks (exclude very low/high and near walls)
  bool heightCheck =
      (player.pos.z < 300.0f) || (player.pos.z > CEILING_Z - 300.0f);
  bool wallDisCheck = ((-SIDE_WALL_X + 700.0f) > player.pos.x) ||
                      ((SIDE_WALL_X - 700.0f) < player.pos.x) ||
                      ((-BACK_WALL_Y + 700.0f) > player.pos.y) ||
                      ((BACK_WALL_Y - 700.0f) < player.pos.y);

  // derive canJump / hasFlipped from CarState fields (expected in Player /
  // CarState)
  bool canJump = true;
  bool hasFlipped = false;
  try {
    canJump = !player.hasJumped;
  } catch (...) { /* fallthrough */
  }
  try {
    hasFlipped = (player.isJumping && player.isFlipping);
  } catch (...) { /* fallthrough */
  }

  // reset tracking if on wall or already flipped
  if (wallDisCheck || hasFlipped) {
    hasReset[carId] = false;
  }

  // only consider flip-resets near ball and not too high/near wall
  if (nearBall && !heightCheck && !wallDisCheck) {
    bool prev = false;
    auto it = prevCanJump.find(carId);
    if (it != prevCanJump.end())
      prev = it->second;

    bool gotReset = (!prev && canJump);

    // ===== Extra conditions to avoid wavedash triggering =====
    bool airborne = player.pos.z > 150.0f; // not basically on the ground

    // grab car up vector from rotation matrix
    Vec carUp(player.rotMat[0][2], player.rotMat[1][2], player.rotMat[2][2]);

    Vec carToBall = state.ball.pos - player.pos;
    bool undersideContact = carToBall.Dot(carUp) < -BALL_RADIUS * 0.5f;

    if (gotReset && airborne && undersideContact) {
      hasReset[carId] = true;
      reward = flipResetR;
    }
  }

  if (hasReset.count(carId) && hasReset.at(carId)) {
    reward += holdFlipResetR;
  }
  prevCanJump[carId] = canJump;

  // ================== DEBUG PRINT ==================
  if (reward > 0.0f) {
    std::cout << "[Debug] Player " << carId
              << " received FlipResetReward: " << reward << std::endl;
  }
  // =================================================

  return reward;
}
class FlipResetEventReward : public Reward {
public:
  float minHeight;
  float maxDist;
  float minUpZ;

  FlipResetEventReward(float minHeight = 150.f, float maxDist = 150.f,
                       float minUpZ = -0.7f)
      : minHeight(minHeight), maxDist(maxDist), minUpZ(minUpZ) {}

  virtual float GetReward(const Player &player, const GameState &state,
                          bool isFinal) override {
    if (!player.prev)
      return 0.f;

    bool gotReset = !player.prev->isOnGround && player.HasFlipOrJump() &&
                    !player.prev->HasFlipOrJump();

    if (gotReset) {

      if (player.pos.z > minHeight && player.rotMat.up.z < minUpZ &&
          (player.pos - state.ball.pos).Length() < maxDist) {
        float reward = 1.f;
        // Print to console because the reward is positive
        std::cout << "FlipResetEventReward: " << reward << std::endl;
        return reward;
      }
    }

    return 0.f;
  }
};
class FlickforMomentumReward : public Reward {
public:
  /**
   * @brief Construct a new FlickReward object.
   *
   * @param height_threshold The maximum vertical distance (Z-axis) from the
   * car's origin to the ball's origin for possession to be considered.
   * @param dist_threshold The maximum planar distance (XY-plane) from the car's
   * origin to the ball's origin for possession to be considered.
   * @param min_z_vel_change The minimum required increase in the ball's
   * vertical velocity to confirm a flick has occurred.
   */
  FlickforMomentumReward(float height_threshold = 160.f,
                         float dist_threshold = 130.f,
                         float min_z_vel_change = 300.f)
      : ball_on_car_height_threshold(height_threshold),
        ball_on_car_dist_threshold(dist_threshold),
        min_ball_z_vel_change(min_z_vel_change) {}

  virtual float GetReward(const Player &player, const GameState &state,
                          bool isFinal) override {
    // We need the previous state to detect changes, so if it's not available,
    // no reward.
    if (!player.prev || !state.prev) {
      return 0;
    }

    // Conditions for a flick:
    // 1. In the PREVIOUS state, the ball was on top of the car.
    // 2. In the CURRENT state, the player has just initiated a flip/dodge.
    // 3. In the CURRENT state, the ball's vertical velocity has significantly
    // increased.

    // Condition 1: Check for ball possession in the previous state.
    const BallState &prev_ball = state.prev->ball;
    const Vec &prev_car_pos = player.prev->pos;

    float height_diff = prev_ball.pos.z - prev_car_pos.z;
    bool is_above =
        height_diff > 0 && height_diff < ball_on_car_height_threshold;
    float dist_2d = prev_ball.pos.Dist2D(prev_car_pos);
    bool is_close = dist_2d < ball_on_car_dist_threshold;
    bool had_possession = is_above && is_close;

    // Condition 2: Check if the player just started flipping.
    bool just_flipped = player.isFlipping && !player.prev->isFlipping;

    if (had_possession && just_flipped) {
      // Condition 3: Check if the ball was launched upwards.
      float z_vel_change = state.ball.vel.z - prev_ball.vel.z;
      if (z_vel_change > min_ball_z_vel_change) {
        // Flick detected, now calculate the reward based on momentum.

        // Part 1: Reward based on the car's forward speed before the flick.
        float forward_speed_prev =
            player.prev->vel.Dot(player.prev->rotMat.forward);

        // Don't reward flicks while reversing.
        if (forward_speed_prev < 0)
          return 0;

        float speed_reward_factor =
            forward_speed_prev / CommonValues::CAR_MAX_SPEED;

        // Part 2: Reward based on how much velocity was imparted to the ball.
        float ball_vel_change_magnitude =
            (state.ball.vel - prev_ball.vel).Length();
        float launch_reward_factor =
            ball_vel_change_magnitude / CommonValues::BALL_MAX_SPEED;

        // Combine the factors. A good flick requires both speed and a strong
        // launch.
        return speed_reward_factor * launch_reward_factor;
      }
    }

    return 0; // No flick detected.
  }

private:
  float ball_on_car_height_threshold;
  float ball_on_car_dist_threshold;
  float min_ball_z_vel_change;
};
class ResetShotReward : public Reward {
private:
  std::map<uint32_t, uint64_t> _tickCountWhenResetObtained;

public:
  ResetShotReward() {}

  virtual void Reset(const GameState &initial_state) override {
    _tickCountWhenResetObtained.clear();
  }

  virtual void PreStep(const GameState &state) override {
    if (!state.lastArena)
      return;
    for (const auto &player : state.players) {
      if (!player.prev)
        continue;

      bool gotReset = !player.prev->isOnGround && player.HasFlipOrJump() &&
                      !player.prev->HasFlipOrJump();
      if (gotReset) {
        _tickCountWhenResetObtained[player.carId] = state.lastArena->tickCount;
      }
    }
  }

  virtual float GetReward(const Player &player, const GameState &state,
                          bool isFinal) override {
    if (!player.prev || !state.prev)
      return 0.f;

    auto it = _tickCountWhenResetObtained.find(player.carId);
    if (it == _tickCountWhenResetObtained.end()) {
      return 0.f;
    }

    bool flipWasUsedForTouch =
        player.ballTouchedStep && !player.isOnGround && !player.hasJumped &&
        player.prev->HasFlipOrJump() && !player.HasFlipOrJump();

    if (flipWasUsedForTouch) {

      float hitForce = (state.ball.vel - state.prev->ball.vel).Length();
      float ballSpeed = state.ball.vel.Length();
      float baseReward =
          (hitForce + ballSpeed) /
          (CommonValues::CAR_MAX_SPEED + CommonValues::BALL_MAX_SPEED);

      uint64_t ticksSinceReset = state.lastArena->tickCount - it->second;
      float timeSinceReset = ticksSinceReset * CommonValues::TICK_TIME;
      float timeBonus = 1.f + std::log1p(timeSinceReset);

      _tickCountWhenResetObtained.erase(it);

      // Calculate the reward
      float reward = baseReward * timeBonus;

      // Print to console if the reward is positive
      if (reward > 0) {
        std::cout << "ResetShotReward: " << reward << std::endl;
      }

      return reward;
    }

    if (player.isOnGround) {
      _tickCountWhenResetObtained.erase(it);
    }

    return 0.f;
  }
};
class AirDribbleWithRollRewardImpl : public Reward {
public:
  // Configurable parameters
  float min_height = 326.0f;
  float max_distance = 250.0f;
  float proximity_weight = 3.0f;
  float velocity_weight = 3.0f;
  float height_weight = 1.0f;
  float touch_bonus = 3.0f;
  float boost_threshold = 20.0f; // Boost amount / 100
  float height_target = 1000.0f;
  float height_target_weight = 1.0f;
  float upward_bonus = 0.1f;
  float air_roll_weight = 0.5f;
  float speed_weight = 1.0f;
  float boost_efficiency_weight = 1.2f;

private:
  // Per-player state
  bool last_ball_touch = false;
  int air_touch_streak = 0;
  float last_roll_input = 0.0f;
  int direction_changes = 0;

public:
  virtual void Reset(const GameState &initialState) override {
    last_ball_touch = false;
    air_touch_streak = 0;
    last_roll_input = 0.0f;
    direction_changes = 0;
  }

  virtual float GetReward(const Player &player, const GameState &state,
                          bool isFinal) override {
    // Reset streak and state if on ground or conditions are not met
    bool is_airborne = !player.isOnGround && player.pos.z >= min_height &&
                       state.ball.pos.z >= min_height;
    float dist = player.pos.Dist(state.ball.pos);
    bool is_close = dist <= max_distance;

    if (!is_airborne || !is_close) {
      air_touch_streak = 0;
      last_ball_touch = player.ballTouchedStep;
      last_roll_input = 0.0f;
      direction_changes = 0;
      return 0.0f;
    }

    // --- Base Reward Components ---
    float prox_reward =
        (1.0f - (dist - CommonValues::BALL_RADIUS) /
                    (max_distance - CommonValues::BALL_RADIUS)) *
        proximity_weight;

    Vec dir_to_ball = (state.ball.pos - player.pos).Normalized();
    float speed_towards_ball = player.vel.Dot(dir_to_ball);
    float vel_reward =
        (speed_towards_ball > 0)
            ? (speed_towards_ball / CommonValues::CAR_MAX_SPEED) *
                  velocity_weight
            : 0;

    float height_reward =
        ((player.pos.z - min_height) / (CommonValues::CEILING_Z - min_height)) *
        height_weight;

    float height_diff = abs(player.pos.z - height_target);
    float height_target_reward =
        (1.0f - height_diff / height_target) * height_target_weight;

    float controlled_speed_reward = 0.0f;
    float player_speed = player.vel.Length();
    if (player_speed >= 800.0f && player_speed <= 1400.0f) {
      controlled_speed_reward = speed_weight;
    } else if (player_speed > 1400.0f) {
      controlled_speed_reward =
          -std::min(0.5f, (player_speed - 1400.0f) / 1000.0f) * speed_weight;
    }

    float boost_conservation_reward =
        (player.boost > 30)
            ? (powf(player.boost / 100.f, 0.7f) * boost_efficiency_weight)
            : 0;

    // --- Air Roll Reward ---
    float air_roll_reward = 0.0f;
    float current_roll_input = player.prevAction.roll;

    if (abs(current_roll_input) > 0.1f && abs(last_roll_input) > 0.1f) {
      if (current_roll_input * last_roll_input < 0) { // Sign change
        direction_changes += 1;
      }
    }
    if (direction_changes > 0)
      direction_changes = std::max(0, direction_changes - 1); // Decay
    last_roll_input = current_roll_input;

    bool is_wiggling = direction_changes > 4; // Check for excessive wiggling
    if (!is_wiggling && abs(current_roll_input) > 0.1f) {
      air_roll_reward = air_roll_weight;
    }

    // --- Combine Rewards ---
    float base_reward = prox_reward + vel_reward + height_reward +
                        height_target_reward + controlled_speed_reward +
                        boost_conservation_reward + air_roll_reward;
    float total_weight = proximity_weight + velocity_weight + height_weight +
                         height_target_weight + speed_weight +
                         boost_efficiency_weight + 1.0f;
    float final_reward = base_reward / total_weight;

    if (player.vel.z > 0)
      final_reward += upward_bonus;

    // --- Touch Bonus ---
    if (!last_ball_touch && player.ballTouchedStep) {
      air_touch_streak += 1;
      float touch_multiplier = 1.0f;
      if (player_speed >= 800.0f && player_speed <= 1400.0f)
        touch_multiplier *= 2.5f;
      if (player.boost > boost_threshold)
        touch_multiplier *= 1.8f;
      if (dist < CommonValues::BALL_RADIUS + 150.0f)
        touch_multiplier *= 1.4f;

      final_reward += touch_bonus * air_touch_streak * touch_multiplier;
    }

    last_ball_touch = player.ballTouchedStep;

    // Final scaling based on height
    final_reward *= std::max(0.1f, std::min(2.0f, state.ball.pos.z / 1000.0f));

    return final_reward;
  }
};
class SpeedBallToGoalReward : public Reward {
public:
  bool ownGoal = false;
  SpeedBallToGoalReward(bool ownGoal = false) : ownGoal(ownGoal) {}

  virtual float GetReward(const Player &player, const GameState &state,
                          bool isFinal) override {
    bool targetOrangeGoal = player.team == Team::BLUE;
    if (ownGoal)
      targetOrangeGoal = !targetOrangeGoal;

    Vec targetPos = targetOrangeGoal ? CommonValues::ORANGE_GOAL_BACK
                                     : CommonValues::BLUE_GOAL_BACK;

    Vec dirToGoal = (targetPos - state.ball.pos).Normalized();
    return state.ball.vel.Dot(dirToGoal) / CommonValues::BALL_MAX_SPEED;
  }
};
class VelocityReward : public Reward {
public:
  bool isNegative;
  VelocityReward(bool isNegative = false) : isNegative(isNegative) {}
  virtual float GetReward(const Player &player, const GameState &state,
                          bool isFinal) {
    return player.vel.Length() / CommonValues::CAR_MAX_SPEED *
           (1 - 2 * isNegative);
  }
};
class EnergyReward : public Reward {
public:
  const double GRAVITY = 650;
  const double MASS = 180;
  virtual float GetReward(const Player &player, const GameState &state,
                          bool isFinal) override {
    auto max_energy = (MASS * GRAVITY * (CEILING_Z - 17.)) +
                      (0.5 * MASS * (CAR_MAX_SPEED * CAR_MAX_SPEED));
    double energy = 0;
    double velocity = player.vel.Length();

    if (player.HasFlipOrJump()) {
      energy += 0.35 * MASS * (292 * 292);
    }
    if (player.HasFlipOrJump() and !player.isOnGround) {
      double dodge_impulse = (velocity <= 1700) ? (500 + (velocity / 17))
                                                : (600 - (velocity - 1700));
      dodge_impulse = std::max(dodge_impulse - 25, 0.0);
      energy += 0.9 * 0.5 * MASS * (dodge_impulse * dodge_impulse);
      energy += 0.35 * MASS * 550. * 550.;
    }
    // height
    energy += MASS * GRAVITY * (player.pos.z - 17.) *
              0.75; // fudge factor to reduce height
    // KE
    energy += 0.5 * MASS * velocity * velocity;
    // boost
    energy += 7.97e5 * player.boost;
    double norm_energy = player.isDemoed ? 0.0f : (energy / max_energy);
    return norm_energy;
  }
};
class DoubleTapReward : public Reward {
public:
  DoubleTapReward(float dblTapR = 1.0f, int maxIntervalTicks = 240)
      : dblTapR(dblTapR), maxIntervalTicks(maxIntervalTicks) {}

  void Reset(const GameState & /*initialState*/) override {
    curTick = 0;
    lastTouchTick.clear();
    lastTouchWasBackboard.clear();
  }

  float GetReward(const Player &player, const GameState &state,
                  bool /*isFinal*/) override {
    uint32_t carId = player.carId;
    float reward = 0.0f;

    curTick++;

    // detect ball touch using Player flags
    bool ballTouch = player.ballTouchedStep || player.ballTouchedTick;
    if (!ballTouch)
      return 0.0f;

    // check if last touch was a backboard hit
    if (lastTouchWasBackboard.count(carId) && lastTouchWasBackboard[carId]) {
      int dt = curTick - lastTouchTick[carId];
      if (dt > 3 && dt < maxIntervalTicks) {
        reward = dblTapR;
        std::cout << "[Debug] Player " << carId
                  << " got DOUBLE TAP reward: " << reward << " (Δticks=" << dt
                  << ")" << std::endl;
      }
    }

    // detect if this touch is off the backboard
    bool hitBackboard = false;
    const auto &bpos = state.ball.pos;
    const float BACK_WALL_Y = CommonValues::BACK_WALL_Y;
    if (fabs(bpos.y) > BACK_WALL_Y - 200.0f && bpos.z > 300.0f) {
      hitBackboard = true;
    }

    lastTouchTick[carId] = curTick;
    lastTouchWasBackboard[carId] = hitBackboard;

    return reward;
  }

private:
  int curTick = 0;
  std::unordered_map<uint32_t, int> lastTouchTick;
  std::unordered_map<uint32_t, bool> lastTouchWasBackboard;

  float dblTapR;
  int maxIntervalTicks;
};
class VelocityBallToGoalReward : public Reward {
public:
  bool ownGoal = false;
  VelocityBallToGoalReward(bool ownGoal = false) : ownGoal(ownGoal) {}

  virtual float GetReward(const Player &player, const GameState &state,
                          bool isFinal) {
    bool targetOrangeGoal = player.team == Team::BLUE;
    if (ownGoal)
      targetOrangeGoal = !targetOrangeGoal;

    Vec targetPos = targetOrangeGoal ? CommonValues::ORANGE_GOAL_BACK
                                     : CommonValues::BLUE_GOAL_BACK;

    Vec ballDirToGoal = (targetPos - state.ball.pos).Normalized();
    return ballDirToGoal.Dot(state.ball.vel / CommonValues::BALL_MAX_SPEED);
  }
};
class VelocityPlayerToBallReward : public Reward {
public:
  virtual float GetReward(const Player &player, const GameState &state,
                          bool isFinal) {
    Vec dirToBall = (state.ball.pos - player.pos).Normalized();
    Vec normVel = player.vel / CommonValues::CAR_MAX_SPEED;
    return dirToBall.Dot(normVel);
  }
};
class AirdribbleSetupReward : public Reward {
public:
  // Tunable parameters for the reward function
  const float MIN_BALL_HEIGHT = CommonValues::BALL_RADIUS + 15.f;
  const float MIN_UPWARD_VELOCITY = 100.f;
  const float MAX_DISTANCE_TO_BALL = 500.f;
  const float IDEAL_HEIGHT_DIFFERENCE =
      150.f; // Bot should be this far below the ball
  const float MAX_REWARDED_POP_VEL = 1500.f;

  virtual float GetReward(const Player &player, const GameState &state,
                          bool isFinal) override {
    // We need the previous state to analyze the result of the touch.
    if (!player.prev || !player.ballTouchedStep) {
      return 0.f;
    }

    const auto &prevBallState = state.prev->ball;
    const auto &ballState = state.ball;
    const auto &playerState = player;

    // 1. The Pop-up: Ensure the ball is popped into the air with upward
    // velocity.
    if (ballState.pos.z < MIN_BALL_HEIGHT ||
        ballState.vel.z < MIN_UPWARD_VELOCITY) {
      return 0.f;
    }

    // Reward for imparting upward velocity on the ball.
    float popReward =
        RS_CLAMP(ballState.vel.z / MAX_REWARDED_POP_VEL, 0.f, 1.f);

    // 2. Player Follow-up: Ensure the player is also airborne and following the
    // ball.
    if (playerState.isOnGround) {
      return 0.f;
    }

    // Reward for being close to the ball.
    float distToBall = playerState.pos.Dist(ballState.pos);
    if (distToBall > MAX_DISTANCE_TO_BALL) {
      return 0.f;
    }
    float proximityReward = 1.f - (distToBall / MAX_DISTANCE_TO_BALL);

    // Reward for having velocity directed towards the ball.
    Vec dirToBall = (ballState.pos - playerState.pos).Normalized();
    float velocityAlignmentReward =
        (playerState.vel.Normalized().Dot(dirToBall) + 1.f) / 2.f;

    // 3. Positioning: Reward for being underneath the ball.
    if (playerState.pos.z >= ballState.pos.z) {
      return 0.f;
    }

    // Use a Gaussian function to give max reward at an ideal height difference.
    float heightDiff = ballState.pos.z - playerState.pos.z;
    float underBallReward =
        exp(-pow(heightDiff - IDEAL_HEIGHT_DIFFERENCE, 2) / (2 * pow(50, 2)));

    // 4. Goal Alignment: Reward for setting up towards the opponent's goal.
    Vec opponentGoal = (player.team == Team::BLUE)
                           ? CommonValues::ORANGE_GOAL_BACK
                           : CommonValues::BLUE_GOAL_BACK;
    Vec ballToGoalDir = (opponentGoal - ballState.pos).Normalized();
    float goalAlignmentReward =
        (ballState.vel.Normalized().Dot(ballToGoalDir) + 1.f) / 2.f;

    // Combine all reward components.
    // By multiplying, we ensure that all conditions must be met to receive a
    // reward.
    float totalReward = popReward * proximityReward * velocityAlignmentReward *
                        underBallReward * goalAlignmentReward;

    if (isnan(totalReward) || isinf(totalReward)) {
      return 0.f;
    }

    return totalReward;
  }
};
class DoubleTapeReward : public Reward {
private:
  // Enum to track the state of a double tap attempt
  enum class AttemptState {
    IDLE,                   // Not currently attempting a double tap
    AWAITING_BACKBOARD_HIT, // Player has hit the ball towards the backboard
    AWAITING_SECOND_TOUCH,  // Ball has hit the backboard, waiting for player's
                            // follow-up
    AWAITING_GOAL // Player has made the follow-up touch, waiting for a goal
  };

  // Struct to hold the state for each player's attempt
  struct DoubleTapeAttempt {
    AttemptState state = AttemptState::IDLE;
    uint64_t last_update_tick = 0;
    uint64_t floor_hit_tick = 0;
  };

  std::vector<DoubleTapeAttempt> player_attempts;
  bool debug;

public:
  /**
   * @brief Construct a new Double Tap Reward object.
   * @param debug_mode If true, will print messages to the console for tracking
   * double tap events.
   */
  DoubleTapeReward(bool debug_mode = true) : debug(debug_mode) {}

  // Called once when the environment is reset.
  virtual void Reset(const GameState &initialState) override {
    player_attempts.clear();
    player_attempts.resize(initialState.players.size());
  }

  // Called for each player every step to calculate their reward.
  virtual float GetReward(const Player &player, const GameState &state,
                          bool isFinal) override {
    if (!state.prev)
      return 0; // Need previous state for comparisons

    auto &attempt = player_attempts[player.index];

    // This is a safe way to get tick_time, even with tick skip.
    float tick_time = (state.lastTickCount > state.prev->lastTickCount)
                          ? (state.deltaTime /
                             (state.lastTickCount - state.prev->lastTickCount))
                          : (1.f / 120.f);

    // Timeout the attempt if too much time has passed (e.g., 5 seconds)
    if (attempt.state != AttemptState::IDLE &&
        (state.lastTickCount - attempt.last_update_tick) * tick_time > 5.0f) {
      attempt = {}; // Reset state
    }

    // Invalidate attempt if another player touches the ball
    if (state.lastTouchCarID != -1 && state.lastTouchCarID != player.carId &&
        attempt.state != AttemptState::IDLE) {
      attempt = {}; // Reset state
    }

    // State machine for the double tap attempt
    switch (attempt.state) {
    case AttemptState::IDLE: {
      if (player.ballTouchedStep) {
        // Heuristic: Is it a potential setup touch? Ball hit high and fast
        // towards the opponent's backboard.
        bool is_blue_team = player.team == Team::BLUE;
        float ball_y_vel = state.ball.vel.y;
        float ball_z_pos = state.ball.pos.z;

        bool heading_to_opp_backboard = (is_blue_team && ball_y_vel > 1000) ||
                                        (!is_blue_team && ball_y_vel < -1000);

        if (heading_to_opp_backboard && ball_z_pos > 500) {
          attempt.state = AttemptState::AWAITING_BACKBOARD_HIT;
          attempt.last_update_tick = state.lastTickCount;
        }
      }
      break;
    }

    case AttemptState::AWAITING_BACKBOARD_HIT: {
      // Check for a backboard rebound
      float ball_y_pos = state.ball.pos.y;
      if (abs(ball_y_pos) >
          CommonValues::BACK_WALL_Y - CommonValues::BALL_RADIUS * 1.5) {
        float prev_ball_y_vel = state.prev->ball.vel.y;
        float cur_ball_y_vel = state.ball.vel.y;

        // Check if y-velocity has flipped sign, indicating a bounce
        if (std::signbit(prev_ball_y_vel) != std::signbit(cur_ball_y_vel)) {
          if (debug) {
            std::cout << "DEBUG: Player " << player.carId
                      << " got a backboard rebound!" << std::endl;
          }
          attempt.state = AttemptState::AWAITING_SECOND_TOUCH;
          attempt.last_update_tick = state.lastTickCount;
        }
      }
      break;
    }

    case AttemptState::AWAITING_SECOND_TOUCH: {
      if (player.ballTouchedStep) {
        // Player made the second touch
        attempt.state = AttemptState::AWAITING_GOAL;
        attempt.last_update_tick = state.lastTickCount;
        attempt.floor_hit_tick = 0; // Reset floor hit grace period timer
      }
      break;
    }

    case AttemptState::AWAITING_GOAL: {
      // Check if ball hits the floor to start the grace period timer
      if (attempt.floor_hit_tick == 0 &&
          state.ball.pos.z < CommonValues::BALL_RADIUS + 15 &&
          state.prev->ball.vel.z < 0) {
        attempt.floor_hit_tick = state.lastTickCount;
      }

      if (state.goalScored) {
        bool scored_on_correct_goal =
            (player.team != RS_TEAM_FROM_Y(state.ball.pos.y));

        if (scored_on_correct_goal && state.lastTouchCarID == player.carId) {

          bool within_grace_period =
              (attempt.floor_hit_tick == 0) ||
              ((state.lastTickCount - attempt.floor_hit_tick) * tick_time <=
               2.0f);

          if (within_grace_period) {
            if (debug) {
              std::cout << "DEBUG: Player " << player.carId
                        << " scored a DOUBLE TAP!" << std::endl;
            }
            attempt = {}; // Reset state for next attempt
            return 1.0f;
          }
        }
      }
      break;
    }
    }

    return 0; // No reward this step
  }
};
class FaceBallReward : public Reward {
public:
  virtual float GetReward(const Player &player, const GameState &state,
                          bool isFinal) {
    Vec dirToBall = (state.ball.pos - player.pos).Normalized();
    return player.rotMat.forward.Dot(dirToBall);
  }
};
class TouchBallReward : public Reward {
public:
  virtual float GetReward(const Player &player, const GameState &state,
                          bool isFinal) {
    return player.ballTouchedStep;
  }
};
class SpeedReward : public Reward {
public:
  virtual float GetReward(const Player &player, const GameState &state,
                          bool isFinal) {
    return player.vel.Length() / CommonValues::CAR_MAX_SPEED;
  }
};
class PickupBoostRewardImproved : public Reward {
private:
  float boostScale = 100.0f;
  float baseReward = 1.0f;
  float minGain = 0.1f;
  bool useMinGain = false;

public:
  PickupBoostRewardImproved() {}

  PickupBoostRewardImproved(float scale, float base, float min = 0.0f,
                            bool enableMin = false)
      : boostScale(scale), baseReward(base), minGain(min),
        useMinGain(enableMin) {}
  float GetReward(const Player &player, const GameState &state,
                  bool isFinal) override {
    if (!player.prev)
      return 0.0f;
    float boostGain = player.boost - player.prev->boost;
    if (boostGain <= 0)
      return 0.0f;
    if (useMinGain && boostGain < minGain) {
      return 0.0f;
    }
    return (boostGain / boostScale) + baseReward;
  }
  void setScale(float scale) { boostScale = scale; }
  void setBase(float base) { baseReward = base; }
  void setMinGain(float min, bool enable = true) {
    minGain = min;
    useMinGain = enable;
  }
};
class PickupBoostReward : public Reward {
public:
  virtual float GetReward(const Player &player, const GameState &state,
                          bool isFinal) override {
    if (!player.prev)
      return 0;

    if (player.boost > player.prev->boost) {
      return sqrtf(player.boost / 100.f) - sqrtf(player.prev->boost / 100.f);
    } else {
      return 0;
    }
  }
};
class SaveBoostReward : public Reward {
public:
  float exponent;
  SaveBoostReward(float exponent = 0.5f) : exponent(exponent) {}

  virtual float GetReward(const Player &player, const GameState &state,
                          bool isFinal) {
    return RS_CLAMP(powf(player.boost / 100, exponent), 0, 1);
  }
};
class DribbleBumpGoalReward : public Reward {
public:
  DribbleBumpGoalReward(
      float window_seconds = 3.0f, // GEÄNDERT: von 2.5f auf 0.8f
      float bump_reward = 100.0f,  // wird jetzt nur für Demos vergeben
      float goal_bonus = 500.0f, float min_ball_speed = 300.0f,
      float min_wall_dist = 400.0f, float max_pitch_deg = 10.0f,
      float dribble_xy_thresh = 150.0f, float dribble_z_min_margin = 20.0f,
      float dribble_z_max_margin = 200.0f, float dribble_grace_seconds = 0.35f,
      float bump_dist = 200.0f,     // wird nicht mehr verwendet
      float min_rel_speed = 300.0f, // wird nicht mehr verwendet
      float bump_cooldown_seconds = 0.3f, bool disable_in_default_state = false,
      int game_tick_rate = 120, int tick_skip = 8);

  void Reset(const GameState &initialState) override;
  float GetReward(const Player &player, const GameState &state,
                  bool isFinal) override;

private:
  // config
  float window_seconds, bump_reward, goal_bonus;
  float min_ball_speed, min_wall_dist, max_pitch_rad;
  float dribble_xy_thresh, dribble_z_min_margin, dribble_z_max_margin;
  int window_steps, grace_steps, bump_cooldown_steps;
  float bump_dist, min_rel_speed; // werden beibehalten für Kompatibilität
  bool disable_in_default_state;

  // runtime
  int _first_pid;
  int _step;
  int _last_blue, _last_orange;
  int _scored_team; // -1 none, 0 blue, 1 orange
  struct PB {
    bool dribbling;
    bool primed;
    bool window_active;
    int window_until;
    int grace_until;
    int demo_cd;
    bool last_demo_state;
  };
  std::unordered_map<int, PB> _p;

  // helpers
  bool _is_dribbling_now(const Player &player, const GameState &state) const;
  // _bump_event wird nicht mehr benötigt
};
static float _local_dist_to_closest_wall(float x, float y) {
  float dist_side_wall = std::fabs(CommonValues::SIDE_WALL_X - std::fabs(x));
  float dist_back_wall = std::fabs(CommonValues::BACK_WALL_Y - std::fabs(y));
  float x1 = CommonValues::SIDE_WALL_X - 1152.0f,
        y1 = CommonValues::BACK_WALL_Y;
  float x2 = CommonValues::SIDE_WALL_X,
        y2 = CommonValues::BACK_WALL_Y - 1152.0f;
  float A = std::fabs(x) - x1;
  float B = std::fabs(y) - y1;
  float C = x2 - x1;
  float D = y2 - y1;
  float dot = A * C + B * D;
  float len_sq = C * C + D * D;
  float param = (len_sq != 0.0f) ? (dot / len_sq) : -1.0f;
  float xx, yy;
  if (param < 0.0f) {
    xx = x1;
    yy = y1;
  } else if (param > 1.0f) {
    xx = x2;
    yy = y2;
  } else {
    xx = x1 + param * C;
    yy = y1 + param * D;
  }
  float dx = std::fabs(x) - xx;
  float dy = std::fabs(y) - yy;
  float dist_corner_wall = std::sqrt(dx * dx + dy * dy);
  return std::min(dist_corner_wall, std::min(dist_side_wall, dist_back_wall));
}
DribbleBumpGoalReward::DribbleBumpGoalReward(
    float window_seconds, float bump_reward_, float goal_bonus_,
    float min_ball_speed_, float min_wall_dist_, float max_pitch_deg,
    float dribble_xy_thresh, float dribble_z_min_margin,
    float dribble_z_max_margin, float dribble_grace_seconds, float bump_dist_,
    float min_rel_speed_, float bump_cooldown_seconds,
    bool disable_in_default_state_, int game_tick_rate, int tick_skip)
    : window_seconds(window_seconds), bump_reward(bump_reward_),
      goal_bonus(goal_bonus_), min_ball_speed(min_ball_speed_),
      min_wall_dist(min_wall_dist_),
      max_pitch_rad(max_pitch_deg * float(M_PI / 180.0f)),
      dribble_xy_thresh(dribble_xy_thresh),
      dribble_z_min_margin(dribble_z_min_margin),
      dribble_z_max_margin(dribble_z_max_margin), bump_dist(bump_dist_),
      min_rel_speed(min_rel_speed_),
      disable_in_default_state(disable_in_default_state_), _first_pid(-1),
      _step(0), _last_blue(0), _last_orange(0), _scored_team(-1) {
  int steps_per_s = std::max(1, game_tick_rate / std::max(1, tick_skip));
  window_steps = std::max(1, int(std::round(window_seconds * steps_per_s)));
  grace_steps =
      std::max(1, int(std::round(dribble_grace_seconds * steps_per_s)));
  bump_cooldown_steps =
      std::max(1, int(std::round(bump_cooldown_seconds * steps_per_s)));
}
void DribbleBumpGoalReward::Reset(const GameState &initialState) {
  _first_pid = -1;
  _step = 0;
  // score tracking: repo's GameState doesn't expose scoreLine; use goalScored
  // flag as baseline
  _last_blue = initialState.goalScored ? 1 : 0;
  _last_orange = 0;
  _scored_team = -1;
  _p.clear();
}
bool DribbleBumpGoalReward::_is_dribbling_now(const Player &player,
                                              const GameState &state) const {
  const Vec &ballpos = state.ball.pos;
  float dz = ballpos.z - player.pos.z;
  bool horiz_ok = (std::fabs(player.pos.x - ballpos.x) < dribble_xy_thresh &&
                   std::fabs(player.pos.y - ballpos.y) < dribble_xy_thresh);
  float ball_speed = state.ball.vel.Length();
  float pitch_ok = std::fabs(player.rotMat.forward.z) <= max_pitch_rad;
  float wall_dist = _local_dist_to_closest_wall(player.pos.x, player.pos.y);
  bool z_ok = ((CommonValues::BALL_RADIUS + dribble_z_min_margin) < dz &&
               dz < (CommonValues::BALL_RADIUS + dribble_z_max_margin));
  return (player.ballTouchedStep && z_ok && horiz_ok &&
          ball_speed >= min_ball_speed && pitch_ok &&
          wall_dist >= min_wall_dist);
}
float DribbleBumpGoalReward::GetReward(const Player &player,
                                       const GameState &state,
                                       bool /*isFinal*/) {
  if (_first_pid == -1 && state.players.size() > 0)
    _first_pid = state.players[0].carId;
  if (player.carId == _first_pid) {
    _step++;
    // detect new goal via goalScored flag and lastTouchCarID (repo doesn't
    // expose scoreLine)
    int cur_goal_flag = state.goalScored ? 1 : 0;
    if (cur_goal_flag > _last_blue) {
      // a goal event occurred since last check; determine scoring team from
      // lastTouchCarID
      int scorerId = state.lastTouchCarID;
      _scored_team = -1;
      for (const auto &p : state.players) {
        if (p.carId == scorerId) {
          _scored_team = (int)p.team;
          break;
        }
      }
    } else if (cur_goal_flag == 0) {
      _scored_team = -1;
    }
    _last_blue = cur_goal_flag;
  }

  auto it =
      _p.try_emplace(player.carId, PB{false, false, false, -1, -1, 0, false})
          .first;
  auto &st = it->second;

  float rew = 0.0f;

  if (_is_dribbling_now(player, state)) {
    st.grace_until = _step + grace_steps;
    st.dribbling = true;
    st.primed = true;
  } else {
    st.dribbling = (_step <= st.grace_until);
  }

  // NUR NOCH DEMO-ERKENNUNG (keine Bump-Logik mehr)
  bool demo_happened = false;
  bool current_demo_state = player.eventState.demo;

  // Demo ist passiert wenn der State von false auf true wechselt
  if (current_demo_state && !st.last_demo_state) {
    demo_happened = true;
  }
  st.last_demo_state = current_demo_state;

  bool demo_now = false;
  if (st.demo_cd > 0) {
    st.demo_cd--;
  } else {
    if (demo_happened) {
      demo_now = true;
      st.demo_cd = bump_cooldown_steps;
    }
  }

  // Fenster öffnet sich nach dem Dribbling
  if ((!st.dribbling) && st.primed && (!st.window_active)) {
    st.window_active = true;
    st.window_until = _step + window_steps; // 0.8 Sekunden Fenster
  }

  // Demo-Reward nur während Dribbling oder im Fenster danach
  if (demo_now && (st.dribbling || st.window_active)) {
    rew += bump_reward; // Demo-Reward
    // Optional: Fenster verlängern nach Demo
    if (st.window_active) {
      st.window_until = _step + window_steps;
    }
  }

  // Goal-Bonus wenn im Fenster
  if (st.window_active) {
    if (_scored_team != -1 && _scored_team == (int)player.team &&
        _step <= st.window_until) {
      rew += goal_bonus;
      st.window_active = false;
      st.primed = false;
    } else if (_step > st.window_until) {
      st.window_active = false;
      st.primed = false;
    }
  }

  return rew;
}
class AirReward : public Reward {
public:
  virtual float GetReward(const Player &player, const GameState &state,
                          bool isFinal) override {
    return !player.isOnGround;
  }
};
class TouchAccelReward : public Reward {
public:
  constexpr static float MAX_REWARDED_BALL_SPEED = RLGC::Math::KPHToVel(110);

  virtual float GetReward(const Player &player, const GameState &state,
                          bool isFinal) override {
    if (!state.prev)
      return 0;

    if (player.ballTouchedStep) {
      float prevSpeedFrac =
          RS_MIN(1, state.prev->ball.vel.Length() / MAX_REWARDED_BALL_SPEED);
      float curSpeedFrac =
          RS_MIN(1, state.ball.vel.Length() / MAX_REWARDED_BALL_SPEED);

      if (curSpeedFrac > prevSpeedFrac) {
        return (curSpeedFrac - prevSpeedFrac);
      } else {
        // Not speeding up the ball so we don't care
        return 0;
      }
    } else {
      return 0;
    }
  }
};
class DribbleeBumpReward : public Reward {
public:
  // Maximum distance from the ball to be considered "dribbling"
  constexpr static float DRIBBLE_DISTANCE_THRESHOLD = 250.f;
  // Maximum number of ticks since the last touch to be considered "recent"
  constexpr static int RECENT_TOUCH_TICK_LIMIT = 15;

  virtual float GetReward(const Player &player, const GameState &state,
                          bool isFinal) override {
    // Check if a bump event occurred for the player in the current step and if
    // they are on the ground
    if (player.eventState.bump && player.isOnGround) {

      // Verify that the player was the last one to touch the ball
      bool was_last_touch = state.lastTouchCarID == player.carId;

      // Check if the touch was recent enough
      bool recently_touched =
          (state.lastTickCount - player.ballHitInfo.tickCountWhenHit) <
          RECENT_TOUCH_TICK_LIMIT;

      // Calculate distance to the ball
      float ball_distance = (player.pos - state.ball.pos).Length();

      // If all dribbling and bump conditions are met, grant a reward
      if (was_last_touch && recently_touched &&
          ball_distance < DRIBBLE_DISTANCE_THRESHOLD) {
        return 1.0f;
      }
    }
    return 0.0f;
  }
};
class TeamSpacingReward : public Reward {
public:
  float minSpacing;

  // minSpacing: Minimum distance between teammates (in Rocket League units)
  // Players get full reward when >= minSpacing apart, penalty when too close
  TeamSpacingReward(float minSpacing = 1500.0f) : minSpacing(minSpacing) {}

  virtual float GetReward(const Player &player, const GameState &state,
                          bool isFinal) override {
    float totalReward = 0.0f;
    int teammateCount = 0;

    // Check spacing with all teammates
    for (const Player &teammate : state.players) {
      // Skip self and opponents
      if (teammate.carId == player.carId || teammate.team != player.team)
        continue;

      teammateCount++;
      float distance = (player.pos - teammate.pos).Length();

      if (distance >= minSpacing) {
        // Good spacing - full reward
        totalReward += 1.0f;
      } else {
        // Too close - linear penalty based on how close they are
        float ratio = distance / minSpacing; // 0 to 1
        totalReward += ratio;                // Linear reward from 0 to 1
      }
    }

    // Return average reward across all teammates (0 if no teammates)
    return teammateCount > 0 ? totalReward / teammateCount : 0.0f;
  }
};
class KickoffWavedashReward : public RLGC::Reward {
private:
  bool is_kickoff;
  bool has_jumped_on_kickoff;
  bool has_touched_ball_after_kickoff_jump;

public:
  KickoffWavedashReward() { Reset(RLGC::GameState()); }

  // Reset is called at the beginning of each episode.
  // We detect a kickoff by checking the initial ball position.
  virtual void Reset(const RLGC::GameState &initialState) override {
    // A standard kickoff starts with the ball near the ground at the center.
    // We check if the ball's height is below 100 UU as a simple heuristic for a
    // kickoff spawn.
    is_kickoff = initialState.ball.pos.z < 100;
    has_jumped_on_kickoff = false;
    has_touched_ball_after_kickoff_jump = false;
  }

  virtual float GetReward(const RLGC::Player &player,
                          const RLGC::GameState &state, bool isFinal) override {
    // If the sequence is no longer a kickoff, or a goal has been scored, do
    // nothing.
    if (!is_kickoff || state.goalScored) {
      is_kickoff = false; // End the sequence if a goal is scored.
      return 0;
    }

    // If another player touches the ball, this player's kickoff sequence is
    // void.
    if (state.lastTouchCarID != -1 && state.lastTouchCarID != player.carId) {
      is_kickoff = false;
      return 0;
    }

    // STATE 1: Player needs to jump during the kickoff.
    if (!has_jumped_on_kickoff && player.hasJumped) {
      has_jumped_on_kickoff = true;
    }

    // STATE 2: Player needs to touch the ball after having jumped.
    if (has_jumped_on_kickoff && !has_touched_ball_after_kickoff_jump &&
        player.ballTouchedStep) {
      has_touched_ball_after_kickoff_jump = true;
    }

    // STATE 3: Player needs to wavedash after touching the ball.
    if (has_touched_ball_after_kickoff_jump) {
      bool is_wavedash = false;
      if (player.prev) {
        // A wavedash is a flip into the ground.
        // This logic is adapted from the WaveDashRewardV2.
        if (player.isOnGround && player.prev->isFlipping &&
            !player.prev->isOnGround) {
          is_wavedash = true;
        }
      }

      if (is_wavedash) {
        // Sequence complete, give reward and end the sequence for this episode.
        is_kickoff = false;
        return 1.0f;
      }
    }

    return 0;
  }
};
class DribbleReward : public Reward {
public:
  float minBallHeight;
  float maxBallHeight;
  float maxDistance;
  float coeff;

  // Based on SwiftGroundDribbleReward - rewards speed matching between player
  // and ball during dribbles
  DribbleReward(float minBallHeight = 109.0f, float maxBallHeight = 180.0f,
                float maxDistance = 197.0f, float coeff = 2.0f)
      : minBallHeight(minBallHeight), maxBallHeight(maxBallHeight),
        maxDistance(maxDistance), coeff(coeff) {}

  virtual float GetReward(const Player &player, const GameState &state,
                          bool isFinal) override {
    // Check all dribbling conditions
    if (!player.isOnGround)
      return 0.0f;
    if (state.ball.pos.z < minBallHeight || state.ball.pos.z > maxBallHeight)
      return 0.0f;
    if ((player.pos - state.ball.pos).Length() >= maxDistance)
      return 0.0f;

    // Calculate speed reward based on player-ball speed matching
    float playerSpeed = player.vel.Length();
    float ballSpeed = state.ball.vel.Length();
    float playerSpeedNormalized = playerSpeed / CommonValues::CAR_MAX_SPEED;
    float inverseDifference = 1.0f - abs(playerSpeed - ballSpeed);
    float twoSum = playerSpeed + ballSpeed;

    // Avoid division by zero
    if (twoSum == 0.0f)
      return 0.0f;

    float speedReward =
        playerSpeedNormalized + coeff * (inverseDifference / twoSum);
    return speedReward;
  }
};
class FlickReward : public Reward {
public:
  float minFlickSpeed;

  // minFlickSpeed: Minimum ball speed to count as a flick
  FlickReward(float minFlickSpeed = 800.0f) : minFlickSpeed(minFlickSpeed) {}

  virtual float GetReward(const Player &player, const GameState &state,
                          bool isFinal) override {
    if (!state.prev || !player.ballTouchedStep)
      return 0.0f;

    // Check if this was a flick (ball was low, now going fast and upward)
    bool ballWasLow = state.prev->ball.pos.z < 200.0f;
    bool ballNowFast = state.ball.vel.Length() > minFlickSpeed;
    bool ballGoingUp = state.ball.vel.z > 200.0f;

    // Player should have been close to ball when touching
    float prevDistance = (player.pos - state.prev->ball.pos).Length();
    bool playerWasClose = prevDistance < 200.0f;

    if (ballWasLow && ballNowFast && ballGoingUp && playerWasClose) {
      // Reward based on ball speed after flick
      float speedRatio = RS_MIN(1.0f, state.ball.vel.Length() / 2000.0f);
      return speedRatio;
    }

    return 0.0f;
  }
};
class KickoffProximityReward2v22
    : public Reward { // zerosum for NvN, good for any kickoff mode
public:
  KickoffProximityReward2v22() : Reward() {}

  virtual float GetReward(const Player &player, const GameState &state,
                          bool isFinal) override {
    // Check if the ball is still near its kickoff position and not moving much
    if (state.ball.vel.Length() < 1.f) {
      float playerDistToBall = (player.pos - state.ball.pos).Length();

      std::vector<float> opponentDistances;
      for (auto &p : state.players) {
        // Collect distances only for players on the opposing team
        if (p.team != player.team) {
          float opponentDistToBall = (p.pos - state.ball.pos).Length();
          opponentDistances.push_back(opponentDistToBall);
        }
      }

      if (opponentDistances.empty()) {
        // Should not happen in a standard game, but good for safety
        return 0.f;
      }

      // Get the distance of the closest opponent to the ball
      float closestOpponentDist =
          *std::min_element(opponentDistances.begin(), opponentDistances.end());

      // The player gets a positive reward if they are closer to the ball than
      // the closest opponent
      if (playerDistToBall < closestOpponentDist) {
        return 1.0f;
      } else {
        // Otherwise, they get a negative reward (zerosum)
        return -1.0f;
      }
    }

    // If the ball is moving (kickoff has started/ended), the reward is zero
    return 0.0f;
  }
};
class KickoffProximityReward2v2Enhanced : public Reward {
public:
  float goerReward = 1.2f;         // Increased base reward for goer
  float cheaterReward = 0.6f;      // Base reward for strategic cheater
  float dynamicWeight = 0.3f;      // Weight for dynamic adjustments
  float rotationPrepWeight = 0.2f; // Weight for rotation preparation

  virtual float GetReward(const Player &player, const GameState &state,
                          bool isFinal) override {
    // Enhanced kickoff detection - more robust
    if (!IsKickoffActive(state))
      return 0.f;

    float playerDistToBall = (player.pos - state.ball.pos).Length();

    // Enhanced team analysis
    TeamAnalysis analysis = AnalyzeTeamState(player, state);
    if (!analysis.hasTeammate)
      return 0.f;

    // Dynamic role assignment with multiple factors
    PlayerRole role = DeterminePlayerRole(player, analysis, state);

    if (role == PlayerRole::GOER) {
      return CalculateGoerReward(player, analysis, state);
    } else {
      return CalculateCheaterReward(player, analysis, state);
    }
  }

private:
  enum class PlayerRole { GOER, CHEATER };

  struct TeamAnalysis {
    bool hasTeammate = false;
    const Player *teammate = nullptr;
    float teammateDistToBall = 0.f;
    float closestOpponentDist = FLT_MAX;
    float secondOpponentDist = FLT_MAX;
    Vec opponentCenterOfMass =
        Vec(0.f, 0.f, 0.f); // Fixed: explicit float construction
    float avgOpponentSpeed = 0.f;
  };

  bool IsKickoffActive(const GameState &state) {
    // More sophisticated kickoff detection
    float ballSpeed = state.ball.vel.Length();
    float ballHeight = state.ball.pos.z;
    Vec ballPos2D = Vec(state.ball.pos.x, state.ball.pos.y,
                        0.f); // Fixed: explicit float for z

    return (ballSpeed < 2.f && ballHeight < 150.f && ballPos2D.Length() < 50.f);
  }

  TeamAnalysis AnalyzeTeamState(const Player &player, const GameState &state) {
    TeamAnalysis analysis;
    int opponentCount = 0;
    float totalOpponentSpeed = 0.f;

    for (const auto &p : state.players) {
      if (p.team == player.team && p.carId != player.carId) {
        analysis.teammate = &p;
        analysis.hasTeammate = true;
        analysis.teammateDistToBall = (p.pos - state.ball.pos).Length();
      } else if (p.team != player.team) {
        float opponentDist = (p.pos - state.ball.pos).Length();
        totalOpponentSpeed += p.vel.Length();
        opponentCount++;

        if (opponentDist < analysis.closestOpponentDist) {
          analysis.secondOpponentDist = analysis.closestOpponentDist;
          analysis.closestOpponentDist = opponentDist;
        } else if (opponentDist < analysis.secondOpponentDist) {
          analysis.secondOpponentDist = opponentDist;
        }

        analysis.opponentCenterOfMass =
            analysis.opponentCenterOfMass + p.pos; // Fixed: use + instead of +=
      }
    }

    if (opponentCount > 0) {
      float countFloat = (float)opponentCount; // Fixed: explicit cast
      analysis.opponentCenterOfMass = analysis.opponentCenterOfMass /
                                      countFloat; // Fixed: explicit division
      analysis.avgOpponentSpeed = totalOpponentSpeed / countFloat;
    }

    return analysis;
  }

  PlayerRole DeterminePlayerRole(const Player &player,
                                 const TeamAnalysis &analysis,
                                 const GameState &state) {
    float playerDistToBall = (player.pos - state.ball.pos).Length();

    // Factor 1: Distance to ball (40% weight)
    float distanceScore =
        (playerDistToBall < analysis.teammateDistToBall) ? 0.4f : 0.f;

    // Factor 2: Speed toward ball (30% weight) - Fixed type issues
    Vec playerToBall = (state.ball.pos - player.pos).Normalized();
    Vec teammateToBall = (state.ball.pos - analysis.teammate->pos).Normalized();
    float playerVelToBall = player.vel.Dot(playerToBall);
    float teammateVelToBall = analysis.teammate->vel.Dot(teammateToBall);
    float speedScore = (playerVelToBall > teammateVelToBall) ? 0.3f : 0.f;

    // Factor 3: Boost level consideration (20% weight)
    float boostScore =
        (player.boost > analysis.teammate->boost + 10.f) ? 0.2f : 0.f;

    // Factor 4: Spawn position advantage (10% weight)
    float spawnScore =
        CalculateSpawnAdvantage(player, *analysis.teammate, state) * 0.1f;

    float totalScore = distanceScore + speedScore + boostScore + spawnScore;

    return (totalScore >= 0.5f) ? PlayerRole::GOER : PlayerRole::CHEATER;
  }

  float CalculateSpawnAdvantage(const Player &player, const Player &teammate,
                                const GameState &state) {
    // Advantage based on diagonal vs straight kickoff positioning
    float playerAngleToBall = atan2f(player.pos.y - state.ball.pos.y,
                                     player.pos.x - state.ball.pos.x);
    float teammateAngleToBall = atan2f(teammate.pos.y - state.ball.pos.y,
                                       teammate.pos.x - state.ball.pos.x);

    float angleDiff = fabsf(playerAngleToBall - teammateAngleToBall);

    // Diagonal spawns (corner positions) have advantage for going
    return (angleDiff > (3.14159f / 3.f))
               ? 1.f
               : 0.f; // Fixed: use explicit float for PI
  }

  float CalculateGoerReward(const Player &player, const TeamAnalysis &analysis,
                            const GameState &state) {
    float playerDistToBall = (player.pos - state.ball.pos).Length();

    // Base reward for being closer than opponents
    float baseReward = (playerDistToBall < analysis.closestOpponentDist)
                           ? goerReward
                           : -goerReward * 0.5f;

    // Speed differential bonus - Fixed type issues
    Vec playerToBall = (state.ball.pos - player.pos).Normalized();
    float playerVelToBall = player.vel.Dot(playerToBall);
    float speedBonus =
        RS_CLAMP(playerVelToBall / 2300.f, -0.3f, 0.3f); // Max car speed ~2300

    // Boost usage efficiency (penalize waste, reward conservation for crucial
    // moments)
    float boostEfficiency = 0.f;
    if (player.boost > 50.f && playerDistToBall > 1000.f) {
      boostEfficiency = 0.1f; // Good boost management
    } else if (player.boost < 20.f && playerDistToBall > 800.f) {
      boostEfficiency = -0.15f; // Poor boost management
    }

    // Angle approach bonus (reward straight-line approaches)
    Vec toBall = (state.ball.pos - player.pos).Normalized();
    Vec velocity = player.vel.Normalized();
    float approachAngle = toBall.Dot(velocity);
    float angleBonus = RS_MAX(0.f, approachAngle) * 0.2f;

    return RS_CLAMP(baseReward + speedBonus + boostEfficiency + angleBonus,
                    -1.5f, 1.5f);
  }

  float CalculateCheaterReward(const Player &player,
                               const TeamAnalysis &analysis,
                               const GameState &state) {
    Vec ownGoal = (player.team == Team::BLUE) ? CommonValues::BLUE_GOAL_BACK
                                              : CommonValues::ORANGE_GOAL_BACK;

    // Dynamic ideal position based on game state
    Vec idealPos =
        CalculateDynamicIdealPosition(player, analysis, state, ownGoal);
    float distToIdeal = (player.pos - idealPos).Length();

    // COMPONENT 1: Dynamic positioning (40% weight)
    float positioningReward =
        CalculatePositioningReward(player, idealPos, distToIdeal);

    // COMPONENT 2: Strategic boost management (25% weight)
    float boostReward =
        CalculateStrategicBoostReward(player, state, analysis) * 0.25f;

    // COMPONENT 3: Rotation preparation (20% weight)
    float rotationReward =
        CalculateRotationPreparation(player, analysis, state) *
        rotationPrepWeight;

    // COMPONENT 4: Opponent awareness (10% weight)
    float awarenessReward =
        CalculateOpponentAwareness(player, analysis, state) * 0.1f;

    // COMPONENT 5: Anti-camping with dynamic threshold (5% weight)
    float campingPenalty =
        CalculateDynamicCampingPenalty(player, ownGoal, state) * 0.05f;

    float totalReward = positioningReward + boostReward + rotationReward +
                        awarenessReward + campingPenalty;

    return RS_CLAMP(totalReward, -0.8f, 0.8f);
  }

  Vec CalculateDynamicIdealPosition(const Player &player,
                                    const TeamAnalysis &analysis,
                                    const GameState &state,
                                    const Vec &ownGoal) {
    Vec fieldCenter = Vec(0.f, 0.f, 100.f); // Fixed: explicit floats

    // Base position: 65% toward center from goal (slightly more aggressive than
    // original)
    Vec centerMultiplied =
        Vec(fieldCenter.x * 1.3f, fieldCenter.y * 1.3f,
            fieldCenter.z * 1.3f); // Fixed: manual multiplication
    Vec baseIdeal = (ownGoal + centerMultiplied) * 0.5f;

    // Adjust based on opponent positioning
    Vec opponentThreatVector =
        (analysis.opponentCenterOfMass - ownGoal).Normalized();
    opponentThreatVector =
        Vec(opponentThreatVector.x * 200.f, opponentThreatVector.y * 200.f,
            opponentThreatVector.z * 200.f); // Fixed: manual scaling

    // Adjust based on teammate position (create optimal spacing)
    Vec teammateOffset = Vec(0.f, 0.f, 0.f); // Fixed: explicit floats
    if (analysis.teammate) {
      Vec teammatePos = analysis.teammate->pos;
      float teammateDistFromCenter = (teammatePos - fieldCenter).Length();

      // If teammate is far from center, position closer to support
      if (teammateDistFromCenter > 1500.f) {
        Vec direction = (teammatePos - baseIdeal).Normalized();
        teammateOffset = Vec(direction.x * 300.f, direction.y * 300.f,
                             direction.z * 300.f); // Fixed: manual scaling
      }
    }

    // Final position with adjustments - Fixed: manual scaling
    Vec threatAdjustment =
        Vec(opponentThreatVector.x * 0.3f, opponentThreatVector.y * 0.3f,
            opponentThreatVector.z * 0.3f);
    Vec teammateAdjustment =
        Vec(teammateOffset.x * 0.2f, teammateOffset.y * 0.2f,
            teammateOffset.z * 0.2f);
    Vec adjustedIdeal = baseIdeal + threatAdjustment + teammateAdjustment;

    // Clamp to reasonable field boundaries
    adjustedIdeal.x = RS_CLAMP(adjustedIdeal.x, -3000.f, 3000.f);
    adjustedIdeal.y = RS_CLAMP(adjustedIdeal.y, -4000.f, 4000.f);
    adjustedIdeal.z = RS_MAX(adjustedIdeal.z, 17.f); // Ground level

    return adjustedIdeal;
  }

  float CalculatePositioningReward(const Player &player, const Vec &idealPos,
                                   float distToIdeal) {
    float optimalRadius = 600.f;
    float acceptableRadius = 1200.f;
    float maxRadius = 2000.f;

    if (distToIdeal <= optimalRadius) {
      // Excellent positioning
      return 0.5f * (1.f - (distToIdeal / optimalRadius));
    } else if (distToIdeal <= acceptableRadius) {
      // Good positioning with gradual falloff
      float ratio =
          (distToIdeal - optimalRadius) / (acceptableRadius - optimalRadius);
      return 0.5f * (1.f - ratio) * 0.7f;
    } else if (distToIdeal <= maxRadius) {
      // Poor but acceptable positioning
      float ratio =
          (distToIdeal - acceptableRadius) / (maxRadius - acceptableRadius);
      return -0.1f * ratio;
    } else {
      // Very poor positioning
      return -0.3f;
    }
  }

  float CalculateStrategicBoostReward(const Player &player,
                                      const GameState &state,
                                      const TeamAnalysis &analysis) {
    // Find strategically important boost pads
    float bestBoostValue = 0.f;

    for (int i = 0; i < CommonValues::BOOST_LOCATIONS_AMOUNT; i++) {
      const Vec &boostPos = CommonValues::BOOST_LOCATIONS[i];

      if (boostPos.z > 72.0f) { // Large boost pad
        float distToBoost = (player.pos - boostPos).Length();

        // Value boost based on multiple factors
        float accessibility = 1.f - RS_CLAMP(distToBoost / 1500.f, 0.f, 1.f);
        float strategicValue =
            CalculateBoostStrategicValue(boostPos, analysis, state);
        float denyValue = CalculateBoostDenialValue(boostPos, analysis);

        float totalValue = accessibility * (strategicValue + denyValue);
        bestBoostValue = RS_MAX(bestBoostValue, totalValue);
      }
    }

    // Boost level consideration
    float boostLevelFactor = 1.f;
    if (player.boost < 30.f) {
      boostLevelFactor = 1.5f; // More urgent need for boost
    } else if (player.boost > 80.f) {
      boostLevelFactor = 0.5f; // Less urgent need
    }

    return bestBoostValue * boostLevelFactor;
  }

  float CalculateBoostStrategicValue(const Vec &boostPos,
                                     const TeamAnalysis &analysis,
                                     const GameState &state) {
    // Boost pads closer to expected ball trajectory are more valuable
    Vec ballToBoost = (boostPos - state.ball.pos);
    float distToBall = ballToBoost.Length();

    // Corner boosts are generally more valuable for rotations
    bool isCornerBoost =
        (fabsf(boostPos.x) > 2500.f && fabsf(boostPos.y) > 3500.f);

    float baseValue = isCornerBoost ? 0.8f : 0.6f;
    float proximityValue = 1.f - RS_CLAMP(distToBall / 3000.f, 0.f, 1.f);

    return baseValue * (0.3f + proximityValue * 0.7f);
  }

  float CalculateBoostDenialValue(const Vec &boostPos,
                                  const TeamAnalysis &analysis) {
    // Value boost based on opponent accessibility
    float opponentDistToBoost =
        (analysis.opponentCenterOfMass - boostPos).Length();

    return RS_CLAMP(1.f - (opponentDistToBoost / 2000.f), 0.f, 0.3f);
  }

  float CalculateRotationPreparation(const Player &player,
                                     const TeamAnalysis &analysis,
                                     const GameState &state) {
    if (!analysis.teammate)
      return 0.f;

    // Reward positioning that allows quick transition to support teammate
    Vec teammatePos = analysis.teammate->pos;
    Vec supportPosition = CalculateOptimalSupportPosition(
        teammatePos, state.ball.pos, player.team);

    float distToSupport = (player.pos - supportPosition).Length();
    float supportReadiness = 1.f - RS_CLAMP(distToSupport / 1000.f, 0.f, 1.f);

    // Velocity alignment for quick rotation
    Vec toSupport = (supportPosition - player.pos).Normalized();
    float velocityAlignment =
        RS_MAX(0.f, player.vel.Normalized().Dot(toSupport));

    return (supportReadiness * 0.7f + velocityAlignment * 0.3f);
  }

  Vec CalculateOptimalSupportPosition(const Vec &teammatePos,
                                      const Vec &ballPos, Team team) {
    Vec ownGoal = (team == Team::BLUE) ? CommonValues::BLUE_GOAL_BACK
                                       : CommonValues::ORANGE_GOAL_BACK;

    // Position that forms good triangle with teammate and goal
    Vec teammateToGoal = (ownGoal - teammatePos).Normalized();
    Vec perpendicular = Vec(-teammateToGoal.y, teammateToGoal.x, 0.f)
                            .Normalized(); // Fixed: explicit float

    // Fixed: manual vector arithmetic
    Vec goalOffset = Vec(teammateToGoal.x * 800.f, teammateToGoal.y * 800.f,
                         teammateToGoal.z * 800.f);
    Vec perpOffset = Vec(perpendicular.x * 600.f, perpendicular.y * 600.f,
                         perpendicular.z * 600.f);
    Vec supportPos = teammatePos + goalOffset + perpOffset;

    return supportPos;
  }

  float CalculateOpponentAwareness(const Player &player,
                                   const TeamAnalysis &analysis,
                                   const GameState &state) {
    // Reward positioning that maintains good sight lines to opponents
    Vec playerToOpponentCenter =
        (analysis.opponentCenterOfMass - player.pos).Normalized();
    Vec playerToBall = (state.ball.pos - player.pos).Normalized();

    float awarenessAngle = playerToOpponentCenter.Dot(playerToBall);

    // Good awareness when you can see both ball and opponents
    return RS_CLAMP(awarenessAngle * 0.5f + 0.5f, 0.f, 1.f);
  }

  float CalculateDynamicCampingPenalty(const Player &player, const Vec &ownGoal,
                                       const GameState &state) {
    float distToGoal = (player.pos - ownGoal).Length();

    // Dynamic threshold based on game state
    float minDistFromGoal = 800.f; // Base minimum

    // Adjust based on ball position
    float ballDistFromGoal = (state.ball.pos - ownGoal).Length();
    if (ballDistFromGoal < 2000.f) {
      minDistFromGoal *= 0.7f; // Allow closer positioning when ball is near
    }

    if (distToGoal < minDistFromGoal) {
      float penalty = -0.4f * (1.f - (distToGoal / minDistFromGoal));
      return penalty;
    }

    return 0.f;
  }
};
class KickoffProximityRewardV2 : public Reward {
public:
  virtual float GetReward(const Player &player, const GameState &state,
                          bool isFinal) override {
    // Check if ball is at kickoff position (0, 0, z)
    if (abs(state.ball.pos.x) > 1.0f || abs(state.ball.pos.y) > 1.0f) {
      return 0.0f; // Not a kickoff situation
    }

    // Calculate player's distance to ball
    float playerDistToBall = (player.pos - state.ball.pos).Length();

    // Find closest opponent distance to ball
    float closestOpponentDist = FLT_MAX;
    bool foundOpponent = false;

    for (const Player &opponent : state.players) {
      // Skip teammates and self
      if (opponent.team == player.team || opponent.carId == player.carId) {
        continue;
      }

      foundOpponent = true;
      float opponentDistToBall = (opponent.pos - state.ball.pos).Length();

      if (opponentDistToBall < closestOpponentDist) {
        closestOpponentDist = opponentDistToBall;
      }
    }

    // If no opponents found, return neutral
    if (!foundOpponent) {
      return 0.0f;
    }

    // Return 1 if player is closer, -1 if opponent is closer
    return (playerDistToBall < closestOpponentDist) ? 1.0f : -1.0f;
  }
};
class StrongTouchReward : public Reward {
public:
  float minRewardedVel, maxRewardedVel;
  StrongTouchReward(float minSpeedKPH = 20, float maxSpeedKPH = 130) {
    minRewardedVel = RLGC::Math::KPHToVel(minSpeedKPH);
    maxRewardedVel = RLGC::Math::KPHToVel(maxSpeedKPH);
  }

  virtual float GetReward(const Player &player, const GameState &state,
                          bool isFinal) override {
    if (!state.prev)
      return 0;

    if (player.ballTouchedStep) {
      float hitForce = (state.ball.vel - state.prev->ball.vel).Length();
      if (hitForce < minRewardedVel)
        return 0;

      return RS_MIN(1, hitForce / maxRewardedVel);
    } else {
      return 0;
    }
  }
};
class FRReward : public Reward {
public:
  float maxDist;
  float minHeight;
  float rewardScale;
  float wallMargin;

  FRReward(float maxDist = 300.0f, float minHeight = 1000.0f,
           float rewardScale = 1.0f, float wallMargin = 150.0f)
      : maxDist(maxDist), minHeight(minHeight), rewardScale(rewardScale),
        wallMargin(wallMargin) {}

  float CalculateCloseness(const Player &player, const BallState &ball) {
    // Gate 1: No closeness score if the ball is too low
    if (ball.pos.z < minHeight)
      return 0.0f;

    float dist = (player.pos - ball.pos).Length();

    // Gate 2: No closeness score if the car is too far from the ball
    if (dist > maxDist)
      return 0.0f;

    Vec carDown = -player.rotMat.up;
    Vec dirToBall = (ball.pos - player.pos).Normalized();

    // Gate 3: Alignment check (must be pointing down toward the ball)
    float alignment = carDown.Dot(dirToBall);

    if (alignment <= 0)
      return 0.0f;

    // Continuous score calculation (closer distance = higher score)
    float distScore = 1.0f - (dist / maxDist);

    return distScore * alignment;
  }

  virtual float GetReward(const Player &player, const GameState &state,
                          bool isFinal) override {
    if (!player.prev || !state.prev)
      return 0.0f;

    // 🛑 CRITICAL GATE: Block reward if the agent already has a usable
    // flip/jump.
    if (player.HasFlipOrJump())
      return 0.0f;

    if (player.isOnGround)
      return 0.0f;

    // Wall Check as a GATE (No reward if car is too close to walls)
    using namespace RLGC::CommonValues;
    if (std::abs(player.pos.x) >= (SIDE_WALL_X - wallMargin) ||
        std::abs(player.pos.y) >= (BACK_WALL_Y - wallMargin)) {
      return 0.0f;
    }

    // Calculate how much the agent's closeness/alignment has improved since the
    // last tick
    float currentCloseness = CalculateCloseness(player, state.ball);
    float prevCloseness = CalculateCloseness(*player.prev, state.prev->ball);

    // Reward is only given for IMPROVEMENT (RS_MAX(0.0f, ...))
    float reward = RS_MAX(0.0f, currentCloseness - prevCloseness);

    return reward * rewardScale;
  }
};

class DribbleBumpReward : public Reward {
public:
  float dribbleDistance;
  float carBallHeightDiff;
  float maxTimeSinceDribble;
  float baseReward;
  float speedBonus;

  std::unordered_map<uint32_t, bool> isDribbling;
  std::unordered_map<uint32_t, float> timeSinceDribble;
  std::unordered_map<uint32_t, float> lastDribbleSpeed;

  DribbleBumpReward(float dribbleDistance = 197.0f,
                    float carBallHeightDiff = 110.0f,
                    float maxTimeSinceDribble = 2.5f, float baseReward = 1.0f,
                    float speedBonus = 0.5f)
      : dribbleDistance(dribbleDistance), carBallHeightDiff(carBallHeightDiff),
        maxTimeSinceDribble(maxTimeSinceDribble), baseReward(baseReward),
        speedBonus(speedBonus) {}

  virtual void Reset(const GameState &initialState) override {
    isDribbling.clear();
    timeSinceDribble.clear();
    lastDribbleSpeed.clear();

    for (const auto &player : initialState.players) {
      isDribbling[player.carId] = false;
      timeSinceDribble[player.carId] = 999.0f;
      lastDribbleSpeed[player.carId] = 0.0f;
    }
  }

  virtual void PreStep(const GameState &state) override {
    for (const auto &player : state.players) {
      uint32_t carId = player.carId;

      Vec ballCarVec = state.ball.pos - player.pos;
      bool isCurrentlyDribbling =
          (ballCarVec.Length() < dribbleDistance &&
           state.ball.pos.z > (player.pos.z + carBallHeightDiff));

      if (isDribbling[carId] && !isCurrentlyDribbling) {
        timeSinceDribble[carId] = 0.0f;
        lastDribbleSpeed[carId] = player.vel.Length();
      } else if (isCurrentlyDribbling) {
        timeSinceDribble[carId] = 999.0f;
      } else if (timeSinceDribble[carId] < 999.0f) {
        timeSinceDribble[carId] += state.deltaTime;
      }

      isDribbling[carId] = isCurrentlyDribbling;
    }
  }

  virtual float GetReward(const Player &player, const GameState &state,
                          bool isFinal) override {
    uint32_t carId = player.carId;

    if (!player.eventState.bump) {
      return 0.0f;
    }

    if (timeSinceDribble[carId] > maxTimeSinceDribble) {
      return 0.0f;
    }

    float reward = baseReward;

    float timeFactor = 1.0f - (timeSinceDribble[carId] / maxTimeSinceDribble);
    reward *= (1.0f + timeFactor * 0.5f);

    float speedFactor = lastDribbleSpeed[carId] / CommonValues::CAR_MAX_SPEED;
    reward *= (1.0f + speedFactor * speedBonus);

    float ballDistance = (state.ball.pos - player.pos).Length();
    if (ballDistance < dribbleDistance * 1.5f) {
      reward *= 1.25f;
    }

    if (player.eventState.demo) {
      reward *= 1.5f;
    }

    return RS_CLAMP(reward, 0.0f, 3.0f);
  }
};
inline float cosine_similarity(const Vec &a, const Vec &b) {
  float dotProduct = a.Dot(b);
  float lenA = a.Length();
  float lenB = b.Length();
  if (lenA == 0 || lenB == 0)
    return 0.0f;
  return dotProduct / (lenA * lenB);
}

inline Vec Normalize(const Vec &v) {
  float len = v.Length();
  if (len == 0)
    return v;
  return v / len;
}

inline float Dot(const Vec &a, const Vec &b) { return a.Dot(b); }

class AirDRB_v3 : public Reward {
public:
  struct DribbleSession {
    bool active;
    float startTime;
    float lastTouchTime;
    float maxBallHeight;
    float startAlong;
    float bestAlong;
    float pendingReward;
    float lastEndTime;
    float lastQuality;

    DribbleSession()
        : active(false), startTime(0.0f), lastTouchTime(0.0f),
          maxBallHeight(0.0f), startAlong(0.0f), bestAlong(0.0f),
          pendingReward(0.0f), lastEndTime(-1.0f), lastQuality(0.0f) {}
  };

  float elapsedTime;
  std::unordered_map<int, DribbleSession> sessions;

  AirDRB_v3() : elapsedTime(0.0f) {}

  virtual void Reset(const GameState &initialState) override {
    elapsedTime = 0.0f;
    sessions.clear();
    for (const auto &p : initialState.players) {
      sessions[p.carId] = DribbleSession();
    }
  }

  virtual void PreStep(const GameState &s) override {
    float dt = s.deltaTime;
    if (dt <= 0.0f)
      dt = CommonValues::TICK_TIME;
    elapsedTime += dt;

    Vec blueGoal = CommonValues::BLUE_GOAL_BACK;
    Vec orangeGoal = CommonValues::ORANGE_GOAL_BACK;
    float fieldLenGlobal = (orangeGoal - blueGoal).Length();
    if (fieldLenGlobal <= 0.0f)
      return;

    for (const auto &p : s.players) {
      DribbleSession &ses = sessions[p.carId];

      Vec ownGoal = (p.team == Team::BLUE) ? blueGoal : orangeGoal;
      Vec enemyGoal = (p.team == Team::BLUE) ? orangeGoal : blueGoal;
      Vec axis = enemyGoal - ownGoal;
      float axisLen = axis.Length();
      if (axisLen <= 0.0f)
        continue;
      Vec dirToGoal = axis / axisLen;

      float along = dirToGoal.Dot(s.ball.pos - ownGoal);
      float u = along / axisLen;

      Vec diff = s.ball.pos - p.pos;
      float dist = diff.Length();
      float verticalDiff = s.ball.pos.z - p.pos.z;

      const float minBallZ = 400.0f;
      const float maxBallZ = 2000.0f;
      const float maxDistStart = 350.0f;
      const float maxDistActive = 450.0f;
      const float minVerticalAbove = 60.0f;
      const float maxNoTouchTime = 0.8f;
      const float minSessionDuration = 0.25f;

      bool candidateTouch = p.ballTouchedStep && !p.isOnGround &&
                            s.ball.pos.z > minBallZ &&
                            verticalDiff > minVerticalAbove &&
                            dist < maxDistStart && u > 0.15f && u < 1.1f;

      if (candidateTouch) {
        if (!ses.active) {
          ses.active = true;
          ses.startTime = elapsedTime;
          ses.startAlong = along;
          ses.bestAlong = along;
          ses.maxBallHeight = s.ball.pos.z;
        }
        ses.lastTouchTime = elapsedTime;
        if (s.ball.pos.z > ses.maxBallHeight)
          ses.maxBallHeight = s.ball.pos.z;
        if (along > ses.bestAlong)
          ses.bestAlong = along;
      }

      if (ses.active) {
        float timeSinceTouch = elapsedTime - ses.lastTouchTime;
        bool stillClose = dist < maxDistActive &&
                          s.ball.pos.z > minBallZ * 0.75f && u > 0.10f &&
                          u < 1.2f;

        if (timeSinceTouch > maxNoTouchTime || !stillClose) {
          float duration = ses.lastTouchTime - ses.startTime;
          if (duration >= minSessionDuration) {
            float forwardGain = (ses.bestAlong - ses.startAlong) / axisLen;
            forwardGain = RS_CLAMP(forwardGain, 0.0f, 1.0f);

            float h =
                RS_CLAMP((ses.maxBallHeight - minBallZ) / (maxBallZ - minBallZ),
                         0.0f, 1.0f);

            float quality = forwardGain * h;
            if (quality > 0.0f) {
              float baseReward = quality;
              ses.pendingReward += baseReward;
              if (ses.pendingReward > 1.0f)
                ses.pendingReward = 1.0f;
              ses.lastEndTime = elapsedTime;
              ses.lastQuality = quality;
            }
          }
          ses.active = false;
          ses.startTime = 0.0f;
          ses.lastTouchTime = 0.0f;
          ses.maxBallHeight = 0.0f;
          ses.startAlong = 0.0f;
          ses.bestAlong = 0.0f;
        }
      }
    }
  }

  virtual float GetReward(const Player &p, const GameState &s,
                          bool isFinal) override {
    (void)isFinal;
    auto it = sessions.find(p.carId);
    if (it == sessions.end())
      return 0.0f;
    DribbleSession &ses = it->second;

    float r = 0.0f;

    if (ses.pendingReward > 0.0f) {
      r += ses.pendingReward;
      ses.pendingReward = 0.0f;
    }

    if (p.eventState.goal) {
      Vec blueGoal = CommonValues::BLUE_GOAL_BACK;
      Vec orangeGoal = CommonValues::ORANGE_GOAL_BACK;
      Vec ownGoal = (p.team == Team::BLUE) ? blueGoal : orangeGoal;
      Vec enemyGoal = (p.team == Team::BLUE) ? orangeGoal : blueGoal;
      float axisLen = (enemyGoal - ownGoal).Length();
      if (axisLen > 0.0f) {
        bool wasRecentDribble =
            (ses.active && ses.maxBallHeight > 0.0f) ||
            (ses.lastEndTime > 0.0f && (elapsedTime - ses.lastEndTime) < 1.0f &&
             ses.lastQuality > 0.0f);

        if (wasRecentDribble) {
          float heightFactor = RS_CLAMP(s.ball.pos.z / 2000.0f, 0.0f, 1.0f);
          float goalBonus = 1.0f + heightFactor;
          r += goalBonus;

          ses.lastEndTime = -1.0f;
          ses.lastQuality = 0.0f;
          ses.active = false;
        }
      }
    }

    return r;
  }
};

// --- merged reward classes from other files ---
class WaveDashRewardV2 : public Reward {
public:
  WaveDashRewardV2() = default;

  virtual void Reset(const GameState &initialState) override {
    playerData.clear();
    for (const auto &player : initialState.players) {
      playerData[player.carId] = {0, 0, false};
    }
  }

  virtual float GetReward(const Player &player, const GameState &state,
                          bool isFinal) override {
    const float FLIP_TORQUE_TIME = 0.65f;
    const float DOUBLEJUMP_MAX_DELAY = 1.25f;
    const float MAX_FORWARD_SPEED = CommonValues::CAR_MAX_SPEED;

    float reward = 0.0f;
    auto &playerInfo = playerData[player.carId];

    if (player.pos.z < 100 && player.isOnGround &&
        playerInfo.airTime < DOUBLEJUMP_MAX_DELAY && playerInfo.hasFlipped &&
        playerInfo.flipTime > 0.0f) {
      Vec dirToBall = (state.ball.pos - player.pos).Normalized();
      Vec normVel = player.vel / MAX_FORWARD_SPEED;
      float dashReward =
          (FLIP_TORQUE_TIME - player.flipTime) / FLIP_TORQUE_TIME;
      float speedReward = player.vel.Length() / MAX_FORWARD_SPEED;

      reward = dashReward * speedReward;
    }

    playerInfo.airTime = player.airTimeSinceJump;
    playerInfo.flipTime = player.flipTime;
    playerInfo.hasFlipped = player.hasFlipped;

    return reward;
  }

private:
  struct PlayerInfo {
    float airTime;
    float flipTime;
    bool hasFlipped;
  };

  std::unordered_map<int, PlayerInfo> playerData;
};

class WallReward : public Reward {
public:
  WallReward() : dist_to_ball(), on_wall(false), lost_jump(false), reward(0) {}

  virtual void Reset(const GameState &initialState) override {
    on_wall = false;
    lost_jump = false;
    reward = 0;
  }

  virtual float GetReward(const Player &player, const GameState &state,
                          bool isFinal) override {
    if (!player.hasJumped) {
      if (!lost_jump) {
        lost_jump = true;
      }
    }

    float dist_to_ball_reward =
        dist_to_ball.GetReward(player, state, isFinal) * 0.04f;

    // Starting encouragement
    if (state.ball.pos.x < -4000 || state.ball.pos.x > 4000 ||
        (state.ball.pos.y > 5020 && state.ball.pos.x > -900 &&
         state.ball.pos.x < 900) ||
        (state.ball.pos.y < -5020 && state.ball.pos.x > -900 &&
         state.ball.pos.x < 900)) {
      reward += dist_to_ball_reward * 2;
    }

    if (player.pos.x < -4000 || player.pos.x > 4000 ||
        (player.pos.y > 5020 && player.pos.x > -900 && player.pos.x < 900) ||
        (player.pos.y < -5020 && player.pos.x > -900 && player.pos.x < 900)) {

      float addup = (player.pos.z / CommonValues::CEILING_Z) * 2;
      dist_to_ball_reward *= (10 + addup);
      if (!on_wall) {
        on_wall = true;
      } else {
        dist_to_ball_reward *= 10;
      }
    }

    if (player.hasFlipped && lost_jump) {
      if ((player.pos.x < -4000 || player.pos.x > 4000) &&
          state.ball.pos.z > 500 && on_wall) {
        dist_to_ball_reward *= 20;
      }
    }

    reward += dist_to_ball_reward;

    if (!player.hasJumped) {
      if (!lost_jump) {
        lost_jump = true;
      }

      float ball_speed = state.ball.vel.Length();
      float ball_speed_reward = ball_speed / CommonValues::BALL_MAX_SPEED;
      reward += ball_speed_reward / 10;
    }

    if (player.ballTouchedStep) {
      reward += 5;
    }

    if (player.hasJumped) {
      lost_jump = false;
    }
    return reward;
  }

  virtual float GetFinalReward(const Player &player, const GameState &state) {
    return 0;
  }

private:
  DistToBallReward dist_to_ball;
  bool on_wall;
  bool lost_jump;
  float reward;

  // Rewards wall play including climbing walls, aerial touches off walls, and
  // ball touches while on walls.
};

class Kickoff2v2Reward : public Reward {
public:
  Kickoff2v2Reward() : Reward() {}

  virtual float GetReward(const Player &player, const GameState &state,
                          bool isFinal) override {
    if (state.ball.vel.Length() >= 1.f) {
      return 0.0f;
    }

    std::vector<const Player *> teammates;
    for (const auto &p : state.players) {
      if (p.team == player.team && p.carId != player.carId) {
        teammates.push_back(&p);
      }
    }

    if (teammates.size() != 1) {
      return 0.0f;
    }

    const Player *teammate = teammates[0];

    float playerDistToBall = (player.pos - state.ball.pos).Length();
    float teammateDistToBall = (teammate->pos - state.ball.pos).Length();

    bool playerShouldGo = false;
    if (playerDistToBall < teammateDistToBall) {
      playerShouldGo = true;
    } else if (playerDistToBall > teammateDistToBall) {
      playerShouldGo = false;
    } else {
      playerShouldGo = (player.pos.x < teammate->pos.x);
    }

    float reward = 0.0f;

    if (playerShouldGo) {
      Vec dirToBall = (state.ball.pos - player.pos).Normalized();
      Vec normVel = player.vel / CommonValues::CAR_MAX_SPEED;
      float velTowardBall = dirToBall.Dot(normVel);
      reward = RS_MAX(0.0f, velTowardBall);
    } else {
      if (player.prevAction.boost > 0.5f) {
        reward = -1.0f;
      } else {
        Vec dirToTeammate = (teammate->pos - player.pos).Normalized();
        Vec normVel = player.vel / CommonValues::CAR_MAX_SPEED;
        float velTowardTeammate = dirToTeammate.Dot(normVel);
        float distToTeammate = (player.pos - teammate->pos).Length();
        float optimalDist = 1000.0f; // Optimal following distance
        float distReward =
            1.0f -
            RS_MIN(1.0f, fabsf(distToTeammate - optimalDist) / optimalDist);
        reward = RS_MAX(0.0f, velTowardTeammate) * 0.5f + distReward * 0.5f;
      }
    }

    return reward;
  }
};

class AerialChallenge : public Reward {
public:
  AerialChallenge(float heightThreshold = 300.0f, float touchRewardBase = 10.0f,
                  float heightMultiplier = 0.5f)
      : heightThreshold(heightThreshold), touchRewardBase(touchRewardBase),
        heightMultiplier(heightMultiplier) {}

  virtual float GetReward(const Player &player, const GameState &state,
                          bool isFinal) override {
    float reward = 0.0f;
    Vec ballPos = state.ball.pos;
    Vec playerPos = player.pos;

    if (ballPos.z > heightThreshold) {
      float distance = (playerPos - ballPos).Length();
      float closestDistance = std::numeric_limits<float>::max();
      for (const auto &otherPlayer : state.players) {
        float otherDistance = (otherPlayer.pos - ballPos).Length();
        if (otherDistance < closestDistance) {
          closestDistance = otherDistance;
        }
      }

      if (distance == closestDistance) {
        reward += 1.0f;
        reward += (ballPos.z - heightThreshold) * heightMultiplier;
      }

      if (player.ballTouchedStep) {
        reward += touchRewardBase * (ballPos.z / heightThreshold);
      }
    }

    return reward;
  }

private:
  float heightThreshold;
  float touchRewardBase;
  float heightMultiplier;
};

class PositiveRollReward : public Reward {
public:
  PositiveRollReward(float height_threshold = 300.0f,
                     float distance_threshold = 500.0f)
      : height_threshold(height_threshold),
        distance_threshold(distance_threshold) {}

  virtual float GetReward(const Player &player, const GameState &state,
                          bool isFinal) override {
    float reward = 0.0f;
    if (player.pos.z > height_threshold &&
        (player.pos - state.ball.pos).Length() < distance_threshold &&
        player.prevAction.roll > 0.0f) {
      reward = 1.0f;
    }
    return reward;
  }

private:
  float height_threshold;
  float distance_threshold;
};

class DistToBallReward : public Reward {
public:
  virtual float GetReward(const Player &player, const GameState &state,
                          bool isFinal) override {
    Vec ballToPlayer = state.ball.pos - player.pos;
    float lengthSquared = ballToPlayer.x * ballToPlayer.x +
                          ballToPlayer.y * ballToPlayer.y +
                          ballToPlayer.z * ballToPlayer.z;
    float distance = 1.0f / std::sqrt(lengthSquared);
    return distance * 100.0f;
  }
};

class AirTouchReward : public Reward {
public:
  const float MAX_TIME_IN_AIR = 1.75f;

  AirTouchReward() {}

  virtual void Reset(const GameState &initialState) override {}

  virtual float GetReward(const Player &player, const GameState &state,
                          bool isFinal) override {
    if (player.ballTouchedStep) {
      float airTimeFraction =
          std::min(player.airTime, MAX_TIME_IN_AIR) / MAX_TIME_IN_AIR;
      float heightFraction = state.ball.pos.z / CommonValues::CEILING_Z;

      return std::min(airTimeFraction, heightFraction);
    }
    return 0.0f;
  }
};

class MultiTouchAerialVelocityChange : public Reward {
public:
  MultiTouchAerialVelocityChange(float aerial_w = 0.1f, int exp = 2,
                                 float min_aerial = 500.0f,
                                 float touch_w = 0.3f, float vel_w = 0.8f)
      : aerial_w(aerial_w), exp(exp), min_aerial(min_aerial), touch_w(touch_w),
        vel_w(vel_w), last_vel(Vec(0, 0, 0)) {}

  virtual void Reset(const GameState &initialState) override {
    last_vel = Vec(0, 0, 0);
    ball_count = 0;
    ball_touch_vel_change = 0;
  }

  virtual float GetReward(const Player &player, const GameState &state,
                          bool isFinal) override {
    constexpr float BALL_RADIUS = 92.75f;
    constexpr float CEILING_Z = 2048.0f;
    float reward = 0;
    ball_touch_vel_change = 0;
    float reward_height = 0;
    float reward_vel = 0;

    if (player.ballTouchedStep && !player.isOnGround) {
      if (state.ball.pos.z > min_aerial) {
        Vec vel_difference = last_vel - state.ball.vel;
        ball_touch_vel_change += vel_difference.Length() / 4600.0f;
        ball_count += 1;
        reward_height += std::pow((state.ball.pos.z * aerial_w), exp) /
                         (CEILING_Z - BALL_RADIUS);
        reward += std::pow(ball_count, exp) * touch_w;
        reward_vel += ball_touch_vel_change * std::pow(ball_count, exp) * vel_w;
      } else {
        ball_touch_vel_change = 0;
        ball_count = 0;
        reward_height = 0;
        reward = 0;
        reward_vel = 0;
      }
    }

    if (player.ballTouchedStep && !player.isOnGround &&
        player.boostFraction > 0) {
      reward += 1.0f;
    }

    last_vel = state.ball.vel;

    return reward + reward_vel + reward_height;
  }

private:
  float aerial_w;
  int exp;
  float min_aerial;
  float touch_w;
  float vel_w;
  Vec last_vel;
  int ball_count = 0;
  float ball_touch_vel_change = 0;
};

class HeightTouchReward : public Reward {
public:
  HeightTouchReward(float min_height = 92.0f, float exp = 0.2f,
                    float coop_dist = 0.0f)
      : min_height(min_height), exp(exp), cooperation_dist(coop_dist) {}

  bool cooperation_detector(const Player &player, const GameState &state) {
    for (const auto &p : state.players) {
      if (p.carId != player.carId &&
          (player.pos - p.pos).Length() < cooperation_dist) {
        return true;
      }
    }
    return false;
  }

  virtual float GetReward(const Player &player, const GameState &state,
                          bool isFinal) override {
    float reward = 0.0f;
    if (player.ballTouchedStep) {
      if (state.ball.pos.z >= min_height) {
        if (!player.isOnGround || cooperation_dist < 90.0f ||
            !cooperation_detector(player, state)) {
          if (player.isOnGround) {
            reward += std::clamp(5000.0f, 0.0001f, (state.ball.pos.z - 92.0f)) *
                      std::pow(exp, 2);
          } else {
            reward +=
                std::clamp(500.0f, 1.0f, std::pow(state.ball.pos.z, exp * 2));
          }
        }
      } else if (!player.isOnGround) {
        reward += 1.0f;
      }
    }
    return reward;
  }

private:
  float min_height;
  float exp;
  float cooperation_dist;
};

class MawkzyFlickReward : public Reward {
private:
  struct PlayerState {
    bool ballControlPhase = false;
    bool airRollPhase = false;
    bool flickPhase = false;
    bool flickCompleted = false;
    Vec ballPosition = Vec(0, 0, 0);
    Vec playerForward = Vec(0, 0, 0);
    Vec playerRight = Vec(0, 0, 0);
    float airRollTime = 0.0f;
    float flickPower = 0.0f;
    float airRollDirection = 0.0f; // -1 = gauche, +1 = droite
    bool wasOnGround = false;
    bool hadFlipBefore = false;
    Vec lastBallVelocity = Vec(0, 0, 0);
  };
  std::map<uint32_t, PlayerState> playerStates;

public:
  float ballProximityThreshold;
  float minAirRollTime;
  float minFlickPower;
  float targetFlickAngle;
  float angleTolerance;
  float ballControlReward;
  float airRollReward;
  float flickReward;
  float powerBonus;
  float precisionBonus;
  float angleBonus;
  float directionBonus;
  bool debug;

  MawkzyFlickReward(float ballProximityThreshold = 250.0f,
                    float minAirRollTime = 0.2f, float minFlickPower = 800.0f,
                    float targetFlickAngle = 45.0f,
                    float angleTolerance = 15.0f,
                    float ballControlReward = 1.0f, float airRollReward = 2.0f,
                    float flickReward = 4.0f, float powerBonus = 2.0f,
                    float precisionBonus = 2.0f, float angleBonus = 3.0f,
                    float directionBonus = 2.5f, bool debug = false)
      : ballProximityThreshold(ballProximityThreshold),
        minAirRollTime(minAirRollTime), minFlickPower(minFlickPower),
        targetFlickAngle(targetFlickAngle), angleTolerance(angleTolerance),
        ballControlReward(ballControlReward), airRollReward(airRollReward),
        flickReward(flickReward), powerBonus(powerBonus),
        precisionBonus(precisionBonus), angleBonus(angleBonus),
        directionBonus(directionBonus), debug(debug) {}

  virtual void Reset(const GameState &initialState) override {
    playerStates.clear();
    for (const auto &player : initialState.players) {
      PlayerState &state = playerStates[player.carId];
      state.ballControlPhase = false;
      state.airRollPhase = false;
      state.flickPhase = false;
      state.flickCompleted = false;
      state.ballPosition = Vec(0, 0, 0);
      state.playerForward = player.rotMat.forward;
      state.playerRight = player.rotMat.right;
      state.airRollTime = 0.0f;
      state.flickPower = 0.0f;
      state.airRollDirection = 0.0f;
      state.wasOnGround = player.isOnGround;
      state.hadFlipBefore = player.HasFlipOrJump();
      state.lastBallVelocity = Vec(0, 0, 0);
    }
  }

  virtual float GetReward(const Player &player, const GameState &state,
                          bool isFinal) override {
    if (!state.prev)
      return 0.0f;

    int carId = player.carId;
    if (playerStates.find(carId) == playerStates.end()) {
      PlayerState &newState = playerStates[player.carId];
      newState.ballControlPhase = false;
      newState.airRollPhase = false;
      newState.flickPhase = false;
      newState.flickCompleted = false;
      newState.ballPosition = Vec(0, 0, 0);
      newState.playerForward = player.rotMat.forward;
      newState.playerRight = player.rotMat.right;
      newState.airRollTime = 0.0f;
      newState.flickPower = 0.0f;
      newState.airRollDirection = 0.0f;
      newState.wasOnGround = player.isOnGround;
      newState.hadFlipBefore = player.HasFlipOrJump();
      newState.lastBallVelocity = Vec(0, 0, 0);
    }

    PlayerState &st = playerStates[carId];
    float reward = 0.0f;

    Vec playerPos = player.pos;
    Vec ballPos = state.ball.pos;
    float distanceToBall = (playerPos - ballPos).Length();
    bool isOnGround = player.isOnGround;
    Vec currentForward = player.rotMat.forward;
    Vec currentRight = player.rotMat.right;

    // === PHASE 1: CONTRÔLE DE LA BALLE ===
    if (distanceToBall < ballProximityThreshold && !st.ballControlPhase) {
      st.ballControlPhase = true;
      reward += ballControlReward;

      if (debug) {
        printf("[MawkzyFlick] car_id=%d PHASE 1: Contrôle de la balle! Reward: "
               "%.2f\n",
               carId, ballControlReward);
      }
    }

    // === PHASE 2: AIR ROLL DIRECTIONNEL ===
    if (st.ballControlPhase && !isOnGround && !st.airRollPhase) {
      // Détecter l'air roll via le changement de rotation
      if (st.playerForward.Length() > 0) {
        float rotationChange = fabsf(currentForward.Dot(st.playerForward));

        if (rotationChange < 0.8f) { // Rotation significative
          st.airRollPhase = true;

          // DÉTECTER LA DIRECTION DE L'AIR ROLL
          Vec carRight = player.rotMat.right;
          float airRollDirection = carRight.z; // -1 = gauche, +1 = droite

          // Normaliser la direction
          if (abs(airRollDirection) > 0.1f) {
            st.airRollDirection = (airRollDirection > 0) ? 1.0f : -1.0f;

            if (debug) {
              printf("[MawkzyFlick] car_id=%d PHASE 2: Air roll %s détecté! "
                     "Direction: %.1f\n",
                     carId, (st.airRollDirection > 0) ? "DROITE" : "GAUCHE",
                     st.airRollDirection);
            }
          }

          reward += airRollReward;
        }
      }
    }

    // === PHASE 3: FLICK 45° ===
    if (st.airRollPhase && !st.flickPhase) {
      bool hasFlipNow = player.HasFlipOrJump();

      if (st.hadFlipBefore && !hasFlipNow) {
        st.flickPhase = true;
        st.lastBallVelocity = state.prev ? state.prev->ball.vel : Vec(0, 0, 0);

        if (debug) {
          printf("[MawkzyFlick] car_id=%d PHASE 3: Flick déclenché! Direction "
                 "air roll: %.1f\n",
                 carId, st.airRollDirection);
        }
      }
    }

    // === PHASE 4: VÉRIFICATION COMPLÈTE ===
    if (st.flickPhase && !st.flickCompleted && player.ballTouchedStep) {
      // 1. VÉRIFIER LA PUISSANCE DU FLICK
      float ballSpeedChange = 0.0f;
      if (state.prev) {
        ballSpeedChange = (state.ball.vel - st.lastBallVelocity).Length();
        st.flickPower = ballSpeedChange;
      }

      if (ballSpeedChange > minFlickPower) {
        reward += flickReward;
        reward += (ballSpeedChange - minFlickPower) * powerBonus / 1000.0f;

        if (debug) {
          printf(
              "[MawkzyFlick] car_id=%d Puissance flick: %.0f, Reward: %.2f\n",
              carId, ballSpeedChange, reward);
        }
      }

      // 2. VÉRIFIER L'ANGLE DU FLICK (45°)
      float flickAngle = CalculateFlickAngle(player, state);
      float angleAccuracy = CalculateAngleAccuracy(flickAngle);

      if (angleAccuracy > 0.7f) { // Angle proche de 45°
        reward += angleBonus * angleAccuracy;

        if (debug) {
          printf("[MawkzyFlick] car_id=%d Angle flick: %.1f° (cible: 45°), "
                 "Accuracy: %.2f, Bonus: %.2f\n",
                 carId, flickAngle, angleAccuracy, angleBonus * angleAccuracy);
        }
      }

      // 3. VÉRIFIER LA DIRECTION DU FLICK (correspond à l'air roll)
      bool directionMatches =
          CheckDirectionMatch(st.airRollDirection, state.ball.vel);

      if (directionMatches) {
        reward += directionBonus;

        if (debug) {
          printf("[MawkzyFlick] car_id=%d DIRECTION PARFAITE! Air roll %s ? "
                 "Flick %s, Bonus: %.2f\n",
                 carId, (st.airRollDirection > 0) ? "DROITE" : "GAUCHE",
                 (st.airRollDirection > 0) ? "DROITE" : "GAUCHE",
                 directionBonus);
        }
      }

      // 4. VÉRIFIER LA PRÉCISION VERS LE BUT
      Vec goalDirection = GetGoalDirection(player.team);
      Vec ballDirection = state.ball.vel.Normalized();
      float precision = ballDirection.Dot(goalDirection);

      if (precision > 0.6f) { // Précision vers le but
        reward += precisionBonus * precision;

        if (debug) {
          printf("[MawkzyFlick] car_id=%d Précision vers le but: %.2f, Bonus: "
                 "%.2f\n",
                 carId, precision, precisionBonus * precision);
        }
      }

      // 5. BONUS FINAL POUR MAWKZY FLICK PARFAIT
      if (angleAccuracy > 0.8f && directionMatches && precision > 0.7f) {
        float perfectBonus = 5.0f;
        reward += perfectBonus;

        if (debug) {
          printf("[MawkzyFlick] car_id=%d ?? MAWKZY FLICK PARFAIT! Bonus "
                 "final: %.2f\n",
                 carId, perfectBonus);
        }
      }

      st.flickCompleted = true;
    }

    // Récompense continue pour maintenir l'air roll
    if (st.airRollPhase && !isOnGround) {
      st.airRollTime += 1.0f / 120.0f;
      if (st.airRollTime > minAirRollTime) {
        reward += 0.1f; // Récompense continue
      }
    }

    // Reset si le joueur touche le sol
    if (st.wasOnGround && !isOnGround) {
      st.airRollTime = 0.0f;
    }

    // Mise à jour de l'état
    st.ballPosition = ballPos;
    st.playerForward = currentForward;
    st.playerRight = currentRight;
    st.wasOnGround = isOnGround;
    st.hadFlipBefore = player.HasFlipOrJump();

    return reward;
  }

private:
  // Calculer l'angle du flick (0° = droit, 90° = vertical)
  float CalculateFlickAngle(const Player &player,
                            const GameState &state) const {
    Vec carForward =
        Vec(player.rotMat.forward.x, player.rotMat.forward.y, 0.0f);
    Vec ballVelocity = Vec(state.ball.vel.x, state.ball.vel.y, 0.0f);

    if (carForward.Length() < 0.1f || ballVelocity.Length() < 100.0f) {
      return 0.0f;
    }

    carForward = carForward.Normalized();
    ballVelocity = ballVelocity.Normalized();

    float dotProduct =
        RS_MAX(-1.0f, RS_MIN(1.0f, carForward.Dot(ballVelocity)));
    float angle = acosf(dotProduct);
    return angle * 180.0f / M_PI;
  }

  // Calculer la précision de l'angle (0-1, 1 = parfait 45°)
  float CalculateAngleAccuracy(float flickAngle) const {
    float angleDiff = abs(flickAngle - targetFlickAngle);
    if (angleDiff <= angleTolerance) {
      return 1.0f - (angleDiff / angleTolerance);
    }
    return 0.0f;
  }

  // Vérifier que la direction du flick correspond à l'air roll
  bool CheckDirectionMatch(float airRollDirection,
                           const Vec &ballVelocity) const {
    if (abs(airRollDirection) < 0.1f)
      return false;

    // Air roll DROITE (+1) ? Flick doit aller vers la DROITE (X positif)
    // Air roll GAUCHE (-1) ? Flick doit aller vers la GAUCHE (X négatif)
    if (airRollDirection > 0) {        // Droite
      return ballVelocity.x > 200.0f;  // Vitesse X positive
    } else {                           // Gauche
      return ballVelocity.x < -200.0f; // Vitesse X négative
    }
  }

  Vec GetGoalDirection(Team team) {
    if (team == Team::BLUE) {
      return (CommonValues::ORANGE_GOAL_CENTER - Vec(0, 0, 0)).Normalized();
    } else {
      return (CommonValues::BLUE_GOAL_CENTER - Vec(0, 0, 0)).Normalized();
    }
  }
};

class PressureFlickReward : public Reward {
public:
  const float PANIC_DISTANCE = 700.0f;
  const float MIN_FLICK_SPEED = 1000.0f;
  const float TARGET_FLICK_SPEED = 2920.0f;
  const float EXPONENT = 2.5f;

  virtual float GetReward(const Player &player, const GameState &state,
                          bool isFinal) override {
    if (!player.ballTouchedStep || !player.isFlipping)
      return 0.0f;

    if (!state.prev)
      return 0.0f;

    // Distance
    float dist = player.prev->pos.Dist(state.prev->ball.pos);
    if (dist > 250.0f)
      return 0.0f;

    // Speed Match
    float speedDiff =
        std::abs(player.prev->vel.Length() - state.prev->ball.vel.Length());
    if (speedDiff > 500.0f)
      return 0.0f;

    // Distance Check
    float closestOppDist = 100000.0f;
    for (const auto &p : state.players) {
      if (p.team != player.team) {
        float d = player.pos.Dist(p.pos);
        if (d < closestOppDist)
          closestOppDist = d;
      }
    }
    if (closestOppDist > PANIC_DISTANCE)
      return 0.0f;

    // Result Check
    bool targetOrange = player.team == Team::BLUE;
    Vec targetPos = targetOrange ? CommonValues::ORANGE_GOAL_BACK
                                 : CommonValues::BLUE_GOAL_BACK;
    Vec dirToGoal = (targetPos - state.ball.pos).Normalized();
    float velTowardsGoal = state.ball.vel.Dot(dirToGoal);

    if (velTowardsGoal > MIN_FLICK_SPEED) {
      float ratio = velTowardsGoal / TARGET_FLICK_SPEED;
      float reward = std::pow(ratio, EXPONENT);
      return std::min(reward, 2.0f);
    }

    return 0.0f;
  }
};

class HalfFlipReward : public Reward {
private:
  float reward_value;
  float min_speed_gain;
  float min_facing_before;
  float min_facing_after;
  int min_ticks;
  bool debug;

  struct PlayerState {
    bool backflip_started = false;
    bool backflip_canceled = false;
    bool air_roll_detected = false;
    bool halfflip_completed = false;
    bool rewarded = false;
    float pre_flip_vel = 0.0f;
    float post_flip_vel = 0.0f;
    int flip_tick = 0;
    float facing_before = 0.0f;
    Vec initial_rotation = Vec(0, 0, 0);
    Vec rotation_at_cancel = Vec(0, 0, 0);
    Vec final_rotation = Vec(0, 0, 0);
  };

  std::unordered_map<int, PlayerState> player_states;

public:
  HalfFlipReward(float reward_value = 8.0f, float min_speed_gain = 300.0f,
                 float min_facing_before = -0.5f, float min_facing_after = 0.5f,
                 int min_ticks = 4, bool debug = false)
      : reward_value(reward_value), min_speed_gain(min_speed_gain),
        min_facing_before(min_facing_before),
        min_facing_after(min_facing_after), min_ticks(min_ticks), debug(debug) {
  }

  virtual void Reset(const GameState &initial_state) override {
    player_states.clear();
  }

  virtual float GetReward(const Player &player, const GameState &state,
                          bool isFinal) override {
    int car_id = player.carId;
    if (player_states.find(car_id) == player_states.end()) {
      player_states[car_id] = PlayerState{};
    }

    PlayerState &st = player_states[car_id];
    float reward = 0.0f;

    // Direction vers la balle
    Vec ball_dir = state.ball.pos - player.pos;
    ball_dir = ball_dir.Normalized();

    // Direction de la voiture
    Vec forward = player.rotMat.forward;
    float facing_ball = forward.Dot(ball_dir);

    // Détection du début du backflip (saut + rotation vers l'arrière)
    if (!st.backflip_started && player.isJumping && !player.isOnGround) {
      // Vérifier si la voiture commence à se retourner (rotation pitch
      // négative)
      Vec car_up = player.rotMat.up;
      if (car_up.z < 0.3f) { // La voiture commence à se retourner
        st.backflip_started = true;
        st.flip_tick = 0;
        st.pre_flip_vel = -forward.Dot(player.vel);
        st.facing_before = facing_ball;
        st.initial_rotation = Vec(car_up.x, car_up.y, car_up.z);
      }
    }

    // Détection du cancel du backflip
    if (st.backflip_started && !st.backflip_canceled) {
      st.flip_tick++;

      // Le cancel se caractérise par une rotation qui s'arrête
      Vec car_up = player.rotMat.up;
      float rotation_change = fabsf(car_up.z - st.initial_rotation.z);

      if (st.flip_tick >= min_ticks && rotation_change < 0.1f) {
        st.backflip_canceled = true;
        st.rotation_at_cancel = Vec(car_up.x, car_up.y, car_up.z);
      }
    }

    // Détection de l'air roll pour se retourner
    if (st.backflip_canceled && !st.air_roll_detected) {
      Vec car_up = player.rotMat.up;
      Vec car_forward = player.rotMat.forward;

      // L'air roll se caractérise par une rotation continue autour de l'axe
      // forward
      float rotation_progress = fabsf(car_up.z - st.rotation_at_cancel.z);

      if (rotation_progress > 0.3f) { // Rotation significative détectée
        st.air_roll_detected = true;
      }
    }

    // Vérification du halfflip complet
    if (st.backflip_canceled && st.air_roll_detected &&
        !st.halfflip_completed) {
      // Vérifier que la voiture fait face à la balle maintenant
      if (facing_ball > min_facing_after) {
        st.halfflip_completed = true;
        st.post_flip_vel = forward.Dot(player.vel);
        float speed_gain = st.post_flip_vel + st.pre_flip_vel;

        if (speed_gain > min_speed_gain) {
          reward = reward_value;
        }
      }
    }

    // Reset si au sol ou trop longtemps après le flip
    if (player.isOnGround || st.flip_tick > 50) {
      st.backflip_started = false;
      st.backflip_canceled = false;
      st.air_roll_detected = false;
      st.halfflip_completed = false;
      st.rewarded = false;
      st.flip_tick = 0;
    }

    return reward;
  }
};

class ShadowDefenseReward : public Reward {
public:
  float maxBallGoalDistance;
  float maxLateralOffset;
  float idealSpacingFromBall;
  float spacingTolerance;
  ShadowDefenseReward(float maxBallGoalDistance = 3500.f,
                      float maxLateralOffset = 1200.f,
                      float idealSpacingFromBall = 1200.f,
                      float spacingTolerance = 700.f)
      : maxBallGoalDistance(maxBallGoalDistance),
        maxLateralOffset(maxLateralOffset),
        idealSpacingFromBall(idealSpacingFromBall),
        spacingTolerance(spacingTolerance) {}

  virtual float GetReward(const Player &player, const GameState &state,
                          bool isFinal) {
    Vec goalCenter = player.team == Team::BLUE
                         ? CommonValues::BLUE_GOAL_CENTER
                         : CommonValues::ORANGE_GOAL_CENTER;
    float goalDirection = player.team == Team::BLUE ? -1.f : 1.f;

    Vec goalToBall = state.ball.pos - goalCenter;
    if (goalDirection * goalToBall.y < 0.f)
      return 0.f;

    float ballDist = goalToBall.Length();
    if (ballDist > maxBallGoalDistance || ballDist < 100.f)
      return 0.f;

    Vec goalDir = goalToBall / ballDist;

    Vec goalToPlayer = player.pos - goalCenter;
    float playerDepth = goalDir.Dot(goalToPlayer);
    if (playerDepth < 0.f || playerDepth > ballDist + 250.f)
      return 0.f;

    Vec lateral = goalToPlayer - goalDir * playerDepth;
    float lateralOffset = lateral.Length();
    float lateralWeight =
        1.f - RS_CLAMP(lateralOffset / maxLateralOffset, 0.f, 1.f);
    if (lateralWeight <= 0.f)
      return 0.f;

    float spacing = ballDist - playerDepth;
    float spacingWeight =
        1.f - RS_CLAMP(fabsf(spacing - idealSpacingFromBall) / spacingTolerance,
                       0.f, 1.f);
    if (spacingWeight <= 0.f)
      return 0.f;

    Vec toBall = state.ball.pos - player.pos;
    float facingWeight = 0.f;
    if (toBall.Length() > 100.f)
      facingWeight =
          RS_MAX(0.f, player.rotMat.forward.Dot(toBall.Normalized()));

    float velocityWeight = 0.f;
    if (player.vel.Length() > 100.f) {
      Vec desired = toBall.Normalized();
      Vec defensiveDir = Vec(0.f, goalDirection, 0.f);
      Vec blendBase = desired * 0.6f + defensiveDir * 0.4f;
      if (blendBase.Length() < 1e-3f)
        blendBase = desired;
      Vec blended = blendBase.Normalized();
      velocityWeight = RS_MAX(0.f, player.vel.Normalized().Dot(blended));
    }

    float controlWeight = 0.4f * facingWeight + 0.6f * velocityWeight;
    return lateralWeight * spacingWeight * controlWeight;
  }
};

class DribbleAirdribbleBumpDemoRewardv1 : public Reward {
public:
  virtual float GetReward(const Player &player, const GameState &state,
                          bool isFinal) {
    float reward = 0.f;
    if (state.ball.pos.y > 4600 &&
        std::abs(state.ball.pos.x) < GOAL_WIDTH_FROM_CENTER &&
        state.ball.vel.y > 0) { // if the ball is in front of the opponents net
                                // and going towards it
      if (player.pos.y > state.ball.pos.y &&
          std::abs(player.pos.x) <
              GOAL_WIDTH_FROM_CENTER) { // if the player is in front of the ball
                                        // and in front of the net
        reward +=
            0.001f; // With the reward being 2000 atm, this should reward it 2
                    // per step, or 60 per second. (this might be to high)
        if (player.eventState.bump) { // if the player bumped the opponent
          reward += 0.5f * (state.ball.vel.y /
                            1000); // 6000 is max ball speed but will rarely be
                                   // that fast. scaling by 1000 should be good
                                   // because player speed averages around 1400
        }
        if (player.eventState.demo) { // if the player demoed the opponent
          reward += 1.0f * (state.ball.vel.y /
                            1000); // this scales it by the speed of the ball
                                   // towards the net / opponents side
        }
      }
    }
    return reward;
  }
};

class DribbleAirdribbleBumpDemoReward : public Reward {
private:
  std::unordered_map<uint32_t, uint64_t> lastBumpTick;
  std::unordered_map<uint32_t, uint64_t> lastDemoTick;

public:
  float positionRewardWeight;
  float bumpRewardWeight;
  float demoRewardWeight;
  DribbleAirdribbleBumpDemoReward(
      float positionWeight = 0.001f, // Increased from 0.001f
      float bumpWeight = 0.5f,       // Increased from 0.5f
      float demoWeight = 1.0f        // Increased from 1.0f
      )
      : positionRewardWeight(positionWeight), bumpRewardWeight(bumpWeight),
        demoRewardWeight(demoWeight) {}
  virtual void Reset(const GameState &initialState) override {
    lastBumpTick.clear();
    lastDemoTick.clear();
  }
  virtual float GetReward(const Player &player, const GameState &state,
                          bool isFinal) override {
    if (!state.lastArena)
      return 0.0f;
    float reward = 0.0f;
    Vec targetGoal = (player.team == Team::BLUE)
                         ? CommonValues::ORANGE_GOAL_BACK
                         : CommonValues::BLUE_GOAL_BACK;
    float goalY = targetGoal.y;
    float goalWidthHalf = CommonValues::GOAL_WIDTH_FROM_CENTER;
    float distToGoalY = std::abs(state.ball.pos.y - goalY);
    bool ballNearOpponentGoal = distToGoalY < 1500.0f;
    bool ballInGoalWidth = std::abs(state.ball.pos.x) < goalWidthHalf + 500.0f;
    Vec toGoal = targetGoal - state.ball.pos;
    float toGoalLen = toGoal.Length();
    Vec dirToGoal = (toGoalLen > 1e-6f) ? toGoal / toGoalLen : Vec(0, 0, 0);
    float ballSpeedTowardGoal = state.ball.vel.Dot(dirToGoal);
    bool ballMovingTowardGoal = ballSpeedTowardGoal > 50.0f;
    if (ballNearOpponentGoal && ballInGoalWidth && ballMovingTowardGoal) {
      float playerDistToGoal = (player.pos - targetGoal).Length();
      float ballDistToGoal = (state.ball.pos - targetGoal).Length();
      bool playerBehindBall = playerDistToGoal > ballDistToGoal - 200.0f;
      float heightDiff = std::abs(player.pos.z - state.ball.pos.z);
      bool playerAtReasonableHeight =
          heightDiff < 500.0f || player.pos.z < 300.0f;
      float playerDistToGoalY = std::abs(player.pos.y - goalY);
      bool playerNearOpponentGoal = playerDistToGoalY < 2000.0f;
      bool playerInAttackingZone =
          std::abs(player.pos.x) < goalWidthHalf + 1000.0f;
      if (playerBehindBall && playerNearOpponentGoal && playerInAttackingZone &&
          playerAtReasonableHeight) {
        float distanceBonus =
            1.0f - RS_CLAMP(playerDistToGoal / 3000.0f, 0.0f, 1.0f);
        float heightBonus = 1.0f - RS_CLAMP(heightDiff / 400.0f, 0.0f, 1.0f);
        reward += positionRewardWeight * distanceBonus * heightBonus;
        uint64_t currentTick = state.lastArena->tickCount;
        if (player.eventState.bump) {
          auto it = lastBumpTick.find(player.carId);
          if (it == lastBumpTick.end() || currentTick > it->second) {
            lastBumpTick[player.carId] = currentTick;
            float ballSpeedNorm = RS_CLAMP(
                ballSpeedTowardGoal / CommonValues::CAR_MAX_SPEED, 0.0f, 1.0f);
            float goalProximityBonus =
                1.0f - RS_CLAMP(ballDistToGoal / 5000.0f, 0.0f, 1.0f);
            reward +=
                bumpRewardWeight * ballSpeedNorm * (1.0f + goalProximityBonus);
          }
        }
        if (player.eventState.demo) {
          auto it2 = lastDemoTick.find(player.carId);
          if (it2 == lastDemoTick.end() || currentTick > it2->second) {
            lastDemoTick[player.carId] = currentTick;
            float ballSpeedNorm2 = RS_CLAMP(
                ballSpeedTowardGoal / CommonValues::CAR_MAX_SPEED, 0.0f, 1.0f);
            float goalProximityBonus2 =
                1.0f - RS_CLAMP(ballDistToGoal / 5000.0f, 0.0f, 1.0f);
            reward += demoRewardWeight * ballSpeedNorm2 *
                      (1.0f + goalProximityBonus2 * 1.5f);
          }
        }
      }
    }
    return reward;
  }
};

class Team2v2SpacingReward : public Reward {
private:
  float optimalDistance;
  float minDistance;
  float maxDistance;
  float ballProximityWeight;
  float fieldCoverageWeight;
  float defensiveSpacingWeight;

public:
  Team2v2SpacingReward(float optimalDistance = 2000.0f,
                       float minDistance = 800.0f, float maxDistance = 4000.0f,
                       float ballProximityWeight = 0.3f,
                       float fieldCoverageWeight = 0.4f,
                       float defensiveSpacingWeight = 0.3f)
      : optimalDistance(optimalDistance), minDistance(minDistance),
        maxDistance(maxDistance), ballProximityWeight(ballProximityWeight),
        fieldCoverageWeight(fieldCoverageWeight),
        defensiveSpacingWeight(defensiveSpacingWeight) {}

  virtual float GetReward(const Player &player, const GameState &state,
                          bool isFinal) override {
    const Player *teammate = nullptr;
    for (const auto &p : state.players) {
      if (p.team == player.team && p.carId != player.carId) {
        teammate = &p;
        break;
      }
    }

    if (!teammate)
      return 0.0f;

    float reward = 0.0f;

    float distanceToTeammate = (player.pos - teammate->pos).Length();
    float spacingReward = 0.0f;

    if (distanceToTeammate < minDistance) {
      spacingReward = -1.0f * (1.0f - distanceToTeammate / minDistance);
    } else if (distanceToTeammate > maxDistance) {
      spacingReward =
          -0.5f * ((distanceToTeammate - maxDistance) / optimalDistance);
    } else {
      float distanceFromOptimal =
          std::abs(distanceToTeammate - optimalDistance);
      spacingReward = 1.0f - (distanceFromOptimal / optimalDistance);
    }

    float playerBallDist = (player.pos - state.ball.pos).Length();
    float teammateBallDist = (teammate->pos - state.ball.pos).Length();
    float ballProximityReward = 0.0f;

    float ballDistDiff = std::abs(playerBallDist - teammateBallDist);
    if (ballDistDiff > 500.0f) {
      ballProximityReward = 0.5f;
    } else if (ballDistDiff < 200.0f && playerBallDist < 1000.0f &&
               teammateBallDist < 1000.0f) {
      ballProximityReward = -0.3f;
    }

    float fieldCoverageReward = 0.0f;
    Vec fieldCenter = {0.0f, 0.0f, 0.0f};
    Vec playerToCenter = player.pos - fieldCenter;
    Vec teammateToCenter = teammate->pos - fieldCenter;

    float angleBetween =
        std::abs(std::atan2(playerToCenter.y, playerToCenter.x) -
                 std::atan2(teammateToCenter.y, teammateToCenter.x));
    if (angleBetween > 3.14159265f)
      angleBetween = 2 * 3.14159265f - angleBetween;

    if (angleBetween > 3.14159265f / 3) {
      fieldCoverageReward = std::min(1.0f, angleBetween / 3.14159265f);
    }

    float defensiveReward = 0.0f;
    Vec ownGoal = (player.team == Team::BLUE)
                      ? CommonValues::BLUE_GOAL_CENTER
                      : CommonValues::ORANGE_GOAL_CENTER;

    float playerGoalDist = (player.pos - ownGoal).Length();
    float teammateGoalDist = (teammate->pos - ownGoal).Length();

    bool ballInOpponentHalf =
        (player.team == Team::BLUE && state.ball.pos.y > 0) ||
        (player.team == Team::ORANGE && state.ball.pos.y < 0);

    if (ballInOpponentHalf) {
      float goalDistDiff = std::abs(playerGoalDist - teammateGoalDist);
      if (goalDistDiff > 1500.0f) {
        defensiveReward = 0.4f;
      }
    } else {
      if (playerGoalDist > 2000.0f && teammateGoalDist > 2000.0f) {
        defensiveReward = 0.2f;
      }
    }

    reward = spacingReward * (1.0f - ballProximityWeight - fieldCoverageWeight -
                              defensiveSpacingWeight) +
             ballProximityReward * ballProximityWeight +
             fieldCoverageReward * fieldCoverageWeight +
             defensiveReward * defensiveSpacingWeight;

    return std::clamp(reward, -1.0f, 1.0f);
  }
};

class StrategicDemoReward : public Reward {
public:
  // Paramètres configurables
  float goalZoneThreshold; // Distance Y depuis le but pour considérer la "zone
                           // de but" (défaut: 2500 UU)
  float demoMultiplier;  // Multiplicateur pour les demos vs bumps (défaut: 2.0)
  float ballInZoneBonus; // Bonus quand la balle est aussi dans la zone
                         // offensive (défaut: 0.5)
  float maxDistanceFromGoal; // Distance max pour calculer le scaling (défaut:
                             // 5000 UU)
  float baseReward; // Récompense de base pour un bump/demo (défaut: 0.3)
  float maxReward;  // Récompense maximale (défaut: 1.5)

  /**
   * Constructeur avec valeurs par défaut
   * @param goalZoneThreshold Distance Y depuis le but pour la "zone
   * stratégique"
   * @param demoMultiplier Multiplicateur pour les demos (vs bumps simples)
   * @param ballInZoneBonus Bonus si la balle est en position offensive
   * @param maxDistanceFromGoal Distance de référence pour le scaling
   * @param baseReward Récompense de base
   * @param maxReward Plafond de récompense
   */
  StrategicDemoReward(float goalZoneThreshold = 2500.0f,
                      float demoMultiplier = 2.0f, float ballInZoneBonus = 0.5f,
                      float maxDistanceFromGoal = 5000.0f,
                      float baseReward = 0.3f, float maxReward = 1.5f)
      : goalZoneThreshold(goalZoneThreshold), demoMultiplier(demoMultiplier),
        ballInZoneBonus(ballInZoneBonus),
        maxDistanceFromGoal(maxDistanceFromGoal), baseReward(baseReward),
        maxReward(maxReward) {}

  virtual void Reset(const GameState &initialState) override {
    // Pas d'état persistant à réinitialiser
  }

  virtual float GetReward(const Player &player, const GameState &state,
                          bool isFinal) override {
    float reward = 0.0f;

    // Vérifier si le joueur a effectué un bump ou demo ce step
    bool didBump = player.eventState.bump;
    bool didDemo = player.eventState.demo;

    if (!didBump && !didDemo) {
      return 0.0f;
    }

    // Trouver la victime (adversaire qui a été bumped/demoed)
    const Player *victim = nullptr;
    for (const auto &otherPlayer : state.players) {
      if (otherPlayer.team != player.team) {
        // Vérifier si cet adversaire a été bumped ou demoed
        if (otherPlayer.eventState.bumped || otherPlayer.eventState.demoed) {
          victim = &otherPlayer;
          break;
        }
      }
    }

    // Si pas de victime trouvée, retourner récompense de base (cas rare)
    if (!victim) {
      return didDemo ? baseReward * demoMultiplier : baseReward;
    }

    // Déterminer le but adverse (celui qu'on attaque)
    // Blue attaque Orange (Y positif), Orange attaque Blue (Y négatif)
    Vec enemyGoalCenter = (player.team == Team::BLUE)
                              ? CommonValues::ORANGE_GOAL_CENTER
                              : CommonValues::BLUE_GOAL_CENTER;

    // Calculer la distance de la victime à son propre but
    // Plus la victime est proche de son but (qu'on attaque), plus c'est
    // stratégique
    float victimDistToGoal =
        (victim->pos.To2D() - enemyGoalCenter.To2D()).Length();

    // Normaliser la distance (0 = au but, 1 = très loin)
    float distanceRatio =
        RS_CLAMP(victimDistToGoal / maxDistanceFromGoal, 0.0f, 1.0f);

    // Inverser pour que proche du but = récompense plus élevée
    float proximityFactor = 1.0f - distanceRatio;

    // Vérifier si la victime était dans la zone du but (très stratégique)
    float goalY = (player.team == Team::BLUE) ? CommonValues::BACK_WALL_Y
                                              : -CommonValues::BACK_WALL_Y;
    bool victimInGoalZone = std::abs(victim->pos.y - goalY) < goalZoneThreshold;

    // Bonus si dans la zone du but
    float zoneBonus = victimInGoalZone ? 1.5f : 1.0f;

    // Vérifier si la balle est en position offensive (dans le tiers offensif)
    float offensiveThreshold = CommonValues::BACK_WALL_Y / 3.0f;
    bool ballInOffensiveZone = false;
    if (player.team == Team::BLUE) {
      ballInOffensiveZone = state.ball.pos.y > offensiveThreshold;
    } else {
      ballInOffensiveZone = state.ball.pos.y < -offensiveThreshold;
    }

    // Calculer le bonus balle
    float ballBonus = ballInOffensiveZone ? (1.0f + ballInZoneBonus) : 1.0f;

    // Multiplicateur demo vs bump
    float actionMultiplier = didDemo ? demoMultiplier : 1.0f;

    // Bonus supplémentaire si la victime avait le potentiel de défendre
    // (était entre la balle et le but, face au jeu)
    float defenderBonus = 1.0f;
    Vec ballToGoal = (enemyGoalCenter - state.ball.pos).Normalized();
    Vec victimToBall = (state.ball.pos - victim->pos).Normalized();
    float alignmentWithDefense = ballToGoal.Dot(victimToBall);
    if (alignmentWithDefense > 0.3f && victimInGoalZone) {
      // La victime était probablement en train de défendre
      defenderBonus = 1.3f;
    }

    // Calcul final de la récompense
    reward = baseReward *
             (1.0f + proximityFactor) * // Plus proche du but = plus de reward
             zoneBonus *                // Bonus zone de but
             ballBonus *                // Bonus balle offensive
             actionMultiplier *         // Demo vs bump
             defenderBonus;             // Défenseur neutralisé

    // Appliquer le plafond
    reward = RS_MIN(reward, maxReward);

    return reward;
  }
};

class KickoffProximityReward2v2 : public Reward {
public:
  float goerReward = 1.0f;    // Reward for player going for kickoff
  float cheaterReward = 0.5f; // Reward for player staying back

  virtual float GetReward(const Player &player, const GameState &state,
                          bool isFinal) override {
    // Only during kickoff (ball stationary and centered)
    if (state.ball.vel.Length() > 1.f)
      return 0.f;

    float playerDistToBall = (player.pos - state.ball.pos).Length();

    // Find teammate and opponents
    const Player *teammate = nullptr;
    float closestOpponentDist = FLT_MAX;

    for (const auto &p : state.players) {
      if (p.team == player.team && p.carId != player.carId) {
        teammate = &p;
      } else if (p.team != player.team) {
        float opponentDist = (p.pos - state.ball.pos).Length();
        closestOpponentDist = RS_MIN(closestOpponentDist, opponentDist);
      }
    }

    if (!teammate)
      return 0.f;

    float teammateDistToBall = (teammate->pos - state.ball.pos).Length();

    // Determine role: are you the "goer" (closest to ball on your team)?
    // Role determination with tiebreaker
    bool isGoer = (playerDistToBall < teammateDistToBall) ||
                  (playerDistToBall == teammateDistToBall &&
                   player.pos.x < teammate->pos.x);

    if (isGoer) {
      // GOER ROLE: Reward for being faster than opponents
      return (playerDistToBall < closestOpponentDist) ? goerReward
                                                      : -goerReward;
    } else {
      // CHEATER ROLE: IMPROVED - Strategic follow-up positioning, not goal
      // camping
      return CalculateCheaterReward(player, state);
    }
  }

private:
  float CalculateCheaterReward(const Player &player, const GameState &state) {
    Vec ownGoal = (player.team == Team::BLUE) ? CommonValues::BLUE_GOAL_BACK
                                              : CommonValues::ORANGE_GOAL_BACK;

    // Define strategic zones for follow-up play
    Vec fieldCenter = Vec(0, 0, 100);
    Vec idealCheaterPos =
        (ownGoal + fieldCenter) * 0.6f; // 60% towards center from goal

    float distToIdealPos = (player.pos - idealCheaterPos).Length();
    float distToOwnGoal = (player.pos - ownGoal).Length();

    // COMPONENT 1: Ideal positioning reward (strategic follow-up position)
    float idealPosReward = 0.0f;
    float idealRadius = 800.f;
    if (distToIdealPos <= idealRadius) {
      idealPosReward = 0.4f * (1.f - (distToIdealPos / idealRadius));
    } else {
      float maxAcceptableDist = 1500.f;
      idealPosReward =
          0.4f * RS_MAX(0.f, 1.f - ((distToIdealPos - idealRadius) /
                                    maxAcceptableDist));
    }

    // COMPONENT 2: Boost proximity bonus (preparation for next play)
    float boostBonus = CalculateBoostProximityBonus(player, state);

    // COMPONENT 3: Anti-goal-camping penalty (discourages passive play)
    float goalCampPenalty = 0.0f;
    float minDistFromGoal = 1000.f;
    if (distToOwnGoal < minDistFromGoal) {
      goalCampPenalty = -0.2f * (1.f - (distToOwnGoal / minDistFromGoal));
    }

    // COMPONENT 4: Field awareness bonus
    float awarenessBonus = CalculateFieldAwarenessBonus(player, state, ownGoal);

    return RS_CLAMP(idealPosReward + boostBonus + goalCampPenalty +
                        awarenessBonus,
                    -0.5f, 0.5f);
  }

  float CalculateBoostProximityBonus(const Player &player,
                                     const GameState &state) {
    // Find closest large boost pad using your framework's boost locations
    // Large boost pads are the ones with z=73.0 (6 total: 4 corners + 2
    // mid-field)
    float closestBoostDist = FLT_MAX;

    for (int i = 0; i < CommonValues::BOOST_LOCATIONS_AMOUNT; i++) {
      const Vec &boostPos = CommonValues::BOOST_LOCATIONS[i];

      // Filter for large boost pads (z=73.0 indicates 100% boost pads)
      if (boostPos.z > 72.0f) {
        float dist = (player.pos - boostPos).Length();
        closestBoostDist = RS_MIN(closestBoostDist, dist);
      }
    }

    if (closestBoostDist <= 1000.f) {
      return 0.1f * (1.f - (closestBoostDist / 1000.f));
    }
    return 0.0f;
  }

  float CalculateFieldAwarenessBonus(const Player &player,
                                     const GameState &state,
                                     const Vec &ownGoal) {
    Vec toBall = (state.ball.pos - player.pos).Normalized();
    Vec toGoal = (ownGoal - player.pos).Normalized();

    float angleDot = toBall.Dot(toGoal);

    // Best when positioned to quickly transition between ball focus and goal
    // defense
    if (angleDot >= -0.7f && angleDot <= 0.2f) {
      return 0.05f;
    }
    return 0.0f;
  }
};

class ProgressiveDribblePopBumpReward : public Reward {
private:
  struct PlayerState {
    bool isDribbling = false;
    int dribbleTouches = 0;
    int ticksSinceLastDribbleTouch = 0;
    bool justPopped = false;
    int ticksSincePop = 0;
    bool eligibleForBumpReward = false;
    int ticksSinceEligible = 0;
  };

  std::map<uint32_t, PlayerState> states;

public:
  float dribbleProgressReward; // Continuous reward for dribbling
  float popReward;             // Reward for successful pop
  float bumpReward;            // Reward for bump after pop
  float demoReward;            // Reward for demo after pop
  float sequenceCompleteBonus; // Extra bonus for full sequence

  ProgressiveDribblePopBumpReward(float dribbleProgressReward = 0.1f,
                                  float popReward = 3.0f,
                                  float bumpReward = 15.0f,
                                  float demoReward = 25.0f,
                                  float sequenceCompleteBonus = 5.0f)
      : dribbleProgressReward(dribbleProgressReward), popReward(popReward),
        bumpReward(bumpReward), demoReward(demoReward),
        sequenceCompleteBonus(sequenceCompleteBonus) {}

  virtual void Reset(const GameState &initialState) override { states.clear(); }

  virtual float GetReward(const Player &player, const GameState &state,
                          bool isFinal) override {
    if (!state.prev)
      return 0.0f;

    auto &pState = states[player.carId];

    pState.ticksSinceLastDribbleTouch++;
    pState.ticksSincePop++;
    pState.ticksSinceEligible++;

    float reward = 0.0f;

    // STEP 1: Reward dribbling progress
    float distToBall = (player.pos - state.ball.pos).Length();
    bool isDribblingNow = player.isOnGround && state.ball.pos.z <= 250.0f &&
                          distToBall <= 180.0f && player.vel.Length() >= 300.0f;

    if (isDribblingNow) {
      reward += dribbleProgressReward;

      if (player.ballTouchedStep) {
        if (pState.ticksSinceLastDribbleTouch < 60) {
          pState.dribbleTouches++;
        } else {
          pState.dribbleTouches = 1;
        }
        pState.ticksSinceLastDribbleTouch = 0;
      }
    }

    pState.isDribbling = isDribblingNow;

    // STEP 2: Reward pop
    bool wasDribblingRecently = pState.ticksSinceLastDribbleTouch <= 20;
    bool hadEnoughTouches = pState.dribbleTouches >= 2;
    bool ballWentHigh =
        state.ball.pos.z >= 300.0f && state.prev->ball.pos.z < 300.0f;
    bool ballMovingUp = state.ball.vel.z >= 400.0f;

    if (wasDribblingRecently && hadEnoughTouches && ballWentHigh &&
        ballMovingUp && player.ballTouchedStep) {
      reward += popReward;
      pState.justPopped = true;
      pState.ticksSincePop = 0;
      pState.eligibleForBumpReward = true;
      pState.ticksSinceEligible = 0;
    }

    if (pState.ticksSincePop > 5) {
      pState.justPopped = false;
    }

    if (pState.ticksSinceEligible > 90) {
      pState.eligibleForBumpReward = false;
      pState.dribbleTouches = 0;
    }

    // STEP 3: Reward bump/demo after pop
    if (pState.eligibleForBumpReward &&
        (player.eventState.bump || player.eventState.demo)) {

      bool didDemo = player.eventState.demo;
      reward += didDemo ? demoReward : bumpReward;

      // Full sequence bonus!
      reward += sequenceCompleteBonus;

      pState.eligibleForBumpReward = false;
      pState.dribbleTouches = 0;
    }

    return reward;
  }
};

class FlickReward45DegreeV2 : public Reward {
private:
  struct PlayerState {
    bool wasOnGround;
    bool ballWasOnRoof;
    float timeSinceBallOnRoof;
    Vec lastCarPos;

    PlayerState()
        : wasOnGround(true), ballWasOnRoof(false), timeSinceBallOnRoof(0.0f),
          lastCarPos(0, 0, 0) {}
  };

  static constexpr float TICK_SKIP = 4.0f;
  static constexpr float DT = (1.0f / 120.0f) * TICK_SKIP;
  std::unordered_map<uint32_t, PlayerState> playerStates;

public:
  float minBallSpeed;
  float maxBallSpeed;
  float angleTolerance;
  float roofDistanceMax;
  float maxTimeSinceRoof;

  FlickReward45DegreeV2(float minBallSpeed = 1200.0f,
                        float maxBallSpeed = 3500.0f,
                        float angleTolerance = 20.0f,
                        float roofDistanceMax = 150.0f,
                        float maxTimeSinceRoof = 0.3f)
      : minBallSpeed(minBallSpeed), maxBallSpeed(maxBallSpeed),
        angleTolerance(angleTolerance), roofDistanceMax(roofDistanceMax),
        maxTimeSinceRoof(maxTimeSinceRoof) {}

  virtual void Reset(const GameState &initialState) override {
    playerStates.clear();
    for (const auto &player : initialState.players) {
      PlayerState state;
      state.wasOnGround = player.isOnGround;
      state.ballWasOnRoof = false;
      state.timeSinceBallOnRoof = 999.0f;
      state.lastCarPos = player.pos;
      playerStates[player.carId] = state;
    }
  }

  virtual float GetReward(const Player &player, const GameState &state,
                          bool isFinal) override {
    uint32_t carId = player.carId;

    // Initialize if needed
    if (playerStates.find(carId) == playerStates.end()) {
      PlayerState newState;
      newState.wasOnGround = player.isOnGround;
      newState.ballWasOnRoof = false;
      newState.timeSinceBallOnRoof = 999.0f;
      newState.lastCarPos = player.pos;
      playerStates[carId] = newState;
    }

    PlayerState &ps = playerStates[carId];

    // Check if ball is currently on car roof
    Vec carToBall = state.ball.pos - player.pos;
    float distToBall = carToBall.Length();
    Vec carUp = player.rotMat.up;

    // Ball is "on roof" if it's close, above the car, and aligned with car's up
    // vector
    bool ballOnRoofNow = false;
    if (distToBall < roofDistanceMax) {
      float heightAboveCar = carToBall.Dot(carUp);
      if (heightAboveCar > 50.0f && heightAboveCar < 150.0f) {
        // Check if ball is roughly above car (not to the side)
        Vec carToBallHorizontal = carToBall - (carUp * heightAboveCar);
        if (carToBallHorizontal.Length() < 80.0f) {
          ballOnRoofNow = true;
        }
      }
    }

    // Update timing
    if (ballOnRoofNow) {
      ps.timeSinceBallOnRoof = 0.0f;
      ps.ballWasOnRoof = true;
    } else {
      ps.timeSinceBallOnRoof += DT;
    }

    float reward = 0.0f;

    // Detect flick: player touched ball recently after having it on roof
    if (player.ballTouchedStep && ps.ballWasOnRoof &&
        ps.timeSinceBallOnRoof < maxTimeSinceRoof) {

      // Check if player just left ground (jumped for flick)
      bool justJumped = ps.wasOnGround && !player.isOnGround;

      if (!state.prev) {
        ps.wasOnGround = player.isOnGround;
        ps.lastCarPos = player.pos;
        return 0.0f;
      }

      // Check ball velocity for 45-degree angle
      float ballSpeed = state.ball.vel.Length();

      if (ballSpeed >= minBallSpeed) {
        Vec ballVelNorm = state.ball.vel.Normalized();

        // Calculate angle from horizontal
        float horizontalSpeed = sqrtf(ballVelNorm.x * ballVelNorm.x +
                                      ballVelNorm.y * ballVelNorm.y);
        float angleDeg = atan2f(ballVelNorm.z, horizontalSpeed) * 57.2957795f;

        // Check if angle is near 45 degrees
        float angleDiff = fabsf(angleDeg - 45.0f);

        if (angleDiff <= angleTolerance) {
          // Power bonus (speed-based)
          float powerBonus = RS_MIN(1.0f, ballSpeed / maxBallSpeed);

          // Angle precision bonus
          float anglePrecision = 1.0f - (angleDiff / angleTolerance);

          // Height bonus (higher flicks are better)
          float heightBonus = RS_MIN(1.0f, state.ball.pos.z / 500.0f);

          // Jump bonus (if player jumped for the flick)
          float jumpBonus = justJumped ? 1.5f : 1.0f;

          // Timing bonus (quicker flick after roof = better)
          float timingBonus =
              1.0f - (ps.timeSinceBallOnRoof / maxTimeSinceRoof);
          timingBonus = RS_MAX(0.3f, timingBonus);

          reward = powerBonus * anglePrecision * heightBonus * jumpBonus *
                   timingBonus;

          // Reset state after successful flick
          ps.ballWasOnRoof = false;
          ps.timeSinceBallOnRoof = 999.0f;
        }
      }
    }

    // Update state for next frame (ALWAYS, not just when reward is 0!)
    ps.wasOnGround = player.isOnGround;
    ps.lastCarPos = player.pos;

    return reward;
  }
};

class InfiniteWalldashRewardV4 : public Reward {
private:
  struct PlayerState {
    bool wasOnWall = false;
    bool dashUsed = false;
    bool resetObtained = false;
    int consecutiveWallDashes = 0;
    float lastDashTime = 0.0f;
    Vec lastWallPosition = Vec(0, 0, 0);
    bool hadFlipBefore = false;
    float wallStayTime = 0.0f;
    int dashCount = 0;
    float totalDashTime = 0.0f;
    bool isSpamming = false;
  };
  std::map<uint32_t, PlayerState> playerStates;

public:
  float wallHeightThreshold;
  float maxTimeBetweenDashAndReset;
  float dashReward;
  float resetReward;
  float consecutiveBonus;
  float wallStayPenalty;
  float spamBonus;
  float speedBonus;
  float frequencyBonus;
  bool debug;

  InfiniteWalldashRewardV4(float wallHeightThreshold = 100.0f,
                           float maxTimeBetweenDashAndReset = 4.0f,
                           float dashReward = 12.0f, float resetReward = 16.0f,
                           float consecutiveBonus = 8.0f,
                           //	float wallStayPenalty = -0.2f,
                           float spamBonus = 20.0f, float speedBonus = 120.0f,
                           float frequencyBonus = 6.0f, bool debug = false)
      : wallHeightThreshold(wallHeightThreshold),
        maxTimeBetweenDashAndReset(maxTimeBetweenDashAndReset),
        dashReward(dashReward), resetReward(resetReward),
        consecutiveBonus(consecutiveBonus), wallStayPenalty(wallStayPenalty),
        spamBonus(spamBonus), speedBonus(speedBonus),
        frequencyBonus(frequencyBonus), debug(debug) {}

  virtual void Reset(const GameState &initialState) override {
    playerStates.clear();
    for (const auto &player : initialState.players) {
      PlayerState &state = playerStates[player.carId];
      state.wasOnWall = false;
      state.dashUsed = false;
      state.resetObtained = false;
      state.consecutiveWallDashes = 0;
      state.lastDashTime = 0.0f;
      state.lastWallPosition = Vec(0, 0, 0);
      state.hadFlipBefore = player.HasFlipOrJump();
      state.wallStayTime = 0.0f;
      state.dashCount = 0;
      state.totalDashTime = 0.0f;
      state.isSpamming = false;
    }
  }

  virtual float GetReward(const Player &player, const GameState &state,
                          bool isFinal) override {
    if (!state.prev)
      return 0.0f;

    uint32_t carId = player.carId;
    if (playerStates.find(carId) == playerStates.end()) {
      PlayerState &newState = playerStates[carId];
      newState.wasOnWall = false;
      newState.dashUsed = false;
      newState.resetObtained = false;
      newState.consecutiveWallDashes = 0;
      newState.lastDashTime = 0.0f;
      newState.lastWallPosition = Vec(0, 0, 0);
      newState.hadFlipBefore = player.HasFlipOrJump();
      newState.wallStayTime = 0.0f;
      newState.dashCount = 0;
      newState.totalDashTime = 0.0f;
      newState.isSpamming = false;
    }

    PlayerState &st = playerStates[carId];
    float reward = 0.0f;
    float currentTime = 0.0f; // Simuler le temps

    // Vérifier si le joueur est sur un mur
    bool isOnWall = IsOnWall(player.pos);
    bool hasFlip = player.HasFlipOrJump();

    // === PHASE 1: DÉTECTION DU DASH DEPUIS LE MUR ===
    if (DetectDashFromWall(player, st)) {
      reward += dashReward;
      st.dashUsed = true;
      st.lastDashTime = currentTime;
      st.dashCount++;

      if (debug) {
        printf("[WallDashV2] car_id=%d DASH #%d! Reward: %.2f\n", carId,
               st.dashCount, dashReward);
      }
    }

    // === PHASE 2: DÉTECTION DU RESET SUR LE MUR ===
    if (DetectWallReset(player, st)) {
      reward += resetReward;
      st.resetObtained = true;
      st.consecutiveWallDashes++;

      // Bonus pour les wall dashes consécutifs
      if (st.consecutiveWallDashes > 1) {
        float consecutiveReward = consecutiveBonus * st.consecutiveWallDashes;
        reward += consecutiveReward;

        if (debug) {
          printf(
              "[WallDashV2] car_id=%d WALL DASH CONSÉCUTIF #%d! Bonus: %.2f\n",
              carId, st.consecutiveWallDashes, consecutiveReward);
        }
      }

      // === BONUS SPAM WALLDASH ===
      // Plus le bot fait de dashes rapidement, plus il est récompensé
      float timeSinceDash = currentTime - st.lastDashTime;
      if (timeSinceDash < 1.0f) { // Dash rapide
        float spamReward = spamBonus * (1.0f - timeSinceDash);
        reward += spamReward;

        if (debug) {
          printf("[WallDashV2] car_id=%d SPAM WALLDASH! Temps: %.2fs, Bonus: "
                 "%.2f\n",
                 carId, timeSinceDash, spamReward);
        }
      }

      // === BONUS FRÉQUENCE ===
      // Récompense pour faire beaucoup de dashes
      if (st.dashCount >= 3) {
        float frequencyReward = frequencyBonus * (st.dashCount / 3.0f);
        reward += frequencyReward;

        if (debug && st.dashCount % 3 == 0) {
          printf("[WallDashV2] car_id=%d FRÉQUENCE ÉLEVÉE! %d dashes, Bonus: "
                 "%.2f\n",
                 carId, st.dashCount, frequencyReward);
        }
      }

      // === BONUS VITESSE ===
      // Récompense pour la vitesse du mouvement
      float playerSpeed = player.vel.Length();
      if (playerSpeed > 800.0f) {
        float speedReward = speedBonus * (playerSpeed / 1000.0f);
        reward += speedReward;

        if (debug && playerSpeed > 1200.0f) {
          printf("[WallDashV2] car_id=%d VITESSE ÉLEVÉE! %.0f, Bonus: %.2f\n",
                 carId, playerSpeed, speedReward);
        }
      }

      if (debug) {
        printf("[WallDashV2] car_id=%d RESET SUR LE MUR! Reward: %.2f\n", carId,
               resetReward);
      }
    }

    // === PHASE 3: GESTION DU TEMPS SUR LE MUR ===
    if (isOnWall) {
      st.wallStayTime += 1.0f / 120.0f; // 120 ticks par seconde

      // Pénalité plus douce pour le spam walldash
      if (st.wallStayTime > 2.0f) { // Après 2 secondes (plus permissif)
        reward += wallStayPenalty;

        if (debug && st.wallStayTime > 3.0f) {
          printf("[WallDashV2] car_id=%d RESTE TROP LONGTEMPS SUR LE MUR! "
                 "Pénalité: %.2f\n",
                 carId, wallStayPenalty);
        }
      }
    } else {
      st.wallStayTime = 0.0f;
    }

    // === PHASE 4: RÉINITIALISATION SI LE JOUEUR TOUCHE LE SOL ===
    if (player.isOnGround) {
      // Récompense finale basée sur la performance
      if (st.dashCount >= 5) {
        float finalBonus = 10.0f;
        reward += finalBonus;

        if (debug) {
          printf("[WallDashV2] car_id=%d 🎯 SESSION WALLDASH TERMINÉE! %d "
                 "dashes, Bonus final: %.2f\n",
                 carId, st.dashCount, finalBonus);
        }
      }

      // Reset des compteurs
      st.consecutiveWallDashes = 0;
      st.dashUsed = false;
      st.resetObtained = false;
      st.wallStayTime = 0.0f;
      st.dashCount = 0;
      st.totalDashTime = 0.0f;
      st.isSpamming = false;
    }

    // Mise à jour de l'état
    st.wasOnWall = isOnWall;
    st.hadFlipBefore = hasFlip;
    if (isOnWall) {
      st.lastWallPosition = player.pos;
    }

    return reward;
  }

private:
  bool IsOnWall(const Vec &playerPos) {
    // Doit être assez haut
    if (playerPos.z < wallHeightThreshold) {
      return false;
    }

    // Vérifier la proximité avec les murs latéraux (Y)
    float sideWallY = fabsf(fabsf(playerPos.y) - CommonValues::BACK_WALL_Y / 2);
    if (sideWallY < 150.0f) {
      return true;
    }

    // Vérifier la proximité avec les murs de fond (X)
    float endWallX = fabsf(fabsf(playerPos.x) - CommonValues::SIDE_WALL_X / 2);
    if (endWallX < 150.0f) {
      return true;
    }

    return false;
  }

  bool DetectDashFromWall(const Player &player, PlayerState &st) {
    // Conditions pour détecter un dash depuis le mur :
    // 1. Était sur un mur
    // 2. Avait un flip
    // 3. N'a plus de flip (l'a utilisé)
    // 4. N'est plus sur le mur (a sauté)

    return (st.wasOnWall && st.hadFlipBefore && !player.HasFlipOrJump() &&
            !IsOnWall(player.pos));
  }

  bool DetectWallReset(const Player &player, PlayerState &st) {
    // Conditions pour détecter un reset sur le mur :
    // 1. Avait utilisé un dash (n'avait plus de flip)
    // 2. Est maintenant sur un mur
    // 3. A récupéré son flip

    return (st.dashUsed && IsOnWall(player.pos) && player.HasFlipOrJump());
  }
};

class FlickeReward : public Reward {
public:
  // Tweakable parameters for flick detection
  float min_dribble_height;
  float max_dribble_height;
  float max_dribble_dist;
  float max_dribble_rel_vel;
  float min_flick_speed_boost;
  float setup_reward;
  float flick_reward_scaler;

private:
  // We'll track players who have just jumped with the ball and are ready to
  // flick. We store the tick count of when they jumped to implement a timeout.
  std::unordered_map<uint32_t, uint64_t> players_in_flick_setup;
  static constexpr uint64_t FLICK_TIMEOUT_TICKS =
      120; // 1 second timeout to complete the flick

  // Helper to determine if a player is in a stable dribble state
  bool IsDribbling(const Player &player, const GameState &state) {
    if (!player.isOnGround)
      return false;

    // Check ball height relative to the ground
    if (state.ball.pos.z < min_dribble_height ||
        state.ball.pos.z > max_dribble_height) {
      return false;
    }

    // Check horizontal distance from car to ball
    if (player.pos.Dist2D(state.ball.pos) > max_dribble_dist) {
      return false;
    }

    // Check relative velocity
    if ((player.vel - state.ball.vel).Length() > max_dribble_rel_vel) {
      return false;
    }

    return true;
  }

public:
  FlickeReward(
      float setup_reward = 1.0f, float flick_reward_scaler = 4.0f,
      float min_dribble_height =
          115.f, // A bit above ball radius + car height offset
      float max_dribble_height = 200.f,
      float max_dribble_dist = 150.f,     // Ball should be close
      float max_dribble_rel_vel = 500.f,  // Ball and car should move together
      float min_flick_speed_boost = 200.f // Must add at least this much speed
      )
      : setup_reward(setup_reward), flick_reward_scaler(flick_reward_scaler),
        min_dribble_height(min_dribble_height),
        max_dribble_height(max_dribble_height),
        max_dribble_dist(max_dribble_dist),
        max_dribble_rel_vel(max_dribble_rel_vel),
        min_flick_speed_boost(min_flick_speed_boost) {}

  virtual void Reset(const GameState &initialState) override {
    players_in_flick_setup.clear();
  }

  virtual float GetReward(const Player &player, const GameState &state,
                          bool isFinal) override {
    // We need previous states to detect state changes
    if (!player.prev || !player.prev->prev || !state.prev ||
        !state.prev->prev) {
      return 0.0f;
    }

    float reward = 0.0f;
    uint32_t carId = player.carId;

    auto it = players_in_flick_setup.find(carId);
    if (it != players_in_flick_setup.end()) {
      // Player was in flick setup state
      uint64_t jump_tick = it->second;

      // Check for timeout or landing
      if (player.isOnGround ||
          state.lastTickCount > jump_tick + FLICK_TIMEOUT_TICKS) {
        players_in_flick_setup.erase(it);
      }
      // Check if they flicked and hit the ball
      else if (player.ballTouchedStep && player.isFlipping &&
               !player.HasFlipOrJump()) {
        // Get ball speed from before the jump to measure speed increase
        float ball_speed_before_jump = state.prev->prev->ball.vel.Length();
        float ball_speed_after_flick = state.ball.vel.Length();

        float delta_speed = ball_speed_after_flick - ball_speed_before_jump;

        if (delta_speed > min_flick_speed_boost) {
          // Reward is scaled by how much speed was added
          reward =
              (delta_speed / CommonValues::CAR_MAX_SPEED) * flick_reward_scaler;
        }
        players_in_flick_setup.erase(it);
      }
    } else {
      // Player was not in setup, check if they are starting a flick now
      bool was_dribbling = IsDribbling(*player.prev, *state.prev);
      bool just_jumped = !player.isOnGround && player.prev->isOnGround;

      if (was_dribbling && just_jumped) {
        // Player has initiated a flick, give setup reward
        players_in_flick_setup[carId] = state.lastTickCount;
        reward = setup_reward;
      }
    }

    return reward;
  }
};

class SwiftGroundDribbleReward : public Reward {
public:
  const float BALL_RADIUS = 92.75f;
  float reward = 0.f;

  virtual float GetReward(const Player &player, const GameState &state,
                          bool isFinal) override {
    if (state.ball.pos.z > 100 && (BALL_RADIUS + 20) < state.ball.pos.z &&
        state.ball.pos.z < (BALL_RADIUS + 200) &&
        std::abs(std::abs(player.pos.x) - std::abs(state.ball.pos.x)) < 150 &&
        std::abs(std::abs(player.pos.y) - std::abs(state.ball.pos.y)) < 150) {

      reward += 0.2f;

      if (player.ballTouchedStep) {
        reward += 0.8f;
      }
    }

    return reward;
  }
};

class StrategicDribbleBumpReward : public Reward {
private:
  struct PlayerState {
    bool wasDribbling = false;
    bool isDribbling = false;
    uint64_t ticksSinceBump = 999;
    uint64_t ticksSinceDribbleEnd = 999;
    bool hadStrategicBump = false;
    uint32_t bumpedOpponentId = 0;
    Vec ballPosAtBump;
    float dribbleProgress =
        0.0f; // How close ball is to opponent goal when dribbling
    bool hadContactLastStep = false;
    uint32_t lastContactedOpponentId = 0;
  };

  std::unordered_map<uint32_t, PlayerState> playerStates;

  // Parameters
  float baseBumpReward;
  float demoBonus;
  float opponentNearGoalBonus;
  float dribbleProgressBonus;
  float savePenalty;
  float goalBonus;
  float dribbleTowardGoalBonus;
  float approachOpponentReward; // Reward for moving toward opponent
  float contactReward;          // Reward for making contact with opponent
  float maxOpponentGoalDist;
  float maxDribbleDist;
  float minDribbleHeight;
  float maxDribbleHeight;
  float maxContactDist;     // Distance to consider as "contact"
  uint64_t saveCheckWindow; // Ticks to check for save after bump

  bool IsDribbling(const Player &player, const GameState &state) const {
    if (!player.isOnGround)
      return false;
    if (state.ball.pos.z < minDribbleHeight ||
        state.ball.pos.z > maxDribbleHeight)
      return false;
    float dist2D = player.pos.Dist2D(state.ball.pos);
    if (dist2D > maxDribbleDist)
      return false;
    return true;
  }

  bool IsOpponentNearGoal(const Player &opponent, Team opponentTeam) const {
    Vec opponentGoal = (opponentTeam == Team::BLUE)
                           ? CommonValues::BLUE_GOAL_CENTER
                           : CommonValues::ORANGE_GOAL_CENTER;
    return opponent.pos.Dist2D(opponentGoal) < maxOpponentGoalDist;
  }

  bool IsDribblingTowardGoal(const Player &player,
                             const GameState &state) const {
    Vec opponentGoal = (player.team == Team::BLUE)
                           ? CommonValues::ORANGE_GOAL_CENTER
                           : CommonValues::BLUE_GOAL_CENTER;

    Vec toGoal = (opponentGoal - state.ball.pos).Normalized();
    Vec ballVel = state.ball.vel.Normalized();

    // Check if ball is moving toward opponent goal
    return toGoal.Dot(ballVel) > 0.3f;
  }

  bool IsBallMovingTowardOpponentGoal(const Player &player,
                                      const GameState &state) const {
    Vec opponentGoal = (player.team == Team::BLUE)
                           ? CommonValues::ORANGE_GOAL_CENTER
                           : CommonValues::BLUE_GOAL_CENTER;

    Vec toGoal = opponentGoal - state.ball.pos;
    float toGoalDist = toGoal.Length();
    if (toGoalDist < 1e-6f)
      return false; // Ball is at goal

    Vec toGoalDir = toGoal / toGoalDist;

    float ballVelLen = state.ball.vel.Length();
    if (ballVelLen < 1e-6f)
      return false; // Ball not moving

    Vec ballVelDir = state.ball.vel / ballVelLen;

    // Check if ball velocity is toward opponent goal (dot product > 0 means
    // moving toward)
    return toGoalDir.Dot(ballVelDir) > 0.0f;
  }

  float CalculateDribbleProgress(const Player &player,
                                 const GameState &state) const {
    Vec opponentGoal = (player.team == Team::BLUE)
                           ? CommonValues::ORANGE_GOAL_CENTER
                           : CommonValues::BLUE_GOAL_CENTER;

    float distToGoal = state.ball.pos.Dist2D(opponentGoal);
    // Normalize: closer to goal = higher progress (max distance ~8000)
    return RS_MAX(0.0f, 1.0f - (distToGoal / 8000.0f));
  }

public:
  StrategicDribbleBumpReward(
      float baseBumpReward = 5.0f, float demoBonus = 3.0f,
      float opponentNearGoalBonus = 4.0f, float dribbleProgressBonus = 2.0f,
      float savePenalty = -8.0f, float goalBonus = 15.0f,
      float dribbleTowardGoalBonus = 2.0f, float approachOpponentReward = 0.5f,
      float contactReward = 2.0f, float maxOpponentGoalDist = 2000.0f,
      float maxDribbleDist = 200.0f, float minDribbleHeight = 109.0f,
      float maxDribbleHeight = 250.0f, float maxContactDist = 300.0f,
      uint64_t saveCheckWindow = 180 // ~1.5 seconds at 120 tick rate
      )
      : baseBumpReward(baseBumpReward), demoBonus(demoBonus),
        opponentNearGoalBonus(opponentNearGoalBonus),
        dribbleProgressBonus(dribbleProgressBonus), savePenalty(savePenalty),
        goalBonus(goalBonus), dribbleTowardGoalBonus(dribbleTowardGoalBonus),
        approachOpponentReward(approachOpponentReward),
        contactReward(contactReward), maxOpponentGoalDist(maxOpponentGoalDist),
        maxDribbleDist(maxDribbleDist), minDribbleHeight(minDribbleHeight),
        maxDribbleHeight(maxDribbleHeight), maxContactDist(maxContactDist),
        saveCheckWindow(saveCheckWindow) {}

  virtual void Reset(const GameState &initialState) override {
    playerStates.clear();
    for (const auto &player : initialState.players) {
      playerStates[player.carId] = PlayerState();
    }
  }

  virtual float GetReward(const Player &player, const GameState &state,
                          bool isFinal) override {
    auto &pState = playerStates[player.carId];
    float reward = 0.0f;

    // Update dribbling state
    bool currentlyDribbling = IsDribbling(player, state);
    pState.wasDribbling = pState.isDribbling;
    pState.isDribbling = currentlyDribbling;

    // Update timers
    if (pState.ticksSinceBump < 999)
      pState.ticksSinceBump++;
    if (!currentlyDribbling && pState.wasDribbling) {
      pState.ticksSinceDribbleEnd = 0;
    } else if (pState.ticksSinceDribbleEnd < 999) {
      pState.ticksSinceDribbleEnd++;
    }

    // Update dribble progress when dribbling
    if (currentlyDribbling) {
      pState.dribbleProgress = CalculateDribbleProgress(player, state);
    }

    // Check for bump/demo events (only trigger once per bump)
    bool justBumped = player.eventState.bump && !pState.hadStrategicBump;
    bool justDemoed = player.eventState.demo && !pState.hadStrategicBump;

    // Check if this is a strategic bump (dribbling toward goal)
    if ((justBumped || justDemoed) && !pState.hadStrategicBump) {
      bool wasDribblingRecently =
          pState.wasDribbling || pState.ticksSinceDribbleEnd < 60;
      bool dribblingTowardGoal = IsDribblingTowardGoal(player, state);

      if (wasDribblingRecently && dribblingTowardGoal) {
        // Find the closest opponent that was bumped (most likely target)
        const Player *bumpedOpponent = nullptr;
        float closestDist = FLT_MAX;

        for (const auto &opponent : state.players) {
          if (opponent.team != player.team && opponent.eventState.bumped) {
            float dist = player.pos.Dist(opponent.pos);
            if (dist < closestDist) {
              closestDist = dist;
              bumpedOpponent = &opponent;
            }
          }
        }

        if (bumpedOpponent) {
          pState.bumpedOpponentId = bumpedOpponent->carId;
          pState.ballPosAtBump = state.ball.pos;
          pState.hadStrategicBump = true;
          pState.ticksSinceBump = 0;

          // Base reward for strategic bump
          reward += baseBumpReward;

          // Demo bonus
          if (justDemoed) {
            reward += demoBonus;
          }

          // Opponent near goal bonus
          if (IsOpponentNearGoal(*bumpedOpponent, bumpedOpponent->team)) {
            reward += opponentNearGoalBonus;
          }

          // Dribble progress bonus (closer to goal = more reward)
          if (pState.dribbleProgress > 0.5f) {
            reward += dribbleProgressBonus * pState.dribbleProgress;
          }

          // Dribbling toward goal bonus
          reward += dribbleTowardGoalBonus;
        }
      }
    }

    // Check for save after strategic bump (penalty)
    if (pState.hadStrategicBump && pState.ticksSinceBump < saveCheckWindow) {
      // Check if opponent saved
      for (const auto &opponent : state.players) {
        if (opponent.carId == pState.bumpedOpponentId &&
            opponent.eventState.save) {
          reward += savePenalty;
          pState.hadStrategicBump = false; // Reset after penalty
          pState.ticksSinceBump = 999;
        }
      }
    }

    // Goal bonus (if goal scored after strategic bump)
    if (pState.hadStrategicBump &&
        pState.ticksSinceBump < saveCheckWindow * 2) {
      if (state.goalScored) {
        // Check if our team scored
        for (const auto &p : state.players) {
          if (p.team == player.team && p.eventState.goal) {
            reward += goalBonus;
            pState.hadStrategicBump = false; // Reset after goal
            pState.ticksSinceBump = 999;
            break;
          }
        }
      }
    }

    // Reset strategic bump state after window expires
    if (pState.ticksSinceBump >= saveCheckWindow * 2) {
      pState.hadStrategicBump = false;
    }

    // Small reward for maintaining dribble toward goal
    if (currentlyDribbling && IsDribblingTowardGoal(player, state)) {
      reward += 0.1f * pState.dribbleProgress;
    }

    // Check if ball is moving toward opponent goal - REQUIRED for
    // approach/contact rewards
    bool ballMovingTowardGoal = IsBallMovingTowardOpponentGoal(player, state);

    // If ball is not moving toward opponent goal, skip approach/contact rewards
    if (!ballMovingTowardGoal) {
      // Update contact state
      if (!player.eventState.bump && !player.eventState.demo) {
        // Only reset if we're not in close proximity to any opponent
        bool stillNearOpponent = false;
        for (const auto &opponent : state.players) {
          if (opponent.team != player.team) {
            if (player.pos.Dist(opponent.pos) < maxContactDist * 1.5f) {
              stillNearOpponent = true;
              break;
            }
          }
        }
        if (!stillNearOpponent) {
          pState.hadContactLastStep = false;
        }
      }
      return reward; // Return early - no approach/contact rewards if ball not
                     // moving toward goal
    }

    // Reward for approaching opponents and detect contact
    for (const auto &opponent : state.players) {
      if (opponent.team == player.team)
        continue;

      float distToOpponent = player.pos.Dist(opponent.pos);

      // Check if moving toward opponent
      Vec toOpponentVec = opponent.pos - player.pos;
      float toOpponentDist = toOpponentVec.Length();
      if (toOpponentDist < 1e-6f)
        continue; // Skip if cars are on top of each other

      Vec toOpponent = toOpponentVec / toOpponentDist;

      float playerVelLen = player.vel.Length();
      if (playerVelLen < 1e-6f)
        continue; // Skip if not moving

      Vec playerVelDir = player.vel / playerVelLen;
      float velTowardOpponent = player.vel.Dot(toOpponent);

      // Reward for moving toward opponent (scaled by speed and proximity)
      if (velTowardOpponent > 0.0f && distToOpponent < 2000.0f) {
        float approachReward =
            approachOpponentReward *
            (velTowardOpponent / CommonValues::CAR_MAX_SPEED);
        // Scale by proximity (closer = more reward)
        float proximityScale = RS_MAX(0.0f, 1.0f - (distToOpponent / 2000.0f));
        reward += approachReward * proximityScale;
      }

      // Detect contact (either through bump event or close proximity)
      bool madeContact = false;

      // Check for bump/demo events (explicit contact)
      // Only count if player initiated the contact (player has bump/demo event)
      if ((player.eventState.bump || player.eventState.demo) &&
          (opponent.eventState.bumped || opponent.eventState.demoed)) {
        // Verify this is the opponent we contacted
        if (distToOpponent < maxContactDist * 2.0f) {
          madeContact = true;
          pState.lastContactedOpponentId = opponent.carId;
        }
      }

      // Also detect contact through proximity (cars very close together)
      // This catches cases where cars make contact but might not trigger bump
      // events immediately
      if (!madeContact && distToOpponent < maxContactDist) {
        // Check if we're moving toward each other (collision likely)
        float opponentVelLen = opponent.vel.Length();
        Vec opponentVelDir = (opponentVelLen > 1e-6f)
                                 ? (opponent.vel / opponentVelLen)
                                 : Vec(0, 0, 0);
        float relativeSpeed = (player.vel - opponent.vel).Length();

        // If cars are close and moving toward each other, consider it contact
        if (relativeSpeed > 500.0f && toOpponent.Dot(playerVelDir) > 0.3f) {
          // Only reward if we haven't already rewarded for this contact
          if (!pState.hadContactLastStep ||
              pState.lastContactedOpponentId != opponent.carId) {
            madeContact = true;
            pState.lastContactedOpponentId = opponent.carId;
          }
        }
      }

      // Reward for making contact (only once per contact)
      if (madeContact && (!pState.hadContactLastStep ||
                          pState.lastContactedOpponentId != opponent.carId)) {
        reward += contactReward;

        // Bonus if contact happened while dribbling toward goal
        if (currentlyDribbling && IsDribblingTowardGoal(player, state)) {
          reward += contactReward * 0.5f;
        }

        // Bonus if opponent is near their goal
        if (IsOpponentNearGoal(opponent, opponent.team)) {
          reward += contactReward * 0.5f;
        }

        // Mark that we had contact this step
        pState.hadContactLastStep = true;
      }
    }

    // Update contact state (reset if no contact this step)
    if (!player.eventState.bump && !player.eventState.demo) {
      // Only reset if we're not in close proximity to any opponent
      bool stillNearOpponent = false;
      for (const auto &opponent : state.players) {
        if (opponent.team != player.team) {
          if (player.pos.Dist(opponent.pos) < maxContactDist * 1.5f) {
            stillNearOpponent = true;
            break;
          }
        }
      }
      if (!stillNearOpponent) {
        pState.hadContactLastStep = false;
      }
    }

    return reward;
  }
};

class TeammateBumpPenaltyReward : public Reward {
private:
  std::unordered_map<int, uint32_t> previousContactCarID;
  std::unordered_map<int, float> previousContactTimer;

public:
  TeammateBumpPenaltyReward() {}

  virtual void Reset(const GameState &initialState) override {
    previousContactCarID.clear();
    previousContactTimer.clear();
  }

  virtual float GetReward(const Player &player, const GameState &state,
                          bool isFinal) override {
    float reward = 0.0f;

    uint32_t currentContactCarID = player.carContact.otherCarID;
    float currentContactTimer = player.carContact.cooldownTimer;

    auto prevCarIDIter = previousContactCarID.find(player.carId);
    auto prevTimerIter = previousContactTimer.find(player.carId);

    bool hadPreviousState = (prevCarIDIter != previousContactCarID.end() &&
                             prevTimerIter != previousContactTimer.end());

    if (currentContactTimer > 0 && currentContactCarID != 0) {
      bool isNewBump = false;

      if (!hadPreviousState) {
        isNewBump = true;
      } else {
        float prevTimer = prevTimerIter->second;
        uint32_t prevCarID = prevCarIDIter->second;

        isNewBump = (currentContactTimer > prevTimer) ||
                    (currentContactCarID != prevCarID && prevTimer <= 0);
      }

      if (isNewBump) {
        for (const auto &otherPlayer : state.players) {
          if (otherPlayer.carId == currentContactCarID) {
            if (otherPlayer.team == player.team) {
              reward -= 1.0f;
            }
            break;
          }
        }
      }
    }

    previousContactCarID[player.carId] = currentContactCarID;
    previousContactTimer[player.carId] = currentContactTimer;

    return reward;
  }
};

class RaptorTeamSpacingReward : public Reward {
public:
  double minSpacing;

  RaptorTeamSpacingReward(double minSpacing = 1550.f)
      : minSpacing(std::clamp(minSpacing, 0.0000001,
                              std::numeric_limits<double>::infinity())) {}

  virtual float GetReward(const Player &player, const GameState &state,
                          bool isFinal) {
    double reward = 0.f;
    for (auto _player : state.players) {
      if ((_player.team == player.team) && (_player.carId != player.carId) &&
          !player.isDemoed && !_player.isDemoed) {
        double separation = player.pos.Dist(_player.pos);
        if (separation < minSpacing) {
          reward -= 1 - (separation / minSpacing);
        }
      }
    }
    return reward;
  }
};

class HyperNoStackReward : public Reward {
private:
  struct PlayerState {
    bool wasOnWall = false;
    bool dashUsed = false;
    bool hadFlipBefore = false;
    bool wasFlipping = false;
    Vec prevVel = Vec(0, 0, 0);
    // int lastDemoCount = 0; // Removed: Not supported by current Player struct
  };

  std::map<uint32_t, PlayerState> playerStates;

  // --- Configuration ---
  float wallHeightThreshold;

  // Base Rewards
  float dashRewardBase;
  float resetRewardBase;
  float wavedashRewardBase;
  float zapDashBaseReward;
  // float demoReward; // Removed temporarily to fix build

  // Quality Scaling
  float accelerationScalar;

  // Penalties
  float wallStayPenalty;
  float supersonicBoostPenalty;

  // Zap Dash Config
  float zapMinSpeedGain;
  float zapMinNoseDown;
  float zapMinFwdDot;

  bool debug;

public:
  HyperNoStackReward(
      // Base Values
      float dashRewardBase = 12.0f, float resetRewardBase = 16.0f,
      float wavedashRewardBase = 10.0f, float zapDashBaseReward = 15.0f,
      // float demoReward = 30.0f,
      // Scaling
      float accelerationScalar = 1.0f,
      // Penalties
      float wallStayPenalty = -0.2f, float supersonicBoostPenalty = -0.1f,
      // Debug
      bool debug = false)
      : wallHeightThreshold(100.0f), dashRewardBase(dashRewardBase),
        resetRewardBase(resetRewardBase),
        wavedashRewardBase(wavedashRewardBase),
        zapDashBaseReward(zapDashBaseReward),
        // demoReward(demoReward),
        accelerationScalar(accelerationScalar),
        wallStayPenalty(wallStayPenalty),
        supersonicBoostPenalty(supersonicBoostPenalty), zapMinSpeedGain(500.0f),
        zapMinNoseDown(0.5f), zapMinFwdDot(0.7f), debug(debug) {}

  virtual void Reset(const GameState &initialState) override {
    playerStates.clear();
    for (const auto &player : initialState.players) {
      InitState(player);
    }
  }

  virtual float GetReward(const Player &player, const GameState &state,
                          bool isFinal) override {
    if (!state.prev)
      return 0.0f;

    if (playerStates.find(player.carId) == playerStates.end()) {
      InitState(player);
    }
    PlayerState &st = playerStates[player.carId];

    float totalReward = 0.0f;
    float dt = 1.0f / 120.0f;

    // --- PRE-CALCULATIONS ---
    bool isOnWall = IsOnWall(player.pos);
    bool hasFlip = player.HasFlipOrJump();

    // Calculate Acceleration Quality
    float acceleration = (player.vel - st.prevVel).Length() / dt;
    float normAccel = std::min(acceleration / 5000.0f, 2.0f);
    float qualityMult = 1.0f + (normAccel * accelerationScalar);

    // =========================================================
    // MECHANIC DETECTION
    // =========================================================

    // 1. DEMOLITION (Disabled to fix build error)
    /*
    if (player.match_demolitions > st.lastDemoCount) {
            totalReward += demoReward * qualityMult;
            if (debug) printf("ID:%d | DEMO | Reward: %.2f\n", player.carId,
    demoReward * qualityMult);
    }
    */

    // 2. ZAP DASH
    float zapRatio = CheckZapDash(player);
    if (zapRatio > 0.0f) {
      float r = zapDashBaseReward * zapRatio * qualityMult;
      totalReward += r;
      if (debug)
        printf("ID:%d | ZapDash | Reward: %.2f\n", player.carId, r);
    }

    // 3. WALL DASH (LAUNCH)
    if (st.wasOnWall && st.hadFlipBefore && !hasFlip && !isOnWall) {
      st.dashUsed = true;
      float r = dashRewardBase * qualityMult;
      totalReward += r;
      if (debug)
        printf("ID:%d | WallDash_Launch | Reward: %.2f\n", player.carId, r);
    }

    // 4. WALL DASH (RESET)
    if (st.dashUsed && isOnWall && hasFlip) {
      st.dashUsed = false; // Cycle complete
      float r = resetRewardBase * qualityMult;
      totalReward += r;
      if (debug)
        printf("ID:%d | WallDash_Reset | Reward: %.2f\n", player.carId, r);
    }

    // 5. WAVEDASH
    const Player *prevPlayer = nullptr;
    for (const auto &p : state.prev->players) {
      if (p.carId == player.carId) {
        prevPlayer = &p;
        break;
      }
    }

    bool justLanded =
        player.isOnGround && prevPlayer && !prevPlayer->isOnGround;
    if (justLanded && (player.isFlipping || st.wasFlipping)) {
      float r = wavedashRewardBase * qualityMult;
      totalReward += r;
      if (debug)
        printf("ID:%d | Wavedash | Reward: %.2f\n", player.carId, r);
    }

    // =========================================================
    // PENALTIES
    // =========================================================

    // 1. Supersonic Boost Waste
    if (prevPlayer && player.isSupersonic && player.boost < prevPlayer->boost) {
      totalReward += supersonicBoostPenalty;
    }

    // 2. Conditional Wall Stay Penalty
    if (isOnWall && st.dashUsed && player.vel.Length() < 600.0f) {
      totalReward += wallStayPenalty;
    }

    // =========================================================
    // STATE UPDATES
    // =========================================================
    st.wasOnWall = isOnWall;
    st.hadFlipBefore = hasFlip;
    st.wasFlipping = player.isFlipping;
    st.prevVel = player.vel;
    // st.lastDemoCount = player.match_demolitions; // Disabled

    if (player.isOnGround) {
      st.dashUsed = false;
    }

    return totalReward;
  }

private:
  void InitState(const Player &p) {
    PlayerState newState;
    newState.prevVel = p.vel;
    newState.hadFlipBefore = p.HasFlipOrJump();
    // newState.lastDemoCount = p.match_demolitions; // Disabled
    playerStates[p.carId] = newState;
  }

  float CheckZapDash(const Player &current) {
    if (!current.prev || !current.prev->prev || !current.prev->prev->prev)
      return 0.0f;

    const Player *p0 = &current;
    const Player *p1 = current.prev;
    const Player *p2 = p1->prev;
    const Player *p3 = p2->prev;

    bool wasAirborneT3 = !p3->isOnGround;
    bool landedT2 = wasAirborneT3 && p2->isOnGround;
    bool noseDown = p2->rotMat.forward.z < -zapMinNoseDown;
    bool wasMovingDown = p2->vel.z < -200.0f;
    bool executingFlip = p0->isFlipping;

    if (landedT2 && noseDown && wasMovingDown && executingFlip) {
      Vec carFwd = p1->rotMat.forward;
      Vec velDir = p0->vel.Normalized();
      if (carFwd.Dot(velDir) > zapMinFwdDot) {
        float gain = p0->vel.Length() - p2->vel.Length();
        if (gain > zapMinSpeedGain) {
          float bonus = (p0->vel.Length() > 2200.f) ? 1.5f : 1.0f;
          return (gain / 1000.0f) * bonus;
        }
      }
    }
    return 0.0f;
  }

  bool IsOnWall(const Vec &pos) {
    if (pos.z < wallHeightThreshold)
      return false;
    if (std::abs(std::abs(pos.x) - 4096.0f) < 150.0f)
      return true;
    if (std::abs(std::abs(pos.y) - 5120.0f) < 150.0f)
      return true;
    return false;
  }
};

class AirDribbleReward : public Reward {
public:
  float minHeight;
  float maxHeight;
  float distanceThreshold;

  AirDribbleReward(float minHeight = 350.0f, float maxHeight = 1500.0f,
                   float distanceThreshold = 400.0f)
      : minHeight(minHeight), maxHeight(maxHeight),
        distanceThreshold(distanceThreshold) {}

  virtual float GetReward(const Player &player, const GameState &state,
                          bool isFinal) override {
    if (player.isOnGround)
      return 0.0f;

    if (state.ball.pos.z < minHeight)
      return 0.0f;

    float dist = (state.ball.pos - player.pos).Length();
    if (dist > distanceThreshold)
      return 0.0f;

    bool targetOrangeGoal = player.team == Team::BLUE;
    Vec targetPos = targetOrangeGoal ? CommonValues::ORANGE_GOAL_BACK
                                     : CommonValues::BLUE_GOAL_BACK;
    Vec toGoal = (targetPos - state.ball.pos).Normalized();

    float velProj = state.ball.vel.Dot(toGoal);

    if (velProj <= 0.0f)
      return 0.0f;

    float velReward = velProj / CommonValues::BALL_MAX_SPEED;

    float clampedHeight = std::min(state.ball.pos.z, maxHeight);
    float heightScale = clampedHeight / maxHeight;

    return velReward * heightScale;
  }
};

class KickoffSpeedflipReward : public Reward {
private:
  struct PlayerState {
    bool kickoffDetected = false;
    bool jumpUsed = false;
    float jumpTime = -1.0f;
    bool diagonalInputDetected = false;
    bool flipCancelDetected = false;
    bool speedflipCompleted = false;
    Vec lastPos = Vec(0, 0, 0);
    float lastTime = 0.0f;
    Vec velocityBeforeFlip = Vec(0, 0, 0);
    Vec velocityAfterFlip = Vec(0, 0, 0);
  };
  std::map<uint32_t, PlayerState> playerStates;
  float kickoffStartTime = -1.0f;
  int frameCount = 0;

public:
  float kickoffDetectionTime;
  float maxJumpDelay;
  float minSpeedThreshold;
  float diagonalInputThreshold;
  float cancelThreshold;
  float maxCancelDelay;
  float rewardValue;
  bool debug;

  KickoffSpeedflipReward(float kickoffDetectionTime = 3.0f,
                         float maxJumpDelay = 0.5f,
                         float minSpeedThreshold = 1200.0f,
                         float diagonalInputThreshold = 0.3f,
                         float cancelThreshold = -0.5f,
                         float maxCancelDelay = 0.3f, float rewardValue = 5.0f,
                         bool debug = false)
      : kickoffDetectionTime(kickoffDetectionTime), maxJumpDelay(maxJumpDelay),
        minSpeedThreshold(minSpeedThreshold),
        diagonalInputThreshold(diagonalInputThreshold),
        cancelThreshold(cancelThreshold), maxCancelDelay(maxCancelDelay),
        rewardValue(rewardValue), debug(debug) {}

  virtual void Reset(const GameState &initialState) override {
    playerStates.clear();
    kickoffStartTime = -1.0f;
    frameCount = 0;
    for (const auto &player : initialState.players) {
      PlayerState &state = playerStates[player.carId];
      state.kickoffDetected = false;
      state.jumpUsed = false;
      state.jumpTime = -1.0f;
      state.diagonalInputDetected = false;
      state.flipCancelDetected = false;
      state.speedflipCompleted = false;
      state.lastPos = player.pos;
      state.lastTime = 0.0f;
      state.velocityBeforeFlip = Vec(0, 0, 0);
      state.velocityAfterFlip = Vec(0, 0, 0);
    }
  }

  virtual float GetReward(const Player &player, const GameState &state,
                          bool isFinal) override {
    uint32_t carId = player.carId;

    if (playerStates.find(carId) == playerStates.end()) {
      PlayerState &newState = playerStates[carId];
      newState.kickoffDetected = false;
      newState.jumpUsed = false;
      newState.jumpTime = -1.0f;
      newState.diagonalInputDetected = false;
      newState.flipCancelDetected = false;
      newState.speedflipCompleted = false;
      newState.lastPos = player.pos;
      newState.lastTime = 0.0f;
      newState.velocityBeforeFlip = Vec(0, 0, 0);
      newState.velocityAfterFlip = Vec(0, 0, 0);
    }

    PlayerState &st = playerStates[carId];
    float reward = 0.0f;
    float currentTime = frameCount * 0.016f;
    float dt = (st.lastTime > 0) ? (currentTime - st.lastTime) : 0.016f;
    frameCount++;

    // DÃ©tecter le kickoff
    if (!st.kickoffDetected && IsKickoffPosition(player.pos)) {
      st.kickoffDetected = true;
      kickoffStartTime = currentTime;
    }

    // DÃ©tecter le saut (avec condition kickoff)
    if (st.kickoffDetected && !st.jumpUsed && player.HasFlipOrJump()) {
      float jumpDelay =
          (kickoffStartTime >= 0) ? (currentTime - kickoffStartTime) : 0;
      if (jumpDelay <= maxJumpDelay) {
        st.jumpUsed = true;
        st.jumpTime = currentTime;
        st.velocityBeforeFlip =
            player.vel; // Sauvegarder la vÃ©locitÃ© avant le flip
      }
    }

    // DÃ©tecter l'input diagonal aprÃ¨s le saut (via la vÃ©locitÃ©)
    if (st.jumpUsed && !st.diagonalInputDetected) {
      // VÃ©rifier si la voiture se dÃ©place en diagonal (composantes X et Y
      // significatives)
      float velX = fabsf(player.vel.x);
      float velY = fabsf(player.vel.y);
      float velZ = fabsf(player.vel.z);

      // Le speedflip au kickoff a des composantes X et Y Ã©levÃ©es avec peu de
      // Z
      if (velX > diagonalInputThreshold * 1000.0f &&
          velY > diagonalInputThreshold * 1000.0f && velZ < 500.0f) {
        st.diagonalInputDetected = true;
      }
    }

    // DÃ©tecter le flip cancel (via la vÃ©locitÃ© et la rotation)
    if (st.jumpUsed && st.diagonalInputDetected && !st.flipCancelDetected) {
      float cancelDelay = (st.jumpTime >= 0) ? (currentTime - st.jumpTime) : 0;
      if (cancelDelay <= maxCancelDelay) {
        // Le flip cancel se caractÃ©rise par une vÃ©locitÃ© Z qui reste basse
        // et une rotation de la voiture qui s'arrÃªte
        if (fabsf(player.vel.z) < 300.0f && player.rotMat.up.z > 0.7f) {
          st.flipCancelDetected = true;
          st.velocityAfterFlip =
              player.vel; // Sauvegarder la vÃ©locitÃ© aprÃ¨s le flip
        }
      }
    }

    // VÃ©rifier le speedflip complet (accÃ©lÃ©ration et direction)
    if (st.jumpUsed && st.diagonalInputDetected && st.flipCancelDetected &&
        !st.speedflipCompleted) {
      // Calculer l'accÃ©lÃ©ration due au speedflip
      float speedBefore = st.velocityBeforeFlip.Length();
      float speedAfter = st.velocityAfterFlip.Length();
      float acceleration = (speedAfter - speedBefore) / dt;

      // VÃ©rifier que la vitesse est suffisante et qu'il y a eu accÃ©lÃ©ration
      if (speedAfter >= minSpeedThreshold && acceleration > 500.0f) {
        st.speedflipCompleted = true;
        reward = rewardValue;
      }
    }

    st.lastPos = player.pos;
    st.lastTime = currentTime;

    return reward;
  }

private:
  bool IsKickoffPosition(const Vec &playerPos) {
    // Positions de kickoff exactes
    Vec kickoffPositions[] = {
        Vec(0, -5120, 0),     // Centre
        Vec(-2048, -5120, 0), // Gauche
        Vec(2048, -5120, 0),  // Droite
        Vec(-2048, 5120, 0),  // Gauche adverse
        Vec(2048, 5120, 0)    // Droite adverse
    };

    for (const auto &pos : kickoffPositions) {
      if ((playerPos - pos).Length() < 500.0f) {
        return true;
      }
    }
    return false;
  }

  float CalculateSpeed(const Vec &currentPos, const Vec &lastPos, float dt) {
    if (dt <= 0)
      return 0.0f;
    Vec displacement = currentPos - lastPos;
    return displacement.Length() / dt;
  }
};

class JumpTouchReward : public Reward {
public:
  float minHeight, maxHeight, range;

  JumpTouchReward(float minHeight = 150.75f, float maxHeight = 300.0f)
      : minHeight(minHeight), maxHeight(maxHeight) {
    range = maxHeight - minHeight;
  }

  virtual float GetReward(const Player &player, const GameState &state,
                          bool isFinal) override {
    if (player.ballTouchedStep && !player.isOnGround &&
        state.ball.pos.z >= minHeight) {
      return (state.ball.pos.z - minHeight) / range;
    }
    return 0;
  }
};

class AerialReward : public Reward {
public:
  AerialReward()
      : in_air_reward(0), last_touch_pos(std::nullopt), lost_jump(false),
        reward(0), dist_to_ball(), on_wall(false) {}

  virtual void Reset(const GameState &initialState) override {
    last_touch_pos = std::nullopt;
    lost_jump = false;
    reward = 0;
  }

  virtual float GetReward(const Player &player, const GameState &state,
                          bool isFinal) override {
    // Starting dribble reward
    if (130 < state.ball.pos.z && state.ball.pos.z < 180 &&
        player.ballTouchedStep) {
      float ball_speed = state.ball.vel.Length();
      if (ball_speed < 2000) {
        reward += 2;
      }
    }

    if (state.ball.pos.z > 230) {
      if (player.ballTouchedStep) {
        reward += 1;
        last_touch_pos = state.ball.pos;

        if (220 < state.ball.pos.z && state.ball.pos.z < 400) {
          float ball_speed = state.ball.vel.Length();
          float ball_speed_reward = ball_speed / CommonValues::BALL_MAX_SPEED;
          reward += ball_speed_reward;
        }

        if (!player.hasJumped) {
          if (!lost_jump) {
            lost_jump = true;
          }

          float ball_speed = state.ball.vel.Length();
          float ball_speed_reward =
              ball_speed / (10 * CommonValues::BALL_MAX_SPEED);
          reward += ball_speed_reward;
        }
      } else if (player.pos.z > 300) {
        Vec ballToPlayer = state.ball.pos - player.pos;
        float dist = ballToPlayer.Length() - CommonValues::BALL_RADIUS;
        float dist_reward = (dist * dist) / 600000.0f + 0.5f;
        if (dist_reward < 0.6f) {
          float height_reward = state.ball.pos.z / CommonValues::CEILING_Z;
          reward += height_reward;
          float dist_to_ball_reward =
              dist_to_ball.GetReward(player, state, isFinal) * 10;
          reward += dist_to_ball_reward;
        }
      }
      return reward;
    } else {
      reward = 0;
      if (lost_jump) {
        lost_jump = false;
      }
      return reward;
    }
  }

private:
  int in_air_reward;
  std::optional<Vec> last_touch_pos;
  bool lost_jump;
  float reward;
  DistToBallReward dist_to_ball;
  bool on_wall;
};

// Rewards the bot for recovering quickly to defense after a failed shot
// when the opponent has possession and no one is defending the goal
class GoodRecoveryReward : public Reward {
private:
  struct PlayerState {
    bool hadPossessionRecently = false;
    int ticksSinceLostPossession = 0;
    bool wasMovingTowardEnemyGoal = false;
    Vec lastPos = Vec(0, 0, 0);
  };

  std::map<uint32_t, PlayerState> playerStates;

  // Configuration
  float recoverySpeedWeight;  // Reward for moving toward own goal quickly
  float positioningWeight;    // Reward for getting between ball and goal
  float goalProximityWeight;  // Reward for reaching defensive position
  float possessionLossWindow; // Ticks to consider "recent" possession loss
  float goalDefenseRadius;    // Radius to consider "in goal" for defenders

public:
  GoodRecoveryReward(
      float recoverySpeedWeight = 1.0f, float positioningWeight = 1.5f,
      float goalProximityWeight = 0.5f,
      float possessionLossWindow = 180.0f, // ~1.5 seconds at 120 ticks/sec
      float goalDefenseRadius = 1500.0f)
      : recoverySpeedWeight(recoverySpeedWeight),
        positioningWeight(positioningWeight),
        goalProximityWeight(goalProximityWeight),
        possessionLossWindow(possessionLossWindow),
        goalDefenseRadius(goalDefenseRadius) {}

  virtual void Reset(const GameState &initialState) override {
    playerStates.clear();
    for (const auto &player : initialState.players) {
      playerStates[player.carId] = PlayerState();
    }
  }

  virtual float GetReward(const Player &player, const GameState &state,
                          bool isFinal) override {
    if (playerStates.find(player.carId) == playerStates.end()) {
      playerStates[player.carId] = PlayerState();
    }
    auto &pState = playerStates[player.carId];

    Vec ownGoal = (player.team == Team::BLUE)
                      ? CommonValues::BLUE_GOAL_CENTER
                      : CommonValues::ORANGE_GOAL_CENTER;
    Vec enemyGoal = (player.team == Team::BLUE)
                        ? CommonValues::ORANGE_GOAL_CENTER
                        : CommonValues::BLUE_GOAL_CENTER;

    // === Track possession state ===
    bool hasPossession = player.ballTouchedStep > 0;
    bool movingTowardEnemyGoal = false;

    if (player.vel.Length() > 100.0f) {
      Vec dirToEnemyGoal = (enemyGoal - player.pos).Normalized();
      movingTowardEnemyGoal =
          player.vel.Normalized().Dot(dirToEnemyGoal) > 0.5f;
    }

    // Detect if we just lost possession after attacking
    if (pState.hadPossessionRecently && !hasPossession) {
      pState.ticksSinceLostPossession++;
    }

    // Update possession tracking
    if (hasPossession && movingTowardEnemyGoal) {
      pState.hadPossessionRecently = true;
      pState.wasMovingTowardEnemyGoal = true;
      pState.ticksSinceLostPossession = 0;
    }

    // Reset if too much time passed
    if (pState.ticksSinceLostPossession > possessionLossWindow) {
      pState.hadPossessionRecently = false;
      pState.wasMovingTowardEnemyGoal = false;
    }

    // === Check if opponent has possession ===
    bool opponentHasPossession = false;
    float closestOpponentDist = FLT_MAX;
    float ourClosestDist = FLT_MAX;

    for (const auto &p : state.players) {
      float distToBall = p.pos.Dist(state.ball.pos);
      if (p.team != player.team) {
        if (distToBall < closestOpponentDist) {
          closestOpponentDist = distToBall;
        }
        // Check if opponent touched ball recently
        if (p.ballTouchedStep > 0) {
          opponentHasPossession = true;
        }
      } else {
        if (distToBall < ourClosestDist) {
          ourClosestDist = distToBall;
        }
      }
    }

    // Opponent is closer to ball = they have possession
    if (closestOpponentDist < ourClosestDist - 200.0f) {
      opponentHasPossession = true;
    }

    // === Check if goal is undefended ===
    bool goalUndefended = true;
    for (const auto &p : state.players) {
      if (p.team == player.team && p.carId != player.carId) {
        float distToOwnGoal = p.pos.Dist(ownGoal);
        if (distToOwnGoal < goalDefenseRadius) {
          goalUndefended = false;
          break;
        }
      }
    }

    // Also check if ball is threatening (moving toward our goal)
    Vec ballToOwnGoal = ownGoal - state.ball.pos;
    float ballMovingTowardGoal =
        state.ball.vel.Normalized().Dot(ballToOwnGoal.Normalized());
    bool ballThreatening = ballMovingTowardGoal > 0.3f;

    // === Calculate reward ===
    float reward = 0.0f;

    // Only apply recovery reward if:
    // 1. We recently lost possession after attacking
    // 2. Opponent has the ball OR ball is threatening
    // 3. Goal is undefended
    bool shouldRecover =
        pState.wasMovingTowardEnemyGoal &&
        pState.ticksSinceLostPossession > 0 &&
        pState.ticksSinceLostPossession < possessionLossWindow &&
        (opponentHasPossession || ballThreatening) && goalUndefended;

    if (shouldRecover) {
      // 1. Reward for moving toward own goal
      Vec dirToOwnGoal = (ownGoal - player.pos).Normalized();
      float speedTowardGoal = player.vel.Dot(dirToOwnGoal);
      if (speedTowardGoal > 0) {
        float speedReward = speedTowardGoal / CommonValues::CAR_MAX_SPEED;
        reward += speedReward * recoverySpeedWeight;
      }

      // 2. Reward for getting between ball and own goal
      Vec ballToGoal = ownGoal - state.ball.pos;
      Vec ballToPlayer = player.pos - state.ball.pos;

      if (ballToGoal.Length() > 1e-6f && ballToPlayer.Length() > 1e-6f) {
        float alignment =
            ballToGoal.Normalized().Dot(ballToPlayer.Normalized());
        if (alignment > 0) {
          // Player is between ball and goal
          reward += alignment * positioningWeight;
        }
      }

      // 3. Reward for reaching defensive position
      float distToOwnGoal = player.pos.Dist(ownGoal);
      if (distToOwnGoal < goalDefenseRadius) {
        float proximityFactor = 1.0f - (distToOwnGoal / goalDefenseRadius);
        reward += proximityFactor * goalProximityWeight;

        // Reset state once we reach defense
        if (proximityFactor > 0.7f) {
          pState.hadPossessionRecently = false;
          pState.wasMovingTowardEnemyGoal = false;
        }
      }

      // 4. Bonus for supersonic recovery
      if (player.isSupersonic && speedTowardGoal > 0) {
        reward += 0.5f;
      }
    }

    pState.lastPos = player.pos;
    return reward;
  }
};

// Rewards musty flicks - a mechanic where the car tilts backward and backflips
// to flick the ball with the nose of the car
class MustyFlickReward : public Reward {
private:
  struct PlayerState {
    bool wasInSetupPosition = false; // Car tilted back with ball on top
    float setupStartTick = 0;
    float maxBackwardTilt = 0.0f; // Track how much they committed to the tilt
    bool wasFlipping = false;
    Vec ballVelBeforeFlick = Vec(0, 0, 0);
  };

  std::map<uint32_t, PlayerState> playerStates;

  // Configuration
  float minTiltAngle;     // Minimum backward tilt to start setup (radians)
  float maxBallDistance;  // Max distance from ball during setup
  float minBallHeight;    // Ball must be above car
  float setupReward;      // Reward for getting into position
  float executionReward;  // Reward for executing the flick
  float powerBonus;       // Bonus based on ball speed after flick
  float setupWindowTicks; // How long setup position can be held

public:
  MustyFlickReward(float minTiltAngle = 0.4f, // ~23 degrees backward tilt
                   float maxBallDistance = 250.0f,
                   float minBallHeight = 50.0f, // Ball should be above car
                   float setupReward = 0.5f, float executionReward = 5.0f,
                   float powerBonus = 3.0f,
                   float setupWindowTicks = 60.0f // ~0.5 seconds
                   )
      : minTiltAngle(minTiltAngle), maxBallDistance(maxBallDistance),
        minBallHeight(minBallHeight), setupReward(setupReward),
        executionReward(executionReward), powerBonus(powerBonus),
        setupWindowTicks(setupWindowTicks) {}

  virtual void Reset(const GameState &initialState) override {
    playerStates.clear();
    for (const auto &player : initialState.players) {
      playerStates[player.carId] = PlayerState();
    }
  }

  virtual float GetReward(const Player &player, const GameState &state,
                          bool isFinal) override {
    if (!player.prev)
      return 0.0f;

    if (playerStates.find(player.carId) == playerStates.end()) {
      playerStates[player.carId] = PlayerState();
    }
    auto &pState = playerStates[player.carId];

    float reward = 0.0f;

    Vec ballPos = state.ball.pos;
    Vec carPos = player.pos;
    Vec carToBall = ballPos - carPos;
    float distToBall = carToBall.Length();

    // Get car orientation - forward vector and up vector
    Vec carForward = player.rotMat.forward;
    Vec carUp = player.rotMat.up;

    // Calculate pitch angle (positive = nose up, negative = nose down)
    // When the car tilts backward for a musty, the forward vector points more
    // upward
    float pitchAngle = std::asin(carForward.z);

    // Check if ball is above and slightly in front of the car
    bool ballAboveCar = carToBall.z > minBallHeight;
    bool ballClose = distToBall < maxBallDistance;

    // Detect if car is tilted backward (nose up) - musty setup position
    bool isBackwardTilt = pitchAngle > minTiltAngle;

    // Check if doing a backflip (flipping with backward pitch input)
    bool isBackflipping = player.isFlipping && player.prevAction.pitch > 0.5f;
    bool justStartedFlip = player.isFlipping && !pState.wasFlipping;

    // === SETUP PHASE ===
    // Reward getting into musty position (ball on car, car tilted back)
    if (ballAboveCar && ballClose && isBackwardTilt && !player.isOnGround) {
      if (!pState.wasInSetupPosition) {
        pState.wasInSetupPosition = true;
        pState.setupStartTick = 0;
        pState.maxBackwardTilt = pitchAngle;
        pState.ballVelBeforeFlick = state.ball.vel;
      }
      pState.setupStartTick++;
      pState.maxBackwardTilt = std::max(pState.maxBackwardTilt, pitchAngle);

      // Small continuous reward for maintaining setup position
      float tiltFactor =
          std::min(1.0f, pitchAngle / 1.0f); // Cap at ~57 degrees
      float proximityFactor = 1.0f - (distToBall / maxBallDistance);
      reward += setupReward * tiltFactor * proximityFactor * 0.1f;
    }

    // === EXECUTION PHASE ===
    // Reward the actual musty flick
    if (pState.wasInSetupPosition && justStartedFlip && isBackflipping) {
      // Check if within the setup window
      if (pState.setupStartTick < setupWindowTicks) {
        // Base execution reward scaled by how committed the tilt was
        float tiltCommitment = std::min(1.0f, pState.maxBackwardTilt / 0.8f);
        reward += executionReward * tiltCommitment;
      }
    }

    // === POWER BONUS ===
    // Reward for ball velocity increase after the flick
    if (pState.wasFlipping && !player.isFlipping && player.ballTouchedStep) {
      // Flick just ended with ball touch
      Vec velChange = state.ball.vel - pState.ballVelBeforeFlick;
      float speedIncrease = velChange.Length();

      // Bonus for upward velocity (characteristic of musty flicks)
      float upwardBonus =
          std::max(0.0f, state.ball.vel.z / CommonValues::BALL_MAX_SPEED);

      if (speedIncrease > 500.0f) {
        float powerFactor = std::min(1.0f, speedIncrease / 2000.0f);
        reward += powerBonus * powerFactor * (1.0f + upwardBonus);
      }

      // Reset setup state
      pState.wasInSetupPosition = false;
      pState.maxBackwardTilt = 0.0f;
    }

    // Reset setup if conditions no longer met (ball too far, landed, etc.)
    if (pState.wasInSetupPosition) {
      if (!ballClose || player.isOnGround ||
          pState.setupStartTick > setupWindowTicks * 2) {
        pState.wasInSetupPosition = false;
        pState.maxBackwardTilt = 0.0f;
      }
    }

    // Track flip state for next tick
    pState.wasFlipping = player.isFlipping;

    return reward;
  }
};

// Rewards breezy flicks - a mechanic where the car is upside down
// and does a tornado spin (air roll + side flip) to flick the ball
class BreeziFlickReward : public Reward {
private:
  struct PlayerState {
    bool wasInSetupPosition = false;
    float setupStartTick = 0;
    float maxInversion = 0.0f; // Track how upside down they got
    bool wasFlipping = false;
    bool wasAirRolling = false;
    Vec ballVelBeforeFlick = Vec(0, 0, 0);
  };

  std::map<uint32_t, PlayerState> playerStates;

  // Configuration
  float minInversionAngle; // How upside down the car needs to be (radians from
                           // vertical)
  float maxBallDistance;   // Max distance from ball during setup
  float minHeight;         // Minimum height above ground
  float setupReward;       // Reward for getting into position
  float executionReward;   // Reward for executing the flick
  float powerBonus;        // Bonus based on ball speed after flick
  float tornadoBonus;      // Extra reward for air rolling during flip
  float setupWindowTicks;  // How long setup position can be held

public:
  BreeziFlickReward(
      float minInversionAngle = 2.5f, // ~143 degrees (almost upside down)
      float maxBallDistance = 300.0f, float minHeight = 200.0f,
      float setupReward = 0.5f, float executionReward = 6.0f,
      float powerBonus = 3.0f, float tornadoBonus = 2.0f,
      float setupWindowTicks = 90.0f // ~0.75 seconds
      )
      : minInversionAngle(minInversionAngle), maxBallDistance(maxBallDistance),
        minHeight(minHeight), setupReward(setupReward),
        executionReward(executionReward), powerBonus(powerBonus),
        tornadoBonus(tornadoBonus), setupWindowTicks(setupWindowTicks) {}

  virtual void Reset(const GameState &initialState) override {
    playerStates.clear();
    for (const auto &player : initialState.players) {
      playerStates[player.carId] = PlayerState();
    }
  }

  virtual float GetReward(const Player &player, const GameState &state,
                          bool isFinal) override {
    if (!player.prev)
      return 0.0f;

    if (playerStates.find(player.carId) == playerStates.end()) {
      playerStates[player.carId] = PlayerState();
    }
    auto &pState = playerStates[player.carId];

    float reward = 0.0f;

    Vec ballPos = state.ball.pos;
    Vec carPos = player.pos;
    Vec carToBall = ballPos - carPos;
    float distToBall = carToBall.Length();

    // Get car orientation
    Vec carUp = player.rotMat.up;
    Vec worldUp = Vec(0, 0, 1);

    // Calculate inversion angle - how upside down the car is
    // Dot product of car up with world up: 1 = right side up, -1 = upside down
    float upDot = carUp.Dot(worldUp);
    float inversionAngle = std::acos(std::max(-1.0f, std::min(1.0f, upDot)));

    // Check conditions
    bool isInverted = inversionAngle > minInversionAngle;
    bool ballClose = distToBall < maxBallDistance;
    bool isAirborne = !player.isOnGround && carPos.z > minHeight;

    // Check for side flip (tornado spin component)
    bool isSideFlipping =
        player.isFlipping && (std::abs(player.prevAction.yaw) > 0.5f ||
                              std::abs(player.prevAction.roll) > 0.5f);
    bool justStartedFlip = player.isFlipping && !pState.wasFlipping;

    // Check for air roll input
    bool isAirRolling = std::abs(player.prevAction.roll) > 0.3f;

    // === SETUP PHASE ===
    // Reward getting into breezy position (upside down near ball in air)
    if (isAirborne && isInverted && ballClose) {
      if (!pState.wasInSetupPosition) {
        pState.wasInSetupPosition = true;
        pState.setupStartTick = 0;
        pState.maxInversion = inversionAngle;
        pState.ballVelBeforeFlick = state.ball.vel;
      }
      pState.setupStartTick++;
      pState.maxInversion = std::max(pState.maxInversion, inversionAngle);

      // Small continuous reward for maintaining inverted position near ball
      float inversionFactor =
          (inversionAngle - minInversionAngle) / (M_PI - minInversionAngle);
      float proximityFactor = 1.0f - (distToBall / maxBallDistance);
      reward += setupReward * inversionFactor * proximityFactor * 0.1f;

      // Bonus for air rolling while in setup (building momentum for tornado)
      if (isAirRolling) {
        reward += 0.05f;
      }
    }

    // === EXECUTION PHASE ===
    // Reward the actual breezy flick (tornado spin while inverted)
    if (pState.wasInSetupPosition && justStartedFlip && isSideFlipping) {
      // Check if within the setup window
      if (pState.setupStartTick < setupWindowTicks) {
        // Base execution reward scaled by inversion commitment
        float inversionCommitment =
            std::min(1.0f, (pState.maxInversion - minInversionAngle) /
                               (M_PI - minInversionAngle));
        reward += executionReward * (0.5f + 0.5f * inversionCommitment);

        // Extra bonus for true tornado spin (air rolling into the flip)
        if (pState.wasAirRolling || isAirRolling) {
          reward += tornadoBonus;
        }
      }
    }

    // === POWER BONUS ===
    // Reward for ball velocity increase after the flick
    if (pState.wasFlipping && !player.isFlipping && player.ballTouchedStep) {
      Vec velChange = state.ball.vel - pState.ballVelBeforeFlick;
      float speedIncrease = velChange.Length();

      // Breezy flicks often send ball sideways or with spin
      float horizontalSpeed =
          Vec(state.ball.vel.x, state.ball.vel.y, 0).Length();
      float horizontalBonus = horizontalSpeed / CommonValues::BALL_MAX_SPEED;

      if (speedIncrease > 400.0f) {
        float powerFactor = std::min(1.0f, speedIncrease / 2000.0f);
        reward += powerBonus * powerFactor * (1.0f + 0.5f * horizontalBonus);
      }

      // Reset setup state
      pState.wasInSetupPosition = false;
      pState.maxInversion = 0.0f;
    }

    // Reset setup if conditions no longer met
    if (pState.wasInSetupPosition) {
      if (!ballClose || player.isOnGround ||
          pState.setupStartTick > setupWindowTicks * 2) {
        pState.wasInSetupPosition = false;
        pState.maxInversion = 0.0f;
      }
    }

    // Track state for next tick
    pState.wasFlipping = player.isFlipping;
    pState.wasAirRolling = isAirRolling;

    return reward;
  }
};

// --- Added from Monkey (1).cpp ---
class HoldAirRollWhileApproachingBallXYReward : public Reward {
public:
    explicit HoldAirRollWhileApproachingBallXYReward(
        // ===== Global multiplier (internal scale) =====
        float scale = 1.0f,

        // ===== Common: airborne判定 threshold =====
        float minAirZ = 180.f,

        // =========================================================
        // A) Aerial touch combo
        // =========================================================
        float comboWindowSec = 0.40f,
        float minBallZForCombo = 150.f,
        float reward1st = 0.0f,
        float reward2nd = 0.0f,
        float reward3rd = 0.0f,
        float reward4plus = 0.0f,

        // (Optional) Filter sloppy touches: if Δvz < this, don't count towards combo (<=0 disables)
        float comboMinDeltaVz = 0.f,

        // =========================================================
        // B) Strongly reward "pushing the ball upward" via Δvz on touch
        // =========================================================
        float upMinDeltaVz = 120.f,
        float upBaseReward = 0.25f,
        float upPerDeltaVz = 0.0020f,
        float upMaxReward  = 1.50f,

        // =========================================================
        // C) Touch while "approaching toward the ball" in the air
        // =========================================================
        bool  approachUseXY = true,
        float approachMinSpeed = 800.f,
        float approachBaseReward = 0.20f,
        float approachSpeedBonusScale = 0.00020f,
        float approachMaxBonus = 0.40f,
        float approachMaxDist = 3000.f,

        // =========================================================
        // D) Face ball (time-proportional)
        // =========================================================
        float faceRewardPerSec = 0.02f,
        float faceZMargin = 50.f,
        float faceAlignThr = 0.80f,
        float facePower = 2.0f,
        float faceMaxDist = 5000.f,

        // =========================================================
        // E) Air-roll reward (no ball touch required)
        // =========================================================
        float rollHoldSec = 0.20f,
        float rollRateThr = 1.3f,
        float rollRewardPerSec = 90.0f,
        float rollSwitchWindowSec = 0.5f,
        float rollBaseSwitchPenalty = 30.0f,
        int   rollMaxSwitchCount = 10,

        // =========================================================
        // Added: active range (XY)
        // =========================================================
        float maxActiveXYDist = 1400.f,

        // =========================================================
        // ★追加（最後に置くことで既存呼び出しを壊さない）
        // ボールが敵陣側（相手ゴール側のY方向）へこの速度以上のときのみ有効（km/h）
        // =========================================================
        float minBallTowardEnemyKPH = 15.f
    )
        : _scale(scale)
        , _minAirZ(minAirZ)

        , _comboWindowSec(comboWindowSec)
        , _minBallZForCombo(minBallZForCombo)
        , _reward1st(reward1st)
        , _reward2nd(reward2nd)
        , _reward3rd(reward3rd)
        , _reward4plus(reward4plus)
        , _comboMinDeltaVz(comboMinDeltaVz)

        , _upMinDeltaVz(upMinDeltaVz)
        , _upBaseReward(upBaseReward)
        , _upPerDeltaVz(upPerDeltaVz)
        , _upMaxReward(upMaxReward)

        , _approachUseXY(approachUseXY)
        , _approachMinSpeed(approachMinSpeed)
        , _approachBaseReward(approachBaseReward)
        , _approachSpeedBonusScale(approachSpeedBonusScale)
        , _approachMaxBonus(approachMaxBonus)
        , _approachMaxDist(approachMaxDist)

        , _faceRewardPerSec(faceRewardPerSec)
        , _faceZMargin(faceZMargin)
        , _faceAlignThr(faceAlignThr)
        , _facePower(facePower)
        , _faceMaxDist(faceMaxDist)

        , _rollHoldSec(rollHoldSec)
        , _rollRateThr(rollRateThr)
        , _rollRewardPerSec(rollRewardPerSec)
        , _rollSwitchWindowSec(rollSwitchWindowSec)
        , _rollBaseSwitchPenalty(rollBaseSwitchPenalty)
        , _rollMaxSwitchCount(rollMaxSwitchCount)

        , _maxActiveXYDist(maxActiveXYDist)
        , _minBallTowardEnemySpeed(minBallTowardEnemyKPH * (100.f / 3.6f)) // km/h -> uu/s (cm/s)
    {}

    float GetReward(const Player& player, const GameState& state, bool /*isFinal*/) override {
        St& st = _st[player.carId];
        st.t += state.deltaTime;

        if (!st.init) {
            st.init = true;
            st.prevBallVz = state.ball.vel.z;
            return 0.f;
        }

        // =========================================================
        // Gate 0) ボールが敵陣側へ 15km/h 以上（Y方向）で動いている時のみ有効
        // Blue: 敵陣は +Y / Orange: 敵陣は -Y として判定
        // =========================================================
        {
            const float teamSign = (player.team == Team::BLUE) ? 1.f : -1.f;
            const float towardEnemy = state.ball.vel.y * teamSign; // +なら敵陣方向
            if (_minBallTowardEnemySpeed > 0.f && towardEnemy < _minBallTowardEnemySpeed) {
                ResetRoll(st);
                st.combo = 0;
                st.lastTouchTime = -999.f;
                st.prevBallVz = state.ball.vel.z;
                return 0.f;
            }
        }

        // =========================================================
        // Gate 1) XY距離（遠いなら無効）
        // =========================================================
        {
            const float dxA = state.ball.pos.x - player.pos.x;
            const float dyA = state.ball.pos.y - player.pos.y;
            const float xy2 = dxA*dxA + dyA*dyA;
            if (_maxActiveXYDist > 0.f && xy2 > _maxActiveXYDist * _maxActiveXYDist) {
                ResetRoll(st);
                st.combo = 0;
                st.lastTouchTime = -999.f;
                st.prevBallVz = state.ball.vel.z;
                return 0.f;
            }
        }

        // =========================================================
        // Gate 2) 車体zがボールより高いなら無効
        // =========================================================
        if (player.pos.z > state.ball.pos.z) {
            ResetRoll(st);
            st.combo = 0;
            st.lastTouchTime = -999.f;
            st.prevBallVz = state.ball.vel.z;
            return 0.f;
        }

        float r = 0.f;

        const bool inAir = (player.pos.z >= _minAirZ);

        // Ball relative vector
        const float dx = state.ball.pos.x - player.pos.x;
        const float dy = state.ball.pos.y - player.pos.y;
        const float dz = state.ball.pos.z - player.pos.z;

        const float dist2 = dx*dx + dy*dy + dz*dz;
        const float dist  = (dist2 > 1.f) ? std::sqrt(dist2) : 0.f;

        const float xy2   = dx*dx + dy*dy;
        const float xyDist= (xy2 > 1.f) ? std::sqrt(xy2) : 0.f;

        const bool ballAbove = (state.ball.pos.z > player.pos.z + _faceZMargin);

        // =========================================================
        // D) Facing assistance (XY版)
        // =========================================================
        if (inAir && ballAbove && xyDist > 1.f && xyDist <= _faceMaxDist) {
            const float inv = 1.f / xyDist;
            const float dirx = dx * inv;
            const float diry = dy * inv;

            float fx = player.rotMat.forward.x;
            float fy = player.rotMat.forward.y;
            const float fn2 = fx*fx + fy*fy;
            if (fn2 > 1e-6f) {
                const float finv = 1.f / std::sqrt(fn2);
                fx *= finv; fy *= finv;

                const float align = fx*dirx + fy*diry;
                if (align > _faceAlignThr) {
                    float s = (align - _faceAlignThr) / (1.f - _faceAlignThr);
                    s = std::clamp(s, 0.f, 1.f);
                    s = std::pow(s, _facePower);
                    r += (_faceRewardPerSec * state.deltaTime) * s;
                }
            }
        }

        // =========================================================
        // Touch event
        // =========================================================
        const bool touched = player.ballTouchedStep;

        if (touched && inAir) {
            const float deltaVz = state.ball.vel.z - st.prevBallVz;

            // B) Δvz (pushed upward)
            if (deltaVz >= _upMinDeltaVz) {
                float upR = _upBaseReward + _upPerDeltaVz * (deltaVz - _upMinDeltaVz);
                upR = std::min(upR, _upMaxReward);
                r += upR;
            }

            // C) Touch while moving toward the ball
            {
                const float adx = dx;
                const float ady = dy;
                const float adz = _approachUseXY ? 0.f : dz;

                const float ad2 = adx*adx + ady*ady + adz*adz;
                if (ad2 > 1.f) {
                    const float ad = std::sqrt(ad2);
                    if (ad <= _approachMaxDist) {
                        const float inv = 1.f / ad;
                        const float dirx = adx * inv;
                        const float diry = ady * inv;
                        const float dirz = adz * inv;

                        const float approachSpeed =
                            player.vel.x * dirx + player.vel.y * diry + player.vel.z * dirz;

                        if (approachSpeed >= _approachMinSpeed) {
                            float bonus = approachSpeed * _approachSpeedBonusScale;
                            bonus = std::min(bonus, _approachMaxBonus);
                            r += _approachBaseReward + bonus;
                        }
                    }
                }
            }

            // A) Aerial combo
            if (state.ball.pos.z >= _minBallZForCombo) {
                if (_comboMinDeltaVz > 0.f) {
                    if (deltaVz < _comboMinDeltaVz) {
                        st.combo = 0;
                        st.lastTouchTime = -999.f;
                    }
                }

                const bool inWindow = (st.t - st.lastTouchTime) <= _comboWindowSec;
                st.combo = inWindow ? (st.combo + 1) : 1;
                st.lastTouchTime = st.t;

                float comboR = 0.f;
                if (st.combo == 1) comboR = _reward1st;
                else if (st.combo == 2) comboR = _reward2nd;
                else if (st.combo == 3) comboR = _reward3rd;
                else comboR = _reward4plus;

                r += comboR;
            } else {
                st.combo = 0;
                st.lastTouchTime = -999.f;
            }
        }

        // =========================================================
        // E) Aerial roll reward (no ball touch required)
        // =========================================================
        {
            const bool rollActive = inAir;

            if (!rollActive) {
                ResetRoll(st);
            } else {
                const float rollRate = player.rotMat.forward.Dot(player.angVel);

                int sign = 0;
                if (rollRate > _rollRateThr) sign = +1;
                else if (rollRate < -_rollRateThr) sign = -1;

                if (sign == 0) {
                    ResetRoll(st);
                } else if (st.rollLastSign == 0) {
                    st.rollLastSign = sign;
                    st.rollSameDirTime = 0.f;
                } else if (sign != st.rollLastSign) {
                    st.rollLastSign = sign;
                    st.rollSameDirTime = 0.f;

                    st.rollSwitchTimes.push_back(st.t);
                    while (!st.rollSwitchTimes.empty() && (st.t - st.rollSwitchTimes.front()) > _rollSwitchWindowSec) {
                        st.rollSwitchTimes.pop_front();
                    }

                    int cnt = (int)st.rollSwitchTimes.size();
                    cnt = std::min(cnt, _rollMaxSwitchCount);

                    r -= _rollBaseSwitchPenalty * (float)cnt;
                } else {
                    st.rollSameDirTime += state.deltaTime;
                    if (st.rollSameDirTime >= _rollHoldSec) {
                        r += (_rollRewardPerSec * state.deltaTime);
                    }
                }
            }
        }

        st.prevBallVz = state.ball.vel.z;
        return r * _scale;
    }

private:
    struct St {
        bool init = false;
        float t = 0.f;

        // combo
        float lastTouchTime = -999.f;
        int combo = 0;

        // deltaVz
        float prevBallVz = 0.f;

        // roll stability
        int rollLastSign = 0;
        float rollSameDirTime = 0.f;
        std::deque<float> rollSwitchTimes;
    };

    std::unordered_map<uint32_t, St> _st;

    void ResetRoll(St& st) const {
        st.rollLastSign = 0;
        st.rollSameDirTime = 0.f;
        st.rollSwitchTimes.clear();
    }

    // global
    float _scale;
    float _minAirZ;

    // A) combo
    float _comboWindowSec;
    float _minBallZForCombo;
    float _reward1st, _reward2nd, _reward3rd, _reward4plus;
    float _comboMinDeltaVz;

    // B) upward impulse
    float _upMinDeltaVz, _upBaseReward, _upPerDeltaVz, _upMaxReward;

    // C) approach touch
    bool  _approachUseXY;
    float _approachMinSpeed, _approachBaseReward, _approachSpeedBonusScale, _approachMaxBonus, _approachMaxDist;

    // D) face ball
    float _faceRewardPerSec, _faceZMargin, _faceAlignThr, _facePower, _faceMaxDist;

    // E) roll stability
    float _rollHoldSec, _rollRateThr, _rollRewardPerSec;
    float _rollSwitchWindowSec, _rollBaseSwitchPenalty;
    int   _rollMaxSwitchCount;

    // gates
    float _maxActiveXYDist;
    float _minBallTowardEnemySpeed; // uu/s (cm/s)
};

} // namespace RLGC

/*
=== UNIQUE CLASSES IN New Text Document.txt.h (syntax repair needed) ===
RippleDemoReward
PopResetReward
SimpleFlipResetLearnReward
SimpleResetTeachReward
MaktufResetReward
MaktufResetRewardContinuous
RipplesFlipResetFreestyleChain
VelocityBallToGoalMouthReward
OffensivedemoRewardV1
ConsolidatedDemoReward
GoalFrontBumpDemoReward
DemoBumpNearBallReward
BumpChainReward
FastDribblePopBumpReward
DemoChaseReward
FlyToGoalKeepHigh
WavedashReward
ExistReward
ShieldzKORew
KickoffDashReward
KickoffProximityReward
InAirMultiTouchesReward
AirdribbleRewardV1
AirdribbleRewardV2
FlyToGoal
VelocityPlayerToNearestPlayerReward
FacePlayerReward
*/
