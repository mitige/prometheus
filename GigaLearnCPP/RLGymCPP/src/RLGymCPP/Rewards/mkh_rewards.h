#pragma once
#include "Reward.h"
#include "../Math.h"
#include <algorithm>
#include <cmath>
#include <unordered_map>
#include <unordered_set>
#include <string>
#include <mutex>


namespace RLGC {

// Globals for WallAirdribbleSetupReward wall-jump blocking (single instance across TUs)
inline std::mutex g_wall_jump_block_mutex;
inline std::unordered_map<uint32_t, int> g_wall_jump_block_until;

static inline bool IsOnGroundApprox(const Player& p) { 
    return p.isOnGround || p.pos.z <= 35.0f; 
}

inline bool PlayerHasFlipAvailable(const Player& player) { 
    return !player.hasFlipped && !player.hasDoubleJumped; 
}

inline bool PlayerGainedFlipThisStep(const Player& player) { 
    const Player* prev = player.prev; 
    bool hasNow = PlayerHasFlipAvailable(player); 
    bool hadPrev = prev ? PlayerHasFlipAvailable(*prev) : hasNow; 
    return hasNow && !hadPrev; 
}

// Precise ground dribble detection
// Conditions:
// - On ground (approx)
// - Ball above car between 90..220 uu
// - Horizontal offset under ball small (<140 uu)
// - Relative ball-car speed small (<600 uu/s)
// - Car up vector points toward ball (dot > 0.6)
static inline bool IsDribblingPrecise(const Player& p, const GameState& s) {
	const Vec& bc = s.ball.pos;
	const Vec& pc = p.pos;
	if (!IsOnGroundApprox(p)) return false;

	const float vertical = bc.z - pc.z;
	if (vertical < 90.0f || vertical > 220.0f) return false;

	const float dx = bc.x - pc.x;
	const float dy = bc.y - pc.y;
	const float horiz = std::sqrt(dx * dx + dy * dy);
	if (horiz > 140.0f) return false;

	const Vec relVel = s.ball.vel - p.vel;
	if (relVel.Length() > 600.0f) return false;

	Vec toBall = (bc - pc).Normalized();
	float upAlign = p.rotMat.up.Dot(toBall);
	if (upAlign < 0.6f) return false;

	return true;
}

	// PlayerDataEventReward removed (duplicate of CommonRewards.h)

class ResetShotReward : public Reward {
public:
    float forceMultiplier;
    float speedMultiplier;
    float windowTime;
    float heightReward;
    float goalDirWeight;
    float cooldown;

    ResetShotReward(float forceMultiplier = 1.0f, float speedMultiplier = 1.25f, float windowTime = 0.0f, float heightReward = 1.5f, float goalDirWeight = 4.0f, float cooldown = 1.5f)
        : forceMultiplier(forceMultiplier), speedMultiplier(speedMultiplier), windowTime(windowTime), heightReward(heightReward), goalDirWeight(goalDirWeight), cooldown(cooldown) {}

    void Reset(const GameState&) override {
        prevPlayerVel.clear();
        prevBallVel.clear();
        prevHasFlipped.clear();
        prevHasFlipAvailable.clear();
        prevGotReset.clear();
        resetWindows.clear();
    }

    float GetReward(const Player& player, const GameState& state, bool) override {
        float reward = 0.0f;

        // Reset context (no time limit, no penalties on delay): open when a flip reset is obtained in air
        auto& window = resetWindows[player.carId];
        window.timeSinceLastReward += state.deltaTime;

        bool hasFlipAvailable = player.HasFlipOrJump();
        bool gotResetNow = player.GotFlipReset();
        bool prevGot = prevGotReset[player.carId];
        if (player.prev && !prevGot && gotResetNow && window.timeSinceLastReward > cooldown) {
            window.active = true;
            window.usedFlip = false;
            window.timeSinceActive = 0.0f;
            window.ballPosAtReset = state.ball.pos;
            window.seenFlipAvailable = false;
        }

        // We must see the flip become available (meaning we are not mid-flip from the setup)
        // before we can consider a subsequent flip as the "shot".
        if (window.active && !window.seenFlipAvailable) {
            if (PlayerHasFlipAvailable(player)) {
                window.seenFlipAvailable = true;
            }
        }

        // Failure condition: if we drop below 200uu before flipping, consider reset-use failed
        if (window.active && !window.usedFlip && player.pos.z < 100.0f) {
            reward -= 1.0f;
            window.active = false;
        }

        // While reset context is active and before flip is used, accumulate hold time and penalize jumping
        if (window.active && !window.usedFlip) {
            window.timeSinceActive += state.deltaTime;
        }

        bool currentlyHasFlipped = player.hasFlipped && !player.isOnGround && !player.hasJumped;
        bool usedFlipFromReset = window.active && window.seenFlipAvailable && currentlyHasFlipped && !PlayerHasFlipAvailable(player);
        if (usedFlipFromReset && player.ballTouchedStep) {
            float distToBallNow = (state.ball.pos - player.pos).Length();
            Vec prevPVel = Vec(0, 0, 0);
            Vec prevBVel = Vec(0, 0, 0);

            auto pIt = prevPlayerVel.find(player.carId);
            auto bIt = prevBallVel.find(player.carId);
            if (pIt != prevPlayerVel.end()) prevPVel = pIt->second;
            if (bIt != prevBallVel.end()) prevBVel = bIt->second;

            float playerVelChange = (player.vel - prevPVel).Length();
            float ballVelChange = (state.ball.vel - prevBVel).Length();

            float playerVelChangeNorm = std::clamp(playerVelChange / CommonValues::CAR_MAX_SPEED, 0.0f, 1.0f);
            float ballVelChangeNorm = std::clamp(ballVelChange / CommonValues::BALL_MAX_SPEED, 0.0f, 1.0f);
            float ballSpeedNorm = std::clamp(state.ball.vel.Length() / CommonValues::BALL_MAX_SPEED, 0.0f, 1.0f);

            float inferredHitForce = playerVelChangeNorm + ballVelChangeNorm;

            // Prioritize vertical velocity component ("pop") over raw speed
            float ballUpVel = state.ball.vel.z;
            float verticalBonus = (ballUpVel > 0) ? (ballUpVel / CommonValues::BALL_MAX_SPEED) * 8.0f : 0.0f; // Reward popping up

            float base = (inferredHitForce * speedMultiplier + ballSpeedNorm * forceMultiplier) * 0.5f; // reduced base power relevance
            float holdBonus = window.timeSinceActive * windowTime;
            float hBonus = (player.pos.z / CommonValues::CEILING_Z) * heightReward;

            // Goal alignment bonus
            bool targetOrangeGoal = player.team == Team::BLUE;
            Vec targetPos = targetOrangeGoal ? CommonValues::ORANGE_GOAL_BACK : CommonValues::BLUE_GOAL_BACK;
            Vec toGoal = (targetPos - state.ball.pos).Normalized();
            Vec ballDir = state.ball.vel.Length() > 1e-6f ? state.ball.vel.Normalized() : Vec(0,0,0);
            float goalAlign = toGoal.Dot(ballDir);
            float gBonus = goalAlign * (goalDirWeight * 0.25f); // Greatly reduced goal bias to allow setups

            reward = base + holdBonus + hBonus + gBonus + verticalBonus;

            window.usedFlip = true;
            window.active = false;
            window.timeSinceLastReward = 0.0f;
        }

        prevHasFlipped[player.carId] = player.hasFlipped;
        prevHasFlipAvailable[player.carId] = hasFlipAvailable;
        prevGotReset[player.carId] = gotResetNow;
        prevPlayerVel[player.carId] = player.vel;
        prevBallVel[player.carId] = state.ball.vel;

        return reward;
    }

private:
    std::unordered_map<uint32_t, Vec> prevPlayerVel;
    std::unordered_map<uint32_t, Vec> prevBallVel;
    std::unordered_map<uint32_t, bool> prevHasFlipped;
    std::unordered_map<uint32_t, bool> prevHasFlipAvailable;
    std::unordered_map<uint32_t, bool> prevGotReset;
    struct ResetWindow { bool active = false; bool usedFlip = false; float timeSinceActive = 0.0f; Vec ballPosAtReset; float timeSinceLastReward = 999.0f; bool seenFlipAvailable = false; };
    std::unordered_map<uint32_t, ResetWindow> resetWindows;
};

    class ArsenalResetReward : public Reward {
        private:
            mutable std::unordered_map<uint32_t, float> prevOrientationScore;
            
            // Track flip reset windows for shot rewards
            struct FlipWindow {
                bool hasFlip = false;
                float timeRemaining = 0.0f;
                Vec ballPosAtReset;
            };
            mutable std::unordered_map<uint32_t, FlipWindow> flipWindows;
            
        public:
            // --- Konfiguration ---
            // Aktivierungs-Schwellenwerte
            const float ACTIVATION_MIN_BALL_HEIGHT = 300.0f;
            const float ACTIVATION_MAX_DIST_TO_BALL = 300.0f;
            const float MIN_PLAYER_HEIGHT_FOR_EVENT = 300.0f;
            const float WALL_MARGIN = 500.0f;
        
            // Gewichtungen für Shaping-Rewards
            const float ORIENTATION_WEIGHT = 0.001f; // Reduced - getting close more important than perfect alignment
            const float POSITION_WEIGHT = 0.001f;
            const float PROXIMITY_WEIGHT = 0.001f;    // Increased - must get close, not farm from distance
            
            // Minimum down alignment threshold (like Saturn's 0.42)
            const float MIN_DOWN_ALIGNMENT = 0.5f; // Must be at least 50% aligned
        
            // Bonus für das erfolgreiche Event
            const float EVENT_BONUS = 20.0f;  // High bonus to make completing flip reset clearly worth it
            
            // Shot reward configuration
            const float SHOT_WINDOW_TIME = 0.0f;        // Time to use flip after getting it
            const float SHOT_REWARD = 0.0f;            // Base reward for using flip
            const float SHOT_MIN_BALL_MOVEMENT = 0.0f; // Ball must move at least this much
            
            void Reset(const GameState&) override { 
                prevOrientationScore.clear();
                flipWindows.clear();
            }
        
            virtual float GetReward(const Player& player, const GameState& state, bool isFinal) override {
                // --- 1. Vorbedingungen prüfen (Aktivierung) ---
                // Reward ist nur aktiv, wenn der Spieler in der Luft ist, der Ball hoch genug ist
                // und sie sich nahe beieinander befinden.
                if (player.isOnGround || state.ball.pos.z < ACTIVATION_MIN_BALL_HEIGHT || (player.pos.Dist(state.ball.pos) > ACTIVATION_MAX_DIST_TO_BALL)) {
                    prevOrientationScore.erase(player.carId); // Clear tracking when not active
                    return 0.0f;
                }
        
                float reward = 0.0f;
        
                // --- 2. Ideale Ausrichtung und Position für Shaping ---
                // Idealerweise ist das Auto direkt über dem Ball und "upside-down".
                // Der "Down"-Vektor des Autos (-rotMat.up) sollte also direkt auf den Ball zeigen.
                Vec idealDownVector = (state.ball.pos - player.pos).Normalized();
        
                // --- 3. Shaping Rewards (Kontinuierliche Belohnung) ---
        
                // **Orientierungs-Reward:** Belohnt die korrekte Ausrichtung der Auto-Unterseite zum Ball.
                // Kosinus-Ähnlichkeit zwischen dem "Down"-Vektor des Autos und der Richtung zum Ball.
                float downAlignment = (-player.rotMat.up).Dot(idealDownVector); // Raw dot product [-1, 1]
                
                // Must meet minimum alignment threshold (like Saturn's approach)
                if (downAlignment < MIN_DOWN_ALIGNMENT) {
                    prevOrientationScore.erase(player.carId); // Not aligned enough
                    return 0.0f;
                }
                
                // Scale to [0, 1] for reward calculation
                float orientationScore = (downAlignment + 1.0f) / 2.0f;
                
                // Graduated commitment multiplier - tighter ranges to discourage farming
                float distToBall = player.pos.Dist(state.ball.pos);
                float commitmentMultiplier = 0.01f;
                if (distToBall < 140.0f) commitmentMultiplier = 3.0f;      // Very close - must commit hard
                else if (distToBall < 200.0f) commitmentMultiplier = 2.0f; // Close - reduced
                else if (distToBall < 280.0f) commitmentMultiplier = 1.0f; // Approaching - minimal
                
                reward += orientationScore * ORIENTATION_WEIGHT * commitmentMultiplier;
                
                // DISABLED: Anti-flip-away penalty makes bot too hesitant to commit
                prevOrientationScore[player.carId] = orientationScore;
        
                // **Positions-Reward:** Belohnt die vertikale Positionierung über dem Ball.
                // Ideal ist eine Höhe von ca. 95-110 Einheiten über dem Ball.
                float heightDifference = player.pos.z - state.ball.pos.z;
                float idealHeight = -95.0f; // Ein guter Mittelwert
                float positionError = std::abs(heightDifference - idealHeight);
                float positionReward = exp(-0.02f * positionError); // Exponentieller Abfall bei Abweichung
                reward += positionReward * POSITION_WEIGHT;
        
                // **Annäherungs-Reward:** Belohnt es, nahe am Ball zu sein.
                // (distToBall already calculated above)
                float proximityReward = exp(-0.01f * distToBall);
                reward += proximityReward * PROXIMITY_WEIGHT;
                
                // **Closing Speed Reward:** Reward HIGH closing speed for impact, not soft catches
                // Flip resets need HARD contact with wheels - soft touches don't trigger reset
                Vec toBallVec = state.ball.pos - player.pos;
                if (distToBall > 1e-3f && distToBall < 300.0f) {
                    Vec approachDir = toBallVec / distToBall;
                    
                    // Calculate relative velocity (how fast gap is closing)
                    Vec relativeVel = player.vel - state.ball.vel;
                    float closingSpeed = relativeVel.Dot(approachDir);
                    
                    if (closingSpeed > 500.0f) {
                        // HIGH closing speed - will create impact for flip reset
                        float closingNorm = std::min((closingSpeed - 500.0f) / 1000.0f, 1.0f);
                        reward += closingNorm * 0.8f; // Very high reward for fast impact
                    } else if (closingSpeed > 250.0f) {
                        // Medium closing speed - okay but not ideal
                        float closingNorm = (closingSpeed - 200.0f) / 300.0f;
                        reward += closingNorm * 0.3f; // Lower reward
                    } else if (closingSpeed < 100.0f) {
                        // Too slow - catching/convoying, not impacting
                        reward -= 0.2f; // Penalty for soft approach
                    }
                    // 100-200 range: no reward/penalty (neutral)
                }
        
                // --- 4. Flip Reset Tracking & Rewards ---
                auto& window = flipWindows[player.carId];
                bool currentlyHasFlip = player.HasFlipOrJump();
                
                // Detect flip reset gained
                bool gotReset = player.prev && !player.prev->HasFlipOrJump() && currentlyHasFlip && !player.isOnGround;
        
                if (gotReset) {
                    // Zusätzliche Bedingungen 
                    bool heightCheck = player.pos.z > MIN_PLAYER_HEIGHT_FOR_EVENT && player.pos.z < (CommonValues::CEILING_Z - 300.0f);
                    bool wallDistCheck = std::abs(player.pos.x) < (CommonValues::SIDE_WALL_X - WALL_MARGIN) && std::abs(player.pos.y) < (CommonValues::BACK_WALL_Y - WALL_MARGIN);
                    bool nearBallCheck = distToBall < 170.0f; // Harter Schwellenwert für das Event
                    bool alignmentCheck = downAlignment >= MIN_DOWN_ALIGNMENT; // Must be properly aligned
        
                    if (heightCheck && wallDistCheck && nearBallCheck && alignmentCheck) {
                        reward += EVENT_BONUS;
                        
                        // CRITICAL: Big bonus for high closing speed at moment of flip reset
                        // This encourages HARD impact, not soft catching
                        Vec toBallVec = state.ball.pos - player.pos;
                        float distToBall = toBallVec.Length();
                        if (distToBall > 1e-3f) {
                            Vec relativeVel = player.vel - state.ball.vel;
                            float closingSpeed = relativeVel.Dot(toBallVec / distToBall);
                            
                            if (closingSpeed > 300.0f) {
                                // High impact speed at reset - exactly what we want!
                                float impactBonus = std::min(closingSpeed / 1500.0f, 1.0f) * 10.0f;
                                reward += impactBonus; // Up to 10.0 bonus for fast impact
                            }
                        }
                        
                        // Open shot window
                        window.hasFlip = true;
                        window.timeRemaining = SHOT_WINDOW_TIME;
                        window.ballPosAtReset = state.ball.pos;
                    }
                }
                
                // Track flip window
                if (window.hasFlip) {
                    window.timeRemaining -= state.deltaTime;
                    
                    // Check if flip was used (lost flip or landed)
                    if (!currentlyHasFlip || player.isOnGround) {
                        // Check if ball was touched while having flip
                        if (player.ballTouchedStep && window.timeRemaining > 0.0f) {
                            // Calculate ball movement since reset
                            float ballMovement = (state.ball.pos - window.ballPosAtReset).Length();
                            
                            // Check if toward goal
                            bool targetOrangeGoal = (player.team == Team::BLUE);
                            Vec goalPos = targetOrangeGoal ? CommonValues::ORANGE_GOAL_CENTER : CommonValues::BLUE_GOAL_CENTER;
                            Vec toGoal = (goalPos - state.ball.pos).Normalized();
                            Vec ballDir = state.ball.vel.Normalized();
                            float shotAlignment = ballDir.Dot(toGoal);
                            
                            // Reward shot if ball moved and is somewhat toward goal
                            if (ballMovement > SHOT_MIN_BALL_MOVEMENT && shotAlignment > 0.2f) {
                                float alignmentBonus = shotAlignment * 10.0f; // Up to 10.0 bonus
                                float speedBonus = std::min(state.ball.vel.Length() / 2000.0f, 1.0f) * 10.0f; // Up to 10.0
                                reward += SHOT_REWARD + alignmentBonus + speedBonus; // Total: 15 to 35
                            }
                        }
                        window.hasFlip = false;
                    }
                    
                    // Close window if expired
                    if (window.timeRemaining <= 0.0f) {
                        window.hasFlip = false;
                    }
                }
        
                return reward;
            }
        };

class KubiContinuousFlipResetReward : public Reward {
private:
    mutable std::unordered_map<uint32_t, float> prevOrientationScore;
    mutable std::unordered_map<uint32_t, float> prevDistToBall;
    mutable std::unordered_map<uint32_t, float> airTimeTracker;

public:
	// main configuration parameters
    const float ACTIVATION_MIN_BALL_HEIGHT = 300.0f;
    const float ACTIVATION_MAX_DIST_TO_BALL = 300.0f;
    const float MIN_PLAYER_HEIGHT_FOR_EVENT = 300.0f;
    const float WALL_MARGIN = 500.0f;
    const float ORIENTATION_WEIGHT = 0.0f;
    const float POSITION_WEIGHT = 0.0f;
    const float PROXIMITY_WEIGHT = 0.0f;

	// configuration for posture and event bonuses
    const float POSTURE_BONUS_WEIGHT = 0.04f;
    const float POSTURE_BONUS_DIST_THRESH = 400.0f;
    const float BELLY_UP_THRESHOLD = 0.5f;
    const float MAX_NOSE_DOWN_ANGLE_RAD = 70.0f * (float(M_PI) / 180.0f);
    const float NOSE_UP_TOLERANCE_RAD = 15.0f * (float(M_PI) / 180.0f);
    const float MIN_DOWN_ALIGNMENT = 0.5f;
    const float EVENT_BONUS = 20.0f;

    // configureration parameters
    const float MIN_CLOSING_SPEED = 110.0f; // mindestannäherungsgeschwindigkeit in uu/s
    const float NOSE_ALIGNMENT_BONUS_WEIGHT = 1.0f; //gewichtung für den nasen ausrichtungs Bonus
    const float MIN_DIST_FOR_SPEED_CHECK = 100.0f; // distanz, unter der die geschwindigkeitsprüfung ignoriert wird
    const float NOSE_ALIGNMENT_FULL_BONUS_THRESHOLD = 0.75f; // schwellenwert für vollen Bonus

    // New: Speed reward parameters
    const float FAST_RESET_REWARD_WEIGHT = 15.0f;
    const float FAST_RESET_TIME_THRESHOLD = 2.0f; // Seconds context for "fast" reward

    // Anti-farming parameters
    const float LOITERING_PENALTY = -0.05f; // Small penalty per step to discourage waiting
    const float SHAPING_DECAY_RATE = 1.0f;   // Exponential decay rate for shaping rewards
    const float MAX_SEQUENCE_TIME = 5.0f;    // Maximum time allowed for a single sequence

    void Reset(const GameState& initialState) override {
        prevOrientationScore.clear();
        prevDistToBall.clear();
		// so mehr zeug noch initialisieren
        for (const auto& p : initialState.players) {
            prevDistToBall[p.carId] = (initialState.ball.pos - p.pos).Length();
            airTimeTracker[p.carId] = 0.0f;
        }
    }

    virtual float GetReward(const Player& player, const GameState& state, bool isFinal) override {
        if (!player.prev) return 0.0f;

		// activation check
        if (player.isOnGround || state.ball.pos.z < ACTIVATION_MIN_BALL_HEIGHT || (player.pos.Dist(state.ball.pos) > ACTIVATION_MAX_DIST_TO_BALL)) {
            prevOrientationScore.erase(player.carId);
            prevDistToBall.erase(player.carId);
            airTimeTracker[player.carId] = 0.0f;
            return 0.0f;
        }

        // nähert der bastard sich an?
        float distToBall = player.pos.Dist(state.ball.pos);
        auto it = prevDistToBall.find(player.carId);
        if (it == prevDistToBall.end()) {
            prevDistToBall[player.carId] = distToBall;
            return 0.0f;
        }
        float lastDist = it->second;
        prevDistToBall[player.carId] = distToBall;
        if (state.deltaTime <= 1e-6f) {
            return 0.0f; // division durch Null vermeiden, weil sonst bumm
        }
        float closingSpeed = (lastDist - distToBall) / state.deltaTime;
        //bedingung: Wenn die Distanz > 100 ist, wird eine mindestannäherungsgeschwindigkeit benötigt.
        if (distToBall > MIN_DIST_FOR_SPEED_CHECK && closingSpeed < MIN_CLOSING_SPEED) {
			return 0.0f; // wenn nicht schnell genug annähert, kein Reward
        }

        float reward = 0.0f;
        float timeSpent = airTimeTracker[player.carId];
        airTimeTracker[player.carId] += state.deltaTime;

        // Hard timeout check
        if (timeSpent > MAX_SEQUENCE_TIME) {
            return -0.1f; // Penalty for loitering too long
        }

        // Calculate shaping decay factor
        float shapingFactor = std::exp(-SHAPING_DECAY_RATE * timeSpent);
        reward += LOITERING_PENALTY; // Constant pressure to finish

        // main shaping rewards berechnen
        Vec idealDownVector = (state.ball.pos - player.pos).Normalized();
        float downAlignment = (-player.rotMat.up).Dot(idealDownVector);

        if (downAlignment < MIN_DOWN_ALIGNMENT) {
            prevOrientationScore.erase(player.carId);
            return 0.0f;
        }

        float orientationScore = (downAlignment + 1.0f) / 2.0f;
        float commitmentMultiplier = 0.0f;
        if (distToBall < 180.0f) commitmentMultiplier = 5.0f;
        else if (distToBall < 250.0f) commitmentMultiplier = 3.0f;
        else if (distToBall < 350.0f) commitmentMultiplier = 1.5f;
        reward += (orientationScore * ORIENTATION_WEIGHT * commitmentMultiplier) * shapingFactor;
        if (distToBall < 200.0f && prevOrientationScore.count(player.carId)) {
            float orientationDrop = prevOrientationScore[player.carId] - orientationScore;
            if (orientationDrop > 0.2f) {
                reward -= orientationDrop * 3.0f;
            }
        }
        prevOrientationScore[player.carId] = orientationScore;
        float heightDifference = player.pos.z - state.ball.pos.z;
        float idealHeight = -95.0f;
        float positionError = std::abs(heightDifference - idealHeight);
        float positionReward = exp(-0.02f * positionError);
        reward += (positionReward * POSITION_WEIGHT) * shapingFactor;
        float proximityReward = exp(-0.01f * distToBall);
        reward += (proximityReward * PROXIMITY_WEIGHT) * shapingFactor;

        // holding bums und nasen bums addieren
        if (distToBall <= POSTURE_BONUS_DIST_THRESH) {
            float bellyUpFactor = std::max(0.0f, (-player.rotMat.up.z - BELLY_UP_THRESHOLD) / (1.0f - BELLY_UP_THRESHOLD));
            if (bellyUpFactor > 0.0f) {
                float noseZ = player.rotMat.forward.z;
                float maxNoseDownZ = sin(MAX_NOSE_DOWN_ANGLE_RAD);
                float maxNoseUpZ = sin(NOSE_UP_TOLERANCE_RAD);
                float noseAngleFactor = 0.0f;
                if (noseZ <= 0 && std::abs(noseZ) <= maxNoseDownZ) {
                    noseAngleFactor = 1.0f - (std::abs(noseZ) / maxNoseDownZ);
                }
                else if (noseZ > 0 && noseZ <= maxNoseUpZ) {
                    noseAngleFactor = 1.0f - (noseZ / maxNoseUpZ);
                }
                noseAngleFactor = std::max(0.0f, noseAngleFactor);
                if (noseAngleFactor > 0.0f) {
                    float postureBonus = bellyUpFactor * noseAngleFactor;
                    reward += (postureBonus * POSTURE_BONUS_WEIGHT) * shapingFactor;
                }

				// added nose alignment bonus
                Vec playerFwd2D = Vec(player.rotMat.forward.x, player.rotMat.forward.y, 0);
                Vec ballVel2D = Vec(state.ball.vel.x, state.ball.vel.y, 0);

                float playerFwdLen = playerFwd2D.Length();
                float ballVelLen = ballVel2D.Length();

                if (playerFwdLen > 1e-3f && ballVelLen > 1e-3f) {
                    Vec playerFwdNorm = playerFwd2D / playerFwdLen;
                    Vec ballVelNorm = ballVel2D / ballVelLen;
                    float alignment = playerFwdNorm.Dot(ballVelNorm);

                    if (alignment > 0) {
                        float scaled_alignment = (alignment >= NOSE_ALIGNMENT_FULL_BONUS_THRESHOLD) ? 1.0f : (alignment / NOSE_ALIGNMENT_FULL_BONUS_THRESHOLD);
                        reward += (scaled_alignment * NOSE_ALIGNMENT_BONUS_WEIGHT) * shapingFactor;
                    }
                }
            }
        }


        // event rewards stuff
        bool gotReset = !player.prev->HasFlipOrJump() && player.HasFlipOrJump() && !player.isOnGround;
        if (gotReset) {
            bool heightCheck = player.pos.z > MIN_PLAYER_HEIGHT_FOR_EVENT && player.pos.z < (CommonValues::CEILING_Z - 300.0f);
            bool wallDistCheck = std::abs(player.pos.x) < (CommonValues::SIDE_WALL_X - WALL_MARGIN) && std::abs(player.pos.y) < (CommonValues::BACK_WALL_Y - WALL_MARGIN);
            bool nearBallCheck = distToBall < 170.0f;
            bool alignmentCheck = downAlignment >= MIN_DOWN_ALIGNMENT;
            if (heightCheck && wallDistCheck && nearBallCheck && alignmentCheck) {
                reward += EVENT_BONUS;

                // Speed-based reward for faster resets
                float timeSpent = airTimeTracker[player.carId];
                if (timeSpent < FAST_RESET_TIME_THRESHOLD) {
                    float speedReward = FAST_RESET_REWARD_WEIGHT * std::max(0.0f, 1.0f - (timeSpent / FAST_RESET_TIME_THRESHOLD));
                    reward += speedReward;
                }
                
                // Reset tracker for possible chain
                airTimeTracker[player.carId] = 0.0f;
            }
        }

        return reward;
    }
};

	// KickoffSpeedflipReward: rewards a completed speedflip sequence at kickoff
	class MKHKickoffSpeedflipReward : public Reward {
	public:
		// Parameters mirroring the Python version
		float kickoff_detection_time;   // seconds window to consider kickoff sequence
		float max_jump_delay;           // seconds after kickoff to register first jump
		float min_speed_threshold;      // uu/s threshold to consider speedflip successful
		float diagonal_input_threshold; // abs(pitch) and abs(yaw) must exceed this
		float cancel_threshold;         // pitch must be below this (negative) to count as cancel
		float max_cancel_delay;         // seconds after jump to detect cancel
		float reward_value;             // reward given once when completed

		MKHKickoffSpeedflipReward(
			float kickoff_detection_time_ = 3.0f,
			float max_jump_delay_ = 0.5f,
			float min_speed_threshold_ = 1200.0f,
			float diagonal_input_threshold_ = 0.3f,
			float cancel_threshold_ = -0.5f,
			float max_cancel_delay_ = 0.3f,
			float reward_value_ = 8.0f)
			: kickoff_detection_time(kickoff_detection_time_),
			  max_jump_delay(max_jump_delay_),
			  min_speed_threshold(min_speed_threshold_),
			  diagonal_input_threshold(diagonal_input_threshold_),
			  cancel_threshold(cancel_threshold_),
			  max_cancel_delay(max_cancel_delay_),
			  reward_value(reward_value_) {}

		virtual void Reset(const GameState& initialState) override {
			_states.clear();
		}

		virtual void PreStep(const GameState& state) override {
			// advance timers per player
			for (auto& p : state.players) {
				auto& st = _states[p.carId];
				st.elapsed += state.deltaTime;
				if (st.kickoff_detected) {
					st.time_since_kickoff += state.deltaTime;
				}
				if (st.jump_used) {
					st.time_since_jump += state.deltaTime;
				}
			}
		}

		virtual float GetReward(const Player& player, const GameState& state, bool isFinal) override {
			auto& st = _states[player.carId];
			float reward = 0.0f;
		
			// Detect kickoff based on spawn positions and early time window
			if (!st.kickoff_detected) {
				if (IsKickoffPosition(player.pos) && IsBallAtKickoff(state)) {
					st.kickoff_detected = true;
					st.time_since_kickoff = 0.0f;
					st.had_jump = !player.isJumping; // initial: not currently jumping
				}
			}
		
			// Only proceed if within kickoff window
			if (st.kickoff_detected && st.time_since_kickoff <= kickoff_detection_time && !st.speedflip_completed) {
				// Detect first jump (transition had_jump true -> currently jumping) within max_jump_delay
				bool jump_edge = (st.had_jump && player.isJumping);
				if (!st.jump_used && jump_edge && st.time_since_kickoff <= max_jump_delay) {
					st.jump_used = true;
					st.time_since_jump = 0.0f;
				}
			
				// After jump, detect diagonal input (abs(pitch) & abs(yaw) > threshold)
				if (st.jump_used && !st.diagonal_input_detected) {
					float ap = fabsf(player.prevAction.pitch);
					float ay = fabsf(player.prevAction.yaw);
					if (ap > diagonal_input_threshold && ay > diagonal_input_threshold) {
						st.diagonal_input_detected = true;
					}
				}
			
				// Detect flip cancel (pitch below cancel_threshold) within max_cancel_delay after jump
				if (st.jump_used && st.diagonal_input_detected && !st.flip_cancel_detected) {
					if (st.time_since_jump <= max_cancel_delay) {
						if (player.prevAction.pitch < cancel_threshold) {
							st.flip_cancel_detected = true;
						}
					}
				}
			
				// Completion check: speed threshold met after sequence
				if (st.jump_used && st.diagonal_input_detected && st.flip_cancel_detected) {
					float speed = player.vel.Length();
					if (speed >= min_speed_threshold) {
						st.speedflip_completed = true;
						reward = reward_value;
					}
				}
			}
		
			// Update trackers for next step
			st.had_jump = !player.isJumping;
			st.last_pos = player.pos;
		
			return reward;
		}

		private:
		struct PerPlayerState {
			bool kickoff_detected = false;
			bool jump_used = false;
			bool diagonal_input_detected = false;
			bool flip_cancel_detected = false;
			bool speedflip_completed = false;
			bool had_jump = true;
			float time_since_kickoff = 0.0f;
			float time_since_jump = 0.0f;
			float elapsed = 0.0f;
			Vec last_pos = {};
		};

		std::unordered_map<uint32_t, PerPlayerState> _states;

		static bool IsBallAtKickoff(const GameState& state) {
			const float TOL = 40.0f;
			return fabsf(state.ball.pos.x) < TOL && fabsf(state.ball.pos.y) < TOL; // near center
		}

		static bool NearPos(const Vec& a, const Vec& b, float tol) {
			return (a - b).Length() < tol;
		}

		static bool IsKickoffPosition(const Vec& p) {
			const float Y = CommonValues::BACK_WALL_Y; // ~5120
			const Vec positions[] = {
				{ 0.0f, -Y, 0.0f },
				{ -2048.0f, -Y, 0.0f },
				{  2048.0f, -Y, 0.0f },
				{ -2048.0f,  Y, 0.0f },
				{  2048.0f,  Y, 0.0f }
			};
			const float TOL = 700.0f; // generous tolerance
			for (const auto& pos : positions) {
				if (NearPos(p, pos, TOL)) return true;
			}
			return false;
		}
	};

	// Typedefs removed (duplicate of CommonRewards.h)

	// Rewards a goal by anyone on the team
	// NOTE: Already zero-sum
	// Standard rewards removed (duplicate of CommonRewards.h: GoalReward, VelocityReward, VelocityBallToGoalReward, VelocityPlayerToBallReward, FaceBallReward, TouchBallReward, SpeedReward)

// StrategicDemoReward - Encourages demos but respects game context
    class StrategicDemoReward : public Reward {
    public:
        float goalZoneThreshold;
        float demoMultiplier;
        float ballInZoneBonus;
        float maxDistanceFromGoal;
        float baseReward;
        float maxReward;

        StrategicDemoReward(
            float goalZoneThreshold = 2500.0f,
            float demoMultiplier = 2.0f,
            float ballInZoneBonus = 0.5f,
            float maxDistanceFromGoal = 5000.0f,
            float baseReward = 0.3f,
            float maxReward = 1.5f)
            : goalZoneThreshold(goalZoneThreshold),
            demoMultiplier(demoMultiplier),
            ballInZoneBonus(ballInZoneBonus),
            maxDistanceFromGoal(maxDistanceFromGoal),
            baseReward(baseReward),
            maxReward(maxReward) {
        }

        virtual void Reset(const GameState& initialState) override {}

        virtual float GetReward(const Player& player, const GameState& state, bool isFinal) override {
            bool didBump = player.eventState.bump;
            bool didDemo = player.eventState.demo;

            if (!didBump && !didDemo) return 0.0f;

            const Player* victim = nullptr;
            for (const auto& otherPlayer : state.players) {
                if (otherPlayer.team != player.team) {
                    if (otherPlayer.eventState.bumped || otherPlayer.eventState.demoed) {
                        victim = &otherPlayer;
                        break;
                    }
                }
            }

            if (!victim) return didDemo ? baseReward * demoMultiplier : baseReward;

            Vec enemyGoalCenter = (player.team == Team::BLUE) ? CommonValues::ORANGE_GOAL_CENTER : CommonValues::BLUE_GOAL_CENTER;
            float victimDistToGoal = (victim->pos.To2D() - enemyGoalCenter.To2D()).Length();
            float distanceRatio = RS_CLAMP(victimDistToGoal / maxDistanceFromGoal, 0.0f, 1.0f);
            float proximityFactor = 1.0f - distanceRatio;

            float goalY = (player.team == Team::BLUE) ? CommonValues::BACK_WALL_Y : -CommonValues::BACK_WALL_Y;
            bool victimInGoalZone = std::abs(victim->pos.y - goalY) < goalZoneThreshold;
            float zoneBonus = victimInGoalZone ? 1.5f : 1.0f;

            float offensiveThreshold = CommonValues::BACK_WALL_Y / 3.0f;
            bool ballInOffensiveZone = (player.team == Team::BLUE) ? (state.ball.pos.y > offensiveThreshold) : (state.ball.pos.y < -offensiveThreshold);
            float ballBonus = ballInOffensiveZone ? (1.0f + ballInZoneBonus) : 1.0f;

            float actionMultiplier = didDemo ? demoMultiplier : 1.0f;

            float defenderBonus = 1.0f;
            Vec ballToGoal = (enemyGoalCenter - state.ball.pos).Normalized();
            Vec victimToBall = (state.ball.pos - victim->pos).Normalized();
            if (ballToGoal.Dot(victimToBall) > 0.3f && victimInGoalZone) {
                defenderBonus = 1.3f;
            }

            float reward = baseReward * (1.0f + proximityFactor) * zoneBonus * ballBonus * actionMultiplier * defenderBonus;
            return RS_MIN(reward, maxReward);
        }
    };

    class ApproachForDemoReward : public Reward {
    public:
        float maxRewardDistance;
        float goalZoneY;
        float rewardScale;

        ApproachForDemoReward(float maxRewardDistance = 800.0f, float goalZoneY = 1500.0f, float rewardScale = 0.1f)
            : maxRewardDistance(maxRewardDistance), goalZoneY(goalZoneY), rewardScale(rewardScale) {
        }

        virtual void Reset(const GameState& initialState) override {}

        virtual float GetReward(const Player& player, const GameState& state, bool isFinal) override {
            bool inOffensiveZone = (player.team == Team::BLUE) ? (player.pos.y > goalZoneY) : (player.pos.y < -goalZoneY);
            if (!inOffensiveZone) return 0.0f;

            float closestDistance = FLT_MAX;
            const Player* closestEnemy = nullptr;

            for (const auto& otherPlayer : state.players) {
                if (otherPlayer.team == player.team || otherPlayer.isDemoed) continue;

                bool enemyInDefensiveZone = (player.team == Team::BLUE) ? (otherPlayer.pos.y > goalZoneY) : (otherPlayer.pos.y < -goalZoneY);
                if (!enemyInDefensiveZone) continue;

                float dist = (player.pos - otherPlayer.pos).Length();
                if (dist < closestDistance) {
                    closestDistance = dist;
                    closestEnemy = &otherPlayer;
                }
            }

            if (!closestEnemy || closestDistance > maxRewardDistance) return 0.0f;

            float proximityReward = RS_CLAMP(1.0f - (closestDistance / maxRewardDistance), 0.0f, 1.0f);
            Vec dirToEnemy = (closestEnemy->pos - player.pos).Normalized();
            float speedTowardsEnemy = player.vel.Dot(dirToEnemy);
            float velocityBonus = RS_CLAMP(speedTowardsEnemy / CommonValues::CAR_MAX_SPEED, 0.0f, 1.0f);
            float supersonicBonus = player.isSupersonic ? 1.5f : 1.0f;

            return proximityReward * (1.0f + velocityBonus) * supersonicBonus * rewardScale;
        }
    };

	// AerialReward: encourages controlled aerial positioning rather than just jump touches.
	// - Gives no reward below `minBallZ` or if car is below `minCarZ`.
	// - Base reward is `baseReward` at `minBallZ` and scales linearly with ball height
	//   up to 2x at the ceiling (CommonValues::CEILING_Z by default).
	// - Multiplies by how well the car nose points toward the ball (dot in [0..1]).
	class AerialReward : public Reward {
	public:
		float minBallZ;     // minimum ball height to start rewarding (uu)
		float minCarZ;      // minimum car height to start rewarding (uu)
		float baseReward;   // base reward at minBallZ with perfect orientation
		float ceilingZ;     // height considered as ceiling for scaling (uu)
		float orientPower;  // shaping for orientation factor (>=1 tightens requirement)

		AerialReward(
			float minBallZ = 150.0f,
			float minCarZ = 120.0f,
			float baseReward = 1.0f,
			float ceilingZ = CommonValues::CEILING_Z,
			float orientPower = 1.0f)
			: minBallZ(minBallZ), minCarZ(minCarZ), baseReward(baseReward), ceilingZ(ceilingZ), orientPower(orientPower) {}

		virtual float GetReward(const Player& player, const GameState& state, bool /*isFinal*/) override {
			const float bZ = state.ball.pos.z;
			const float cZ = player.pos.z;

			// Height gates
			if (bZ < minBallZ || cZ < minCarZ)
				return 0.0f;

			// Linear height scaling: 1.0 at minBallZ -> 2.0 at ceilingZ
			float denom = RS_MAX(1.0f, ceilingZ - minBallZ);
			float t = RS_CLAMP((bZ - minBallZ) / denom, 0.0f, 1.0f);
			float heightScale = 1.0f + t; // [1..2]

			// Orientation toward the ball: dot in [0..1]
			Vec toBall = state.ball.pos - player.pos;
			float dist = toBall.Length();
			if (dist < 1e-4f)
				return 0.0f;
			Vec dirToBall = toBall / dist;
			float align = player.rotMat.forward.Dot(dirToBall);
			float orient = RS_CLAMP(align, 0.0f, 1.0f);
			if (orientPower != 1.0f)
				orient = powf(orient, orientPower);

			return baseReward * heightScale * orient;
		}
	};

	// Energy-based reward combining gravitational potential, kinetic energy, jump/flip potential, and boost
	class EnergyReward : public Reward {
	public:
		float mass;          // car mass in kg (default 180)
		float heightFudge;   // scale to reduce height contribution
		float minZ;          // reference ground height
		float jumpSpeed;     // effective speed equivalent for jump energy term
		float flipSpeed;     // effective speed equivalent for flip energy term
		float boostCoeff;    // coefficient to convert boost amount to energy units

		EnergyReward(
			float mass_ = 180.0f,
			float heightFudge_ = 0.75f,
			float minZ_ = 17.0f,
			float jumpSpeed_ = 292.0f,
			float flipSpeed_ = 550.0f,
			float boostCoeff_ = 7.97e6f
		) : mass(mass_), heightFudge(heightFudge_), minZ(minZ_), jumpSpeed(jumpSpeed_), flipSpeed(flipSpeed_), boostCoeff(boostCoeff_) {}

		virtual float GetReward(const Player& player, const GameState& state, bool isFinal) override {
			const float MASS = mass;
			const float G_MAG = fabsf(CommonValues::GRAVITY_Z);
			const float CEILING_Z = CommonValues::CEILING_Z;
			const float CAR_MAX_SPEED = CommonValues::CAR_MAX_SPEED;

			// Maximum reference energy (height to ceiling + max kinetic)
			const float max_energy = (MASS * G_MAG * (CEILING_Z - minZ)) + (0.5f * MASS * (CAR_MAX_SPEED * CAR_MAX_SPEED));

			float energy = 0.0f;

			// Jump potential: if the player can still jump (on ground or still within double-jump window)
			if (player.HasFlipOrJump()) {
				energy += 0.35f * MASS * jumpSpeed * jumpSpeed;
			}

			// Flip/double-jump potential: if a flip or double jump hasn't been consumed yet
			if (!player.hasFlipped && !player.hasDoubleJumped) {
				energy += 0.35f * MASS * flipSpeed * flipSpeed;
			}

			// Height contribution (using |g|) with a fudge factor to reduce height influence
			const float heightAboveMin = RS_MAX(player.pos.z - minZ, 0.0f);
			energy += MASS * G_MAG * heightAboveMin * heightFudge;

			// Kinetic energy from linear velocity
			const float v = player.vel.Length();
			energy += 0.5f * MASS * v * v;

			// Boost contribution (boost is 0..100 in this framework)
			energy += boostCoeff * player.boost;

			const float std_energy = player.isDemoed ? 0.0f : (energy / max_energy);
			return std_energy;
		}
	};

	// WavedashReward removed (duplicate of CommonRewards.h)

	class AirRollReward : public Reward {
		private:
			struct AirRollState {
				float lastDir = 0.0f;
				float switchPenalty = 0.0f;
				uint64_t lastTickSeen = 0ULL;
				uint64_t lastRollTick = 0ULL;
			};
	
		public:
			AirRollReward(
				float rewardRight = 0.5f,
				float rewardLeft = 1.5f,
				float minHeight = 320.0f,
				float maxBallDistance = 1000.0f,
				float alignmentThreshold = 0.2f,
				float rollThreshold = 0.5f,
				float minApproachSpeed = 300.0f,
				float fastApproachMultiplier = 1.01f,
				float fastApproachAlignment = 0.01f,
				float switchPenaltyIncrement = 1.0f,
				float switchPenaltyDecayPerTick = 0.05f,
				float penaltyScale = 0.5f,
				uint64_t directionResetTicks = 60ULL
			) :
				_rewardRight(rewardRight),
				_rewardLeft(rewardLeft),
				_minHeight(minHeight),
				_maxBallDistance(std::max(0.0f, maxBallDistance)),
				_alignmentThreshold(alignmentThreshold),
				_rollThreshold(rollThreshold),
				_minApproachSpeed(minApproachSpeed),
				_fastApproachMultiplier(std::max(1.0f, fastApproachMultiplier)),
				_fastApproachAlignment(std::max(0.0f, fastApproachAlignment)),
				_switchPenaltyIncrement(std::max(0.0f, switchPenaltyIncrement)),
				_switchPenaltyDecayPerTick(std::max(0.0f, switchPenaltyDecayPerTick)),
				_penaltyScale(std::max(0.0f, penaltyScale)),
				_directionResetTicks(directionResetTicks) {
				if (_directionResetTicks == 0ULL) {
					_directionResetTicks = 60ULL;
				}
			}
	
			void Reset(const GameState& initialState) override {
				_states.clear();
			}
	
			float GetReward(const Player& player, const GameState& state, bool isFinal) override {
				AirRollState& st = _states[player.carId];
				_decayState(st, state.lastTickCount);
	
				if (player.isOnGround)
					return 0.0f;
	
				if (player.pos.z < _minHeight || state.ball.pos.z < _minHeight)
					return 0.0f;
	
				Vec toBall = state.ball.pos - player.pos;
				float distToBall = toBall.Length();
				if (distToBall < 1e-3f)
					return 0.0f;
				if (_maxBallDistance > 0.0f && distToBall > _maxBallDistance)
					return 0.0f;
				Vec dirToBall = toBall / distToBall;
	
				float speed = player.vel.Length();
				if (speed < 1e-3f)
					return 0.0f;
	
				float approachSpeed = dirToBall.Dot(player.vel);
				if (approachSpeed < _minApproachSpeed)
					return 0.0f;
	
				float alignment = approachSpeed / (speed + 1e-6f);
				if (alignment < _alignmentThreshold) {
					bool fastApproach = approachSpeed >= _minApproachSpeed * _fastApproachMultiplier;
					if (!(fastApproach && alignment >= _fastApproachAlignment))
						return 0.0f;
				}
	
				float rollInput = player.prevAction[4];
				if (std::fabs(rollInput) < _rollThreshold)
					return 0.0f;
	
				float direction = rollInput > 0.0f ? 1.0f : -1.0f;
				if (st.lastDir != 0.0f && direction != st.lastDir) {
					st.switchPenalty = std::min(6.0f, st.switchPenalty + _switchPenaltyIncrement);
				}
				st.lastDir = direction;
				st.lastRollTick = state.lastTickCount;
	
				float penaltyMultiplier = 1.0f;
				if (_penaltyScale > 0.0f) {
					penaltyMultiplier = std::max(0.0f, 1.0f - _penaltyScale * st.switchPenalty);
				}
	
				float baseReward = direction > 0.0f ? _rewardRight : _rewardLeft;
				return baseReward * penaltyMultiplier;
			}
	
		private:
			void _decayState(AirRollState& st, uint64_t currentTick) {
				if (st.lastTickSeen == 0ULL) {
					st.lastTickSeen = currentTick;
				} else if (currentTick > st.lastTickSeen) {
					uint64_t deltaTicks = currentTick - st.lastTickSeen;
					if (_switchPenaltyDecayPerTick > 0.0f) {
						float decayAmount = _switchPenaltyDecayPerTick * static_cast<float>(deltaTicks);
						st.switchPenalty = std::max(0.0f, st.switchPenalty - decayAmount);
					}
					st.lastTickSeen = currentTick;
				}
	
				if (_directionResetTicks > 0ULL && st.lastRollTick > 0ULL && currentTick > st.lastRollTick) {
					uint64_t ticksSinceRoll = currentTick - st.lastRollTick;
					if (ticksSinceRoll >= _directionResetTicks && st.switchPenalty <= 1e-3f) {
						st.lastDir = 0.0f;
						st.lastRollTick = 0ULL;
					}
				}
			}
	
			float _rewardRight;
			float _rewardLeft;
			float _minHeight;
			float _maxBallDistance;
			float _alignmentThreshold;
			float _rollThreshold;
			float _minApproachSpeed;
			float _fastApproachMultiplier;
			float _fastApproachAlignment;
			float _switchPenaltyIncrement;
			float _switchPenaltyDecayPerTick;
			float _penaltyScale;
			uint64_t _directionResetTicks;
			std::unordered_map<uint32_t, AirRollState> _states;
		};

	// PickupBoostReward, SaveBoostReward, AirReward removed (duplicate of CommonRewards.h)

	class DribbleReward : public Reward {
	public:
		virtual float GetReward(const Player& player, const GameState& state, bool isFinal) override {
			const float MIN_BALL_HEIGHT = 109.0f;
			const float MAX_BALL_HEIGHT = 180.0f;
			const float MAX_DISTANCE = 197.0f;
			const float SPEED_MATCH_FACTOR = 2.0f; // Adjust this value to control the importance of speed matching
			const float CAR_MAX_SPEED = CommonValues::CAR_MAX_SPEED;

			if (player.isOnGround && state.ball.pos.z >= MIN_BALL_HEIGHT && state.ball.pos.z <= MAX_BALL_HEIGHT && (player.pos - state.ball.pos).Length() < MAX_DISTANCE) {
				float playerSpeed = player.vel.Length();
				float ballSpeed = state.ball.vel.Length();
				float speedMatchReward = ((playerSpeed/CAR_MAX_SPEED) + SPEED_MATCH_FACTOR * (1.0f - std::abs(playerSpeed - ballSpeed) / (playerSpeed + ballSpeed))) / 2.0f;
				return speedMatchReward; // Reward for successful dribbling, with a bonus for speed matching, normalized to 1
			} else {
				return 0.0f; // No reward
			}
		}
	};

class TeammateBumpPenaltyReward : public Reward {
private:
    std::unordered_map<int, uint32_t> previousContactCarID;
    std::unordered_map<int, float> previousContactTimer;

public:
    TeammateBumpPenaltyReward() {}

    virtual void Reset(const GameState& initialState) override {
        previousContactCarID.clear();
        previousContactTimer.clear();
    }

    virtual float GetReward(const Player& player, const GameState& state, bool isFinal) override {
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
            }
            else {
                float prevTimer = prevTimerIter->second;
                uint32_t prevCarID = prevCarIDIter->second;

                isNewBump = (currentContactTimer > prevTimer) ||
                    (currentContactCarID != prevCarID && prevTimer <= 0);
            }

            if (isNewBump) {
                for (const auto& otherPlayer : state.players) {
                    if (otherPlayer.carId == currentContactCarID) {
                        if (otherPlayer.team == player.team) {
                            reward = -1.0f;
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

	class FlyToGoal : public Reward {
	public:
		// ============================================================================
		// PARAMETRI BASE
		// ============================================================================


		// --- Gating & reward base (per-tick) ---
		float min_air_height = 320.0f;
		float max_ball_dist = 450.0f;              // Leggermente aumentato (era 380)
		float per_tick_scale = 0.08f;              // Aumentato da 0.05

		// --- Touch bonus (esponenziale verso porta) ---
		float touch_base = 10.0f;                  // Aumentato da 15
		float lambda_goal = 5.0f;                 // Più aggressivo (era 10)
		float min_touch_factor = 0.0f;             // ZERO per tocchi sbagliati (era 0.02)
		float max_delta_v = 2500.0f;
		bool  use_impulse_delta = true;



		// --- Field position scaling ---
		float min_field_mult = 0.35f;              // Più permissivo (era 0.27)
		float max_field_mult = 1.35f;              // Leggermente più alto
		float optimal_start_dist = 6000.0f;
		float too_far_penalty_dist = 10000.0f;
		bool  use_field_scaling = true;

		// --- Bonus "sotto la palla" (solo per touch) ---
		float under_ball_min_gap = 180.0f;         // Leggermente aumentato
		float under_ball_max_gap = 450.0f;
		float under_touch_boost = 0.5f;          //  RIDOTTO da 0.6

		// ============================================================================
		// PARAMETRI ALLINEAMENTO DIREZIONE TARGET (dal documento)
		// ============================================================================
		float w_dir_ball = 0.35f;                // Peso verso la palla
		float w_dir_goal = 0.75f;                // Peso verso la porta (palla→porta)
		float min_goal_push_dot = 0.30f;           //  AUMENTATO da 0.15 (più stringente!)
		float cross_track_penalty = 0.20f;         //  AUMENTATO da 0.06 (3x più forte)
		float opposite_dir_penalty = 0.35f;        //  AUMENTATO da 0.12 (3x più forte)

		// ============================================================================
		// 🆕 PARAMETRI SHOT QUALITY (nuovo sistema semplificato)
		// ============================================================================
		float shot_quality_bonus = 8.0f;           // Bonus per posizionamento ottimale
		float optimal_approach_angle = 25.0f;      // Angolo ottimale approach (gradi)
		float shot_quality_distance_max = 0.8f;    // Max dist come % di max_ball_dist

		// 🆕 Height optimization
		float optimal_shot_height = 380.0f;        // Altezza ottimale per shot
		float shot_height_bonus = 4.0f;            // Bonus aggiuntivo per altezza corretta

		// ============================================================================
		// STATE TRACKING
		// ============================================================================
		struct PlayerAirRollState {
			// Shot quality tracking
			float best_shot_quality = 0.0f;
			int   ticks_in_good_position = 0;
		};
		std::unordered_map<uint32_t, PlayerAirRollState> player_states;

		// ============================================================================
		// HELPER FUNCTIONS
		// ============================================================================

		inline static float clamp01(float x) {
			return std::max(0.0f, std::min(1.0f, x));
		}

		std::string GetName() override {
			return "FlyToGoal";
		}

		void Reset(const GameState& initialState) override {
			player_states.clear();
		}

		float GetFieldPositionMultiplier(const Player& player, const Vec& ball_pos) {
			if (!use_field_scaling) return 1.0f;

			Vec oppGoal = (player.team == RocketSim::Team::BLUE)
				? CommonValues::ORANGE_GOAL_CENTER
				: CommonValues::BLUE_GOAL_CENTER;
			Vec ownGoal = (player.team == RocketSim::Team::BLUE)
				? CommonValues::BLUE_GOAL_CENTER
				: CommonValues::ORANGE_GOAL_CENTER;

			float dist_to_opp_goal = (ball_pos - oppGoal).Length();
			float field_mult = 1.0f;

			if (dist_to_opp_goal <= optimal_start_dist) {
				float proximity_factor = 1.0f - (dist_to_opp_goal / optimal_start_dist);
				field_mult = 1.0f + (max_field_mult - 1.0f) * proximity_factor;
			}
			else if (dist_to_opp_goal <= too_far_penalty_dist) {
				float penalty_factor = (dist_to_opp_goal - optimal_start_dist) /
					(too_far_penalty_dist - optimal_start_dist);
				field_mult = 1.0f - (1.0f - min_field_mult) * penalty_factor;
			}
			else {
				field_mult = min_field_mult;
			}

			// Penalità own half ridotta
			float midfield_y = (oppGoal.y + ownGoal.y) / 2.0f;
			bool in_own_half = (player.team == RocketSim::Team::BLUE && ball_pos.y < midfield_y) ||
				(player.team == RocketSim::Team::ORANGE && ball_pos.y > midfield_y);
			if (in_own_half) {
				field_mult *= 0.85f;  // Era 0.7, ora più permissivo
			}

			return std::max(min_field_mult, std::min(max_field_mult, field_mult));
		}

		// 🆕 NUOVO: Calcola qualità posizionamento per shot
		float GetShotQuality(const Vec& player_pos, const Vec& ball_pos,
			const Vec& ball_to_goal_dir, float ball_height) {
			// Approach angle quality
			Vec player_to_ball = ball_pos - player_pos;
			player_to_ball.z = 0.0f;  // Solo orizzontale
			float dist = player_to_ball.Length();

			if (dist < 1e-6f) return 0.5f;

			Vec approach_dir = player_to_ball / dist;
			float dot = approach_dir.Dot(ball_to_goal_dir);
			dot = std::max(-1.0f, std::min(1.0f, dot));

			float angle_rad = std::acos(dot);
			float angle_deg = angle_rad * 180.0f / 3.14159265f;

			// Premia approach laterale/dietro palla (migliore per shot)
			float angle_quality = 0.0f;
			if (angle_deg < optimal_approach_angle) {
				angle_quality = 1.0f;
			}
			else if (angle_deg < optimal_approach_angle + 40.0f) {
				float ratio = (angle_deg - optimal_approach_angle) / 40.0f;
				angle_quality = 1.0f - 0.6f * ratio;
			}
			else {
				angle_quality = 0.4f;
			}

			// Height quality (premia altezze ottimali per shot)
			float height_diff = std::abs(ball_height - optimal_shot_height);
			float height_quality = 1.0f;
			if (height_diff > 400.0f) {
				height_quality = 0.5f;
			}
			else {
				height_quality = 1.0f - 0.5f * (height_diff / 400.0f);
			}

			return angle_quality * height_quality;
		}

		// ============================================================================
		// MAIN REWARD FUNCTION
		// ============================================================================

		float GetReward(const Player& player, const GameState& state, bool /*isFinal*/) override {
			auto& ar_state = player_states[player.carId];

			// ========================================================================
			// GATING
			// ========================================================================
			bool passed_gates = true;
			if (player.isOnGround || player.pos.z < min_air_height) {
				passed_gates = false;
			}

			float dist = (state.ball.pos - player.pos).Length();
			if (dist > max_ball_dist) {
				passed_gates = false;
			}

			if (!passed_gates) {
				// Reset state when outside gating
				ar_state.best_shot_quality = 0.0f;
				ar_state.ticks_in_good_position = 0;
				return 0.0f;
			}

			float field_mult = GetFieldPositionMultiplier(player, state.ball.pos);

			// ========================================================================
			// DIREZIONI BASE
			// ========================================================================
			Vec toBall = state.ball.pos - player.pos;
			Vec toBall2D(toBall.x, toBall.y, 0.0f);
			float d2 = toBall2D.Length();
			Vec dirToBall2D = (d2 > 1e-6f) ? (toBall2D / d2) : Vec(0, 0, 0);

			Vec v2D(player.vel.x, player.vel.y, 0.0f);
			float v2 = v2D.Length();
			Vec vHat2D = (v2 > 1e-6f) ? (v2D / v2) : Vec(0, 0, 0);

			// ========================================================================
			// DIREZIONE TARGET BLEND (palla→porta + verso palla)
			// ========================================================================
			Vec oppGoal = (player.team == RocketSim::Team::BLUE)
				? CommonValues::ORANGE_GOAL_CENTER
				: CommonValues::BLUE_GOAL_CENTER;

			Vec ballToGoal2D(oppGoal.x - state.ball.pos.x, oppGoal.y - state.ball.pos.y, 0.0f);
			float bgL = ballToGoal2D.Length();
			Vec dirBallToGoal2D = (bgL > 1e-6f) ? (ballToGoal2D / bgL) : Vec(0, 0, 0);

			// Blend dinamico: più vicino alla palla → più peso verso palla
			float nearBall01 = clamp01((max_ball_dist - std::min(max_ball_dist, d2)) / max_ball_dist);
			float wB = w_dir_ball * (0.6f + 0.4f * nearBall01);
			float wG = w_dir_goal * (1.0f - 0.3f * nearBall01);

			Vec desired2D = wB * dirToBall2D + wG * dirBallToGoal2D;
			float desL = desired2D.Length();
			if (desL > 1e-6f) desired2D = desired2D / desL;

			// Allineamento velocità e muso alla direzione target
			Vec fwd2D(player.rotMat.forward.x, player.rotMat.forward.y, 0.0f);
			float f2 = fwd2D.Length();
			if (f2 > 1e-6f) fwd2D = fwd2D / f2;

			float align_vel_desired = clamp01(vHat2D.Dot(desired2D));
			float align_fwd_desired = clamp01(fwd2D.Dot(desired2D));

			// Cross-track: componente laterale rispetto a desired2D
			float cross = std::sqrt(std::max(0.0f, 1.0f - align_vel_desired * align_vel_desired));
			float cross_pen = cross_track_penalty * cross;

			// Penalità se vai contro la direzione target
			float opposite_pen = 0.0f;
			if (v2 > 50.0f) {
				float rawDot = vHat2D.Dot(desired2D);
				if (rawDot < 0.0f) {
					opposite_pen = opposite_dir_penalty * (-rawDot);
				}
			}

			// ========================================================================
			// ALLINEAMENTI "CLASSICI" (per compatibilità)
			// ========================================================================
			float vlen = player.vel.Length();
			Vec vHat = (vlen > 1e-6f) ? (player.vel / vlen) : Vec(0, 0, 0);
			Vec dirToBall = (dist > 1e-6f) ? ((state.ball.pos - player.pos) / dist) : Vec(0, 0, 0);
			float align_ball = clamp01(vHat.Dot(dirToBall));

			const float R = CommonValues::BALL_RADIUS;
			float proximity = clamp01(1.0f - (dist - R) / std::max(1.0f, (max_ball_dist - R)));

			// ========================================================================
			// REWARD BASE PER-TICK (weighted blend)
			// ========================================================================
			float shaped_core =
				0.50f * align_vel_desired +   // Direzione target
				0.20f * align_fwd_desired +   // Muso allineato
				0.20f * align_ball +          // Ancora verso palla
				0.10f * proximity;            // Prossimità

			shaped_core = clamp01(shaped_core);
			float reward = per_tick_scale * shaped_core * field_mult;

			// Applica penalità (aumentate!)
			reward -= (cross_pen + opposite_pen) * field_mult;

			// ========================================================================
			// 🆕 SHOT QUALITY BONUS
			// ========================================================================
			if (dist < max_ball_dist * shot_quality_distance_max) {
				float shot_quality = GetShotQuality(player.pos, state.ball.pos,
					dirBallToGoal2D, state.ball.pos.z);

				// Reward continuo per posizione ottimale
				reward += shot_quality_bonus * shot_quality * field_mult * 0.04f;

				// Track best position
				ar_state.best_shot_quality = std::max(ar_state.best_shot_quality, shot_quality);

				if (shot_quality > 0.75f) {
					ar_state.ticks_in_good_position++;

					// Bonus milestone per mantenere posizione ottimale
					if (ar_state.ticks_in_good_position == 15) {
						reward += 2.0f * field_mult;
					}
					else if (ar_state.ticks_in_good_position == 30) {
						reward += 3.0f * field_mult;
					}
				}
				else {
					ar_state.ticks_in_good_position = 0;
				}

				// Bonus extra per altezza ottimale
				float height_diff = std::abs(state.ball.pos.z - optimal_shot_height);
				if (height_diff < 150.0f) {
					float height_quality = 1.0f - (height_diff / 150.0f);
					reward += shot_height_bonus * height_quality * field_mult * 0.03f;
				}
			}

			// ========================================================================
			// TOUCH BONUS (esponenziale verso porta, ZERO per tocchi sbagliati)
			// ========================================================================
			if (player.ballTouchedStep) {
				Vec dV = (use_impulse_delta && state.prev)
					? (state.ball.vel - state.prev->ball.vel)
					: state.ball.vel;

				Vec dV2D(dV.x, dV.y, 0.0f);
				float dVlen = dV2D.Length();

				if (dVlen > 1e-6f && bgL > 1e-6f) {
					Vec dVhat = dV2D / dVlen;
					float dot_goal = std::max(-1.0f, std::min(1.0f, dVhat.Dot(dirBallToGoal2D)));

					// Esponenziale più aggressivo
					float e_pos = std::exp(lambda_goal * dot_goal);
					float e_min = std::exp(-lambda_goal);
					float e_max = std::exp(lambda_goal);
					float gain01 = (e_pos - e_min) / (e_max - e_min);

					float mag = clamp01(dVlen / std::max(1.0f, max_delta_v));

					// 🔧 ZERO reward se direzione negativa (era min_touch_factor)
					float dir_factor = (dot_goal > 0.0f) ? gain01 : 0.0f;

					float touch = touch_base * mag * dir_factor * field_mult;

					// Boost se tocco da sotto (SOLO se verso porta)
					if (dot_goal > 0.5f) {
						float vertical_gap = state.ball.pos.z - player.pos.z;
						if (vertical_gap > under_ball_min_gap && dV.z > 0.0f) {
							float gap01 = clamp01((vertical_gap - under_ball_min_gap) /
								std::max(1.0f, under_ball_max_gap - under_ball_min_gap));
							float touch_boost = 1.0f + under_touch_boost * gap01;
							touch *= touch_boost;
						}
					}

					reward += touch;

					// 🆕 Bonus extra per shot perfetti
					if (dot_goal > 0.9f && mag > 0.6f) {
						reward += 5.0f * field_mult;
					}
				}
			}

			// ========================================================================
			// 🆕 RIMOSSO: Secondo "under ball bonus" continuo
			// Era confuso e controproducente - ora solo nel touch bonus
			// ========================================================================



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
        float dashRewardBase = 12.0f,
        float resetRewardBase = 16.0f,
        float wavedashRewardBase = 10.0f,
        float zapDashBaseReward = 15.0f,
        // float demoReward = 30.0f, 
        // Scaling
        float accelerationScalar = 1.0f,
        // Penalties
        float wallStayPenalty = -0.2f,
        float supersonicBoostPenalty = -0.1f,
        // Debug
        bool debug = false
    ) :
        wallHeightThreshold(100.0f),
        dashRewardBase(dashRewardBase), resetRewardBase(resetRewardBase),
        wavedashRewardBase(wavedashRewardBase), zapDashBaseReward(zapDashBaseReward),
        // demoReward(demoReward),
        accelerationScalar(accelerationScalar),
        wallStayPenalty(wallStayPenalty), supersonicBoostPenalty(supersonicBoostPenalty),
        zapMinSpeedGain(500.0f), zapMinNoseDown(0.5f), zapMinFwdDot(0.7f),
        debug(debug)
    {
    }

    virtual void Reset(const GameState& initialState) override {
        playerStates.clear();
        for (const auto& player : initialState.players) {
            InitState(player);
        }
    }

    virtual float GetReward(const Player& player, const GameState& state, bool isFinal) override {
        if (!state.prev) return 0.0f;

        if (playerStates.find(player.carId) == playerStates.end()) {
            InitState(player);
        }
        PlayerState& st = playerStates[player.carId];

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
            if (debug) printf("ID:%d | DEMO | Reward: %.2f\n", player.carId, demoReward * qualityMult);
        }
        */

        // 2. ZAP DASH
        float zapRatio = CheckZapDash(player);
        if (zapRatio > 0.0f) {
            float r = zapDashBaseReward * zapRatio * qualityMult;
            totalReward += r;
            if (debug) printf("ID:%d | ZapDash | Reward: %.2f\n", player.carId, r);
        }

        // 3. WALL DASH (LAUNCH)
        if (st.wasOnWall && st.hadFlipBefore && !hasFlip && !isOnWall) {
            st.dashUsed = true;
            float r = dashRewardBase * qualityMult;
            totalReward += r;
            if (debug) printf("ID:%d | WallDash_Launch | Reward: %.2f\n", player.carId, r);
        }

        // 4. WALL DASH (RESET)
        if (st.dashUsed && isOnWall && hasFlip) {
            st.dashUsed = false; // Cycle complete
            float r = resetRewardBase * qualityMult;
            totalReward += r;
            if (debug) printf("ID:%d | WallDash_Reset | Reward: %.2f\n", player.carId, r);
        }

        // 5. WAVEDASH
        const Player* prevPlayer = nullptr;
        for (const auto& p : state.prev->players) { if (p.carId == player.carId) { prevPlayer = &p; break; } }

        bool justLanded = player.isOnGround && prevPlayer && !prevPlayer->isOnGround;
        if (justLanded && (player.isFlipping || st.wasFlipping)) {
            float r = wavedashRewardBase * qualityMult;
            totalReward += r;
            if (debug) printf("ID:%d | Wavedash | Reward: %.2f\n", player.carId, r);
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
    void InitState(const Player& p) {
        PlayerState newState;
        newState.prevVel = p.vel;
        newState.hadFlipBefore = p.HasFlipOrJump();
        // newState.lastDemoCount = p.match_demolitions; // Disabled
        playerStates[p.carId] = newState;
    }

    float CheckZapDash(const Player& current) {
        if (!current.prev || !current.prev->prev || !current.prev->prev->prev) return 0.0f;

        const Player* p0 = &current;
        const Player* p1 = current.prev;
        const Player* p2 = p1->prev;
        const Player* p3 = p2->prev;

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

    bool IsOnWall(const Vec& pos) {
        if (pos.z < wallHeightThreshold) return false;
        if (std::abs(std::abs(pos.x) - 4096.0f) < 150.0f) return true;
        if (std::abs(std::abs(pos.y) - 5120.0f) < 150.0f) return true;
        return false;
    }
};

	// ============================================================================
	// MKH_Airdribble_reward
	// A lean airdribble reward: gates on air + ball proximity, then gives a small
	// per-tick signal for moving the ball toward the goal and a touch bonus when
	// the player actually hits it in the right direction.  No field scaling, no
	// shot-quality tracking, no milestone counters.
	// ============================================================================
	class MKH_Airdribble_reward : public Reward {
	public:
		// --- Gating ---
		float min_air_height  = 250.0f;   // car must be above this to activate
		float max_ball_dist   = 500.0f;   // car-to-ball distance gate

		// --- Per-tick reward ---
		// Each tick we reward: how well the velocity direction lines up with the
		// ball→goal direction, blended with raw proximity.
		float tick_scale      = 0.05f;    // overall per-tick multiplier

		// --- Touch bonus ---
		// On a touch, reward = touch_base * impulse_magnitude * goal_direction_quality
		// (zero if the touch sends the ball away from goal)
		float touch_base      = 8.0f;
		float max_delta_v     = 2500.0f;  // impulse normalisation cap

		// ============================================================================

		inline static float clamp01(float x) {
			return std::max(0.0f, std::min(1.0f, x));
		}

		std::string GetName() override { return "MKH_Airdribble_reward"; }

		void Reset(const GameState& /*initialState*/) override {}

		float GetReward(const Player& player, const GameState& state, bool /*isFinal*/) override {

			// ---- GATING --------------------------------------------------------
			if (player.isOnGround || player.pos.z < min_air_height)
				return 0.0f;

			float dist = (state.ball.pos - player.pos).Length();
			if (dist > max_ball_dist)
				return 0.0f;

			// ---- SHARED GEOMETRY -----------------------------------------------
			Vec oppGoal = (player.team == RocketSim::Team::BLUE)
				? CommonValues::ORANGE_GOAL_CENTER
				: CommonValues::BLUE_GOAL_CENTER;

			// Direction: ball → opponent goal (2-D is fine, keeps it simpler)
			Vec ballToGoal(oppGoal.x - state.ball.pos.x,
			               oppGoal.y - state.ball.pos.y, 0.0f);
			float bgLen = ballToGoal.Length();
			Vec dirBallToGoal = (bgLen > 1e-6f) ? (ballToGoal / bgLen) : Vec(0, 0, 0);

			// ---- PER-TICK REWARD -----------------------------------------------
			// 1) How well is the car velocity aligned with ball→goal?
			float vlen = player.vel.Length();
			Vec vHat   = (vlen > 1e-6f) ? (player.vel / vlen) : Vec(0, 0, 0);
			float vel_align = clamp01(vHat.x * dirBallToGoal.x + vHat.y * dirBallToGoal.y);

			// 2) Proximity bonus: closer to ball = more reward
			const float R = CommonValues::BALL_RADIUS;
			float proximity = clamp01(1.0f - (dist - R) / std::max(1.0f, max_ball_dist - R));

			float tick_reward = tick_scale * (0.7f * vel_align + 0.3f * proximity);

			// ---- TOUCH BONUS ---------------------------------------------------
			float touch_reward = 0.0f;
			if (player.ballTouchedStep) {
				// Use impulse delta when available (same approach as FlyToGoal)
				Vec dV = (state.prev)
					? (state.ball.vel - state.prev->ball.vel)
					: state.ball.vel;

				Vec dV2D(dV.x, dV.y, 0.0f);
				float dVlen = dV2D.Length();

				if (dVlen > 1e-6f && bgLen > 1e-6f) {
					Vec dVhat = dV2D / dVlen;
					float dot_goal = std::max(-1.0f, std::min(1.0f,
						dVhat.x * dirBallToGoal.x + dVhat.y * dirBallToGoal.y));

					// Only reward touches that send the ball toward goal
					if (dot_goal > 0.0f) {
						float mag       = clamp01(dVlen / std::max(1.0f, max_delta_v));
						touch_reward    = touch_base * mag * dot_goal;
					}
				}
			}

			return tick_reward + touch_reward;
		}
	};

	class TeamSpacingReward_MKH : public Reward {
    public:
        float min_spacing;
        float penalty_strength;

        TeamSpacingReward_MKH(float min_spacing = 1000.0f, float penalty_strength = 1.0f) 
            : min_spacing(min_spacing), penalty_strength(penalty_strength) {}

        void Reset(const GameState& initialState) override {}

        float GetReward(const Player& player, const GameState& state, bool isFinal) override {
            float total_penalty = 0.0f;
            int count = 0;

            for (const auto& other : state.players) {
                if (other.carId == player.carId) continue;
                if (other.team != player.team) continue;

                float dist = (player.pos - other.pos).Length();

                if (dist < min_spacing) {
                    float penalty = 1.0f - (dist / min_spacing);
                    total_penalty += penalty;
                    count++;
                }
            }

            if (count > 0) {
                return -total_penalty * penalty_strength;
            }

            return 0.0f;
        }
    };

	// FlickSpeedReward: reward fast, goalward, upward launches caused by a flip near the ball ("flicks")
	class FlickSpeedReward : public Reward {
	public:
		// Tuning parameters
		float dribbleMaxDist;        // pre-flip: car must be this close to ball to consider a flick
		float dribbleMinBallZ;       // pre-flip: ball must be at least this high (on car)
		float maxRewardedBallKPH;    // cap for ball speed normalization
		float weightSpeedGain;       // weight for speed gain of the ball (delta speed)
		float weightUpward;          // weight for upward launch component
		float weightGoalward;        // weight for forward direction toward opponent goal
		float minQualifyingDeltaKPH; // minimum delta speed (kph) to pay any reward

		FlickSpeedReward(
			float dribbleMaxDist_ = 197.0f,
			float dribbleMinBallZ_ = 109.0f,
			float maxRewardedBallKPH_ = 130.0f,
			float weightSpeedGain_ = 1.0f,
			float weightUpward_ = 0.5f,
			float weightGoalward_ = 0.8f,
			float minQualifyingDeltaKPH_ = 10.0f)
			: dribbleMaxDist(dribbleMaxDist_),
			  dribbleMinBallZ(dribbleMinBallZ_),
			  maxRewardedBallKPH(maxRewardedBallKPH_),
			  weightSpeedGain(weightSpeedGain_),
			  weightUpward(weightUpward_),
			  weightGoalward(weightGoalward_),
			  minQualifyingDeltaKPH(minQualifyingDeltaKPH_) {}

		virtual void Reset(const GameState& initialState) override {
			_states.clear();
		}

		virtual void PreStep(const GameState& state) override {
			for (const auto& p : state.players) {
				auto& st = _states[p.carId];
				st.timeSinceFlip += state.deltaTime;
				// Track pre-flip dribble-like state using original dribble thresholds (always requires on-ground here)
				const float MIN_BALL_HEIGHT = 109.0f;
				const float MAX_BALL_HEIGHT = 180.0f;
				const float MAX_DISTANCE = 197.0f;
				float dist = (p.pos - state.ball.pos).Length();
				st.preflipDribble = p.isOnGround && state.ball.pos.z >= MIN_BALL_HEIGHT && state.ball.pos.z <= MAX_BALL_HEIGHT && dist < MAX_DISTANCE;
				// Remember last ball speed for delta computation
				st.prevBallSpeed = state.ball.vel.Length();
			}
		}

		virtual float GetReward(const Player& player, const GameState& state, bool isFinal) override {
			auto& st = _states[player.carId];
			float rew = 0.0f;

			// Detect flip start edge: prev not flipping -> now flipping
			bool flipEdge = false;
			if (player.prev) {
				if (!player.prev->isFlipping && player.isFlipping) {
					flipEdge = true;
					st.timeSinceFlip = 0.0f;
					// Latch the dribble state at the moment flip starts (allows jump -> flip sequence)
					st.preflipDribbleLatched = st.preflipDribble;
				}
			}

			// Qualifying touch: must have been dribbling pre-flip (latched) and a real flip performed (no time window required)
			bool flipPerformed = flipEdge || player.isFlipping || (player.prev && !player.prev->hasFlipped && player.hasFlipped);
			if (player.ballTouchedStep && st.preflipDribbleLatched && flipPerformed) {
				// Speed gain term
				const float prevSpeed = (state.prev ? state.prev->ball.vel.Length() : st.prevBallSpeed);
				const float curSpeed = state.ball.vel.Length();
				const float deltaSpeed = RS_MAX(0.0f, curSpeed - prevSpeed);
				const float maxVel = RLGC::Math::KPHToVel(maxRewardedBallKPH);
				float speedGainTerm = 0.0f;
				if (deltaSpeed >= RLGC::Math::KPHToVel(minQualifyingDeltaKPH)) {
					speedGainTerm = RS_MIN(1.0f, deltaSpeed / maxVel);
				}

				// Upward component term (favor popping the ball up)
				const float upTerm = RS_CLAMP(state.ball.vel.z / 1800.0f, 0.0f, 1.0f);

				// Goalward component term: project ball velocity toward opponent goal
				bool targetOrange = (player.team == Team::BLUE);
				Vec goalBack = targetOrange ? CommonValues::ORANGE_GOAL_BACK : CommonValues::BLUE_GOAL_BACK;
				Vec toGoalDir = (goalBack - state.ball.pos).Normalized();
				float goalward = RS_CLAMP(toGoalDir.Dot(state.ball.vel / CommonValues::BALL_MAX_SPEED), 0.0f, 1.0f);

				// Combine
				const float totalW = weightSpeedGain + weightUpward + weightGoalward;
				float combined = 0.0f;
				if (totalW > 1e-6f) {
					combined = (weightSpeedGain * speedGainTerm + weightUpward * upTerm + weightGoalward * goalward) / totalW;
				}

				rew += combined;
			}

			// Reset latch on any touch, or when back on ground after a flip sequence without touching
			if (player.ballTouchedStep) {
				st.preflipDribbleLatched = false;
			}
			if (player.isOnGround && st.timeSinceFlip > 0.0f && !player.isFlipping) {
				st.preflipDribbleLatched = false;
			}

			return rew;
		}

	private:
		struct PerPlayerState {
			float timeSinceFlip = 9999.0f;
			bool preflipDribble = false;
			bool preflipDribbleLatched = false;
			float prevBallSpeed = 0.0f;
		};
		std::unordered_map<uint32_t, PerPlayerState> _states;
	};

	// Mostly based on the classic Necto rewards
	// Total reward output for speeding the ball up to MAX_REWARDED_BALL_SPEED is 1.0
	// The bot can do this slowly (putting) or quickly (shooting)
	// TouchAccelReward, StrongTouchReward removed (duplicate of CommonRewards.h)

	// ======= MERGED FROM OLD FRAMEWORK: SimpleJumpTouchReward =======
	class SimpleJumpTouchReward : public Reward {
	public:
		float minHeight;        // Minimum ball height (uu) to count as a valid jump touch
		float rewardAmount;     // Base reward given on a valid touch
		float doubleJumpBonus;  // Extra reward if player has double jumped
		bool requireDoubleJump; // If true, only reward if player has double jumped before the touch

		SimpleJumpTouchReward(
			float minHeight = 100.0f,
			float rewardAmount = 1.0f,
			float doubleJumpBonus = 0.5f,
			bool requireDoubleJump = false)
			: minHeight(minHeight), rewardAmount(rewardAmount), doubleJumpBonus(doubleJumpBonus), requireDoubleJump(requireDoubleJump) {}

		virtual float GetReward(const Player& player, const GameState& state, bool isFinal) override {
			if (player.ballTouchedStep && state.ball.pos.z >= minHeight && (!requireDoubleJump || player.hasDoubleJumped)) {
				float reward = rewardAmount;
				if (player.hasDoubleJumped) {
					reward += doubleJumpBonus;
				}
				return reward;
			}
			return 0.0f;
		}
	};

	// AerialTouchHeightReward: rewards touches that occur while the car is airborne,
	// scaled by the height of the ball. No reward on ground touches.
	class AerialTouchHeightReward : public Reward {
	public:
		float minHeight; // touches below this ball height give 0
		float maxHeight; // touches at/above this ball height give full reward (clamped)
		float range;     // precomputed maxHeight - minHeight (>= 1)

		AerialTouchHeightReward(float minHeight_ = 120.0f,
			float maxHeight_ = CommonValues::CEILING_Z)
			: minHeight(minHeight_), maxHeight(maxHeight_), range(maxHeight_ - minHeight_) {
			if (!(range > 0.0f)) range = 1.0f;
		}

		virtual float GetReward(const Player& player, const GameState& state, bool isFinal) override {
			// Only count true aerial touches (ignore ground/tiny-hop tolerance)
			if (player.ballTouchedStep && !IsOnGroundApprox(player)) {
				float h = state.ball.pos.z;
				if (h <= minHeight) return 0.0f;
				float frac = (h - minHeight) / range;
				// Clamp 0..1
				frac = RS_CLAMP(frac, 0.0f, 1.0f);
				return frac; // 0..1 depending on touch height
			}
			return 0.0f;
		}
	};

	class KickoffProximityReward : public Reward {
	public:
		virtual float GetReward(const Player& player, const GameState& state, bool isFinal) override {
			constexpr float KICKOFF_POS_XY_TOLERANCE = 1.0f;
			if (std::abs(state.ball.pos.x) < KICKOFF_POS_XY_TOLERANCE && std::abs(state.ball.pos.y) < KICKOFF_POS_XY_TOLERANCE) {
				Vec playerPos = player.pos;
				Vec ballPos = state.ball.pos;
				float playerDistSq = (playerPos - ballPos).LengthSq();

				float minOpponentDistSq = std::numeric_limits<float>::max();
				bool opponentFound = false;

				for (const auto& otherPlayer : state.players) {
					if (otherPlayer.carId == player.carId)
						continue;
					if (otherPlayer.team != player.team) {
						opponentFound = true;
						Vec opponentPos = otherPlayer.pos;
						float opponentDistSq = (opponentPos - ballPos).LengthSq();
						minOpponentDistSq = std::min(minOpponentDistSq, opponentDistSq);
					}
				}

				if (!opponentFound || playerDistSq < minOpponentDistSq) {
					if (opponentFound && playerDistSq == minOpponentDistSq)
						return 0.0f;
					return 1.0f;
				} else {
					return -1.0f;
				}
			}
			return 0.0f;
		}
	};

		// Enhanced Air Dribble Reward - rewards players for maintaining close proximity to the ball while airborne
		class EnhancedAirDribbleRewardv99 : public Reward {
		public:
			float min_height;         // Minimum height (uu) to consider aerial dribble shaping
			float max_distance;       // Max car-to-ball distance for non-zero reward
			float proximity_weight;   // Weight for proximity component
			float velocity_weight;    // Weight for velocity-toward-ball component
			float height_weight;      // Weight for height component
			float touch_bonus;        // Additive bonus on aerial touches
			float ball_goal_vel_weight;   // Weight for ball velocity toward opponent goal center
			float goal_alignment_weight;  // Weight for alignment of car->ball with ball->goal
			float flip_penalty;       // Penalty for flipping during aerial touches
 
		private:
			float reward_scaling_denominator; // max_distance - BALL_RADIUS
			float height_scaling_denominator; // CEILING_Z - min_height
 
		public:
			EnhancedAirDribbleRewardv99(
				float minHeight = 300.0f,
				float maxDistance = 400.0f,
				float proximityWeight = 1.8f,
				float velocityWeight = 1.5f,
				float heightWeight = 1.0f,
				float touchBonus = 1.0f,
				float ballGoalVelWeight = 2.0f,
				float goalAlignmentWeight = 2.5f)
				: min_height(minHeight),
				  max_distance(maxDistance),
				  proximity_weight(proximityWeight),
				  velocity_weight(velocityWeight),
				  height_weight(heightWeight),
				  touch_bonus(touchBonus),
				  ball_goal_vel_weight(ballGoalVelWeight),
				  goal_alignment_weight(goalAlignmentWeight) {
				float totalWeight = proximity_weight + velocity_weight + height_weight + ball_goal_vel_weight + goal_alignment_weight;
				if (!(totalWeight > 1e-6f)) totalWeight = 1.0f; // ensure sane division
 
				if (!(max_distance > CommonValues::BALL_RADIUS)) {
					max_distance = CommonValues::BALL_RADIUS + 1.0f;
				}
 
				reward_scaling_denominator = std::max(1e-6f, max_distance - CommonValues::BALL_RADIUS);
				height_scaling_denominator = std::max(1e-6f, CommonValues::CEILING_Z - min_height);
			}
 
			virtual void Reset(const GameState& initialState) override {}
 
			virtual float GetReward(const Player& player, const GameState& state, bool isFinal) override {
				const Vec& car_pos = player.pos;
				const Vec& ball_pos = state.ball.pos;
 
				// Airborne gating for both car and ball
				bool is_airborne = (!player.isOnGround &&
					car_pos.z >= this->min_height &&
					ball_pos.z >= this->min_height);
				if (!is_airborne) return 0.0f;
 
				Vec dist_vec = ball_pos - car_pos;
				float dist = dist_vec.Length();
 
				bool is_close = dist <= this->max_distance;
				if (!is_close) return 0.0f;
 
				// 1) Proximity Reward
				float proximity_factor = std::max(0.0f, 1.0f - (dist - CommonValues::BALL_RADIUS) / this->reward_scaling_denominator);
				float prox_reward = proximity_factor * this->proximity_weight;
 
				// 2) Velocity Towards Ball Reward
				float vel_reward = 0.0f;
				if (dist > 1e-6f) {
					Vec dir_to_ball = dist_vec / dist;
					const Vec& player_vel = player.vel;
					float speed_towards_ball = player_vel.Dot(dir_to_ball);
					if (speed_towards_ball > 1e-6f) {
						vel_reward = (speed_towards_ball / CommonValues::CAR_MAX_SPEED) * this->velocity_weight;
					}
				}
 
				// 3) Height Reward (use car height; alternate is ball height)
				float current_height = car_pos.z;
				float height_factor = std::max(0.0f, (current_height - this->min_height) / this->height_scaling_denominator);
				float height_reward = height_factor * this->height_weight;
 
				// 4) Ball velocity toward opponent goal center
				bool targetOrange = (player.team == Team::BLUE);
				Vec goal_center = targetOrange ? CommonValues::ORANGE_GOAL_CENTER : CommonValues::BLUE_GOAL_CENTER;
				Vec dir_ball_to_goal = (goal_center - ball_pos).Normalized();
				float ball_goal_vel = RS_CLAMP(dir_ball_to_goal.Dot(state.ball.vel / CommonValues::BALL_MAX_SPEED), 0.0f, 1.0f);
				float ball_goal_reward = ball_goal_vel * this->ball_goal_vel_weight;
 
				// 5) Car–ball–goal alignment (encourage touches that push ball toward goal)
				float align_reward = 0.0f;
				if (dist > 1e-6f) {
					Vec dir_car_to_ball = dist_vec / dist;
					float align = RS_CLAMP(dir_car_to_ball.Dot(dir_ball_to_goal), 0.0f, 1.0f);
					align_reward = align * this->goal_alignment_weight;
				}
 
				// Combine and normalize base reward
				float base_reward = prox_reward + vel_reward + height_reward + ball_goal_reward + align_reward;
				float total_weight = this->proximity_weight + this->velocity_weight + this->height_weight + this->ball_goal_vel_weight + this->goal_alignment_weight;
				float normalized_base_reward = (total_weight > 1e-6f) ? (base_reward / total_weight) : base_reward;
 
				// Touch bonus when touching in the air
				float final_reward = normalized_base_reward;
				if (player.ballTouchedStep && !player.isOnGround) {
					final_reward += this->touch_bonus;
				}
 
				return final_reward;
			}
		};

	// WallAirdribbleSetupReward
	class WallAirdribbleSetupReward : public Reward {
	public:
		WallAirdribbleSetupReward(
			float wall_x_thresh = 3800.0f,
			float min_pop_dz = 220.0f,
			float min_ball_speed = 350.0f,
			float min_ball_z_after_pop = 800.0f,
			float max_ball_z_after_pop = 2000.0f,
			int active_ticks = 30,
			float near_xy_thresh = 350.0f,
			float jump_bonus = 0.0f,
			float touch_bonus = 2.0f,
			float wall_jump_bonus = 0.50f,
			int wall_jump_window_ticks = 100,
			float orientation_bonus_scale = 0.02f,
			bool orientation_require_over_car = true,
			float end_height_z = 150.0f,
			float end_height_hysteresis = 5.0f,
			float pop_window_seconds = 0.7f,
			int game_tick_rate = 120,
			int tick_skip = 8,
			float start_boost_min = 0.65f,
			float start_boost_max = 1.0f,
			float proximity_max_dist = 480.0f,
			float proximity_min_ball_z = 150.0f,
			bool require_goal_forward_for_touch = true,
			bool require_goal_forward_for_proximity = true,
			float goal_forward_min_proj = 0.12f
		)
			: wall_x_thresh(wall_x_thresh), min_pop_dz(min_pop_dz), min_ball_speed(min_ball_speed),
			  min_ball_z_after_pop(min_ball_z_after_pop), max_ball_z_after_pop(max_ball_z_after_pop),
			  active_ticks(active_ticks), near_xy_thresh(near_xy_thresh), jump_bonus(jump_bonus),
			  touch_bonus(touch_bonus), wall_jump_bonus(wall_jump_bonus), wall_jump_window_ticks(wall_jump_window_ticks),
			  orientation_bonus_scale(orientation_bonus_scale), orientation_require_over_car(orientation_require_over_car),
			  end_height_z(end_height_z), end_height_hysteresis(end_height_hysteresis),
			  _steps_per_second(std::max(1, game_tick_rate / std::max(1, tick_skip))),
			  _pop_window_steps(std::max(1, int(std::round(pop_window_seconds * _steps_per_second)))),
			  start_boost_min(start_boost_min), start_boost_max(start_boost_max),
			  proximity_max_dist(proximity_max_dist), proximity_min_ball_z(proximity_min_ball_z),
			  require_goal_forward_for_touch(require_goal_forward_for_touch),
			  require_goal_forward_for_proximity(require_goal_forward_for_proximity),
			  goal_forward_min_proj(goal_forward_min_proj),
			  tick(0), _last_ball_vel_z(0.0f), _first_pid(-1),
			  _goal_half_width(892.755f), _goal_height(CommonValues::GOAL_HEIGHT != 0.0f ? CommonValues::GOAL_HEIGHT : 642.775f)
		{
			_player_state.clear();
		}

		void Reset(const GameState& initialState) override {
			tick = 0;
			_player_state.clear();
			_last_ball_vel_z = initialState.ball.vel.z;
			_first_pid = -1;
			// clear per-episode wall-jump block timers
			std::lock_guard<std::mutex> lk(g_wall_jump_block_mutex);
			g_wall_jump_block_until.clear();
		}

		float GetReward(const Player& player, const GameState& state, bool /*isFinal*/) override {
			if (_first_pid == -1 && state.players.size() > 0) _first_pid = state.players[0].carId;
			if (player.carId == _first_pid) tick += 1;

			const Vec& carpos = player.pos;
			const Vec& ballpos = state.ball.pos;
			const Vec& ballvel = state.ball.vel;

			// per-player state map<string,float>
			auto& st = _player_state[player.carId];

			// Initialize defaults when missing
			if (st.empty()) {
				st["active"] = 0.0f;
				st["wall_jump_expires"] = -1.0f;
				st["wall_jump_given"] = 0.0f;
				st["pop_wall_side"] = 0.0f;
				// no reliable has_flip in Player -> track but default 0
				st["last_has_flip"] = 0.0f;
				st["popped_tick"] = -1.0f;
				st["ended_once"] = 0.0f;
				st["pop_deadline"] = -1.0f;
				st["pop_vz0"] = 0.0f;
				st["air_touches"] = 0.0f; // integer-like counter stored as float
			}

			float reward = 0.0f;

			// deactivate if ball fell below end height
			if (st["active"] > 0.5f && ballpos.z < (end_height_z - end_height_hysteresis)) {
				st["active"] = 0.0f;
				st["ended_once"] = 1.0f;
				// no shared-state modifications here
			}

			bool near_wall_car = std::fabs(carpos.x) > wall_x_thresh;
			bool near_wall_ball = std::fabs(ballpos.x) > (wall_x_thresh - 150.0f);
			bool near_wall = near_wall_car && near_wall_ball;

			// use ballTouchedStep and boost fraction (player.boost is 0..100)
			float boost_frac = player.boost / 100.0f;
			if (st["active"] < 0.5f && player.ballTouchedStep && near_wall &&
				(start_boost_min <= boost_frac && boost_frac <= start_boost_max)) {
				st["pop_deadline"] = tick + _pop_window_steps;
				st["pop_vz0"] = ballvel.z;
			}

			// check pop window and set active state
			if (st["active"] < 0.5f && st["pop_deadline"] >= 0 && tick <= (int)st["pop_deadline"] && near_wall) {
				float dvz_since_touch = ballvel.z - st["pop_vz0"];
				float ball_speed = ballvel.Length();
				if (dvz_since_touch > min_pop_dz
					&& min_ball_z_after_pop <= ballpos.z && ballpos.z <= max_ball_z_after_pop
					&& ball_speed >= min_ball_speed) {
					st["active"] = 1.0f;
					st["wall_jump_expires"] = tick + wall_jump_window_ticks;
					st["wall_jump_given"] = 0.0f;
					st["pop_wall_side"] = (carpos.x > 0.0f) ? 1.0f : ((carpos.x < 0.0f) ? -1.0f : 0.0f);
					st["popped_tick"] = tick;
					st["ended_once"] = 0.0f;
					st["pop_deadline"] = -1.0f;
					st["air_touches"] = 0.0f;
				}
			}

			if (st["active"] < 0.5f && st["pop_deadline"] >= 0 && tick > (int)st["pop_deadline"]) {
				st["pop_deadline"] = -1.0f;
			}

			// wall jump bonus: gave wall jump within window
			if (st["active"] > 0.5f
				&& tick <= (int)st["wall_jump_expires"]
				&& !IsOnGroundApprox(player)
				&& st["wall_jump_given"] < 0.5f
				&& st["pop_wall_side"] != 0.0f
				&& ((carpos.x > 0.0f) == (st["pop_wall_side"] > 0.0f))
				&& near_wall_car) {
				reward += wall_jump_bonus;
				st["wall_jump_given"] = 1.0f;

				// start blocking window for FlipResetReward after the wall-jump (1.65s)
				int block_steps = std::max(1, int(std::round(1.65f * _steps_per_second)));
				std::lock_guard<std::mutex> lk(g_wall_jump_block_mutex);
				g_wall_jump_block_until[player.carId] = tick + block_steps;
			}

			// active shaping & bonuses
			if (st["active"] > 0.5f) {
				Vec goal_dir = _goal_dir_vec(player, state);
				Vec ball_dir = ballvel / (ballvel.Length() + 1e-6f);
				float forward = goal_dir.Dot(ball_dir);
				bool over_car = ballpos.z > carpos.z + 50.0f;
				float horiz_dx = ballpos.x - carpos.x;
				float horiz_dy = ballpos.y - carpos.y;
				float horiz_dist = std::sqrt(horiz_dx * horiz_dx + horiz_dy * horiz_dy);
				bool near_xy = horiz_dist <= near_xy_thresh;

				Vec forward_flat = _get_forward_flat(player);
				Vec to_ball_flat = ballpos - carpos;
				to_ball_flat.z = 0.0f;
				float tb_n = std::sqrt(to_ball_flat.x * to_ball_flat.x + to_ball_flat.y * to_ball_flat.y);
				if (tb_n > 1e-6f) {
					to_ball_flat = to_ball_flat / tb_n;
				}
				float orientation_val = std::max(0.0f, forward_flat.Dot(to_ball_flat));
				bool orientation_ok = (over_car || !orientation_require_over_car);
				if (orientation_val > 0.0f && orientation_ok) {
					reward += orientation_bonus_scale * orientation_val;
				}

				if (over_car && near_xy && forward > 0.1f) {
					float h = (ballpos.z - min_ball_z_after_pop) / std::max(1.0f, (max_ball_z_after_pop - min_ball_z_after_pop));
					h = std::min(1.0f, std::max(0.0f, h));
					float core = 0.6f * h + 0.4f * std::min(1.0f, std::max(0.0f, forward));
					reward += core;

					// detect jump press from prevAction (Action implements operator[])
					float prev_jump = 0.0f;
					try { prev_jump = player.prevAction[5]; }
					catch (...) { prev_jump = 0.0f; }
					if (prev_jump > 0.5f && jump_bonus > 0.0f) reward += jump_bonus;

					// award air-touch exponential bonus
					if (player.ballTouchedStep && !IsOnGroundApprox(player)) {
						// minimal change: only reward touches when ball is higher than 550
						if (ballpos.z > 550.0f && (!require_goal_forward_for_touch || _ball_moving_toward_goal(player, state))) {
							int t_idx = static_cast<int>(st["air_touches"]);
							int mult = (t_idx >= 3) ? 8 : (1 << t_idx); // 1,2,4,8
							reward += touch_bonus * static_cast<float>(mult);
							st["air_touches"] = static_cast<float>(t_idx + 1);
						}
					}
				}

				// proximity shaping while in air and ball high enough
				if (!IsOnGroundApprox(player) && ballpos.z >= proximity_min_ball_z) {
					bool goal_ok = (!require_goal_forward_for_proximity) || _ball_moving_toward_goal(player, state);
					if (goal_ok) {
						float dx = ballpos.x - carpos.x;
						float dy = ballpos.y - carpos.y;
						float dz = ballpos.z - carpos.z;
						float dist3 = std::sqrt(dx * dx + dy * dy + dz * dz);
						if (dist3 <= proximity_max_dist) {
							float prox_raw = 1.0f - (dist3 / std::max(1e-6f, proximity_max_dist));
							reward += std::min(1.0f, std::max(0.0f, prox_raw));
						}
					}
				}
			}

			// keep last_has_flip placeholder (Player doesn't expose has_flip)
			st["last_has_flip"] = 0.0f;
			return reward;
		}

	private:
		// config
		float wall_x_thresh;
		float min_pop_dz;
		float min_ball_speed;
		float min_ball_z_after_pop;
		float max_ball_z_after_pop;
		int active_ticks;
		float near_xy_thresh;
		float jump_bonus;
		float touch_bonus;
		float wall_jump_bonus;
		int wall_jump_window_ticks;
		float orientation_bonus_scale;
		bool orientation_require_over_car;
		float end_height_z;
		float end_height_hysteresis;

		int _steps_per_second;
		int _pop_window_steps;

		float start_boost_min;
		float start_boost_max;

		float proximity_max_dist;
		float proximity_min_ball_z;

		bool require_goal_forward_for_touch;
		bool require_goal_forward_for_proximity;
		float goal_forward_min_proj;

		int tick;
		std::unordered_map<int, std::unordered_map<std::string, float>> _player_state;
		float _last_ball_vel_z;
		int _first_pid;

		float _goal_half_width;
		float _goal_height;

		// helpers
		Vec _goal_dir_vec(const Player& player, const GameState& state) const {
			Vec target = (player.team == Team::BLUE) ? CommonValues::ORANGE_GOAL_BACK : CommonValues::BLUE_GOAL_BACK;
			Vec d = target - state.ball.pos;
			float n = d.Length();
			if (n < 1e-6f) return Vec{ 0.0f, 0.0f, 0.0f };
			return d / n;
		}

		Vec _get_forward_flat(const Player& player) const {
			Vec f = player.rotMat.forward;
			f.z = 0.0f;
			float n = f.Length();
			if (n > 1e-6f) return f / n;
			return f;
		}

		bool _ball_moving_toward_goal(const Player& player, const GameState& state) const {
			const Vec& ballpos = state.ball.pos;
			Vec vel = state.ball.vel;
			float speed = vel.Length();
			if (speed < 1e-3f) return false;

			Vec goal_center = (player.team == Team::BLUE) ? CommonValues::ORANGE_GOAL_CENTER : CommonValues::BLUE_GOAL_CENTER;
			Vec goal_back = (player.team == Team::BLUE) ? CommonValues::ORANGE_GOAL_BACK : CommonValues::BLUE_GOAL_BACK;
			float goal_y = goal_back.y;

			Vec dir_to_center = goal_center - ballpos;
			float n_dir = dir_to_center.Length();
			if (n_dir < 1e-6f) return false;
			dir_to_center = dir_to_center / n_dir;

			Vec vel_n = vel / speed;
			float proj = vel_n.Dot(dir_to_center);
			if (proj < goal_forward_min_proj) return false;

			float dy = goal_y - ballpos.y;
			if (std::fabs(dy) < 1e-6f) return false;
			// require vel.y to point toward the goal
			if (vel.y * dy <= 0.0f) return false;

			if (std::fabs(vel.y) < 1e-3f) return false;
			float t = dy / vel.y;
			if (t <= 0.0f) return false;

			float x_pred = ballpos.x + vel.x * t;
			float z_pred = ballpos.z + vel.z * t;
			if (std::fabs(x_pred) <= _goal_half_width && z_pred >= 0.0f && z_pred <= _goal_height) return true;
			return false;
		}
	};

	// BouncyAirdribbleReward: rewards a bouncy air dribble toward the opponent goal
	class BouncyAirdribbleReward : public Reward {
	public:
		// Bounciness thresholds
		float minSeparation;       // min gap between car and ball to consider it "bouncy"
		float minRelSpeed;        // |player.vel - ball.vel| threshold

		// Air-dribble gating
		float minBallZ;           // ball z OR rising condition
		float minCarZ;            // minimum car height (uu)
		float minBallRiseZ;       // minimum upward ball vel (uu/s) to satisfy rising condition
		float minCarRiseZ;        // minimum upward car vel (uu/s)
		float maxDistance;        // max allowed car-ball distance
		float minZDirToBall;      // normalize(ball - car).z must exceed this
		float minPlayerGoalSpeed; // player goal-ward speed threshold (uu/s)
		float minBallGoalSpeed;   // ball goal-ward speed threshold (uu/s)
		float minAlign2D;         // alignment of 2D (player->ball) with (ball->goal)
		float maxOverCrossbar;    // predicted z at goal line must be <= crossbar + this
		float shutoffTimeToGoal;  // disable reward if time to goal line <= this (s)

		// Shaping weights
		float wProximity;
		float wAlign2D;
		float wSpeed;
		float wZDir;

		BouncyAirdribbleReward(
			float minSeparation_ = 100.0f,
			float minRelSpeed_ = 300.0f,
			float minBallZ_ = 300.0f,
			float minCarZ_ = 300.0f,
			float minBallRiseZ_ = 60.0f,
			float minCarRiseZ_ = 40.0f,
			float maxDistance_ = 350.0f,
			float minZDirToBall_ = 0.05f,
			float minPlayerGoalSpeed_ = 300.0f,
			float minBallGoalSpeed_ = 400.0f,
			float minAlign2D_ = 0.60f,
			float maxOverCrossbar_ = 300.0f,
			float shutoffTimeToGoal_ = 2.0f,
			float wProximity_ = 1.0f,
			float wAlign2D_ = 1.0f,
			float wSpeed_ = 1.0f,
			float wZDir_ = 0.5f
		)
			: minSeparation(minSeparation_), minRelSpeed(minRelSpeed_),
			  minBallZ(minBallZ_), minCarZ(minCarZ_), minBallRiseZ(minBallRiseZ_), minCarRiseZ(minCarRiseZ_),
			  maxDistance(maxDistance_), minZDirToBall(minZDirToBall_), minPlayerGoalSpeed(minPlayerGoalSpeed_),
			  minBallGoalSpeed(minBallGoalSpeed_), minAlign2D(minAlign2D_), maxOverCrossbar(maxOverCrossbar_),
			  shutoffTimeToGoal(shutoffTimeToGoal_),
			  wProximity(wProximity_), wAlign2D(wAlign2D_), wSpeed(wSpeed_), wZDir(wZDir_) {
			proxDenom = std::max(1e-6f, maxDistance - CommonValues::BALL_RADIUS);
		}

		virtual float GetReward(const Player& player, const GameState& state, bool isFinal) override {
			const Vec& cpos = player.pos;
			const Vec& cvel = player.vel;
			const Vec& bpos = state.ball.pos;
			const Vec& bvel = state.ball.vel;

			// Near-goal shutoff using predicted time to cross goal line
			float tGoal; float zAt; bool movingTowardGoal;
			if (!PredictGoalLineCross(player, bpos, bvel, tGoal, zAt, movingTowardGoal)) return 0.0f; // not toward goal at all
			if (tGoal <= shutoffTimeToGoal) return 0.0f; // near-goal exploration window

			// Crossbar constraint
			const float crossbar = CommonValues::GOAL_HEIGHT;
			if (zAt > crossbar + maxOverCrossbar) return 0.0f;

			// Bouncy condition: ball above car and either separated or relatively fast
			Vec d = bpos - cpos;
			float dist = d.Length();
			if (bpos.z <= cpos.z + 30.0f) return 0.0f;
			Vec rel = cvel - bvel;
			bool bouncy = (dist >= minSeparation) || (rel.Length() >= minRelSpeed);
			if (!bouncy) return 0.0f;

			// Air-dribble gating
			bool heightOK = ((bpos.z >= minBallZ) && (cpos.z >= minCarZ)) || (bvel.z >= minBallRiseZ && cvel.z >= minCarRiseZ);
			if (!heightOK) return 0.0f;
			if (dist > maxDistance || dist < 1e-6f) return 0.0f;

			// Minimum boost based on distance to opponent goal (2D)
			const bool isBlue = (player.team == Team::BLUE);
			const Vec goalBack = isBlue ? CommonValues::ORANGE_GOAL_BACK : CommonValues::BLUE_GOAL_BACK;
			Vec g2d = { goalBack.x - cpos.x, goalBack.y - cpos.y, 0.0f };
			float dist2Goal2D = std::sqrt(g2d.x * g2d.x + g2d.y * g2d.y);
			float minBoostNeeded = 15.0f * std::pow(dist2Goal2D / 5000.0f, 1.75f);
			if (player.boost < minBoostNeeded) return 0.0f;

			// Direction to ball (3D and 2D) and Z component
			Vec dirToBall = d / dist;
			if (dirToBall.z < minZDirToBall) return 0.0f;
			Vec dirToBall2D = { dirToBall.x, dirToBall.y, 0.0f }; NormalizeSafe2D(dirToBall2D);

			// Direction to goal from ball
			Vec toGoalDir3D = (goalBack - bpos).Normalized();
			Vec toGoalDir2D = { toGoalDir3D.x, toGoalDir3D.y, 0.0f }; NormalizeSafe2D(toGoalDir2D);

			// Player and ball moving toward goal with minimum speeds
			float playerGoalSpeed = cvel.Dot(toGoalDir3D);
			float ballGoalSpeed = bvel.Dot(toGoalDir3D);
			if (playerGoalSpeed < minPlayerGoalSpeed || ballGoalSpeed < minBallGoalSpeed) return 0.0f;

			// Alignment 2D check
			float align2D = dirToBall2D.x * toGoalDir2D.x + dirToBall2D.y * toGoalDir2D.y;
			if (align2D < minAlign2D) return 0.0f;

			// Shaping terms
			float proximity = std::max(0.0f, 1.0f - (dist - CommonValues::BALL_RADIUS) / proxDenom);
			float alignTerm = std::max(0.0f, align2D); // already -1..1
			float speedTerm = 0.0f;
			{
				float pNorm = std::min(1.0f, std::max(0.0f, playerGoalSpeed / CommonValues::CAR_MAX_SPEED));
				float bNorm = std::min(1.0f, std::max(0.0f, ballGoalSpeed / CommonValues::BALL_MAX_SPEED));
				speedTerm = 0.6f * bNorm + 0.4f * pNorm;
			}
			float zDirTerm = std::min(1.0f, std::max(0.0f, (dirToBall.z - minZDirToBall) / std::max(1e-3f, 1.0f - minZDirToBall)));

			float totalW = std::max(1e-6f, wProximity + wAlign2D + wSpeed + wZDir);
			float base = (wProximity * proximity + wAlign2D * alignTerm + wSpeed * speedTerm + wZDir * zDirTerm) / totalW;
			return std::max(0.0f, base);
		}

	private:
		float proxDenom;

		static inline void NormalizeSafe2D(Vec& v) {
			float n = std::sqrt(v.x * v.x + v.y * v.y);
			if (n > 1e-6f) { v.x /= n; v.y /= n; }
		}

		// Predict time to cross opponent goal line and z at crossing
		static inline bool PredictGoalLineCross(const Player& player, const Vec& bPos, const Vec& bVel, float& tOut, float& zAtOut, bool& movingTowardOut) {
			const bool targetOrange = (player.team == Team::BLUE);
			const float goalY = targetOrange ? CommonValues::BACK_WALL_Y : -CommonValues::BACK_WALL_Y;
			const float vy = bVel.y;
			movingTowardOut = !((targetOrange && vy <= 1e-6f) || (!targetOrange && vy >= -1e-6f));
			if (!movingTowardOut) return false;
			tOut = (goalY - bPos.y) / RS_MAX(vy, (float)1e-6);
			if (!(tOut > 0.0f) || !std::isfinite(tOut)) return false;
			zAtOut = bPos.z + bVel.z * tOut;
			return true;
		}
	};

	/* Tuning Guide
	Zealan says to start the bot less strict since the bot is shit first, then make it stricter as it improve its mechs
	To make the reward MORE STRICT (harder to earn, for refining the skill):
	  - INCREASE: bouncyDist, bouncyVel, minHeight, 
	              minSpeed, minPlayerToBallZ, ShutoffTime
	   - DECREASE: maxDistFromBall

	 To make the reward LESS STRICT (easier to earn, for initial learning):
	   - DECREASE: bouncyDist, bouncyVel, minHeight,
	               minSpeed, minPlayerToBallZ, ShutoffTime
	   - INCREASE: maxDistFromBall*/

	class BouncyAirDribbleReward : public Reward {
	public:
		float bouncyDistThreshold;
		float bouncyVelThreshold;
		float minHeightForDribble;
		float minSpeedTowardGoal;
		float maxDistFromBall;
		float minPlayerToBallUpwards;
		float nearGoalShutoffTime;

		BouncyAirDribbleReward(
			float bouncyDist = 15.f, float bouncyVel = 50.f, float minHeight = 250.f,
			float minSpeed = 700.f, float maxDist = 360.f, float minPlayerToBallZ = 0.1f,
			float shutoffTime = 2.f) :
			bouncyDistThreshold(bouncyDist), bouncyVelThreshold(bouncyVel), minHeightForDribble(minHeight),
			minSpeedTowardGoal(minSpeed), maxDistFromBall(maxDist), minPlayerToBallUpwards(minPlayerToBallZ),
			nearGoalShutoffTime(shutoffTime) {}

		virtual float GetReward(const Player& player, const GameState& state, bool isFinal) override {

			Vec opponentGoal = (player.team == Team::BLUE) ?
				CommonValues::ORANGE_GOAL_CENTER : CommonValues::BLUE_GOAL_CENTER;

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

			if (!heightConditionMet) return 0.f;

			if (distToBall > maxDistFromBall) return 0.f;

			float distToGoal2D = (player.pos - opponentGoal).Length2D();
			float minBoostRequired = 15.f * std::pow(distToGoal2D / 5000.f, 1.75f);
			if (player.boost < minBoostRequired) return 0.f;

			Vec playerToBallDir = (state.ball.pos - player.pos).Normalized();
			if (playerToBallDir.z < minPlayerToBallUpwards) return 0.f;

			Vec ballToGoalDir = (opponentGoal - state.ball.pos).Normalized();
			if (player.vel.Dot(ballToGoalDir) < minSpeedTowardGoal || state.ball.vel.Dot(ballToGoalDir) < minSpeedTowardGoal) {
				return 0.f;
			}

			Vec playerToBallDir2D = Vec(playerToBallDir.x, playerToBallDir.y, 0).Normalized();
			Vec ballToGoalDir2D = Vec(ballToGoalDir.x, ballToGoalDir.y, 0).Normalized();
			if (playerToBallDir2D.Dot(ballToGoalDir2D) < 0.75) return 0.f; //near the goal value

			float yDistToGoal = std::abs(state.ball.pos.y - opponentGoal.y);
			float timeToGoal = yDistToGoal / std::max(1.f, std::abs(state.ball.vel.y));

			if (timeToGoal > 0) {
				float gravityZ = state.lastArena ? state.lastArena->GetMutatorConfig().gravity.z : -650.f;
				float heightAtGoal = state.ball.pos.z + state.ball.vel.z * timeToGoal + 0.5f * gravityZ * timeToGoal * timeToGoal;
				if (heightAtGoal > CommonValues::GOAL_HEIGHT + 100) return 0.f;
			}
			
			if (timeToGoal < nearGoalShutoffTime) {
				return 0.f;
			}

			return 1.f;
		}
	};

	class VelocityBallToGoalMouthReward : public Reward {
	public:
		bool ownGoal;
		float missSoftness; // softness scale for off-target decay (uu)
		bool useHeight;     // include crossbar check (z) in aim factor

		// ... (rest of the code remains the same)
		// missSoftness ~300–800 works well; smaller = harsher off-target penalty
		VelocityBallToGoalMouthReward(bool ownGoal = false, float missSoftness = 600.0f, bool useHeight = true)
			: ownGoal(ownGoal), missSoftness(missSoftness), useHeight(useHeight) {}

		virtual float GetReward(const Player& player, const GameState& state, bool isFinal) override {
			// Determine target goal side for this player
			bool targetOrangeGoal = player.team == Team::BLUE;
			if (ownGoal) targetOrangeGoal = !targetOrangeGoal;

			const Vec& bPos = state.ball.pos;
			const Vec& bVel = state.ball.vel;
			const float speed = bVel.Length();
			if (speed <= 1e-3f) return 0.0f;

			// Keep the base alignment shaping used widely in prior work
			const Vec targetBack = targetOrangeGoal ? CommonValues::ORANGE_GOAL_BACK : CommonValues::BLUE_GOAL_BACK;
			const Vec toGoalDir = (targetBack - bPos).Normalized();
			float alignment = toGoalDir.Dot(bVel / CommonValues::BALL_MAX_SPEED);

			// Compute where the current ball ray would hit the goal line (mouth plane)
			const float goalY = targetOrangeGoal ? CommonValues::BACK_WALL_Y : -CommonValues::BACK_WALL_Y;
			const float vy = bVel.y;
			// If not moving toward that goal line, there is no on-target component
			if ((targetOrangeGoal && vy <= 1e-6f) || (!targetOrangeGoal && vy >= -1e-6f)) {
				return 0.0f; // don’t incentivize around-the-net/away motion
			}

			const float t = (goalY - bPos.y) / RS_MAX(vy, (float)1e-6);
			if (t <= 0.0f || !std::isfinite(t)) {
				return 0.0f;
			}

			const float x_at = bPos.x + bVel.x * t;
			const float z_at = bPos.z + bVel.z * t;

			const float halfWidth = CommonValues::GOAL_WIDTH_FROM_CENTER;
			const float top = CommonValues::GOAL_HEIGHT;

			// Miss distances outside the mouth rectangle at the goal line
			float missX = fabsf(x_at) - halfWidth; if (missX < 0.0f) missX = 0.0f;
			float missZ = 0.0f;
			if (useHeight) {
				if (z_at < 0.0f) missZ = -z_at; else if (z_at > top) missZ = z_at - top; else missZ = 0.0f;
			}
			const float miss = sqrtf(missX * missX + missZ * missZ);

			// Aim factor: 1 when inside the mouth; decays smoothly when off-target
			const float k = RS_MAX(missSoftness, 1.0f);
			const float aimFactor = 1.0f / (1.0f + (miss / k));

			// Final reward: base alignment scaled by how on-target the shot is
			return alignment * aimFactor;
		}
	};

	class AirDribbleReward : public Reward {
	public:
		AirDribbleReward(
			float minHeight = 250.0f,
			float maxDistance = 400.0f,
			float minApproachSpeed = 150.0f,
			float minGoalProj = 0.10f,
			bool requireAir = true,
			bool requireGoalProgress = false,
			float proximityWeight = 2.0f,
			float velocityWeight = 1.0f,
			float heightWeight = 0.5f,
			float touchBonus = 2.0f
		) : minHeight(minHeight), maxDistance(maxDistance), minApproachSpeed(minApproachSpeed),
			minGoalProj(minGoalProj), requireAir(requireAir), requireGoalProgress(requireGoalProgress),
			proximityWeight(proximityWeight), velocityWeight(velocityWeight), heightWeight(heightWeight),
			touchBonus(touchBonus)
		{
			// Precompute denominators with safety clamps
			proxDenom = std::max(1e-6f, maxDistance - CommonValues::BALL_RADIUS);
			heightDenom = std::max(1e-6f, CommonValues::CEILING_Z - minHeight);
		}

	virtual float GetReward(const Player& player, const GameState& state, bool isFinal) override {
    const Vec& carPos = player.pos;
    const Vec& ballPos = state.ball.pos;
    const Vec& ballVel = state.ball.vel;

    // Height gates
    if (ballPos.z < minHeight || carPos.z < minHeight) return 0.0f;
    if (requireAir && IsOnGroundApprox(player)) return 0.0f;

    // Proximity to ball
    Vec toBall = ballPos - carPos;
    float dist = toBall.Length();
    if (dist > maxDistance || dist < 1e-6f) return 0.0f;
    Vec dirToBall = toBall / dist;

    // Actively speeding toward the ball
    float approach = player.vel.Dot(dirToBall); // positive when moving toward ball
    if (approach <= minApproachSpeed) return 0.0f;

    // Ball making progress toward opponent goal
    if (requireGoalProgress && !BallMovingTowardOppGoal(player, ballPos, ballVel, minGoalProj)) return 0.0f;

    // Shape reward components
    const float invCarMax = 1.0f / RS_MAX(CommonValues::CAR_MAX_SPEED, 1.0f);
    float approachTerm = std::min(1.0f, std::max(0.0f, approach * invCarMax));
    float proximityFactor = std::max(0.0f, 1.0f - (dist - CommonValues::BALL_RADIUS) / proxDenom);
    float heightFactor = std::max(0.0f, (carPos.z - minHeight) / heightDenom);

    // Combine weighted and normalize by total weights
    float baseScore = proximityWeight * proximityFactor + velocityWeight * approachTerm + heightWeight * heightFactor;
    float totalW = std::max(1e-6f, proximityWeight + velocityWeight + heightWeight);
    float reward = baseScore / totalW;

    // Optional touch bonus when all conditions are met
    if (touchBonus > 0.0f && player.ballTouchedStep) {
        reward += touchBonus;
    }

    return std::max(0.0f, reward);
}

private:
float minHeight;
float maxDistance;
float minApproachSpeed;
float minGoalProj;
bool requireAir;
bool requireGoalProgress;
float proximityWeight;
float velocityWeight;
float heightWeight;
float touchBonus;
float proxDenom;
float heightDenom;

static inline bool BallMovingTowardOppGoal(const Player& player, const Vec& ballPos, const Vec& ballVel, float minProj) {
    float speed = ballVel.Length();
    if (speed < 1e-3f) return false;

    // Determine opponent goal direction
    const bool isBlue = (player.team == Team::BLUE);
    const Vec goalCenter = isBlue ? CommonValues::ORANGE_GOAL_CENTER : CommonValues::BLUE_GOAL_CENTER;
    const Vec goalBack = isBlue ? CommonValues::ORANGE_GOAL_BACK : CommonValues::BLUE_GOAL_BACK;

    Vec dirToGoal = (goalCenter - ballPos).Normalized();
    Vec vN = ballVel / speed;
    float proj = vN.Dot(dirToGoal);
    if (proj < minProj) return false;

    // Ensure y-velocity points toward that back wall (avoids away-from-goal movement)
    float dy = goalBack.y - ballPos.y;
    if (std::fabs(dy) < 1e-6f) return false;
    if (ballVel.y * dy <= 0.0f) return false;
    return true;
}

static inline float GoalForwardAmount(const Player& player, const Vec& ballPos, const Vec& ballVel) {
    float speed = ballVel.Length();
    if (speed < 1e-3f) return 0.0f;
    const bool isBlue = (player.team == Team::BLUE);
    const Vec goalBack = isBlue ? CommonValues::ORANGE_GOAL_BACK : CommonValues::BLUE_GOAL_BACK;
    const Vec dirToGoal = (goalBack - ballPos).Normalized();
    return std::max(0.0f, (ballVel / speed).Dot(dirToGoal));
}
};

class AggressiveDemoBumpReward : public Reward {
public:
    virtual float GetReward(const Player& player, const GameState& state, bool isFinal) override {
        float reward = 0.0f;

        // Strategic positioning scale: neutral 1.0 at center, up to 3.0 near either goal,
        // gated by ball field position to reduce camping.
        const bool isBlue = player.team == Team::BLUE;
        const float backY = RLGC::CommonValues::BACK_WALL_Y;
        const float playerSignedYToOpp = isBlue ? player.pos.y : -player.pos.y;
        const float ballSignedYToOpp = isBlue ? state.ball.pos.y : -state.ball.pos.y;
        auto clamp01 = [](float v) { return std::max(0.0f, std::min(1.0f, v)); };

        float closenessOpp = clamp01(playerSignedYToOpp / backY);  // 0 at center -> 1 at opponent back wall
        float closenessOwn = clamp01(-playerSignedYToOpp / backY); // 0 at center -> 1 at own back wall

        // Gate scales by ball location so we only scale up where the play is happening
        float offenseWeight = clamp01(ballSignedYToOpp / backY);
        float defenseWeight = clamp01(-ballSignedYToOpp / backY);

        float baseScale = 1.0f + 2.0f * std::max(offenseWeight * closenessOpp, defenseWeight * closenessOwn);
        // baseScale in [1,3], ~1 at center, up to 3 near goals when ball is also there

        // Engagement shaping: focus on actively closing distance to an opponent
        float bestEngage = 0.0f;
        float nearestDist = 1e9f;
        const float engageRadius = 2500.0f; // ~half-field in lengthwise terms
        const float invCarMax = 1.0f / RLGC::CommonValues::CAR_MAX_SPEED;

        for (const auto& otherPlayer : state.players) {
            if (&player == &otherPlayer || otherPlayer.team == player.team) continue;
            Vec toOpp = otherPlayer.pos - player.pos;
            float dist = toOpp.Length();
            if (dist < 1e-3f) continue;
            Vec dir = toOpp / dist;

            float approach = player.vel.Dot(dir); // + if moving toward opponent
            float proximity = clamp01((engageRadius - dist) / engageRadius);

            float engageTerm = 0.0f;
            if (approach > 0.0f) {
                float approachNorm = std::min(1.0f, approach * invCarMax);
                engageTerm = approachNorm * proximity; // fast and close => high
            }

            bestEngage = std::max(bestEngage, engageTerm);
            nearestDist = std::min(nearestDist, dist);
        }

        // Event rewards scaled by strategic positioning
        if (player.eventState.bump) {
            reward += baseScale * 0.5f; // bump bonus
        }
        if (player.eventState.demo) {
            reward += baseScale * 5.0f; // demo bonus
        }

        // Proactive engagement reward (prevents waiting/camping)
        reward += baseScale * (2.0f * bestEngage);

        // Anti-camping penalty: sitting near own goal without engaging
        float speed = player.vel.Length();
        float antiCampFactor = defenseWeight * closenessOwn; // high when camping in own end while ball is there
        if (antiCampFactor > 0.4f && bestEngage < 0.05f && speed < 400.0f) {
            reward -= (antiCampFactor - 0.4f) * 0.5f; // small deterrent
        }
        return reward;
    }

};

        // AirDribble: rewards controlled aerial dribbles toward the opponent goal, prioritizing on-target shots.
        // Conditions (approx):
        // - Player and ball are airborne enough
        // - Ball is close to the car nose (small relative distance and relative velocity)
        // - Ball velocity points toward opponent back net plane
        // - Predicted intercept with goal plane within posts and under crossbar gets a big multiplier
        // - Penalize/zero when along sidewalls or too high above the crossbar at intercept
        class AirDribble : public Reward {
        public:
            // geometry/time thresholds
            float minCarZ = 180.0f;            // player must be off ground
            float minBallZ = 250.0f;           // ball height indicates aerial
            float maxHorizOffset = 160.0f;     // max horizontal separation car->ball for control
            float minVertOffset = 40.0f;       // ball must be somewhat above car
            float maxVertOffset = 260.0f;      // not sitting too high above car
            float maxRelSpeed = 900.0f;        // relative speed car<->ball for control
            float horizonTime = 1.25f;         // seconds ahead to check goal plane intercept
            float sideWallMargin = 650.0f;     // avoid dribbling along sidewalls

            // shaping weights
            float controlWeight = 0.45f;       // proximity and relative speed control
            float orientWeight = 0.20f;        // nose alignment toward goal
            float ballGoalVelWeight = 0.35f;   // ball vel toward goal
            float onTargetBonus = 2.5f;        // multiplier when projected on target

            float GetReward(const Player& player, const GameState& state, bool) override {
                const Vec& pc = player.pos;
                const Vec& bc = state.ball.pos;
                const Vec& pv = player.vel;
                const Vec& bv = state.ball.vel;

                if (pc.z < minCarZ || bc.z < minBallZ)
                    return 0.0f;

                // sidewall avoidance
                if (std::fabs(bc.x) > (CommonValues::SIDE_WALL_X - sideWallMargin))
                    return 0.0f;

                // control conditions
                const float dx = bc.x - pc.x;
                const float dy = bc.y - pc.y;
                const float dz = bc.z - pc.z;
                const float horiz = std::sqrt(dx*dx + dy*dy);
                if (horiz > maxHorizOffset) return 0.0f;
                if (dz < minVertOffset || dz > maxVertOffset) return 0.0f;

                Vec rel = bv - pv;
                if (rel.Length() > maxRelSpeed) return 0.0f;

                // goal direction and ball velocity toward that goal
                const bool isBlue = (player.team == Team::BLUE);
                const Vec goalBack = isBlue ? CommonValues::ORANGE_GOAL_BACK : CommonValues::BLUE_GOAL_BACK;
                Vec dirToGoal = (goalBack - bc).Normalized();

                float ballGoalVel = 0.0f;
                {
                    float speed = bv.Length();
                    if (speed > 1e-3f) {
                        ballGoalVel = std::max(0.0f, (bv / speed).Dot(dirToGoal));
                    }
                }
                if (ballGoalVel <= 0.0f) return 0.0f; // must be moving generally toward goal

                // player nose alignment toward goal
                float noseAlign = std::max(0.0f, player.rotMat.forward.Dot(dirToGoal));

                // control shaping: closeness and relative speed smallness
                float closeness = std::exp(-0.01f * std::max(0.0f, horiz - 20.0f));
                float relSpeedGood = 1.0f - std::min(1.0f, rel.Length() / (maxRelSpeed + 1e-3f));
                float control = 0.6f * closeness + 0.4f * relSpeedGood;

                // on-target projection to goal plane (uses gravity approximation on z)
                float onTarget = 0.0f;
                float vy = bv.y;
                float dyGoal = goalBack.y - bc.y;
                if ((isBlue && vy > 1e-3f) || (!isBlue && vy < -1e-3f)) {
                    float t = dyGoal / vy; // time to reach goal plane
                    if (t > 0.02f && t <= horizonTime) {
                        float xHit = bc.x + state.ball.vel.x * t;
                        float zHit = bc.z + state.ball.vel.z * t + 0.5f * CommonValues::GRAVITY_Z * t * t;
                        bool withinPosts = std::fabs(xHit) <= CommonValues::GOAL_WIDTH_FROM_CENTER;
                        bool underBar = zHit >= 0.0f && zHit <= CommonValues::GOAL_HEIGHT;
                        if (withinPosts && underBar)
                            onTarget = 1.0f;
                        else if (withinPosts && zHit > CommonValues::GOAL_HEIGHT) {
                            // if clearly above bar at intercept, zero reward
                            return 0.0f;
                        }
                    }
                }

                // aggregate shaping
                float shaped = controlWeight * control + orientWeight * noseAlign + ballGoalVelWeight * ballGoalVel;
                if (shaped <= 0.0f)
                    return 0.0f;

                float bonus = (onTarget > 0.5f) ? onTargetBonus : 1.0f;
                return shaped * bonus;
            }
        };


    // Rewards purely for getting a flip reset, neutral to style/orientation.
    class GeneralFlipResetReward : public Reward {
    public:
        float rewardVal;
        GeneralFlipResetReward(float val = 10.0f) : rewardVal(val) {}
        void Reset(const GameState& s) override {}
        float GetReward(const Player& p, const GameState& s, bool final) override {
            if (!p.prev) return 0.0f;
            // Detect flip reset acquisition in air
            // Using GotFlipReset() helper if available, which checks (hasFlip && !prev.hasFlip && !isOnGround)
            if (p.GotFlipReset()) return rewardVal;
            return 0.0f;
        }
    };

}
