#pragma once
#include <RLGymCPP/Rewards/Reward.h>
#include <RLGymCPP/Math.h>
#include <cmath>
#include <unordered_map>
#include <limits>

namespace RLGC {


    // ===== ENHANCED AIR DRIBBLE REWARD PRO VERSION =====
    class EnhancedAirDribbleReward : public Reward {
    public:
        float min_height;
        float max_distance;
        float proximity_weight;
        float velocity_weight;
        float height_weight;
        float touch_bonus;

    private:
        float reward_scaling_denominator;
        float height_scaling_denominator;

    public:
        EnhancedAirDribbleReward(
            float minHeight = 326.0f,
            float maxDistance = 250.0f,
            float proximityWeight = 1.8f,
            float velocityWeight = 1.0f,
            float heightWeight = 0.5f,
            float touchBonus = 2.0f) :
            min_height(minHeight),
            max_distance(maxDistance),
            proximity_weight(proximityWeight),
            velocity_weight(velocityWeight),
            height_weight(heightWeight),
            touch_bonus(touchBonus)
        {
            float totalWeight = proximity_weight + velocity_weight + height_weight;
            reward_scaling_denominator = RS_MAX(1e-6f, max_distance - CommonValues::BALL_RADIUS);
            height_scaling_denominator = RS_MAX(1e-6f, CommonValues::CEILING_Z - min_height);
        }

        virtual void Reset(const GameState& initialState) override {}

        virtual float GetReward(const Player& player, const GameState& state, bool isFinal) override {
            const Vec& car_pos = player.pos;
            const Vec& ball_pos = state.ball.pos;

            bool is_airborne = (!player.isOnGround &&
                car_pos.z >= this->min_height &&
                ball_pos.z >= this->min_height);

            if (!is_airborne) return 0.0f;

            Vec dist_vec = ball_pos - car_pos;
            float dist = dist_vec.Length();

            bool is_close = dist <= this->max_distance;
            if (!is_close) return 0.0f;

            // 1. Proximity Reward
            float proximity_factor = RS_MAX(0.0f, 1.0f - (dist - CommonValues::BALL_RADIUS) / this->reward_scaling_denominator);
            float prox_reward = proximity_factor * this->proximity_weight;

            // 2. Velocity Towards Ball Reward
            float vel_reward = 0.0f;
            if (dist > 1e-6f) {
                Vec dir_to_ball = dist_vec / dist;
                const Vec& player_vel = player.vel;
                float speed_towards_ball = player_vel.Dot(dir_to_ball);

                if (speed_towards_ball > 0) {
                    vel_reward = (speed_towards_ball / CommonValues::CAR_MAX_SPEED) * this->velocity_weight;
                }
            }

            // 3. Height Reward
            float current_height = car_pos.z;
            float height_factor = RS_MAX(0.0f, (current_height - this->min_height) / this->height_scaling_denominator);
            float height_reward = height_factor * this->height_weight;

            // --- Combine and Normalize Base Reward ---
            float base_reward = prox_reward + vel_reward + height_reward;
            float total_weight = this->proximity_weight + this->velocity_weight + this->height_weight;
            float normalized_base_reward = 0.0f;

            if (total_weight > 1e-6f) {
                normalized_base_reward = base_reward / total_weight;
            }

            // --- Add Touch Bonus ---
            float final_reward = normalized_base_reward;
            if (player.ballTouchedStep) {
                final_reward += this->touch_bonus;
            }

            return final_reward;
        }
    };

    // ===== AIR DRIBBLE WITH ROLL REWARD - NIVEAU PRO ULTRA AVANCÉ =====
#pragma warning(push)
#pragma warning(disable: 4251)
    class FrenchAirDribbleWithRollReward : public Reward {
    private:
        struct PlayerState {
            bool lastBallTouch = false;
            int airTouchStreak = 0;
            float cumulativeRollDirection = 0.0f;
            float lastRollInput = 0.0f;
            float directionChanges = 0.0f;
        };
        std::map<uint32_t, PlayerState> playerStates;

    public:
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
        bool debug;

        FrenchAirDribbleWithRollReward(
            float minHeight = 200.0f,
            float maxDistance = 1200.0f,
            float proximityWeight = 1.0f,
            float velocityWeight = 1.0f,
            float heightWeight = 1.0f,
            float touchBonus = 0.3f,
            float boostThreshold = 0.2f,
            float heightTarget = 1000.0f,
            float heightTargetWeight = 1.0f,
            float upwardBonus = 0.1f,
            float airRollWeight = 1.0f,
            float speedWeight = 0.5f,
            float boostEfficiencyWeight = 0.7f,
            bool debug = false
        ) : minHeight(minHeight), maxDistance(maxDistance), proximityWeight(proximityWeight),
            velocityWeight(velocityWeight), heightWeight(heightWeight), touchBonus(touchBonus),
            boostThreshold(boostThreshold), heightTarget(heightTarget), heightTargetWeight(heightTargetWeight),
            upwardBonus(upwardBonus), airRollWeight(airRollWeight), speedWeight(speedWeight),
            boostEfficiencyWeight(boostEfficiencyWeight), debug(debug) {
        }

        virtual void Reset(const GameState& initialState) override {
            playerStates.clear();
            for (const auto& player : initialState.players) {
                PlayerState& state = playerStates[player.carId];
                state.lastBallTouch = player.ballTouchedStep;
                state.airTouchStreak = 0;
                state.cumulativeRollDirection = 0.0f;
                state.lastRollInput = 0.0f;
                state.directionChanges = 0.0f;
            }
        }

        virtual float GetReward(const Player& player, const GameState& state, bool isFinal) override {
            uint32_t carId = player.carId;

            if (playerStates.find(carId) == playerStates.end()) {
                PlayerState& newState = playerStates[carId];
                newState.lastBallTouch = player.ballTouchedStep;
                newState.airTouchStreak = 0;
                newState.cumulativeRollDirection = 0.0f;
                newState.lastRollInput = 0.0f;
                newState.directionChanges = 0.0f;
            }

            PlayerState& st = playerStates[carId];

            Vec carPos = player.pos;
            Vec ballPos = state.ball.pos;

            // VÉRIFICATION DIRECTION VERS LE BUT ADVERSE
            bool isMovingTowardsGoal = false;
            bool isBallMovingTowardsGoal = false;

            if (player.team == Team::BLUE) {
                // Équipe bleue doit aller vers le but orange (Y positif)
                isMovingTowardsGoal = (ballPos.y > carPos.y + 50.0f);
                // Vérifier que la balle va vers le but orange (vélocité Y positive)
                isBallMovingTowardsGoal = (state.ball.vel.y > 100.0f);
            }
            else {
                // Équipe orange doit aller vers le but bleu (Y négatif)
                isMovingTowardsGoal = (ballPos.y < carPos.y - 50.0f);
                // Vérifier que la balle va vers le but bleu (vélocité Y négative)
                isBallMovingTowardsGoal = (state.ball.vel.y < -100.0f);
            }

            // Si la balle ne va pas vers le but adverse, pas de récompense
            if (!isMovingTowardsGoal || !isBallMovingTowardsGoal) {
                st.airTouchStreak = 0;
                st.lastBallTouch = player.ballTouchedStep;
                st.cumulativeRollDirection = 0.0f;
                st.directionChanges = 0.0f;
                return 0.0f;
            }

            bool isAirborne = (!player.isOnGround &&
                carPos.z >= minHeight &&
                ballPos.z >= minHeight);

            if (!isAirborne) {
                st.airTouchStreak = 0;
                st.lastBallTouch = player.ballTouchedStep;
                st.cumulativeRollDirection = 0.0f;
                st.lastRollInput = 0.0f;
                st.directionChanges = 0.0f;
                return 0.0f;
            }

            Vec distVec = ballPos - carPos;
            float dist = distVec.Length();
            bool isClose = dist <= maxDistance;

            if (!isClose) {
                st.airTouchStreak = 0;
                st.lastBallTouch = player.ballTouchedStep;
                st.cumulativeRollDirection = 0.0f;
                st.lastRollInput = 0.0f;
                st.directionChanges = 0.0f;
                return 0.0f;
            }

            // 1. Proximity Reward
            float proximityDenominator = RS_MAX(1e-6f, maxDistance - CommonValues::BALL_RADIUS);
            float proximityFactor = RS_MAX(0.0f, 1.0f - (dist - CommonValues::BALL_RADIUS) / proximityDenominator);
            float proxReward = proximityFactor * proximityWeight;

            // 2. Velocity Towards Ball Reward
            float velReward = 0.0f;
            if (dist > 1e-6f) {
                Vec dirToBall = distVec / dist;
                Vec playerVel = player.vel;
                float speedTowardsBall = playerVel.Dot(dirToBall);
                if (speedTowardsBall > 0) {
                    velReward = (speedTowardsBall / CommonValues::CAR_MAX_SPEED) * velocityWeight;
                }
            }

            // 3. Height Reward
            float currentHeight = carPos.z;
            float heightDenominator = RS_MAX(1e-6f, CommonValues::CEILING_Z - minHeight);
            float heightFactor = RS_MAX(0.0f, (currentHeight - minHeight) / heightDenominator);
            float heightReward = heightFactor * heightWeight;

            // 4. Height Target Reward
            float heightDiff = abs(currentHeight - heightTarget);
            float heightTargetDenominator = RS_MAX(1e-6f, heightTarget);
            float heightScale = RS_MAX(0.0f, 1.0f - heightDiff / heightTargetDenominator);
            float heightTargetReward = heightScale * heightTargetWeight;

            // 5. Controlled Speed Reward
            float controlledSpeedReward = 0.0f;
            float playerSpeed = player.vel.Length();
            float optimalMinSpeed = 800.0f;
            float optimalMaxSpeed = 1400.0f;

            if (optimalMinSpeed <= playerSpeed && playerSpeed <= optimalMaxSpeed) {
                controlledSpeedReward = speedWeight;
            }
            else if (playerSpeed > optimalMaxSpeed) {
                float excessSpeedPenalty = RS_MIN(0.5f, (playerSpeed - optimalMaxSpeed) / 1000.0f);
                controlledSpeedReward = -excessSpeedPenalty * speedWeight;
            }

            // 6. Boost Conservation Reward
            float boostConservationReward = 0.0f;
            if (player.boost > 0.3f) {
                float boostFactor = powf(player.boost, 0.7f);
                boostConservationReward = boostFactor * boostEfficiencyWeight;
            }

            // 7. Air Roll Reward (sans détection wiggling pour air roll continu)
            float airRollReward = 0.0f;
            Vec carRight = Vec(player.rotMat.right.x, player.rotMat.right.y, player.rotMat.right.z);

            // Simuler l'input de roll via la rotation de la voiture
            float currentRollInput = carRight.z;  // Approximation de l'air roll

            // Récompenser SEULEMENT air roll DROIT
            if (currentRollInput > 0.1f) {  // SEULEMENT air roll droit (positif)
                float baseRollReward = airRollWeight;

                // Bonus boost
                if (player.boost > boostThreshold) {
                    baseRollReward *= 1.4f;
                }

                airRollReward = baseRollReward;
            }

            st.lastRollInput = currentRollInput;

            // Combine base rewards
            float baseReward = proxReward + velReward + heightReward + heightTargetReward +
                controlledSpeedReward + boostConservationReward + airRollReward;
            float totalWeight = proximityWeight + velocityWeight + heightWeight + heightTargetWeight +
                speedWeight + boostEfficiencyWeight + 1.0f;  // +1 pour air roll

            float normalizedBaseReward = (totalWeight > 1e-6f) ? (baseReward / totalWeight) : 0.0f;
            float finalReward = normalizedBaseReward;

            // 8. Upward Bonus
            if (player.vel.z > 0) {
                finalReward += upwardBonus;
            }

            // 9. Position Bonus (below ball + facing ball)
            Vec carToBall = ballPos - carPos;
            Vec carUp = player.rotMat.up;
            bool belowBall = carPos.z < ballPos.z - 30.0f;
            bool facingBall = false;

            if (carToBall.Length() > 0) {
                Vec carToBallNorm = carToBall.Normalized();
                float facingBallDot = carUp.Dot(carToBallNorm);
                facingBall = facingBallDot > 0.7f;
            }

            if (belowBall && facingBall) {
                finalReward += 0.2f;
            }

            // 10. Touch Bonus avec streak
            if (!st.lastBallTouch && player.ballTouchedStep) {
                st.airTouchStreak += 1;

                float touchMultiplier = 1.0f;

                // Multiplicateur vitesse
                if (800.0f <= playerSpeed && playerSpeed <= 1400.0f) {
                    touchMultiplier *= 2.5f;
                }
                else if (playerSpeed > 1400.0f) {
                    float speedPenalty = RS_MIN(0.7f, (playerSpeed - 1400.0f) / 1000.0f);
                    touchMultiplier *= RS_MAX(0.1f, 1.5f - speedPenalty);
                }
                else {
                    touchMultiplier *= 1.2f;
                }

                // Multiplicateur boost
                if (player.boost > boostThreshold) {
                    touchMultiplier *= 1.8f;
                }

                // Multiplicateur proximité
                if (dist < CommonValues::BALL_RADIUS + 150.0f) {
                    touchMultiplier *= 1.4f;
                }

                float touchReward = touchBonus * st.airTouchStreak * touchMultiplier;
                finalReward += touchReward;


            }
            else if (player.isOnGround) {
                st.airTouchStreak = 0;
                st.cumulativeRollDirection = 0.0f;
                st.lastRollInput = 0.0f;
                st.directionChanges = 0.0f;
            }

            st.lastBallTouch = player.ballTouchedStep;

            // 11. Height Scaling
            float heightScaling = ballPos.z / 1000.0f;
            finalReward *= RS_MAX(0.1f, RS_MIN(2.0f, heightScaling));

            return finalReward;
        }
    };
#pragma warning(pop)


    // ===== BOOST SEEKING REWARD - Encourage le bot à chercher activement les boost =====
    class FrenchBoostSeekingReward : public Reward {
    private:
        struct PlayerState {
            float lastBoost = 100.0f;
            Vec lastPosition = Vec(0, 0, 0);
            bool wasMovingTowardsBoost = false;
            float boostSeekingTime = 0.0f;
            Vec targetBoostPos = Vec(0, 0, 0);
        };
        std::map<uint32_t, PlayerState> playerStates;

        // Positions des boost pads (petits)
        std::vector<Vec> smallBoostPositions;
        // Positions des gros boosts
        std::vector<Vec> bigBoostPositions;

    public:
        float boostPickupReward;
        float boostSeekingReward;
        float boostProximityReward;
        float boostDirectionReward;
        float lowBoostThreshold;
        float maxSeekingDistance;
        bool debug;

        FrenchBoostSeekingReward(
            float boostPickupReward = 5.0f,
            float boostSeekingReward = 0.8f,
            float boostProximityReward = 1.2f,
            float boostDirectionReward = 1.5f,
            float lowBoostThreshold = 0.3f,
            float maxSeekingDistance = 800.0f,
            bool debug = false
        ) : boostPickupReward(boostPickupReward), boostSeekingReward(boostSeekingReward),
            boostProximityReward(boostProximityReward), boostDirectionReward(boostDirectionReward),
            lowBoostThreshold(lowBoostThreshold), maxSeekingDistance(maxSeekingDistance), debug(debug) {
            initializeBoostPositions();
        }

        virtual void Reset(const GameState& initialState) override {
            playerStates.clear();
            for (const auto& player : initialState.players) {
                PlayerState& state = playerStates[player.carId];
                state.lastBoost = player.boost;
                state.lastPosition = player.pos;
                state.wasMovingTowardsBoost = false;
                state.boostSeekingTime = 0.0f;
                state.targetBoostPos = Vec(0, 0, 0);
            }
        }

        virtual float GetReward(const Player& player, const GameState& state, bool isFinal) override {
            uint32_t carId = player.carId;

            if (playerStates.find(carId) == playerStates.end()) {
                PlayerState& newState = playerStates[carId];
                newState.lastBoost = player.boost;
                newState.lastPosition = player.pos;
                newState.wasMovingTowardsBoost = false;
                newState.boostSeekingTime = 0.0f;
                newState.targetBoostPos = Vec(0, 0, 0);
            }

            PlayerState& st = playerStates[carId];
            float reward = 0.0f;
            Vec playerPos = player.pos;
            float currentBoost = player.boost;

            // 1. RÉCOMPENSE POUR AVOIR PRIS UN BOOST
            if (currentBoost > st.lastBoost) {
                float boostGained = currentBoost - st.lastBoost;
                reward += boostPickupReward * (boostGained / 100.0f); // Normalisé par 100

                if (debug) {
                    printf("[BoostSeeking] car_id=%d BOOST PICKUP! +%.1f boost, Reward: %.2f\n",
                        carId, boostGained, reward);
                }
            }

            // 2. RÉCOMPENSE POUR CHERCHER LES BOOST QUAND ON EN A BESOIN
            if (currentBoost < lowBoostThreshold) {
                // Trouver le boost le plus proche
                Vec closestBoost = findClosestBoost(playerPos);
                float distanceToBoost = (closestBoost - playerPos).Length();

                if (distanceToBoost <= maxSeekingDistance) {
                    // Récompense de proximité (plus on est proche, mieux c'est)
                    float proximityFactor = 1.0f - (distanceToBoost / maxSeekingDistance);
                    reward += boostProximityReward * proximityFactor;

                    // Récompense pour se diriger vers le boost
                    if (isMovingTowardsBoost(player, closestBoost)) {
                        reward += boostDirectionReward;
                        st.wasMovingTowardsBoost = true;
                        st.boostSeekingTime += 1.0f / 120.0f; // 120 ticks par seconde

                        // Bonus de persévérance
                        if (st.boostSeekingTime > 1.0f) {
                            reward += boostSeekingReward * 0.5f;
                        }

                        if (debug && st.boostSeekingTime > 2.0f) {
                            printf("[BoostSeeking] car_id=%d PERSÉVÉRANCE BOOST! Temps: %.1fs, Reward: %.2f\n",
                                carId, st.boostSeekingTime, reward);
                        }
                    }
                    else {
                        st.wasMovingTowardsBoost = false;
                        st.boostSeekingTime = 0.0f;
                    }

                    st.targetBoostPos = closestBoost;
                }
            }
            else {
                // Reset si on a assez de boost
                st.boostSeekingTime = 0.0f;
                st.wasMovingTowardsBoost = false;
            }

            // Mise à jour de l'état
            st.lastBoost = currentBoost;
            st.lastPosition = playerPos;

            return reward;
        }

    private:
        void initializeBoostPositions() {
            // Gros boosts (100 boost)
            bigBoostPositions = {
                Vec(-3072, -4096, 73),   // Blue corner
                Vec(3072, -4096, 73),    // Blue corner
                Vec(-3072, 4096, 73),    // Orange corner
                Vec(3072, 4096, 73),     // Orange corner
                Vec(0, -4240, 73),       // Blue back
                Vec(0, 4240, 73)         // Orange back
            };

            // Petits boost pads (12 boost) - grille 6x6
            smallBoostPositions.clear();
            for (int x = -2560; x <= 2560; x += 1024) {
                for (int y = -3840; y <= 3840; y += 1280) {
                    // Éviter les positions des gros boosts
                    bool isBigBoost = false;
                    for (const auto& bigBoost : bigBoostPositions) {
                        if ((Vec(x, y, 73) - bigBoost).Length() < 200) {
                            isBigBoost = true;
                            break;
                        }
                    }
                    if (!isBigBoost) {
                        smallBoostPositions.push_back(Vec(x, y, 73));
                    }
                }
            }
        }

        Vec findClosestBoost(const Vec& playerPos) const {
            Vec closestBoost = Vec(0, 0, 0);
            float minDistance = std::numeric_limits<float>::infinity();

            // Vérifier d'abord les gros boosts
            for (const auto& boostPos : bigBoostPositions) {
                float distance = (boostPos - playerPos).Length();
                if (distance < minDistance) {
                    minDistance = distance;
                    closestBoost = boostPos;
                }
            }

            // Vérifier les petits boost pads
            for (const auto& boostPos : smallBoostPositions) {
                float distance = (boostPos - playerPos).Length();
                if (distance < minDistance) {
                    minDistance = distance;
                    closestBoost = boostPos;
                }
            }

            return closestBoost;
        }

        bool isMovingTowardsBoost(const Player& player, const Vec& boostPos) const {
            Vec playerPos = player.pos;
            Vec directionToBoost = (boostPos - playerPos).Normalized();
            Vec playerVelocity = player.vel;

            if (playerVelocity.Length() < 100) {
                return false; // Pas assez de vitesse
            }

            Vec playerVelocityNorm = playerVelocity.Normalized();
            float alignment = playerVelocityNorm.Dot(directionToBoost);

            return alignment > 0.3f; // Alignement suffisant vers le boost
        }
    };


    // SpeedFlipMovementReward - Détecte vraiment le diagonal cancel et speedflip
    class SpeedFlipMovementReward : public Reward {
    private:
        struct PlayerState {
            bool jumpUsed = false;
            float jumpTime = -1.0f;
            bool diagonalInputDetected = false;
            bool flipCancelDetected = false;
            bool speedflipCompleted = false;
            Vec lastPos = Vec(0, 0, 0);
            float lastTime = 0.0f;
            bool hadJump = true;
            Vec velocityBeforeFlip = Vec(0, 0, 0);
            Vec velocityAfterFlip = Vec(0, 0, 0);
        };
        std::map<uint32_t, PlayerState> playerStates;

    public:
        float maxJumpDelay;
        float minSpeedThreshold;
        float diagonalInputThreshold;
        float cancelThreshold;
        float maxCancelDelay;
        float rewardValue;
        bool debug;
        int frameCount;

        SpeedFlipMovementReward(
            float maxJumpDelay = 0.5f,
            float minSpeedThreshold = 1000.0f,
            float diagonalInputThreshold = 0.3f,
            float cancelThreshold = -0.5f,
            float maxCancelDelay = 0.3f,
            float rewardValue = 8.0f,
            bool debug = false
        ) : maxJumpDelay(maxJumpDelay), minSpeedThreshold(minSpeedThreshold),
            diagonalInputThreshold(diagonalInputThreshold), cancelThreshold(cancelThreshold),
            maxCancelDelay(maxCancelDelay), rewardValue(rewardValue), debug(debug), frameCount(0) {
        }

        virtual void Reset(const GameState& initialState) override {
            playerStates.clear();
            frameCount = 0;
            for (const auto& player : initialState.players) {
                PlayerState& state = playerStates[player.carId];
                state.jumpUsed = false;
                state.jumpTime = -1.0f;
                state.diagonalInputDetected = false;
                state.flipCancelDetected = false;
                state.speedflipCompleted = false;
                state.lastPos = player.pos;
                state.lastTime = 0.0f;
                state.hadJump = true;
                state.velocityBeforeFlip = Vec(0, 0, 0);
                state.velocityAfterFlip = Vec(0, 0, 0);
            }
        }

        virtual float GetReward(const Player& player, const GameState& state, bool isFinal) override {
            uint32_t carId = player.carId;

            if (playerStates.find(carId) == playerStates.end()) {
                PlayerState& newState = playerStates[carId];
                newState.jumpUsed = false;
                newState.jumpTime = -1.0f;
                newState.diagonalInputDetected = false;
                newState.flipCancelDetected = false;
                newState.speedflipCompleted = false;
                newState.lastPos = player.pos;
                newState.lastTime = 0.0f;
                newState.hadJump = true;
                newState.velocityBeforeFlip = Vec(0, 0, 0);
                newState.velocityAfterFlip = Vec(0, 0, 0);
            }

            PlayerState& st = playerStates[carId];
            float reward = 0.0f;
            float currentTime = frameCount * 0.016f;
            float dt = (st.lastTime > 0) ? (currentTime - st.lastTime) : 0.016f;
            frameCount++;

            // Détecter le saut (sans condition kickoff)
            if (!st.jumpUsed && player.HasFlipOrJump() && st.hadJump) {
                st.jumpUsed = true;
                st.jumpTime = currentTime;
                st.velocityBeforeFlip = player.vel; // Sauvegarder la vélocité avant le flip

            }

            // Détecter l'input diagonal après le saut (via la vélocité)
            if (st.jumpUsed && !st.diagonalInputDetected) {
                // Vérifier si la voiture se déplace en diagonal (composantes X et Y significatives)
                float velX = fabsf(player.vel.x);
                float velY = fabsf(player.vel.y);
                float velZ = fabsf(player.vel.z);

                // Le speedflip a des composantes X et Y élevées avec peu de Z
                if (velX > diagonalInputThreshold * 1000.0f && velY > diagonalInputThreshold * 1000.0f && velZ < 500.0f) {
                    st.diagonalInputDetected = true;

                }
            }

            // Détecter le flip cancel (via la vélocité et la rotation)
            if (st.jumpUsed && st.diagonalInputDetected && !st.flipCancelDetected) {
                float cancelDelay = (st.jumpTime >= 0) ? (currentTime - st.jumpTime) : 0;
                if (cancelDelay <= maxCancelDelay) {
                    // Le flip cancel se caractérise par une vélocité Z qui reste basse
                    // et une rotation de la voiture qui s'arrête
                    if (fabsf(player.vel.z) < 300.0f && player.rotMat.up.z > 0.7f) {
                        st.flipCancelDetected = true;
                        st.velocityAfterFlip = player.vel; // Sauvegarder la vélocité après le flip

                    }
                }
            }

            // Vérifier le speedflip complet (accélération et direction)
            if (st.jumpUsed && st.diagonalInputDetected && st.flipCancelDetected && !st.speedflipCompleted) {
                // Calculer l'accélération due au speedflip
                float speedBefore = st.velocityBeforeFlip.Length();
                float speedAfter = st.velocityAfterFlip.Length();
                float acceleration = (speedAfter - speedBefore) / dt;

                // Vérifier que la vitesse est suffisante et qu'il y a eu accélération
                if (speedAfter >= minSpeedThreshold && acceleration > 500.0f) {
                    st.speedflipCompleted = true;
                    reward = rewardValue;

                }
            }

            st.lastPos = player.pos;
            st.lastTime = currentTime;
            st.hadJump = player.HasFlipOrJump();

            return reward;
        }

    private:
        float CalculateSpeed(const Vec& currentPos, const Vec& lastPos, float dt) {
            if (dt <= 0) return 0.0f;
            Vec displacement = currentPos - lastPos;
            return displacement.Length() / dt;
        }
    };

    // KickoffSpeedflipReward - Détecte vraiment le speedflip au kickoff
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

        KickoffSpeedflipReward(
            float kickoffDetectionTime = 3.0f,
            float maxJumpDelay = 0.5f,
            float minSpeedThreshold = 1200.0f,
            float diagonalInputThreshold = 0.3f,
            float cancelThreshold = -0.5f,
            float maxCancelDelay = 0.3f,
            float rewardValue = 5.0f,
            bool debug = false
        ) : kickoffDetectionTime(kickoffDetectionTime), maxJumpDelay(maxJumpDelay),
            minSpeedThreshold(minSpeedThreshold), diagonalInputThreshold(diagonalInputThreshold),
            cancelThreshold(cancelThreshold), maxCancelDelay(maxCancelDelay),
            rewardValue(rewardValue), debug(debug) {
        }

        virtual void Reset(const GameState& initialState) override {
            playerStates.clear();
            kickoffStartTime = -1.0f;
            frameCount = 0;
            for (const auto& player : initialState.players) {
                PlayerState& state = playerStates[player.carId];
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

        virtual float GetReward(const Player& player, const GameState& state, bool isFinal) override {
            uint32_t carId = player.carId;

            if (playerStates.find(carId) == playerStates.end()) {
                PlayerState& newState = playerStates[carId];
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

            PlayerState& st = playerStates[carId];
            float reward = 0.0f;
            float currentTime = frameCount * 0.016f;
            float dt = (st.lastTime > 0) ? (currentTime - st.lastTime) : 0.016f;
            frameCount++;

            // Détecter le kickoff
            if (!st.kickoffDetected && IsKickoffPosition(player.pos)) {
                st.kickoffDetected = true;
                kickoffStartTime = currentTime;

            }

            // Détecter le saut (avec condition kickoff)
            if (st.kickoffDetected && !st.jumpUsed && player.HasFlipOrJump()) {
                float jumpDelay = (kickoffStartTime >= 0) ? (currentTime - kickoffStartTime) : 0;
                if (jumpDelay <= maxJumpDelay) {
                    st.jumpUsed = true;
                    st.jumpTime = currentTime;
                    st.velocityBeforeFlip = player.vel; // Sauvegarder la vélocité avant le flip

                }
            }

            // Détecter l'input diagonal après le saut (via la vélocité)
            if (st.jumpUsed && !st.diagonalInputDetected) {
                // Vérifier si la voiture se déplace en diagonal (composantes X et Y significatives)
                float velX = fabsf(player.vel.x);
                float velY = fabsf(player.vel.y);
                float velZ = fabsf(player.vel.z);

                // Le speedflip au kickoff a des composantes X et Y élevées avec peu de Z
                if (velX > diagonalInputThreshold * 1000.0f && velY > diagonalInputThreshold * 1000.0f && velZ < 500.0f) {
                    st.diagonalInputDetected = true;

                }
            }

            // Détecter le flip cancel (via la vélocité et la rotation)
            if (st.jumpUsed && st.diagonalInputDetected && !st.flipCancelDetected) {
                float cancelDelay = (st.jumpTime >= 0) ? (currentTime - st.jumpTime) : 0;
                if (cancelDelay <= maxCancelDelay) {
                    // Le flip cancel se caractérise par une vélocité Z qui reste basse
                    // et une rotation de la voiture qui s'arrête
                    if (fabsf(player.vel.z) < 300.0f && player.rotMat.up.z > 0.7f) {
                        st.flipCancelDetected = true;
                        st.velocityAfterFlip = player.vel; // Sauvegarder la vélocité après le flip

                    }
                }
            }

            // Vérifier le speedflip complet (accélération et direction)
            if (st.jumpUsed && st.diagonalInputDetected && st.flipCancelDetected && !st.speedflipCompleted) {
                // Calculer l'accélération due au speedflip
                float speedBefore = st.velocityBeforeFlip.Length();
                float speedAfter = st.velocityAfterFlip.Length();
                float acceleration = (speedAfter - speedBefore) / dt;

                // Vérifier que la vitesse est suffisante et qu'il y a eu accélération
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
        bool IsKickoffPosition(const Vec& playerPos) {
            // Positions de kickoff exactes
            Vec kickoffPositions[] = {
                Vec(0, -5120, 0),      // Centre
                Vec(-2048, -5120, 0),  // Gauche
                Vec(2048, -5120, 0),   // Droite
                Vec(-2048, 5120, 0),   // Gauche adverse
                Vec(2048, 5120, 0)     // Droite adverse
            };

            for (const auto& pos : kickoffPositions) {
                if ((playerPos - pos).Length() < 500.0f) {
                    return true;
                }
            }
            return false;
        }

        float CalculateSpeed(const Vec& currentPos, const Vec& lastPos, float dt) {
            if (dt <= 0) return 0.0f;
            Vec displacement = currentPos - lastPos;
            return displacement.Length() / dt;
        }
    };

    class InfiniteWalldashReward : public Reward {
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
        };
        std::map<uint32_t, PlayerState> playerStates;

    public:
        float wallHeightThreshold;
        float maxTimeBetweenDashAndReset;
        float dashReward;
        float resetReward;
        float consecutiveBonus;
        float wallStayPenalty;
        float timingBonus;
        bool debug;

        InfiniteWalldashReward(
            float wallHeightThreshold = 100.0f,
            float maxTimeBetweenDashAndReset = 1.5f,
            float dashReward = 3.0f,
            float resetReward = 8.0f,
            float consecutiveBonus = 5.0f,
            float wallStayPenalty = -0.2f,
            float timingBonus = 2.0f,
            bool debug = false
        ) : wallHeightThreshold(wallHeightThreshold),
            maxTimeBetweenDashAndReset(maxTimeBetweenDashAndReset),
            dashReward(dashReward), resetReward(resetReward),
            consecutiveBonus(consecutiveBonus), wallStayPenalty(wallStayPenalty),
            timingBonus(timingBonus), debug(debug) {
        }

        virtual void Reset(const GameState& initialState) override {
            playerStates.clear();
            for (const auto& player : initialState.players) {
                PlayerState& state = playerStates[player.carId];
                state.wasOnWall = false;
                state.dashUsed = false;
                state.resetObtained = false;
                state.consecutiveWallDashes = 0;
                state.lastDashTime = 0.0f;
                state.lastWallPosition = Vec(0, 0, 0);
                state.hadFlipBefore = player.HasFlipOrJump();
                state.wallStayTime = 0.0f;
            }
        }

        virtual float GetReward(const Player& player, const GameState& state, bool isFinal) override {
            if (!state.prev) return 0.0f;

            uint32_t carId = player.carId;
            if (playerStates.find(carId) == playerStates.end()) {
                PlayerState& newState = playerStates[carId];
                newState.wasOnWall = false;
                newState.dashUsed = false;
                newState.resetObtained = false;
                newState.consecutiveWallDashes = 0;
                newState.lastDashTime = 0.0f;
                newState.lastWallPosition = Vec(0, 0, 0);
                newState.hadFlipBefore = player.HasFlipOrJump();
                newState.wallStayTime = 0.0f;
            }

            PlayerState& st = playerStates[carId];
            float reward = 0.0f;
            float currentTime = 0.0f; // Simuler le temps

            // Vérifier si le joueur est sur un mur
            bool isOnWall = IsOnWall(player.pos);
            bool hasFlip = player.HasFlipOrJump();

            // 1. DÉTECTION DU DASH DEPUIS LE MUR (priorité 1)
            if (DetectDashFromWall(player, st)) {
                reward += dashReward;
                st.dashUsed = true;
                st.lastDashTime = currentTime;

                if (debug) {
                    printf("[WallDash] car_id=%d DASH DEPUIS LE MUR! Reward: %.2f\n", carId, dashReward);
                }
            }

            // 2. DÉTECTION DU RESET SUR LE MUR (priorité 2)
            if (DetectWallReset(player, st)) {
                reward += resetReward;
                st.resetObtained = true;
                st.consecutiveWallDashes++;

                // Bonus pour les wall dashes consécutifs
                if (st.consecutiveWallDashes > 1) {
                    float consecutiveReward = consecutiveBonus * st.consecutiveWallDashes;
                    reward += consecutiveReward;

                    if (debug) {
                        printf("[WallDash] car_id=%d WALL DASH CONSÉCUTIF #%d! Bonus: %.2f\n",
                            carId, st.consecutiveWallDashes, consecutiveReward);
                    }
                }

                // Bonus de timing (plus c'est rapide, mieux c'est)
                float timeSinceDash = currentTime - st.lastDashTime;
                if (timeSinceDash < maxTimeBetweenDashAndReset) {
                    float timingReward = timingBonus * (1.0f - timeSinceDash / maxTimeBetweenDashAndReset);
                    reward += timingReward;

                    if (debug) {
                        printf("[WallDash] car_id=%d TIMING PARFAIT! Bonus: %.2f\n", carId, timingReward);
                    }
                }

                if (debug) {
                    printf("[WallDash] car_id=%d RESET SUR LE MUR! Reward: %.2f\n", carId, resetReward);
                }
            }

            // 3. PÉNALITÉ POUR RESTER TROP LONGTEMPS SUR LE MUR
            if (isOnWall) {
                st.wallStayTime += 1.0f / 120.0f; // 120 ticks par seconde

                if (st.wallStayTime > 1.0f) { // Après 1 seconde
                    reward += wallStayPenalty;

                    if (debug && st.wallStayTime > 2.0f) {
                        printf("[WallDash] car_id=%d RESTE TROP LONGTEMPS SUR LE MUR! Pénalité: %.2f\n",
                            carId, wallStayPenalty);
                    }
                }
            }
            else {
                st.wallStayTime = 0.0f;
            }

            // 4. RÉINITIALISATION SI LE JOUEUR TOUCHE LE SOL
            if (player.isOnGround) {
                st.consecutiveWallDashes = 0;
                st.dashUsed = false;
                st.resetObtained = false;
                st.wallStayTime = 0.0f;
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
        bool IsOnWall(const Vec& playerPos) {
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

        bool DetectDashFromWall(const Player& player, PlayerState& st) {
            // Conditions pour détecter un dash depuis le mur :
            // 1. Était sur un mur
            // 2. Avait un flip
            // 3. N'a plus de flip (l'a utilisé)
            // 4. N'est plus sur le mur (a sauté)

            return (st.wasOnWall &&
                st.hadFlipBefore &&
                !player.HasFlipOrJump() &&
                !IsOnWall(player.pos));
        }

        bool DetectWallReset(const Player& player, PlayerState& st) {
            // Conditions pour détecter un reset sur le mur :
            // 1. Avait utilisé un dash (n'avait plus de flip)
            // 2. Est maintenant sur un mur
            // 3. A récupéré son flip

            return (st.dashUsed &&
                IsOnWall(player.pos) &&
                player.HasFlipOrJump());
        }
    };

    // CeilingShotReward - Apprend le ceiling shot avec reset
    class CeilingShotReward : public Reward {
    private:
        struct PlayerState {
            bool ballControlPhase = false;
            bool ceilingApproachPhase = false;
            bool ceilingLandingPhase = false;
            bool resetObtained = false;
            bool resetUsed = false;
            Vec ballPosition = Vec(0, 0, 0);
            float resetTime = 0.0f;
            float ceilingTime = 0.0f;
            bool wasOnCeiling = false;
            bool hadFlipBefore = false;
        };
        std::map<uint32_t, PlayerState> playerStates;

    public:
        float ballProximityThreshold;
        float ceilingHeightThreshold;
        float ballControlReward;
        float ceilingApproachReward;
        float ceilingLandingReward;
        float resetObtainReward;
        float resetUsageReward;
        float shotPowerReward;

        CeilingShotReward(
            float ballProximityThreshold = 300.0f,
            float ceilingHeightThreshold = 1800.0f,
            float ballControlReward = 0.8f,
            float ceilingApproachReward = 1.5f,
            float ceilingLandingReward = 2.0f,
            float resetObtainReward = 3.0f,
            float resetUsageReward = 5.0f,
            float shotPowerReward = 1.5f
        ) : ballProximityThreshold(ballProximityThreshold), ceilingHeightThreshold(ceilingHeightThreshold),
            ballControlReward(ballControlReward), ceilingApproachReward(ceilingApproachReward),
            ceilingLandingReward(ceilingLandingReward), resetObtainReward(resetObtainReward),
            resetUsageReward(resetUsageReward), shotPowerReward(shotPowerReward) {
        }

        virtual void Reset(const GameState& initialState) override {
            playerStates.clear();
            for (const auto& player : initialState.players) {
                playerStates[player.carId] = { false, false, false, false, false, Vec(0,0,0), 0.0f, 0.0f, false, player.HasFlipOrJump() };
            }
        }

        virtual float GetReward(const Player& player, const GameState& state, bool isFinal) override {
            if (!state.prev)
                return 0.0f;

            int carId = player.carId;
            if (playerStates.find(carId) == playerStates.end()) {
                playerStates[carId] = { false, false, false, false, false, Vec(0,0,0), 0.0f, 0.0f, false, player.HasFlipOrJump() };
            }

            PlayerState& st = playerStates[carId];
            float reward = 0.0f;

            Vec playerPos = player.pos;
            Vec ballPos = state.ball.pos;
            float distanceToBall = (playerPos - ballPos).Length();
            bool isOnCeiling = playerPos.z > ceilingHeightThreshold;

            // Phase 1: Contrôle de la balle
            if (distanceToBall < ballProximityThreshold && !st.ballControlPhase) {
                st.ballControlPhase = true;
                reward += ballControlReward;
            }

            // Phase 2: Approche du plafond avec la balle
            if (st.ballControlPhase && playerPos.z > 1000.0f && !st.ceilingApproachPhase) {
                st.ceilingApproachPhase = true;
                reward += ceilingApproachReward;
            }

            // Phase 3: Atterrissage sur le plafond
            if (st.ceilingApproachPhase && isOnCeiling && !st.ceilingLandingPhase) {
                st.ceilingLandingPhase = true;
                reward += ceilingLandingReward;
            }

            // Phase 4: Obtention du reset
            if (st.ceilingLandingPhase && !st.resetObtained) {
                bool hasFlipNow = player.HasFlipOrJump();
                if (st.hadFlipBefore && !hasFlipNow) {
                    // Le joueur avait un flip et l'a utilisé (probablement pour atterrir sur le plafond)
                    st.resetObtained = true;
                    reward += resetObtainReward;
                }
            }

            // Phase 5: Utilisation du reset pour tirer
            if (st.resetObtained && !st.resetUsed && player.ballTouchedStep) {
                // Vérifier la puissance du tir
                if (state.prev) {
                    float ballSpeedChange = (state.ball.vel - state.prev->ball.vel).Length();
                    if (ballSpeedChange > 500.0f) {
                        reward += resetUsageReward;
                        reward += ballSpeedChange * shotPowerReward / 1000.0f;
                        st.resetUsed = true;
                    }
                }
            }

            // Récompense continue pour rester sur le plafond
            if (isOnCeiling) {
                st.ceilingTime += 1.0f / 120.0f;
                if (st.ceilingTime > 0.5f) { // Après 0.5 seconde
                    reward += 0.1f; // Récompense continue
                }
            }

            // Reset si le joueur tombe du plafond
            if (st.wasOnCeiling && !isOnCeiling) {
                st.ceilingTime = 0.0f;
            }

            // Mise à jour de l'état
            st.ballPosition = ballPos;
            st.wasOnCeiling = isOnCeiling;
            st.hadFlipBefore = player.HasFlipOrJump();

            return reward;
        }
    };



    // DoubleResetReward - Apprend le double reset (reset + reset)
    class DoubleResetReward : public Reward {
    private:
        struct PlayerState {
            bool firstResetPhase = false;
            bool firstResetObtained = false;
            bool secondResetPhase = false;
            bool secondResetObtained = false;
            bool doubleResetCompleted = false;
            Vec ballPosition = Vec(0, 0, 0);
            float firstResetTime = 0.0f;
            float secondResetTime = 0.0f;
            bool wasOnGround = false;
            bool hadFlipBefore = false;
            bool hadSecondFlipBefore = false;
        };
        std::map<uint32_t, PlayerState> playerStates;

    public:
        float ballProximityThreshold;
        float minResetHeight;
        float maxTimeBetweenResets;
        float firstResetReward;
        float secondResetReward;
        float doubleResetBonus;
        float positioningReward;

        DoubleResetReward(
            float ballProximityThreshold = 200.0f,
            float minResetHeight = 300.0f,
            float maxTimeBetweenResets = 2.0f,
            float firstResetReward = 5.0f,
            float secondResetReward = 8.0f,
            float doubleResetBonus = 15.0f,
            float positioningReward = 2.0f
        ) : ballProximityThreshold(ballProximityThreshold), minResetHeight(minResetHeight),
            maxTimeBetweenResets(maxTimeBetweenResets), firstResetReward(firstResetReward),
            secondResetReward(secondResetReward), doubleResetBonus(doubleResetBonus),
            positioningReward(positioningReward) {
        }

        virtual void Reset(const GameState& initialState) override {
            playerStates.clear();
            for (const auto& player : initialState.players) {
                playerStates[player.carId] = { false, false, false, false, false, Vec(0,0,0), 0.0f, 0.0f, false, player.HasFlipOrJump(), false };
            }
        }

        virtual float GetReward(const Player& player, const GameState& state, bool isFinal) override {
            if (!state.prev)
                return 0.0f;

            int carId = player.carId;
            if (playerStates.find(carId) == playerStates.end()) {
                playerStates[carId] = { false, false, false, false, false, Vec(0,0,0), 0.0f, 0.0f, false, player.HasFlipOrJump(), false };
            }

            PlayerState& st = playerStates[carId];
            float reward = 0.0f;

            Vec playerPos = player.pos;
            Vec ballPos = state.ball.pos;
            float dist = (playerPos - ballPos).Length();
            bool isOnGround = player.isOnGround;
            bool hasFlip = player.HasFlipOrJump();

            // Phase 1: Premier Reset (positionnement sous la balle)
            if (!st.firstResetPhase && !isOnGround && dist < ballProximityThreshold &&
                ballPos.z > minResetHeight && playerPos.z < ballPos.z - 50.0f) {
                st.firstResetPhase = true;
                reward += positioningReward;
            }

            // Phase 2: Obtention du premier reset
            if (st.firstResetPhase && !st.firstResetObtained) {
                bool hasFlipNow = player.HasFlipOrJump();
                if (st.hadFlipBefore && !hasFlipNow) {
                    // Le joueur avait un flip et l'a utilisé pour le premier reset
                    st.firstResetObtained = true;
                    st.firstResetTime = 0.0f;
                    reward += firstResetReward;
                }
            }

            // Phase 3: Deuxième Reset (utiliser le nouveau flip)
            if (st.firstResetObtained && !st.secondResetPhase) {
                st.firstResetTime += 0.016f; // 60 FPS

                // Vérifier si le joueur se repositionne sous la balle
                bool underBall = (playerPos.z < ballPos.z - 50.0f) && (dist < ballProximityThreshold);
                bool upsideDown = player.rotMat.forward.z < -0.3f; // Voiture à l'envers

                if (underBall && upsideDown && st.firstResetTime <= maxTimeBetweenResets) {
                    st.secondResetPhase = true;
                    reward += positioningReward;
                }
            }

            // Phase 4: Obtention du deuxième reset
            if (st.secondResetPhase && !st.secondResetObtained) {
                bool hasFlipNow = player.HasFlipOrJump();
                if (st.hadSecondFlipBefore && !hasFlipNow) {
                    // Le joueur avait le deuxième flip et l'a utilisé
                    st.secondResetObtained = true;
                    st.doubleResetCompleted = true;
                    reward += secondResetReward;
                    reward += doubleResetBonus; // Bonus énorme pour le double reset !
                }
            }

            // Récompense continue pour maintenir la position sous la balle
            if (st.firstResetPhase && !st.doubleResetCompleted) {
                bool underBall = (playerPos.z < ballPos.z - 50.0f) && (dist < ballProximityThreshold);
                if (underBall) {
                    reward += positioningReward * 0.1f; // Récompense continue faible
                }
            }

            // Reset si le joueur touche le sol
            if (st.wasOnGround && !isOnGround) {
                st.firstResetTime = 0.0f;
                st.secondResetTime = 0.0f;
            }

            // Mise à jour de l'état
            st.ballPosition = ballPos;
            st.wasOnGround = isOnGround;
            st.hadFlipBefore = hasFlip;
            st.hadSecondFlipBefore = hasFlip; // Pour le deuxième reset

            return reward;
        }
    };

    // ===== MAWKZY FLICK REWARD - VRAI FLICK 45° + AIR ROLL DIRECTIONNEL =====
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

        MawkzyFlickReward(
            float ballProximityThreshold = 250.0f,
            float minAirRollTime = 0.2f,
            float minFlickPower = 800.0f,
            float targetFlickAngle = 45.0f,
            float angleTolerance = 15.0f,
            float ballControlReward = 1.0f,
            float airRollReward = 2.0f,
            float flickReward = 4.0f,
            float powerBonus = 2.0f,
            float precisionBonus = 2.0f,
            float angleBonus = 3.0f,
            float directionBonus = 2.5f,
            bool debug = false
        ) : ballProximityThreshold(ballProximityThreshold), minAirRollTime(minAirRollTime),
            minFlickPower(minFlickPower), targetFlickAngle(targetFlickAngle),
            angleTolerance(angleTolerance), ballControlReward(ballControlReward),
            airRollReward(airRollReward), flickReward(flickReward),
            powerBonus(powerBonus), precisionBonus(precisionBonus),
            angleBonus(angleBonus), directionBonus(directionBonus), debug(debug) {
        }

        virtual void Reset(const GameState& initialState) override {
            playerStates.clear();
            for (const auto& player : initialState.players) {
                PlayerState& state = playerStates[player.carId];
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

        virtual float GetReward(const Player& player, const GameState& state, bool isFinal) override {
            if (!state.prev)
                return 0.0f;

            int carId = player.carId;
            if (playerStates.find(carId) == playerStates.end()) {
                PlayerState& newState = playerStates[player.carId];
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

            PlayerState& st = playerStates[carId];
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
                    printf("[MawkzyFlick] car_id=%d PHASE 1: Contrôle de la balle! Reward: %.2f\n",
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
                                printf("[MawkzyFlick] car_id=%d PHASE 2: Air roll %s détecté! Direction: %.1f\n",
                                    carId, (st.airRollDirection > 0) ? "DROITE" : "GAUCHE", st.airRollDirection);
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
                        printf("[MawkzyFlick] car_id=%d PHASE 3: Flick déclenché! Direction air roll: %.1f\n",
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
                        printf("[MawkzyFlick] car_id=%d Puissance flick: %.0f, Reward: %.2f\n",
                            carId, ballSpeedChange, reward);
                    }
                }

                // 2. VÉRIFIER L'ANGLE DU FLICK (45°)
                float flickAngle = CalculateFlickAngle(player, state);
                float angleAccuracy = CalculateAngleAccuracy(flickAngle);

                if (angleAccuracy > 0.7f) { // Angle proche de 45°
                    reward += angleBonus * angleAccuracy;

                    if (debug) {
                        printf("[MawkzyFlick] car_id=%d Angle flick: %.1f° (cible: 45°), Accuracy: %.2f, Bonus: %.2f\n",
                            carId, flickAngle, angleAccuracy, angleBonus * angleAccuracy);
                    }
                }

                // 3. VÉRIFIER LA DIRECTION DU FLICK (correspond à l'air roll)
                bool directionMatches = CheckDirectionMatch(st.airRollDirection, state.ball.vel);

                if (directionMatches) {
                    reward += directionBonus;

                    if (debug) {
                        printf("[MawkzyFlick] car_id=%d DIRECTION PARFAITE! Air roll %s → Flick %s, Bonus: %.2f\n",
                            carId,
                            (st.airRollDirection > 0) ? "DROITE" : "GAUCHE",
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
                        printf("[MawkzyFlick] car_id=%d Précision vers le but: %.2f, Bonus: %.2f\n",
                            carId, precision, precisionBonus * precision);
                    }
                }

                // 5. BONUS FINAL POUR MAWKZY FLICK PARFAIT
                if (angleAccuracy > 0.8f && directionMatches && precision > 0.7f) {
                    float perfectBonus = 5.0f;
                    reward += perfectBonus;

                    if (debug) {
                        printf("[MawkzyFlick] car_id=%d 🎯 MAWKZY FLICK PARFAIT! Bonus final: %.2f\n",
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
        float CalculateFlickAngle(const Player& player, const GameState& state) const {
            Vec carForward = Vec(player.rotMat.forward.x, player.rotMat.forward.y, 0.0f);
            Vec ballVelocity = Vec(state.ball.vel.x, state.ball.vel.y, 0.0f);

            if (carForward.Length() < 0.1f || ballVelocity.Length() < 100.0f) {
                return 0.0f;
            }

            carForward = carForward.Normalized();
            ballVelocity = ballVelocity.Normalized();

            float dotProduct = RS_MAX(-1.0f, RS_MIN(1.0f, carForward.Dot(ballVelocity)));
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
        bool CheckDirectionMatch(float airRollDirection, const Vec& ballVelocity) const {
            if (abs(airRollDirection) < 0.1f) return false;

            // Air roll DROITE (+1) → Flick doit aller vers la DROITE (X positif)
            // Air roll GAUCHE (-1) → Flick doit aller vers la GAUCHE (X négatif)
            if (airRollDirection > 0) { // Droite
                return ballVelocity.x > 200.0f; // Vitesse X positive
            }
            else { // Gauche
                return ballVelocity.x < -200.0f; // Vitesse X négative
            }
        }

        Vec GetGoalDirection(Team team) {
            if (team == Team::BLUE) {
                return (CommonValues::ORANGE_GOAL_CENTER - Vec(0, 0, 0)).Normalized();
            }
            else {
                return (CommonValues::BLUE_GOAL_CENTER - Vec(0, 0, 0)).Normalized();
            }
        }
    };

    // ===== BOOST PICKUP REWARD - Gestion intelligente du boost =====
#pragma warning(push)
#pragma warning(disable: 4251)
    // Player state tracking for BoostPickupReward
    struct BoostPickupPlayerState {
        float prevBoost = 0.0f;
        int lowBoostCounter = 0;
    };

    // Converted from rlgymppo Python implementation
    class BoostPickupReward : public Reward {
    public:
        float rewardValue;
        float lowBoostPenalty;
        int lowBoostPatience;
        float minLow;
        float boostSeekBonus;
        float boostProximityBonus;
        bool debug;

        // Player state tracking by carId
        std::unordered_map<uint32_t, BoostPickupPlayerState> playerStates;

        BoostPickupReward(
            float rewardValue = 3.0f,
            float lowBoostPenalty = -0.5f,
            int lowBoostPatience = 30,
            float minLow = 0.2f,
            float boostSeekBonus = 0.2f,
            float boostProximityBonus = 0.5f,
            bool debug = false
        ) : rewardValue(rewardValue),
            lowBoostPenalty(lowBoostPenalty),
            lowBoostPatience(lowBoostPatience),
            minLow(minLow),
            boostSeekBonus(boostSeekBonus),
            boostProximityBonus(boostProximityBonus),
            debug(debug) {
        }

        virtual void Reset(const GameState& initialState) override {
            playerStates.clear();
            for (const Player& player : initialState.players) {
                BoostPickupPlayerState state;
                state.prevBoost = player.boost / 100.0f; // Convert to 0-1 range like Python
                state.lowBoostCounter = 0;
                playerStates[player.carId] = state;
            }
        }

    private:
        float GetClosestBoostDistance(const Vec& playerPos) {
            float minDistance = FLT_MAX;

            for (int i = 0; i < CommonValues::BOOST_LOCATIONS_AMOUNT; i++) {
                Vec boostPos = CommonValues::BOOST_LOCATIONS[i];
                float distance = (boostPos - playerPos).Length();
                if (distance < minDistance) {
                    minDistance = distance;
                }
            }

            return minDistance;
        }

        bool IsMovingTowardsBoost(const Player& player, const GameState& state) {
            Vec playerPos = player.pos;
            float closestBoostDist = GetClosestBoostDistance(playerPos);

            // Skip if too far from any boost
            if (closestBoostDist > 1000) {
                return false;
            }

            // Find closest boost
            Vec closestBoost;
            float minDist = FLT_MAX;
            bool foundBoost = false;

            for (int i = 0; i < CommonValues::BOOST_LOCATIONS_AMOUNT; i++) {
                Vec boostPos = CommonValues::BOOST_LOCATIONS[i];
                float dist = (boostPos - playerPos).Length();
                if (dist < minDist) {
                    minDist = dist;
                    closestBoost = boostPos;
                    foundBoost = true;
                }
            }

            if (!foundBoost) {
                return false;
            }

            // Calculate direction to boost
            Vec directionToBoost = (closestBoost - playerPos).Normalized();

            // Check player velocity
            Vec playerVelocity = player.vel;
            if (playerVelocity.Length() < 100) {
                return false;
            }

            Vec playerVelocityNorm = playerVelocity.Normalized();

            // Check alignment between velocity and boost direction
            float alignment = playerVelocityNorm.Dot(directionToBoost);

            return alignment > 0.3f;
        }

    public:
        virtual float GetReward(const Player& player, const GameState& state, bool isFinal) override {
            uint32_t carId = player.carId;

            // Initialize player state if not exists
            if (playerStates.find(carId) == playerStates.end()) {
                BoostPickupPlayerState newState;
                newState.prevBoost = player.boost / 100.0f;
                newState.lowBoostCounter = 0;
                playerStates[carId] = newState;
            }

            BoostPickupPlayerState& playerState = playerStates[carId];
            float reward = 0.0f;
            float boost = player.boost / 100.0f; // Convert to 0-1 range like Python
            Vec playerPos = player.pos;

            // Boost pickup detection
            if (boost > playerState.prevBoost) {
                reward = rewardValue;
                playerState.lowBoostCounter = 0;
                if (debug) {
                    RG_LOG("[BoostPickupReward] Pickup détecté pour car_id=" << carId << " : boost " << playerState.prevBoost << " -> " << boost);
                }
            }
            else {
                // Low boost handling
                if (boost < minLow) {
                    playerState.lowBoostCounter++;

                    // Apply penalty after patience period
                    if (playerState.lowBoostCounter > lowBoostPatience) {
                        reward += lowBoostPenalty;
                        if (debug) {
                            RG_LOG("[BoostPickupReward] Pénalité manque de boost car_id=" << carId << " : boost=" << boost << " depuis " << playerState.lowBoostCounter << " frames");
                        }
                    }

                    // Bonus for seeking boost
                    if (IsMovingTowardsBoost(player, state)) {
                        reward += boostSeekBonus;
                        if (debug) {
                            RG_LOG("[BoostPickupReward] Bonus direction boost car_id=" << carId);
                        }
                    }

                    // Proximity bonus
                    float closestBoostDist = GetClosestBoostDistance(playerPos);
                    if (closestBoostDist < 300) {
                        reward += boostProximityBonus;
                        if (debug) {
                            RG_LOG("[BoostPickupReward] Bonus proximité boost car_id=" << carId << " : distance=" << closestBoostDist);
                        }
                    }
                }
                else {
                    // Reset counter if boost is adequate
                    playerState.lowBoostCounter = 0;
                }
            }

            // Update previous boost
            playerState.prevBoost = boost;
            return reward;
        }
    };
#pragma warning(pop)

#pragma warning(pop)

    // ===== INFINITE WALLDASH REWARD V2 - SPAM WALLDASH (PLUS FACILE À APPRENDRE) =====
    class InfiniteWalldashRewardV2 : public Reward {
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

        InfiniteWalldashRewardV2(
            float wallHeightThreshold = 100.0f,
            float maxTimeBetweenDashAndReset = 2.0f,
            float dashReward = 2.0f,
            float resetReward = 6.0f,
            float consecutiveBonus = 3.0f,
            float wallStayPenalty = -0.1f,
            float spamBonus = 4.0f,
            float speedBonus = 2.0f,
            float frequencyBonus = 3.0f,
            bool debug = false
        ) : wallHeightThreshold(wallHeightThreshold),
            maxTimeBetweenDashAndReset(maxTimeBetweenDashAndReset),
            dashReward(dashReward), resetReward(resetReward),
            consecutiveBonus(consecutiveBonus), wallStayPenalty(wallStayPenalty),
            spamBonus(spamBonus), speedBonus(speedBonus),
            frequencyBonus(frequencyBonus), debug(debug) {
        }

        virtual void Reset(const GameState& initialState) override {
            playerStates.clear();
            for (const auto& player : initialState.players) {
                PlayerState& state = playerStates[player.carId];
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

        virtual float GetReward(const Player& player, const GameState& state, bool isFinal) override {
            if (!state.prev) return 0.0f;

            uint32_t carId = player.carId;
            if (playerStates.find(carId) == playerStates.end()) {
                PlayerState& newState = playerStates[carId];
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

            PlayerState& st = playerStates[carId];
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
                    printf("[WallDashV2] car_id=%d DASH #%d! Reward: %.2f\n",
                        carId, st.dashCount, dashReward);
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
                        printf("[WallDashV2] car_id=%d WALL DASH CONSÉCUTIF #%d! Bonus: %.2f\n",
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
                        printf("[WallDashV2] car_id=%d SPAM WALLDASH! Temps: %.2fs, Bonus: %.2f\n",
                            carId, timeSinceDash, spamReward);
                    }
                }

                // === BONUS FRÉQUENCE ===
                // Récompense pour faire beaucoup de dashes
                if (st.dashCount >= 3) {
                    float frequencyReward = frequencyBonus * (st.dashCount / 3.0f);
                    reward += frequencyReward;

                    if (debug && st.dashCount % 3 == 0) {
                        printf("[WallDashV2] car_id=%d FRÉQUENCE ÉLEVÉE! %d dashes, Bonus: %.2f\n",
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
                    printf("[WallDashV2] car_id=%d RESET SUR LE MUR! Reward: %.2f\n", carId, resetReward);
                }
            }

            // === PHASE 3: GESTION DU TEMPS SUR LE MUR ===
            if (isOnWall) {
                st.wallStayTime += 1.0f / 120.0f; // 120 ticks par seconde

                // Pénalité plus douce pour le spam walldash
                if (st.wallStayTime > 2.0f) { // Après 2 secondes (plus permissif)
                    reward += wallStayPenalty;

                    if (debug && st.wallStayTime > 3.0f) {
                        printf("[WallDashV2] car_id=%d RESTE TROP LONGTEMPS SUR LE MUR! Pénalité: %.2f\n",
                            carId, wallStayPenalty);
                    }
                }
            }
            else {
                st.wallStayTime = 0.0f;
            }

            // === PHASE 4: RÉINITIALISATION SI LE JOUEUR TOUCHE LE SOL ===
            if (player.isOnGround) {
                // Récompense finale basée sur la performance
                if (st.dashCount >= 5) {
                    float finalBonus = 10.0f;
                    reward += finalBonus;

                    if (debug) {
                        printf("[WallDashV2] car_id=%d 🎯 SESSION WALLDASH TERMINÉE! %d dashes, Bonus final: %.2f\n",
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
        bool IsOnWall(const Vec& playerPos) {
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

        bool DetectDashFromWall(const Player& player, PlayerState& st) {
            // Conditions pour détecter un dash depuis le mur :
            // 1. Était sur un mur
            // 2. Avait un flip
            // 3. N'a plus de flip (l'a utilisé)
            // 4. N'est plus sur le mur (a sauté)

            return (st.wasOnWall &&
                st.hadFlipBefore &&
                !player.HasFlipOrJump() &&
                !IsOnWall(player.pos));
        }

        bool DetectWallReset(const Player& player, PlayerState& st) {
            // Conditions pour détecter un reset sur le mur :
            // 1. Avait utilisé un dash (n'avait plus de flip)
            // 2. Est maintenant sur un mur
            // 3. A récupéré son flip

            return (st.dashUsed &&
                IsOnWall(player.pos) &&
                player.HasFlipOrJump());
        }
    };

    // ===== AIR ROLL REWARD - Encourage l'air roll dès qu'il est en l'air =====
    class FrenchAirRollReward : public Reward {
    private:
        struct PlayerState {
            float lastRollInput = 0.0f;
            float airRollTime = 0.0f;
        };
        std::map<uint32_t, PlayerState> playerStates;

    public:
        float minHeight;
        float airRollWeight;
        float continuousBonus;
        bool debug;

        FrenchAirRollReward(
            float minHeight = 200.0f,
            float airRollWeight = 2.0f,
            float continuousBonus = 0.5f,
            bool debug = false
        ) : minHeight(minHeight), airRollWeight(airRollWeight),
            continuousBonus(continuousBonus), debug(debug) {
        }

        virtual void Reset(const GameState& initialState) override {
            playerStates.clear();
            for (const auto& player : initialState.players) {
                PlayerState& state = playerStates[player.carId];
                state.lastRollInput = 0.0f;
                state.airRollTime = 0.0f;
            }
        }

        virtual float GetReward(const Player& player, const GameState& state, bool isFinal) override {
            uint32_t carId = player.carId;

            if (playerStates.find(carId) == playerStates.end()) {
                PlayerState& newState = playerStates[carId];
                newState.lastRollInput = 0.0f;
                newState.airRollTime = 0.0f;
            }

            PlayerState& st = playerStates[carId];
            float reward = 0.0f;

            // Vérifier si le bot est en l'air
            bool isAirborne = (!player.isOnGround && player.pos.z >= minHeight);

            if (!isAirborne) {
                st.airRollTime = 0.0f;
                return 0.0f;
            }

            // Détecter l'air roll via la rotation de la voiture
            Vec carRight = Vec(player.rotMat.right.x, player.rotMat.right.y, player.rotMat.right.z);
            float currentRollInput = carRight.z; // Approximation de l'air roll

            // Récompenser SEULEMENT l'air roll DROIT
            if (currentRollInput > 0.1f) {  // SEULEMENT air roll droit (positif)
                float baseReward = airRollWeight;

                reward += baseReward;
                st.airRollTime += 1.0f / 120.0f; // 120 ticks par seconde

                // Bonus pour air roll continu
                if (st.airRollTime > 0.5f) {
                    reward += continuousBonus;
                }

                if (debug && st.airRollTime > 1.0f) {
                    printf("[AirRoll] car_id=%d AIR ROLL EN L'AIR! Direction: %s, Temps: %.1fs, Reward: %.2f\n",
                        carId, (currentRollInput < 0) ? "GAUCHE" : "DROITE", st.airRollTime, reward);
                }
            }
            else {
                st.airRollTime = 0.0f;
            }

            st.lastRollInput = currentRollInput;
            return reward;
        }
    };

    class FrenchBackboardDefenseReward : public Reward {
    public:
        float GetReward(const Player& player, const GameState& state, bool isFinal) override {
            // 1. Vérifier si la balle est en l'air près de son propre but
            if (!IsBallNearOurGoalInAir(player.team, state.ball)) return 0.0f;

            // 2. Vérifier si le bot est sur SON backboard (derrière son but)
            if (!IsOnOwnBackboard(player.pos, player.team)) return 0.0f;

            // 3. Vérifier s'il y a un adversaire en l'air avec la balle
            bool opponentInAirWithBall = false;
            for (const auto& otherPlayer : state.players) {
                if (otherPlayer.team != player.team && !otherPlayer.isOnGround) {
                    float distanceToBall = (otherPlayer.pos - state.ball.pos).Length();
                    if (distanceToBall < 300.0f) { // Adversaire proche de la balle
                        opponentInAirWithBall = true;
                        break;
                    }
                }
            }

            if (!opponentInAirWithBall) return 0.0f;

            // 4. Récompenser la position défensive sur le backboard
            return 1.0f;
        }

    private:
        bool IsOnOwnBackboard(const Vec& pos, Team team) {
            // Vérifier si le bot est sur SON backboard (derrière son but)
            float backboardY = (team == Team::BLUE) ?
                CommonValues::BLUE_GOAL_CENTER.y : CommonValues::ORANGE_GOAL_CENTER.y;

            // Proche du backboard ET assez haut
            float distanceToBackboard = fabsf(pos.y - backboardY);
            return distanceToBackboard < 200.0f && pos.z > 100.0f;
        }

        bool IsBallNearOurGoalInAir(Team team, const PhysState& ball) {
            // Vérifier si la balle est en l'air près de son propre but
            Vec ownGoalCenter = (team == Team::BLUE) ?
                CommonValues::BLUE_GOAL_CENTER : CommonValues::ORANGE_GOAL_CENTER;

            float distanceToOurGoal = (ball.pos - ownGoalCenter).Length();

            // Balle en l'air (>200) ET proche de notre but (<1500)
            return ball.pos.z > 200.0f && distanceToOurGoal < 1500.0f;
        }
    };

    // ===== DOUBLE TAP REWARD - VRAIE DOUBLE TAP (BOT RESTE EN L'AIR) =====
    class FrenchDoubleTapReward : public Reward {
    private:
        struct PlayerState {
            bool airDribblePhase = false;
            bool backboardHitPhase = false;
            bool recoveryPhase = false;
            bool doubleTapCompleted = false;
            Vec lastBallPosition = Vec(0, 0, 0);
            Vec lastBallVelocity = Vec(0, 0, 0);
            float airDribbleTime = 0.0f;
            float backboardHitTime = 0.0f;
            bool wasAirRolling = false;
            bool hadFlipBefore = false;
            Vec backboardHitPosition = Vec(0, 0, 0);
            float backboardHitSpeed = 0.0f;
            bool hasTouchedGround = false; // CRUCIAL: vérifier si le bot a touché le sol
            bool ballHitBackboard = false; // Vérifier si la balle a vraiment touché le backboard
            Vec ballPositionBeforeBackboard = Vec(0, 0, 0);
        };
        std::map<uint32_t, PlayerState> playerStates;

    public:
        float minAirDribbleHeight;
        float minAirDribbleTime;
        float minBackboardHitSpeed;
        float maxRecoveryTime;
        float airDribbleReward;
        float backboardHitReward;
        float recoveryReward;
        float doubleTapBonus;
        float airRollBonus;
        bool debug;

        FrenchDoubleTapReward(
            float minAirDribbleHeight = 400.0f,
            float minAirDribbleTime = 1.0f,
            float minBackboardHitSpeed = 800.0f,
            float maxRecoveryTime = 3.0f,
            float airDribbleReward = 2.0f,
            float backboardHitReward = 5.0f,
            float recoveryReward = 3.0f,
            float doubleTapBonus = 15.0f,
            float airRollBonus = 2.0f,
            bool debug = false) :
            minAirDribbleHeight(minAirDribbleHeight),
            minAirDribbleTime(minAirDribbleTime),
            minBackboardHitSpeed(minBackboardHitSpeed),
            maxRecoveryTime(maxRecoveryTime),
            airDribbleReward(airDribbleReward),
            backboardHitReward(backboardHitReward),
            recoveryReward(recoveryReward),
            doubleTapBonus(doubleTapBonus),
            airRollBonus(airRollBonus),
            debug(debug) {
        }

        virtual void Reset(const GameState& initialState) override {
            playerStates.clear();
        }

        virtual float GetReward(const Player& player, const GameState& state, bool isFinal) override {
            if (!state.prev) return 0.0f;

            int carId = player.carId;
            if (playerStates.find(carId) == playerStates.end()) {
                playerStates[carId] = { false, false, false, false, Vec(0,0,0), Vec(0,0,0), 0.0f, 0.0f, false, false, Vec(0,0,0), 0.0f, false, false, Vec(0,0,0) };
            }

            PlayerState& st = playerStates[carId];
            float reward = 0.0f;
            const Vec& ballPos = state.ball.pos;
            const Vec& ballVel = state.ball.vel;
            const Vec& carPos = player.pos;
            bool isOnGround = player.isOnGround;

            // === VÉRIFICATION CRUCIALE: LE BOT NE DOIT JAMAIS TOUCHER LE SOL ===
            if (isOnGround && (st.airDribblePhase || st.backboardHitPhase || st.recoveryPhase)) {
                st.hasTouchedGround = true;

                // Reset complet si le bot touche le sol
                st.airDribblePhase = false;
                st.backboardHitPhase = false;
                st.recoveryPhase = false;
                st.doubleTapCompleted = false;
                st.hasTouchedGround = false;
                st.ballHitBackboard = false;
                st.airDribbleTime = 0.0f;
                st.backboardHitTime = 0.0f;
                return 0.0f;
            }

            // === PHASE 1: DÉTECTION AIRDRIBBLE AVEC AIR ROLL ===
            if (!st.airDribblePhase && !isOnGround && ballPos.z >= minAirDribbleHeight && !st.hasTouchedGround) {
                // Vérifier si le joueur fait de l'air roll
                bool isAirRolling = IsAirRolling(player, state);

                if (isAirRolling && player.ballTouchedStep) {
                    st.airDribblePhase = true;
                    st.airDribbleTime = 0.0f;
                    st.wasAirRolling = true;
                    st.hasTouchedGround = false; // Reset du flag

                }
            }

            // Récompense continue pour l'airdribble (seulement si pas touché le sol)
            if (st.airDribblePhase && !st.backboardHitPhase && !st.hasTouchedGround) {
                st.airDribbleTime += 1.0f / 120.0f;

                if (st.airDribbleTime >= minAirDribbleTime) {
                    reward += airDribbleReward * 0.1f; // Récompense continue

                    // Bonus pour l'air roll pendant l'airdribble
                    if (IsAirRolling(player, state)) {
                        reward += airRollBonus * 0.1f;
                    }
                }
            }

            // === PHASE 2: DÉTECTION FRAPPE SUR BACKBOARD ENNEMI ===
            if (st.airDribblePhase && !st.backboardHitPhase && player.ballTouchedStep && !st.hasTouchedGround) {
                // Vérifier si la balle va vers le backboard ennemi
                if (IsBallGoingToEnemyBackboard(ballVel, player.team)) {
                    float ballSpeed = ballVel.Length();

                    if (ballSpeed >= minBackboardHitSpeed) {
                        st.backboardHitPhase = true;
                        st.backboardHitTime = 0.0f;
                        st.backboardHitPosition = ballPos;
                        st.backboardHitSpeed = ballSpeed;
                        st.ballPositionBeforeBackboard = ballPos;

                        reward += backboardHitReward;

                    }
                }
            }

            // === DÉTECTION QUE LA BALLE TOUCHE VRAIMENT LE BACKBOARD ===
            if (st.backboardHitPhase && !st.ballHitBackboard && !st.hasTouchedGround) {
                if (HasBallHitBackboard(ballPos, st.ballPositionBeforeBackboard, player.team)) {
                    st.ballHitBackboard = true;

                }
            }

            // === PHASE 3: DÉTECTION RÉCUPÉRATION ET BUT (VRAIE DOUBLE TAP) ===
            if (st.backboardHitPhase && st.ballHitBackboard && !st.doubleTapCompleted && !st.hasTouchedGround) {
                st.backboardHitTime += 1.0f / 120.0f;

                // Vérifier si le joueur récupère la balle après le backboard (TOUJOURS EN L'AIR)
                if (player.ballTouchedStep && st.backboardHitTime > 0.5f) {
                    // Vérifier si c'est un but
                    if (IsGoalScored(state, player.team)) {
                        st.doubleTapCompleted = true;
                        reward += doubleTapBonus;

                        if (debug) {
                            printf("[DoubleTap] car_id=%d 🎯 DOUBLE TAP PARFAITE! Bot jamais redescendu! Bonus: %.2f\n",
                                carId, doubleTapBonus);
                        }
                    }
                    else {
                        // Récompense pour la récupération en l'air
                        reward += recoveryReward;

                    }
                }

                // Timeout si pas de récupération
                if (st.backboardHitTime > maxRecoveryTime) {
                    st.airDribblePhase = false;
                    st.backboardHitPhase = false;
                    st.recoveryPhase = false;
                    st.ballHitBackboard = false;
                }
            }

            // Mise à jour de l'état
            st.lastBallPosition = ballPos;
            st.lastBallVelocity = ballVel;
            st.hadFlipBefore = player.HasFlipOrJump();

            return reward;
        }

    private:
        bool IsAirRolling(const Player& player, const GameState& state) {
            // Détecter l'air roll basique - utiliser les vraies API
            Vec angularVel = player.angVel;

            // Air roll si rotation angulaire significative
            return (fabsf(angularVel.z) > 2.0f || fabsf(angularVel.x) > 2.0f) && !player.isOnGround;
        }

        bool IsBallGoingToEnemyBackboard(const Vec& ballVel, Team team) {
            if (ballVel.Length() < 500.0f) return false;

            Vec goalDirection = GetEnemyGoalDirection(team);
            Vec ballDirection = ballVel.Normalized();

            // Vérifier si la balle va vers le backboard ennemi
            float dot = ballDirection.Dot(goalDirection);
            return dot > 0.7f; // Direction vers le but ennemi
        }

        Vec GetEnemyGoalDirection(Team team) {
            // Direction vers le but ennemi
            if (team == Team::BLUE) {
                return Vec(0, 5120.0f, 0); // But orange (ennemi)
            }
            else {
                return Vec(0, -5120.0f, 0); // But bleu (ennemi)
            }
        }

        bool IsGoalScored(const GameState& state, Team team) {
            // Vérifier si un but a été marqué - utiliser les vraies API
            return state.goalScored;
        }

        bool HasBallHitBackboard(const Vec& currentBallPos, const Vec& previousBallPos, Team team) {
            // Vérifier si la balle a touché le backboard ennemi
            // Le backboard ennemi est le mur derrière le but adverse

            float backboardY = (team == Team::BLUE) ? 5120.0f : -5120.0f; // Position Y du backboard ennemi

            // Vérifier si la balle était d'un côté du backboard et maintenant de l'autre
            bool wasBeforeBackboard = false;
            bool isAfterBackboard = false;

            if (team == Team::BLUE) { // Équipe bleue, backboard ennemi à droite
                wasBeforeBackboard = previousBallPos.y < backboardY;
                isAfterBackboard = currentBallPos.y > backboardY;
            }
            else { // Équipe orange, backboard ennemi à gauche
                wasBeforeBackboard = previousBallPos.y > backboardY;
                isAfterBackboard = currentBallPos.y < backboardY;
            }

            // Vérifier aussi la proximité avec le backboard
            float distanceToBackboard = fabsf(currentBallPos.y - backboardY);
            bool isCloseToBackboard = distanceToBackboard < 200.0f;

            return (wasBeforeBackboard && isAfterBackboard) || isCloseToBackboard;
        }
    };

    // ===== AIR DRIBBLE WITH ROLL REWARD 2 - CONVERSION EXACTE DU PYTHON =====
    class AirDribbleWithRollReward2 : public Reward {
    private:
        static constexpr float TICK_SKIP = 4.0f;  // YOUR TICK SKIP VALUE
        static constexpr float DT = (1.0f / 120.0f) * TICK_SKIP;  // Correct delta time

        struct PlayerState {
            bool lastBallTouch = false;
            int airTouchStreak = 0;
            float totalAirRollTime = 0.0f;
            Vec lastAngVel = Vec(0, 0, 0);
            float consecutiveAirRollTime = 0.0f;
        };
        std::map<uint32_t, PlayerState> playerStates;

    public:
        float minHeight;
        float maxHeight;
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
        bool debug;

        AirDribbleWithRollReward2(
            float minHeight = 200.0f,
            float maxHeight = 1600.0f,
            float maxDistance = 1000.0f,
            float proximityWeight = 0.4f,
            float velocityWeight = 0.3f,
            float heightWeight = 0.5f,
            float touchBonus = 0.3f,
            float boostThreshold = 0.2f,
            float heightTarget = 1000.0f,
            float heightTargetWeight = 1.0f,
            float upwardBonus = 0.1f,
            float airRollWeight = 0.8f,
            float speedWeight = 0.5f,
            float boostEfficiencyWeight = 0.8f,
            bool debug = false
        ) : minHeight(minHeight), maxHeight(maxHeight), maxDistance(maxDistance), proximityWeight(proximityWeight),
            velocityWeight(velocityWeight), heightWeight(heightWeight), touchBonus(touchBonus),
            boostThreshold(boostThreshold), heightTarget(heightTarget), heightTargetWeight(heightTargetWeight),
            upwardBonus(upwardBonus), airRollWeight(airRollWeight), speedWeight(speedWeight),
            boostEfficiencyWeight(boostEfficiencyWeight), debug(debug) {
        }

        virtual void Reset(const GameState& initialState) override {
            playerStates.clear();
            for (const auto& player : initialState.players) {
                PlayerState& state = playerStates[player.carId];
                state.lastBallTouch = player.ballTouchedStep;
                state.airTouchStreak = 0;
                state.totalAirRollTime = 0.0f;
                state.lastAngVel = player.angVel;
                state.consecutiveAirRollTime = 0.0f;
            }
        }

        virtual float GetReward(const Player& player, const GameState& state, bool isFinal) override {
            uint32_t carId = player.carId;

            if (playerStates.find(carId) == playerStates.end()) {
                PlayerState& newState = playerStates[carId];
                newState.lastBallTouch = player.ballTouchedStep;
                newState.airTouchStreak = 0;
                newState.totalAirRollTime = 0.0f;
                newState.lastAngVel = player.angVel;
                newState.consecutiveAirRollTime = 0.0f;
            }

            PlayerState& st = playerStates[carId];

            Vec carPos = player.pos;
            Vec ballPos = state.ball.pos;

            // Vérification airborne avec maxHeight pour éviter le ceiling
            bool isAirborne = (!player.isOnGround &&
                carPos.z >= minHeight &&
                carPos.z <= maxHeight &&
                ballPos.z >= minHeight &&
                ballPos.z <= maxHeight);

            if (!isAirborne) {
                st.airTouchStreak = 0;
                st.lastBallTouch = player.ballTouchedStep;
                st.totalAirRollTime = 0.0f;
                st.consecutiveAirRollTime = 0.0f;
                st.lastAngVel = player.angVel;
                return 0.0f;
            }

            Vec distVec = ballPos - carPos;
            float dist = distVec.Length();
            bool isClose = dist <= maxDistance;

            if (!isClose) {
                st.airTouchStreak = 0;
                st.lastBallTouch = player.ballTouchedStep;
                st.totalAirRollTime = 0.0f;
                st.consecutiveAirRollTime = 0.0f;
                st.lastAngVel = player.angVel;
                return 0.0f;
            }

            // 1. Proximity Reward
            float proximityFactor = RS_MAX(0.0f, 1.0f - (dist - CommonValues::BALL_RADIUS) / RS_MAX(1e-6f, maxDistance - CommonValues::BALL_RADIUS));
            float proxReward = proximityFactor * proximityWeight;

            // 2. Velocity Reward
            float velReward = 0.0f;
            if (dist > 1e-6f) {
                Vec dirToBall = distVec / dist;
                Vec playerVel = player.vel;
                float speedTowardsBall = playerVel.Dot(dirToBall);
                if (speedTowardsBall > 0) {
                    velReward = (speedTowardsBall / CommonValues::CAR_MAX_SPEED) * velocityWeight;
                }
            }

            // 3. Height Reward
            float currentHeight = carPos.z;
            float heightFactor = RS_MAX(0.0f, (currentHeight - minHeight) / RS_MAX(1e-6f, CommonValues::CEILING_Z - minHeight));
            float heightReward = heightFactor * heightWeight;

            // 4. Height Target Reward
            float heightDiff = fabsf(currentHeight - heightTarget);
            float heightScale = RS_MAX(0.0f, 1.0f - heightDiff / heightTarget);
            float heightTargetReward = heightScale * heightTargetWeight;

            // 5. Controlled Speed Reward
            float controlledSpeedReward = 0.0f;
            float playerSpeed = player.vel.Length();
            float optimalMinSpeed = 800.0f;
            float optimalMaxSpeed = 1400.0f;

            if (optimalMinSpeed <= playerSpeed && playerSpeed <= optimalMaxSpeed) {
                controlledSpeedReward = speedWeight;
            }
            else if (playerSpeed > optimalMaxSpeed) {
                float excessSpeedPenalty = RS_MIN(0.5f, (playerSpeed - optimalMaxSpeed) / 1000.0f);
                controlledSpeedReward = -excessSpeedPenalty * speedWeight;
            }

            // 6. Boost Conservation Reward
            float boostConservationReward = 0.0f;
            if (player.boost > 0.3f) {
                float boostFactor = powf(player.boost, 0.7f);
                boostConservationReward = boostFactor * boostEfficiencyWeight;
            }

            // 7. FIXED Air Roll Detection & Reward
            float airRollReward = 0.0f;

            // Detect air roll by measuring roll axis angular velocity
            // In car's local space, roll is around the forward axis (player.rotMat.forward)
            Vec rollAxis = player.rotMat.forward;
            float rollAngVel = player.angVel.Dot(rollAxis);
            float rollSpeed = fabsf(rollAngVel);

            // Minimum angular velocity to be considered "air rolling" (radians/sec)
            constexpr float MIN_ROLL_SPEED = 1.5f;  // ~86 degrees/sec
            constexpr float OPTIMAL_ROLL_SPEED = 4.0f;  // ~229 degrees/sec

            bool isAirRolling = rollSpeed > MIN_ROLL_SPEED;

            if (isAirRolling) {
                // Accumulate air roll time (with tick skip correction)
                st.consecutiveAirRollTime += DT;
                st.totalAirRollTime += DT;

                // Base reward for air rolling
                float rollIntensity = RS_MIN(1.0f, rollSpeed / OPTIMAL_ROLL_SPEED);
                float baseRollReward = rollIntensity * airRollWeight;

                // Bonus for sustained air roll (encourages continuous rolling)
                float sustainedBonus = RS_MIN(0.5f, st.consecutiveAirRollTime / 1.0f);
                baseRollReward *= (1.0f + sustainedBonus);

                // Bonus if boosting while air rolling (proper air dribble technique)
                if (player.boost > boostThreshold) {
                    baseRollReward *= 1.4f;
                }

                // Bonus if close to ball while air rolling
                if (dist < CommonValues::BALL_RADIUS + 200.0f) {
                    baseRollReward *= 1.3f;
                }

                airRollReward = baseRollReward;

                if (debug && airRollReward > 0.1f) {
                    printf("[AirRoll] car_id=%d roll_speed=%.2f time=%.2f reward=%.2f\n",
                        carId, rollSpeed, st.consecutiveAirRollTime, airRollReward);
                }
            }
            else {
                // Reset consecutive time if not air rolling
                st.consecutiveAirRollTime = 0.0f;
            }

            st.lastAngVel = player.angVel;

            // Base reward calculation
            float baseReward = proxReward + velReward + heightReward + heightTargetReward +
                controlledSpeedReward + boostConservationReward + airRollReward;
            float totalWeight = proximityWeight + velocityWeight + heightWeight + heightTargetWeight +
                speedWeight + boostEfficiencyWeight + airRollWeight;

            // Normalized base reward
            float normalizedBaseReward = (totalWeight > 1e-6f) ? (baseReward / totalWeight) : 0.0f;
            float finalReward = normalizedBaseReward;

            // 8. Upward Bonus
            if (player.vel.z > 0) {
                finalReward += upwardBonus;
            }

            // 9. Below Ball + Facing Ball Bonus
            Vec carToBall = ballPos - carPos;
            Vec carUp = player.rotMat.up;
            bool belowBall = carPos.z < ballPos.z - 30.0f;
            bool facingBall = carToBall.Normalized().Dot(carUp) > 0.7f;
            if (belowBall && facingBall) {
                finalReward += 0.2f;
            }

            // 10. Touch Reward
            if (!st.lastBallTouch && player.ballTouchedStep) {
                st.airTouchStreak += 1;

                float touchMultiplier = 1.0f;

                float speed = player.vel.Length();
                if (800.0f <= speed && speed <= 1400.0f) {
                    touchMultiplier *= 2.5f;
                }
                else if (speed > 1400.0f) {
                    float speedPenalty = RS_MIN(0.7f, (speed - 1400.0f) / 1000.0f);
                    touchMultiplier *= RS_MAX(0.1f, 1.5f - speedPenalty);
                }
                else {
                    touchMultiplier *= 1.2f;
                }

                if (player.boost > boostThreshold) {
                    touchMultiplier *= 1.8f;
                }

                if (dist < CommonValues::BALL_RADIUS + 150.0f) {
                    touchMultiplier *= 1.4f;
                }

                // BONUS: Extra multiplier if air rolling during touch
                if (rollSpeed > MIN_ROLL_SPEED) {
                    touchMultiplier *= 1.5f;  // Big bonus for air roll touches!
                }

                float touchReward = touchBonus * st.airTouchStreak * touchMultiplier;
                finalReward += touchReward;

                if (debug) {
                    printf("[Airdribble2] car_id=%d streak=%d touch_reward=%.2f\n",
                        carId, st.airTouchStreak, touchReward);
                }

            }
            else if (player.isOnGround) {
                st.airTouchStreak = 0;
                st.totalAirRollTime = 0.0f;
                st.consecutiveAirRollTime = 0.0f;
            }

            st.lastBallTouch = player.ballTouchedStep;

            // 11. Height Scaling
            float heightScaling = ballPos.z / 1000.0f;
            finalReward *= RS_MAX(0.1f, RS_MIN(2.0f, heightScaling));

            return finalReward;
        }
    };

    // ===== DEMO REWARD - CONVERSION EXACTE DU PYTHON =====
    class DemoReward2 : public Reward {
    private:
        std::unordered_map<uint32_t, bool> prevDemoed;
        float rewardValue;

    public:
        DemoReward2(float rewardValue = 1.0f) : rewardValue(rewardValue) {}

        virtual void Reset(const GameState& initialState) override {
            prevDemoed.clear();
            for (const auto& player : initialState.players) {
                prevDemoed[player.carId] = player.isDemoed;
            }
        }

        virtual float GetReward(const Player& player, const GameState& state, bool isFinal) override {
            float reward = 0.0f;

            for (const auto& opp : state.players) {
                if (opp.team != player.team) {
                    bool prev = false;
                    if (prevDemoed.find(opp.carId) != prevDemoed.end()) {
                        prev = prevDemoed[opp.carId];
                    }

                    if (!prev && opp.isDemoed) {
                        reward += rewardValue;
                    }

                    prevDemoed[opp.carId] = opp.isDemoed;
                }
            }

            return reward;
        }
    };

    // ===== CEILING SHOT REWARD V2 - CONVERSION EXACTE DU PYTHON =====
    class CeilingShotRewardV2 : public Reward {
    private:
        std::unordered_map<uint32_t, bool> previousBallTouched;
        std::unordered_map<uint32_t, bool> previousHasJump;
        std::unordered_map<uint32_t, bool> previousHasFlip;

    public:
        CeilingShotRewardV2() {}

        virtual void Reset(const GameState& initialState) override {
            previousBallTouched.clear();
            previousHasJump.clear();
            previousHasFlip.clear();

            for (const auto& player : initialState.players) {
                previousBallTouched[player.carId] = player.ballTouchedStep;
                previousHasJump[player.carId] = player.HasFlipOrJump();
                previousHasFlip[player.carId] = player.HasFlipOrJump();
            }
        }

        virtual float GetReward(const Player& player, const GameState& state, bool isFinal) override {
            float reward = 0.0f;
            uint32_t carId = player.carId;
            Vec carPos = player.pos;
            Vec ballPos = state.ball.pos;

            // Calcul des distances aux murs
            float wallDistX = RS_MIN(fabsf(carPos.x - 4096.0f), fabsf(carPos.x + 4096.0f));
            float wallDistY = RS_MIN(fabsf(carPos.y - 5120.0f), fabsf(carPos.y + 5120.0f));
            float wallDist = RS_MIN(wallDistX, wallDistY);

            // États du bot
            bool isOnWall = (wallDist < 150.0f && carPos.z > 200.0f);
            bool isOnCeiling = (carPos.z > 1400.0f);
            bool isFalling = (!player.isOnGround && player.vel.z < -200.0f);

            // Distance à la balle
            float distanceToBall = (ballPos - carPos).Length();

            // 1. Bonus pour être sur le mur (si balle proche)
            if (isOnWall && distanceToBall < 400.0f) {
                reward += 0.3f;
            }

            // 2. Bonus pour être sur le ceiling
            if (isOnCeiling) {
                reward += 0.5f;
                if (distanceToBall < 300.0f) {
                    reward += 0.3f;
                }
            }

            // 3. Bonus pour tomber (hauteur > 800)
            if (isFalling && carPos.z > 800.0f) {
                reward += 0.4f;
            }

            // 4. Bonus pour toucher la balle
            if (previousBallTouched.find(carId) != previousBallTouched.end()) {
                if (!previousBallTouched[carId] && player.ballTouchedStep) {
                    if (isOnWall && distanceToBall < 400.0f) {
                        reward += 1.5f;
                    }
                    else if (isOnCeiling) {
                        reward += 2.0f;
                    }
                    else {
                        reward += 0.2f;
                    }
                }
            }

            // 5. Bonus pour utiliser le jump (SEULEMENT si pas sur le ceiling)
            if (previousHasJump.find(carId) != previousHasJump.end()) {
                if (previousHasJump[carId] && !player.HasFlipOrJump()) {
                    if (!isOnCeiling) {
                        reward += 1.0f; // Bonus pour utiliser le jump (pas sur le ceiling)
                    }
                    else {
                        reward -= 0.5f; // Pénalité pour utiliser le jump sur le ceiling
                    }
                }
            }

            // 5b. Bonus pour tomber naturellement du ceiling (sans jump)
            if (isOnCeiling && isFalling && !player.HasFlipOrJump()) {
                reward += 2.0f; // Gros bonus pour tomber naturellement du ceiling
            }

            // 6. Bonus pour utiliser le flip
            if (previousHasFlip.find(carId) != previousHasFlip.end()) {
                if (previousHasFlip[carId] && !player.HasFlipOrJump()) {
                    if (isFalling && !player.isOnGround) {
                        if (distanceToBall < 200.0f) {
                            reward += 3.0f;
                        }
                        else {
                            reward += 1.0f;
                        }
                    }
                    else {
                        reward += 0.2f;
                    }
                }
            }

            // Mise à jour des états précédents
            previousBallTouched[carId] = player.ballTouchedStep;
            previousHasJump[carId] = player.HasFlipOrJump();
            previousHasFlip[carId] = player.HasFlipOrJump();

            return reward;
        }
    };

    // ============================================
    // FlipResetSeekingReward - Guide le bot pour chercher activement des flip resets
    // ============================================
    class FlipResetArsenalReward : public Reward {
    private:
        std::map<uint32_t, bool> lastHasFlip;
        std::map<uint32_t, float> lastDistanceToBall;

    public:
        float maxSeekDistance;
        float minHeight;
        float proximityWeight;
        float positionWeight;
        float orientationWeight;
        bool debug;

        FlipResetArsenalReward(
            float maxSeekDistance = 1500.0f,  // Distance max pour chercher un reset
            float minHeight = 200.0f,         // Hauteur minimum pour chercher
            float proximityWeight = 1.0f,     // Récompense pour se rapprocher
            float positionWeight = 2.0f,      // Récompense pour être sous la balle
            float orientationWeight = 1.5f,   // Récompense pour bonne orientation
            bool debug = false
        ) : maxSeekDistance(maxSeekDistance), minHeight(minHeight), proximityWeight(proximityWeight),
            positionWeight(positionWeight), orientationWeight(orientationWeight), debug(debug) {
        }

        virtual void Reset(const GameState& initialState) override {
            lastHasFlip.clear();
            lastDistanceToBall.clear();
            for (const auto& player : initialState.players) {
                lastHasFlip[player.carId] = player.HasFlipOrJump();
                Vec playerPos = Vec(player.pos.x, player.pos.y, player.pos.z);
                Vec ballPos = Vec(initialState.ball.pos.x, initialState.ball.pos.y, initialState.ball.pos.z);
                lastDistanceToBall[player.carId] = (playerPos - ballPos).Length();
            }
        }

        virtual float GetReward(const Player& player, const GameState& state, bool isFinal) override {
            if (!state.prev) return 0.0f;

            uint32_t carId = player.carId;
            Vec playerPos = Vec(player.pos.x, player.pos.y, player.pos.z);
            Vec ballPos = Vec(state.ball.pos.x, state.ball.pos.y, state.ball.pos.z);

            // 1. Le bot doit être en l'air et ne pas avoir de flip
            if (player.isOnGround || player.HasFlipOrJump()) {
                lastHasFlip[carId] = player.HasFlipOrJump();
                lastDistanceToBall[carId] = (playerPos - ballPos).Length();
                return 0.0f;
            }

            // 2. Hauteur minimum
            if (playerPos.z < minHeight) {
                lastHasFlip[carId] = player.HasFlipOrJump();
                lastDistanceToBall[carId] = (playerPos - ballPos).Length();
                return 0.0f;
            }

            float distanceToBall = (playerPos - ballPos).Length();

            // 3. Distance maximum pour chercher un reset
            if (distanceToBall > maxSeekDistance) {
                lastHasFlip[carId] = player.HasFlipOrJump();
                lastDistanceToBall[carId] = distanceToBall;
                return 0.0f;
            }

            float reward = 0.0f;

            // 4. RÉCOMPENSE PROXIMITÉ : Se rapprocher de la balle
            float lastDist = lastDistanceToBall.find(carId) != lastDistanceToBall.end() ?
                lastDistanceToBall[carId] : distanceToBall;

            if (distanceToBall < lastDist) {
                float proximityBonus = (lastDist - distanceToBall) / 100.0f; // Plus on se rapproche, plus c'est récompensé
                reward += proximityBonus * proximityWeight;

                if (debug) {
                    std::cout << "[FlipResetArsenal] Proximity: " << proximityBonus * proximityWeight
                        << " (dist: " << lastDist << " -> " << distanceToBall << ")" << std::endl;
                }
            }

            // 5. RÉCOMPENSE POSITION : Être sous la balle
            float heightDiff = ballPos.z - playerPos.z;
            if (heightDiff > 0.0f && heightDiff < 100.0f) { // Sweet spot : entre 50 et 300 unités sous la balle
                float positionBonus = RS_MIN(1.0f, heightDiff / 200.0f);
                reward += positionBonus * positionWeight;

                if (debug) {
                    std::cout << "[FlipResetArsenal] Position: " << positionBonus * positionWeight
                        << " (height: " << heightDiff << ")" << std::endl;
                }
            }

            // 6. RÉCOMPENSE ORIENTATION : Dessous de la voiture vers la balle
            Vec carUp = player.rotMat.up;
            Vec carToBall = (ballPos - playerPos).Normalized();

            // Le dessous de la voiture doit pointer vers la balle (carUp inversé)
            Vec carBottom = carUp * -1.0f;
            float orientationDot = carBottom.Dot(carToBall);

            if (orientationDot > 0.5f) { // Bonne orientation
                float orientationBonus = orientationDot * orientationDot; // Quadratique pour récompenser plus les bonnes orientations
                reward += orientationBonus * orientationWeight;

                if (debug) {
                    std::cout << "[FlipResetArsenal] Orientation: " << orientationBonus * orientationWeight
                        << " (dot: " << orientationDot << ")" << std::endl;
                }
            }

            // 7. BONUS SPÉCIAL : Très proche et bien orienté (pré-flip reset)
            if (distanceToBall < 200.0f && heightDiff > 0.0f && heightDiff < 100.0f && orientationDot > 0.7f) {
                reward += 3.0f; // Gros bonus pour être dans la position parfaite

                if (debug) {
                    std::cout << "[FlipResetArsenal] PRE-RESET BONUS: +3.0f !" << std::endl;
                }
            }

            // Mise à jour des états précédents
            lastHasFlip[carId] = player.HasFlipOrJump();
            lastDistanceToBall[carId] = distanceToBall;

            if (debug && reward > 0.0f) {
                std::cout << "[FlipResetArsenal] TOTAL: " << reward << std::endl;
            }

            return reward;
        }
    };

    class FlipResetNormalReward : public Reward {
    private:
        std::map<uint32_t, bool> lastHasFlip;
        std::map<uint32_t, float> lastDistanceToBall;

    public:
        float maxSeekDistance;
        float minHeight;
        float proximityWeight;
        float positionWeight;
        float orientationWeight;
        bool debug;

        FlipResetNormalReward(
            float maxSeekDistance = 1500.0f,  // Distance max pour chercher un reset
            float minHeight = 200.0f,         // Hauteur minimum pour chercher
            float proximityWeight = 1.0f,     // Récompense pour se rapprocher
            float positionWeight = 2.0f,      // Récompense pour être sous la balle
            float orientationWeight = 1.5f,   // Récompense pour bonne orientation
            bool debug = false
        ) : maxSeekDistance(maxSeekDistance), minHeight(minHeight), proximityWeight(proximityWeight),
            positionWeight(positionWeight), orientationWeight(orientationWeight), debug(debug) {
        }

        virtual void Reset(const GameState& initialState) override {
            lastHasFlip.clear();
            lastDistanceToBall.clear();
            for (const auto& player : initialState.players) {
                lastHasFlip[player.carId] = player.HasFlipOrJump();
                Vec playerPos = Vec(player.pos.x, player.pos.y, player.pos.z);
                Vec ballPos = Vec(initialState.ball.pos.x, initialState.ball.pos.y, initialState.ball.pos.z);
                lastDistanceToBall[player.carId] = (playerPos - ballPos).Length();
            }
        }

        virtual float GetReward(const Player& player, const GameState& state, bool isFinal) override {
            if (!state.prev) return 0.0f;

            uint32_t carId = player.carId;
            Vec playerPos = Vec(player.pos.x, player.pos.y, player.pos.z);
            Vec ballPos = Vec(state.ball.pos.x, state.ball.pos.y, state.ball.pos.z);

            // 1. Le bot doit être en l'air et ne pas avoir de flip
            if (player.isOnGround || player.HasFlipOrJump()) {
                lastHasFlip[carId] = player.HasFlipOrJump();
                lastDistanceToBall[carId] = (playerPos - ballPos).Length();
                return 0.0f;
            }

            // 2. Hauteur minimum
            if (playerPos.z < minHeight) {
                lastHasFlip[carId] = player.HasFlipOrJump();
                lastDistanceToBall[carId] = (playerPos - ballPos).Length();
                return 0.0f;
            }

            float distanceToBall = (playerPos - ballPos).Length();

            // 3. Distance maximum pour chercher un reset
            if (distanceToBall > maxSeekDistance) {
                lastHasFlip[carId] = player.HasFlipOrJump();
                lastDistanceToBall[carId] = distanceToBall;
                return 0.0f;
            }

            float reward = 0.0f;

            // 4. RÉCOMPENSE PROXIMITÉ : Se rapprocher de la balle
            float lastDist = lastDistanceToBall.find(carId) != lastDistanceToBall.end() ?
                lastDistanceToBall[carId] : distanceToBall;

            if (distanceToBall < lastDist) {
                float proximityBonus = (lastDist - distanceToBall) / 100.0f; // Plus on se rapproche, plus c'est récompensé
                reward += proximityBonus * proximityWeight;

                if (debug) {
                    std::cout << "[FlipResetNormal] Proximity: " << proximityBonus * proximityWeight
                        << " (dist: " << lastDist << " -> " << distanceToBall << ")" << std::endl;
                }
            }

            // 5. RÉCOMPENSE POSITION : Être PILE EN DESSOUS de la balle (physique réaliste)
            float ballBottomSurface = ballPos.z - CommonValues::BALL_RADIUS; // Surface basse de la balle
            float carTopHeight = playerPos.z + 35.0f; // Dessus de la voiture (environ 35 unités)
            float realHeightDiff = ballBottomSurface - carTopHeight; // Distance réelle pour contact

            if (realHeightDiff > -50.0f && realHeightDiff < 20.0f) { // Zone de contact possible
                // PLUS PROCHE DE LA SURFACE = MIEUX ! 
                float positionBonus = RS_MIN(1.0f, (60.0f - (realHeightDiff + 10.0f)) / 60.0f);

                // BONUS pour être EXACTEMENT en dessous (distance horizontale faible)
                Vec ballPos2D = Vec(ballPos.x, ballPos.y, 0); // Position balle en 2D (sans Z)
                Vec playerPos2D = Vec(playerPos.x, playerPos.y, 0); // Position joueur en 2D
                float horizontalDist = (ballPos2D - playerPos2D).Length();

                if (horizontalDist < 50.0f) { // Très proche horizontalement = pile en dessous
                    float alignmentBonus = 1.0f - (horizontalDist / 100.0f); // Plus proche = plus de bonus
                    positionBonus *= (1.0f + alignmentBonus); // Bonus jusqu'à x2

                    if (debug) {
                        std::cout << "[FlipResetNormal] PILE EN DESSOUS BONUS: x" << (1.0f + alignmentBonus)
                            << " (dist2D: " << horizontalDist << ")" << std::endl;
                    }
                }

                reward += positionBonus * positionWeight;

                if (debug) {
                    std::cout << "[FlipResetNormal] Position: " << positionBonus * positionWeight
                        << " (realHeight: " << realHeightDiff << ")" << std::endl;
                }
            }

            // 6. RÉCOMPENSE ORIENTATION : Dessous de la voiture vers la balle (STRICT pour vrai reset)
            Vec carUp = player.rotMat.up;
            Vec carToBall = (ballPos - playerPos).Normalized();

            // Le dessous de la voiture doit pointer vers la balle (carUp inversé)
            Vec carBottom = carUp * -1.0f;
            float orientationDot = carBottom.Dot(carToBall);

            // PLUS STRICT : doit être vraiment à l'envers (0.7 au lieu de 0.5)
            if (orientationDot > 0.7f) { // Orientation stricte pour VRAI reset
                float orientationBonus = orientationDot * orientationDot; // Quadratique pour récompenser plus les bonnes orientations
                reward += orientationBonus * orientationWeight;

                if (debug) {
                    std::cout << "[FlipResetNormal] VRAI RESET Orientation: " << orientationBonus * orientationWeight
                        << " (dot: " << orientationDot << ")" << std::endl;
                }
            }

            // BONUS pour bonne orientation du nez (vers l'avant de la balle, pas vers la balle)
            Vec carForward = player.rotMat.forward;
            Vec ballVelocity = Vec(state.ball.vel.x, state.ball.vel.y, state.ball.vel.z);

            // Si la balle bouge, récompenser le nez qui suit la direction de la balle
            if (ballVelocity.Length() > 100.0f) { // Balle en mouvement
                Vec ballDirection = ballVelocity.Normalized();
                float forwardDot = carForward.Dot(ballDirection);

                if (forwardDot > 0.3f && orientationDot > 0.7f) { // Nez suit la balle ET roues vers balle
                    float alignmentBonus = forwardDot * 0.5f;
                    reward += alignmentBonus;

                    if (debug) {
                        std::cout << "[FlipResetNormal] NEZ ALIGNMENT BONUS: +" << alignmentBonus
                            << " (forward: " << forwardDot << ")" << std::endl;
                    }
                }
            }

            // 7. BONUS SPÉCIAL : Très proche et bien orienté pour VRAI flip reset (plus strict)
            if (distanceToBall < 150.0f && realHeightDiff > -5.0f && realHeightDiff < 30.0f && orientationDot > 0.85f) {
                reward += 5.0f; // GROS bonus pour la position parfaite de VRAI reset

                if (debug) {
                    std::cout << "[FlipResetNormal] VRAI FLIP RESET BONUS: +5.0f ! (dot: " << orientationDot << ")" << std::endl;
                }
            }

            // Mise à jour des états précédents
            lastHasFlip[carId] = player.HasFlipOrJump();
            lastDistanceToBall[carId] = distanceToBall;

            if (debug && reward > 0.0f) {
                std::cout << "[FlipResetNormal] TOTAL: " << reward << std::endl;
            }

            return reward;
        }
    };

    // ===============================================
    // SATURN FLIP RESET REWARD - Version C++ Optimisée
    // Système complet : Continu + Événement + Shot
    // Avec physique précise et orientation 224°
    // ===============================================
    class AdvancedFlipResetReward : public Reward {
    private:
        struct PlayerState {
            bool lastHasFlipped = false; // true = avait un flip disponible
            bool lastOnGround = false;
            int resetObtainedTime = -1;
            bool hasJumpedSinceReset = false;
            int airTimeSinceReset = 0;
            bool lastBallTouch = false;
            Vec lastBallVel = Vec(0, 0, 0);
            bool flipUsedAfterReset = false;
            int flipUsedTime = 0;
        };

        float continuousWeight;
        float eventWeight;
        float shotWeight;
        float minResetHeight;
        float maxResetDistance;
        float minShotForce;
        float weight;
        bool debug;

        std::unordered_map<int, PlayerState> playerStates;

        // CONSTANTES PHYSIQUES (comme FlipResetNormal)
        static constexpr float BALL_RADIUS = 93.15f;
        static constexpr float CAR_HEIGHT = 36.0f;

    public:
        AdvancedFlipResetReward(float continuousWeight = 1.0f,
            float eventWeight = 2.0f,
            float shotWeight = 3.0f,
            float minResetHeight = 400.0f,
            float maxResetDistance = 200.0f,
            float minShotForce = 500.0f,
            float weight = 1.0f,
            bool debug = false)
            : continuousWeight(continuousWeight), eventWeight(eventWeight), shotWeight(shotWeight),
            minResetHeight(minResetHeight), maxResetDistance(maxResetDistance),
            minShotForce(minShotForce), weight(weight), debug(debug) {
        }

        void Reset(const GameState& initialState) override {
            playerStates.clear();
            for (int i = 0; i < initialState.players.size(); i++) {
                const Player& player = initialState.players[i];
                PlayerState& state = playerStates[player.carId];
                state.lastHasFlipped = player.HasFlipOrJump();
                state.lastOnGround = player.isOnGround;
                state.resetObtainedTime = -1;
                state.hasJumpedSinceReset = false;
                state.airTimeSinceReset = 0;
                state.lastBallTouch = player.ballTouchedStep;
                state.lastBallVel = initialState.ball.vel;
                state.flipUsedAfterReset = false;
                state.flipUsedTime = 0;
            }
        }

        // PHYSIQUE AMÉLIORÉE : Proximité pour flip reset avec rayon de balle
        float GetResetProximity(const Player& player, const GameState& state) {
            // Pour chercher un flip reset, on doit être en l'air SANS flip disponible
            if (player.HasFlipOrJump() || player.isOnGround) {
                return 0.0f;
            }

            Vec playerPos = player.pos;
            Vec ballPos = state.ball.pos;

            // Vérification hauteur minimale
            if (playerPos.z < minResetHeight) {
                return 0.0f;
            }

            // PHYSIQUE PRÉCISE : Distance réelle entre surfaces
            float horizontalDist = sqrtf(powf(ballPos.x - playerPos.x, 2) + powf(ballPos.y - playerPos.y, 2));
            float realHeightDiff = (ballPos.z - BALL_RADIUS) - (playerPos.z + CAR_HEIGHT);
            float realDistance = sqrtf(horizontalDist * horizontalDist + realHeightDiff * realHeightDiff);

            if (realDistance > maxResetDistance) {
                return 0.0f;
            }

            // Vitesse relative vers la balle
            Vec playerVel = player.vel;
            Vec ballVel = state.ball.vel;
            Vec relativeVel = playerVel - ballVel;
            Vec toBall = ballPos - playerPos;
            float toBallLength = toBall.Length();

            if (toBallLength <= 0) {
                return 0.0f;
            }

            Vec toBallNorm = toBall / toBallLength;
            float velTowardsBall = relativeVel.Dot(toBallNorm);

            if (velTowardsBall <= 0) {
                return 0.0f;
            }

            float velTowardsBallNorm = RS_MAX(0.0f, RS_MIN(velTowardsBall / 1000.0f, 1.0f));
            float distanceNorm = RS_MAX(0.0f, 1.0f - realDistance / maxResetDistance);

            // ORIENTATION 224° OPTIMISÉE (selon ton image)
            Vec carUp = player.rotMat.up;
            Vec carForward = player.rotMat.forward;

            // Calculer pitch actuel (plus précis)
            float pitchRadians = atan2(-carUp.z, sqrtf(carForward.x * carForward.x + carForward.y * carForward.y));
            float currentPitch = pitchRadians * 180.0f / M_PI;

            // Normaliser pour angles positifs
            if (currentPitch < 0) {
                currentPitch += 360.0f;
            }

            float targetPitch = 224.0f; // Angle optimal selon ton image
            float angleDiff = fabsf(currentPitch - targetPitch);
            if (angleDiff > 180.0f) {
                angleDiff = 360.0f - angleDiff;
            }

            // Système de bonus progressif pour 224°
            float orientationBonus = 0.1f; // Base très faible

            if (angleDiff < 50.0f) { // Zone d'apprentissage élargie
                float proximityFactor = RS_MAX(0.0f, (50.0f - angleDiff) / 50.0f);
                orientationBonus = 0.1f + 0.6f * proximityFactor; // Entre 0.1 et 0.7

                // Bonus de précision pour être très proche de 224°
                if (angleDiff < 20.0f) {
                    float precisionFactor = RS_MAX(0.0f, (20.0f - angleDiff) / 20.0f);
                    orientationBonus += 0.4f * precisionFactor; // Jusqu'à +0.4

                    // Super bonus pour orientation parfaite
                    if (angleDiff < 10.0f) {
                        orientationBonus += 0.3f; // Bonus max total = 1.4
                        if (debug) {
                            std::cout << "[Saturn224°] ORIENTATION PARFAITE! Angle: " << currentPitch
                                << "°, Diff: " << angleDiff << "°" << std::endl;
                        }
                    }
                }
            }

            // Encourager le progrès vers se retourner (même si pas encore à 224°)
            if (carUp.z > -0.2f) { // Pas encore assez retourné
                float flipProgress = RS_MAX(0.0f, (1.0f - carUp.z) / 1.2f);
                float baseProgressBonus = 0.2f * flipProgress;
                orientationBonus = RS_MAX(orientationBonus, baseProgressBonus);
            }

            return RS_MIN(velTowardsBallNorm, distanceNorm) * orientationBonus;
        }

        // DÉTECTION ÉVÉNEMENT : Flip reset obtenu
        bool DetectResetEvent(const Player& player, const GameState& state) {
            int carId = player.carId;
            if (playerStates.find(carId) == playerStates.end()) {
                return false;
            }

            PlayerState& playerState = playerStates[carId];
            // Flip reset obtenu = on n'avait pas de flip ET maintenant on en a un
            bool gotFlip = !playerState.lastHasFlipped && player.HasFlipOrJump();

            if (!gotFlip) {
                return false;
            }

            // Vérifications physiques précises
            Vec playerPos = player.pos;
            Vec ballPos = state.ball.pos;
            float horizontalDist = sqrtf(powf(ballPos.x - playerPos.x, 2) + powf(ballPos.y - playerPos.y, 2));
            float realHeightDiff = (ballPos.z - BALL_RADIUS) - (playerPos.z + CAR_HEIGHT);
            float realDistance = sqrtf(horizontalDist * horizontalDist + realHeightDiff * realHeightDiff);

            if (realDistance > maxResetDistance) {
                return false;
            }

            bool ballClose = realDistance < maxResetDistance * 0.8f;
            bool sufficientHeight = playerPos.z > minResetHeight;

            Vec carUp = player.rotMat.up;
            bool upsideDown = carUp.z < -0.3f;

            return ballClose && sufficientHeight && upsideDown;
        }

        // DÉTECTION SHOT : Flip utilisé après reset
        bool DetectResetShot(const Player& player) {
            int carId = player.carId;
            if (playerStates.find(carId) == playerStates.end()) {
                return false;
            }

            PlayerState& playerState = playerStates[carId];

            bool hasReset = (!player.isOnGround &&
                !playerState.hasJumpedSinceReset &&
                playerState.resetObtainedTime >= 0);

            // Flip utilisé = on avait un flip ET maintenant on n'en a plus
            bool usedFlip = playerState.lastHasFlipped && !player.HasFlipOrJump();

            return hasReset && usedFlip;
        }

        float GetReward(const Player& player, const GameState& state, bool isFinal) override {
            int carId = player.carId;

            // Initialiser état si nécessaire
            if (playerStates.find(carId) == playerStates.end()) {
                PlayerState& newState = playerStates[carId];
                newState.lastHasFlipped = player.HasFlipOrJump();
                newState.lastOnGround = player.isOnGround;
                newState.resetObtainedTime = -1;
                newState.hasJumpedSinceReset = false;
                newState.airTimeSinceReset = 0;
                newState.lastBallTouch = player.ballTouchedStep;
                newState.lastBallVel = state.ball.vel;
                newState.flipUsedAfterReset = false;
                newState.flipUsedTime = 0;
            }

            PlayerState& playerState = playerStates[carId];
            float reward = 0.0f;

            // 1. RÉCOMPENSE CONTINUE : Proximité pour flip reset
            float resetProximity = GetResetProximity(player, state);
            if (resetProximity > 0) {
                float continuousReward = resetProximity * continuousWeight;
                reward += continuousReward;
            }

            // 2. ÉVÉNEMENT : Flip reset obtenu
            if (DetectResetEvent(player, state)) {
                float eventReward = eventWeight * 2.0f;
                reward += eventReward;

                if (debug) {
                    Vec playerPos = player.pos;
                    Vec ballPos = state.ball.pos;
                    float dist = (ballPos - playerPos).Length();
                    std::cout << "[SaturnFlipReset] RESET OBTENU! Joueur " << carId
                        << " - Hauteur: " << playerPos.z << ", Distance balle: " << dist << std::endl;
                }

                playerState.resetObtainedTime = 0;
                playerState.hasJumpedSinceReset = false;
            }

            // 3. SHOT : Flip utilisé après reset
            if (DetectResetShot(player)) {
                if (debug) {
                    std::cout << "[SaturnFlipReset] FLIP UTILISÉ! Joueur " << carId << " - Reset utilisé" << std::endl;
                }

                playerState.flipUsedAfterReset = true;
                playerState.flipUsedTime = 0;
            }

            // 4. SUIVI POST-FLIP : Touches de balle et pénalités
            if (playerState.flipUsedAfterReset) {
                playerState.flipUsedTime++;

                // Vérifier orientation vers la balle lors du flip
                Vec carForward = player.rotMat.forward;
                Vec toBall = state.ball.pos - player.pos;
                float orientationPenalty = 0.0f;

                float toBallLength = toBall.Length();
                if (toBallLength > 0) {
                    Vec toBallNorm = toBall / toBallLength;
                    float facingBall = carForward.Dot(toBallNorm);

                    if (facingBall < 0.3f) {
                        orientationPenalty = -0.1f;
                        if (debug) {
                            std::cout << "[SaturnFlipReset] ORIENTATION MAUVAISE! Joueur " << carId
                                << " - Facing: " << facingBall << std::endl;
                        }
                    }

                    // Détection backflip (pénalité forte)
                    // PITCH négatif = backflip, pas air roll !
                    // Détection via vélocité angulaire : backflip = rotation pitch négative
                    Vec angularVel = player.angVel;
                    float pitchRotation = carForward.Dot(angularVel); // Rotation autour de l'axe pitch

                    if (pitchRotation < -2.0f && facingBall < 0.5f) { // Backflip détecté
                        float backflipPenalty = -5.0f;
                        orientationPenalty += backflipPenalty;
                        if (debug) {
                            std::cout << "[SaturnFlipReset] BACKFLIP DÉTECTÉ! Joueur " << carId
                                << " - PitchRot: " << pitchRotation << ", Facing: " << facingBall << std::endl;
                        }
                    }
                }

                // Touche de balle après flip
                if (player.ballTouchedStep) {
                    Vec ballVelChange = state.ball.vel - playerState.lastBallVel;
                    float ballVelChangeLength = ballVelChange.Length();
                    float touchBonus = 5.0f;

                    if (ballVelChangeLength > minShotForce) {
                        float forceBonus = RS_MIN(ballVelChangeLength / 400.0f, 8.0f);
                        touchBonus += forceBonus;
                    }

                    touchBonus += 3.0f; // Bonus supplémentaire

                    if (debug) {
                        std::cout << "[SaturnFlipReset] TOUCHE APRÈS RESET! Joueur " << carId
                            << " - Force: " << ballVelChangeLength << ", Touch bonus: " << touchBonus
                            << ", Orientation penalty: " << orientationPenalty << std::endl;
                    }

                    reward += touchBonus + orientationPenalty;
                }
                else {
                    // Pénalité réduite si pas de touche après reset
                    reward -= 1.0f;
                    if (debug && playerState.flipUsedTime % 30 == 0) {
                        std::cout << "[SaturnFlipReset] PAS DE TOUCHE! Joueur " << carId
                            << " - Reset inutile (temps: " << playerState.flipUsedTime << ")" << std::endl;
                    }
                }

                // Reset après un certain temps ou si on touche le sol
                if (playerState.flipUsedTime > 120 || player.isOnGround) { // 2 secondes max
                    playerState.flipUsedAfterReset = false;
                    playerState.flipUsedTime = 0;
                    playerState.resetObtainedTime = -1;
                    playerState.hasJumpedSinceReset = false;
                }
            }

            // Mise à jour des timers
            if (playerState.resetObtainedTime >= 0) {
                playerState.resetObtainedTime++;
            }

            if (player.isOnGround) {
                playerState.hasJumpedSinceReset = true;
                playerState.resetObtainedTime = -1;
            }

            // Mise à jour état précédent
            playerState.lastHasFlipped = player.HasFlipOrJump();
            playerState.lastOnGround = player.isOnGround;
            playerState.lastBallTouch = player.ballTouchedStep;
            playerState.lastBallVel = state.ball.vel;

            return reward * weight;
        }
    };

    class ContinuousFlipResetReward : public Reward {
    public:
        float minHeight;
        float maxDist;

        ContinuousFlipResetReward(float minHeight = 150.f, float maxDist = 200.f) :
            minHeight(minHeight), maxDist(maxDist) {
        }

        virtual float GetReward(const Player& player, const GameState& state, bool isFinal) override {
            if (player.isOnGround || player.HasFlipOrJump()) {
                return 0.f;
            }

            if (player.pos.z < minHeight) {
                return 0.f;
            }

            if (player.rotMat.up.z > 0.3f) {
                return 0.f;
            }

            float distToBall = (player.pos - state.ball.pos).Length();
            if (distToBall > maxDist) {
                return 0.f;
            }

            Vec dirToBall = (state.ball.pos - player.pos).Normalized();
            Vec relVel = player.vel - state.ball.vel;
            float approachSpeed = relVel.Dot(dirToBall);

            bool isClose = distToBall < 150.0f;
            bool isGoingUnder = (player.pos.z < state.ball.pos.z - 50.0f);

            if (approachSpeed <= 0 && !isClose && !isGoingUnder) {
                return 0.f;
            }

            float normSpeed = RS_CLAMP(approachSpeed / CommonValues::CAR_MAX_SPEED, 0.f, 1.f);
            float normAlign = ((-player.rotMat.up).Dot(dirToBall) + 1.f) / 2.f;
            float normDist = 1.f - RS_CLAMP(distToBall / maxDist, 0.f, 1.f);

            float flipProgress = RS_MAX(0.f, (0.3f - player.rotMat.up.z) / 1.3f);

            float underBallBonus = 0.f;
            if (player.pos.z < state.ball.pos.z - 30.f) {
                float heightDiff = (state.ball.pos.z - player.pos.z) / 200.f;
                underBallBonus = RS_CLAMP(heightDiff, 0.f, 1.f) * 0.3f;
            }

            float passingBonus = 0.f;
            if (isClose && player.pos.z < state.ball.pos.z) {
                Vec ballPos2D = Vec(state.ball.pos.x, state.ball.pos.y, 0);
                Vec playerPos2D = Vec(player.pos.x, player.pos.y, 0);
                Vec playerVel2D = Vec(player.vel.x, player.vel.y, 0);

                Vec passingDir = (ballPos2D - playerPos2D).Normalized();
                float passingSpeed = playerVel2D.Dot(passingDir);
                if (passingSpeed > 0) {
                    passingBonus = RS_CLAMP(passingSpeed / 1000.f, 0.f, 0.2f);
                }
            }

            return (normSpeed * 0.2f + normAlign * 0.3f + normDist * 0.15f +
                flipProgress * 0.1f + underBallBonus + passingBonus);
        }
    };


    class FrenchFlipResetEventReward : public Reward {
    public:
        float minHeight;
        float maxDist;
        float minUpZ;

        FrenchFlipResetEventReward(float minHeight = 150.f, float maxDist = 50.f, float minUpZ = -0.5f) :
            minHeight(minHeight), maxDist(maxDist), minUpZ(minUpZ) {
        }

        virtual float GetReward(const Player& player, const GameState& state, bool isFinal) override {
            if (!player.prev) return 0.f;

            bool gotReset = !player.prev->isOnGround && player.HasFlipOrJump() && !player.prev->HasFlipOrJump();

            if (gotReset) {
                float distToBall = (player.pos - state.ball.pos).Length();

                // CONDITIONS DE BASE
                if (player.pos.z > minHeight && player.rotMat.up.z < minUpZ && distToBall < maxDist) {

                    // BONUS BASÉ SUR LA QUALITÉ DU RESET
                    float baseReward = 3.0f; // Récompense de base plus élevée

                    // Bonus pour être très proche de la balle (contact quasi-parfait)
                    if (distToBall < 30.f) {
                        baseReward += 3.0f; // Gros bonus pour contact parfait
                    }
                    else if (distToBall < 50.f) {
                        baseReward += 1.5f; // Bonus moyen pour contact correct
                    }

                    // Bonus pour être bien orienté (plus retourné = mieux)
                    if (player.rotMat.up.z < -0.5f) {
                        baseReward += 1.5f;
                    }

                    // Bonus pour être exactement sous la balle
                    if (player.pos.z < state.ball.pos.z - 30.f) {
                        baseReward += 1.0f;
                    }

                    return baseReward;
                }
            }

            return 0.f;
        }
    };


    class FrenchResetShotReward : public Reward {
    private:

        std::map<uint32_t, uint64_t> _tickCountWhenResetObtained;

    public:
        FrenchResetShotReward() {}

        virtual void Reset(const GameState& initial_state) override {
            _tickCountWhenResetObtained.clear();
        }

        virtual void PreStep(const GameState& state) override {
            if (!state.lastArena) return;
            for (const auto& player : state.players) {
                if (!player.prev) continue;

                bool gotReset = !player.prev->isOnGround && player.HasFlipOrJump() && !player.prev->HasFlipOrJump();
                if (gotReset) {
                    _tickCountWhenResetObtained[player.carId] = state.lastArena->tickCount;
                }
            }
        }

        virtual float GetReward(const Player& player, const GameState& state, bool isFinal) override {
            if (!player.prev || !state.prev) return 0.f;

            auto it = _tickCountWhenResetObtained.find(player.carId);
            if (it == _tickCountWhenResetObtained.end()) {
                return 0.f;
            }

            bool flipWasUsedForTouch = player.ballTouchedStep &&
                !player.isOnGround &&
                !player.hasJumped &&
                player.prev->HasFlipOrJump() &&
                !player.HasFlipOrJump();

            if (flipWasUsedForTouch) {

                float hitForce = (state.ball.vel - state.prev->ball.vel).Length();
                float ballSpeed = state.ball.vel.Length();
                float baseReward = (hitForce + ballSpeed) / (CommonValues::CAR_MAX_SPEED + CommonValues::BALL_MAX_SPEED);

                uint64_t ticksSinceReset = state.lastArena->tickCount - it->second;
                float timeSinceReset = ticksSinceReset * CommonValues::TICK_TIME;
                float timeBonus = 1.f + std::log1p(timeSinceReset);

                _tickCountWhenResetObtained.erase(it);

                return baseReward * timeBonus;
            }

            if (player.isOnGround) {
                _tickCountWhenResetObtained.erase(it);
            }

            return 0.f;
        }
    };

    class Speedflipkickoffv3 : public Reward {
    private:
        static constexpr float TICK_SKIP = 4.0f;
        static constexpr float DT = (1.0f / 120.0f) * TICK_SKIP;  // 0.0333 seconds

        struct PlayerState {
            bool kickoffDetected;
            bool wasOnGround;
            bool flipDetected;
            bool speedflipCompleted;

            Vec kickoffStartPos;
            float timeSinceKickoff;
            float timeWhenFlipped;
            float speedBeforeFlip;
            float speedAfterFlip;
            Vec lastAngVel;

            PlayerState() : kickoffDetected(false), wasOnGround(true), flipDetected(false),
                speedflipCompleted(false), kickoffStartPos(0, 0, 0),
                timeSinceKickoff(0.0f), timeWhenFlipped(0.0f),
                speedBeforeFlip(0.0f), speedAfterFlip(0.0f), lastAngVel(0, 0, 0) {
            }
        };

        std::unordered_map<uint32_t, PlayerState> playerStates;

    public:
        float kickoffDetectionRadius;
        float maxFlipDelay;
        float maxTouchTime;
        float minSpeedThreshold;
        float minFlipAngularSpeed;
        bool debug;

        Speedflipkickoffv3(
            float kickoffDetectionRadius = 300.0f,
            float maxFlipDelay = 0.8f,
            float maxTouchTime = 2.5f,
            float minSpeedThreshold = 1000.0f,
            float minFlipAngularSpeed = 3.0f,
            bool debug = false
        ) : kickoffDetectionRadius(kickoffDetectionRadius), maxFlipDelay(maxFlipDelay),
            maxTouchTime(maxTouchTime), minSpeedThreshold(minSpeedThreshold),
            minFlipAngularSpeed(minFlipAngularSpeed), debug(debug) {
        }

        virtual void Reset(const GameState& initialState) override {
            playerStates.clear();
            for (const auto& player : initialState.players) {
                PlayerState state;
                state.wasOnGround = player.isOnGround;
                state.kickoffStartPos = player.pos;
                playerStates[player.carId] = state;
            }
        }

        bool IsKickoffPosition(const Vec& playerPos, const Vec& ballPos) {
            // Check if ball is at center
            if (ballPos.Length() > 200.0f) {
                return false;
            }

            // Standard kickoff spawn positions
            Vec kickoffPositions[] = {
                Vec(0, -4608, 0),      // Back center
                Vec(0, 4608, 0),       // Back center (orange)
                Vec(-2048, -2560, 0),  // Diagonal left
                Vec(2048, -2560, 0),   // Diagonal right
                Vec(-2048, 2560, 0),   // Diagonal left (orange)
                Vec(2048, 2560, 0),    // Diagonal right (orange)
                Vec(-256, -3840, 0),   // Off-center left
                Vec(256, -3840, 0),    // Off-center right
                Vec(-256, 3840, 0),    // Off-center left (orange)
                Vec(256, 3840, 0)      // Off-center right (orange)
            };

            for (const auto& pos : kickoffPositions) {
                if ((playerPos - pos).Length() < kickoffDetectionRadius) {
                    return true;
                }
            }
            return false;
        }

        virtual float GetReward(const Player& player, const GameState& state, bool isFinal) override {
            uint32_t carId = player.carId;

            // Initialize if needed
            if (playerStates.find(carId) == playerStates.end()) {
                PlayerState newState;
                newState.wasOnGround = player.isOnGround;
                newState.kickoffStartPos = player.pos;
                newState.lastAngVel = player.angVel;
                playerStates[carId] = newState;
            }

            PlayerState& ps = playerStates[carId];
            float reward = 0.0f;

            // 1. Detect kickoff
            if (!ps.kickoffDetected) {
                if (IsKickoffPosition(player.pos, state.ball.pos)) {
                    ps.kickoffDetected = true;
                    ps.kickoffStartPos = player.pos;
                    ps.timeSinceKickoff = 0.0f;

                    if (debug) {
                        printf("[Kickoff] car_id=%d kickoff detected\n", carId);
                    }
                }
            }

            // Update timer
            if (ps.kickoffDetected) {
                ps.timeSinceKickoff += DT;
            }

            // 2. Detect flip (using angular velocity)
            if (ps.kickoffDetected && !ps.flipDetected && ps.timeSinceKickoff < maxFlipDelay) {

                // Detect flip by high angular velocity
                float angSpeed = player.angVel.Length();

                if (angSpeed > minFlipAngularSpeed && !player.isOnGround) {
                    // Check if it's a diagonal flip (speedflip characteristic)
                    // Speedflip has rotation around both pitch and roll axes
                    Vec angVelAbs = Vec(fabsf(player.angVel.x), fabsf(player.angVel.y), fabsf(player.angVel.z));

                    // Diagonal flip has both pitch (y) and roll (x) components
                    bool isDiagonalFlip = (angVelAbs.x > 1.5f && angVelAbs.y > 1.5f);

                    if (isDiagonalFlip) {
                        ps.flipDetected = true;
                        ps.timeWhenFlipped = ps.timeSinceKickoff;
                        ps.speedBeforeFlip = player.vel.Length();

                        if (debug) {
                            printf("[Kickoff] car_id=%d diagonal flip detected at %.2fs, speed=%.0f\n",
                                carId, ps.timeSinceKickoff, ps.speedBeforeFlip);
                        }
                    }
                }
            }

            // 3. Check speed after flip (flip cancel maintains/increases speed)
            if (ps.flipDetected && !ps.speedflipCompleted) {
                float currentSpeed = player.vel.Length();

                // Good speedflip maintains or increases speed
                if (currentSpeed >= ps.speedBeforeFlip * 0.9f) {
                    ps.speedAfterFlip = currentSpeed;
                }
            }

            // 4. Reward on ball touch
            if (ps.kickoffDetected && ps.flipDetected && !ps.speedflipCompleted) {
                if (player.ballTouchedStep && ps.timeSinceKickoff < maxTouchTime) {
                    float currentSpeed = player.vel.Length();

                    // Check if speed is maintained (flip cancel worked)
                    bool goodSpeed = currentSpeed >= minSpeedThreshold;

                    if (goodSpeed) {
                        ps.speedflipCompleted = true;

                        // Calculate bonuses
                        float timeBonus = RS_MAX(0.0f, 1.0f - (ps.timeSinceKickoff / maxTouchTime));
                        float speedBonus = RS_MIN(2.0f, currentSpeed / 2000.0f);
                        float flipTimingBonus = RS_MAX(0.0f, 1.0f - (ps.timeWhenFlipped / maxFlipDelay));

                        // Distance bonus (closer to ball at start = harder kickoff spot)
                        float startDist = (ps.kickoffStartPos - state.ball.pos).Length();
                        float distanceBonus = RS_MIN(1.0f, startDist / 3000.0f);

                        reward = 3.0f + timeBonus + speedBonus + flipTimingBonus + distanceBonus;

                        if (debug) {
                            printf("[Kickoff] car_id=%d SPEEDFLIP SUCCESS! time=%.2fs speed=%.0f reward=%.2f\n",
                                carId, ps.timeSinceKickoff, currentSpeed, reward);
                        }

                        // Reset for next kickoff
                        ps.kickoffDetected = false;
                        ps.flipDetected = false;
                    }
                }
            }

            // Reset if kickoff takes too long or goal scored
            if (ps.timeSinceKickoff > maxTouchTime * 2.0f || state.goalScored) {
                ps.kickoffDetected = false;
                ps.flipDetected = false;
                ps.speedflipCompleted = false;
                ps.timeSinceKickoff = 0.0f;
            }

            // Update state
            ps.wasOnGround = player.isOnGround;
            ps.lastAngVel = player.angVel;

            return reward;
        }
    };


    // Récompense pour Musty Flick
        // Basé sur MustyFlickReward Python
    class FrenchMustyFlickReward : public Reward {
    public:
        float min_ball_speed;
        float max_ball_speed;
        float angle_tolerance;

        FrenchMustyFlickReward(float min_ball_speed = 1200.0f, float max_ball_speed = 3500.0f, float angle_tolerance = 20.0f)
            : min_ball_speed(min_ball_speed), max_ball_speed(max_ball_speed), angle_tolerance(angle_tolerance) {
        }

        virtual float GetReward(const Player& player, const GameState& state, bool isFinal) override {
            if (!state.prev || !player.ballTouchedStep)
                return 0.0f;

            // Musty flick : le joueur doit être en l'air, en flip arrière
            if (player.isOnGround || !player.hasFlipped)
                return 0.0f;

            float ball_speed = state.ball.vel.Length();
            if (ball_speed < min_ball_speed)
                return 0.0f;

            // Vérifier direction du flip (doit être vers l'arrière par rapport au mouvement)
            Vec player_forward = player.rotMat.forward;
            Vec ball_direction = state.ball.vel.Normalized();

            // Pour un musty flick, la balle part généralement vers l'avant alors que le joueur flip vers l'arrière
            float forward_dot = player_forward.Dot(ball_direction);

            if (forward_dot > 0.3f) { // Balle va vers l'avant du joueur
                return RS_MIN(1.0f, ball_speed / max_ball_speed);
            }

            return 0.0f;
        }
    };

    // Récompense pour Speed Flip
    // Basé sur SpeedFlipReward Python
    class FrenchSpeedFlipReward : public Reward {
    private:
        float flip_start_time = 0.0f;
        bool flip_detected = false;

    public:
        float max_flip_duration;
        float min_speed_gain;

        FrenchSpeedFlipReward(float max_flip_duration = 1.5f, float min_speed_gain = 300.0f)
            : max_flip_duration(max_flip_duration), min_speed_gain(min_speed_gain) {
        }

        virtual void Reset(const GameState& initialState) override {
            flip_start_time = 0.0f;
            flip_detected = false;
        }

        virtual float GetReward(const Player& player, const GameState& state, bool isFinal) override {
            if (!player.prev)
                return 0.0f;

            // Détecter début de flip
            if (!flip_detected && player.isFlipping && !player.prev->isFlipping) {
                flip_detected = true;
                flip_start_time = 0.0f;
                return 0.0f;
            }

            if (flip_detected) {
                flip_start_time += state.deltaTime;

                // Fin du flip
                if (!player.isFlipping && player.prev->isFlipping) {
                    if (flip_start_time <= max_flip_duration) {
                        float speed_before = player.prev->vel.Length();
                        float speed_after = player.vel.Length();
                        float speed_gain = speed_after - speed_before;

                        if (speed_gain > min_speed_gain) {
                            flip_detected = false;
                            return RS_MIN(1.0f, speed_gain / 1000.0f);
                        }
                    }
                    flip_detected = false;
                }

                // Timeout
                if (flip_start_time > max_flip_duration) {
                    flip_detected = false;
                }
            }

            return 0.0f;
        }
    };

    // Récompense pour Speed Flip Kickoff
    // Basé sur SpeedFlipKickoffReward Python
    class FrenchSpeedFlipKickoffReward : public Reward {
    public:
        float kickoff_radius;
        float min_final_speed;

        FrenchSpeedFlipKickoffReward(float kickoff_radius = 200.0f, float min_final_speed = 1800.0f)
            : kickoff_radius(kickoff_radius), min_final_speed(min_final_speed) {
        }

        virtual float GetReward(const Player& player, const GameState& state, bool isFinal) override {
            // Vérifier que c'est un kickoff (balle au centre)
            if (state.ball.pos.Length() > kickoff_radius)
                return 0.0f;

            // Récompenser speed flip pendant kickoff
            if (player.hasFlipped && player.vel.Length() > min_final_speed) {
                return 1.0f;
            }

            return 0.0f;
        }
    };

    // Récompense pour Air Dribble avec Air Roll
    // Basé sur AirDribbleWithAirRollReward Python
    class AirDribbleWithAirRollReward : public Reward {
    public:
        float min_height;
        float min_angular_velocity;
        float max_distance_to_ball;

        AirDribbleWithAirRollReward(float min_height = 200.0f, float min_angular_velocity = 1.5f, float max_distance_to_ball = 300.0f)
            : min_height(min_height), min_angular_velocity(min_angular_velocity), max_distance_to_ball(max_distance_to_ball) {
        }

        virtual float GetReward(const Player& player, const GameState& state, bool isFinal) override {
            if (player.isOnGround || state.ball.pos.z < min_height)
                return 0.0f;

            float distance_to_ball = (player.pos - state.ball.pos).Length();
            if (distance_to_ball > max_distance_to_ball)
                return 0.0f;

            float angular_speed = player.angVel.Length();
            if (angular_speed < min_angular_velocity)
                return 0.0f;

            if (player.ballTouchedStep) {
                float height_factor = RS_MIN(1.0f, player.pos.z / 1000.0f);
                float angular_factor = RS_MIN(1.0f, angular_speed / CommonValues::CAR_MAX_ANG_VEL);
                return height_factor * angular_factor;
            }

            return 0.0f;
        }
    };

    // Récompense pour Wall Dribble Start
    // Basé sur WallDribbleStartReward Python
    class WallDribbleStartReward : public Reward {
    public:
        float wall_threshold;
        float min_ball_height;

        WallDribbleStartReward(float wall_threshold = 3500.0f, float min_ball_height = 200.0f)
            : wall_threshold(wall_threshold), min_ball_height(min_ball_height) {
        }

        virtual float GetReward(const Player& player, const GameState& state, bool isFinal) override {
            if (!player.ballTouchedStep)
                return 0.0f;

            // Vérifier que le joueur est sur le mur
            bool on_side_wall = fabsf(player.pos.x) > wall_threshold;
            bool on_back_wall = fabsf(player.pos.y) > 4000.0f;

            if (!on_side_wall && !on_back_wall)
                return 0.0f;

            // Vérifier que la balle est aussi en hauteur
            if (state.ball.pos.z < min_ball_height)
                return 0.0f;

            // Récompenser le contrôle de balle sur le mur
            float distance_to_ball = (player.pos - state.ball.pos).Length();
            if (distance_to_ball < 200.0f) {
                return 1.0f;
            }

            return 0.0f;
        }
    };

    // Récompense pour Ground to Air Dribble
    // Basé sur GroundToAirDribbleReward Python
    class GroundToAirDribbleReward : public Reward {
    private:
        bool ground_control_detected = false;
        float ground_control_time = 0.0f;

    public:
        float min_air_height;
        float max_ground_time;

        GroundToAirDribbleReward(float min_air_height = 200.0f, float max_ground_time = 3.0f)
            : min_air_height(min_air_height), max_ground_time(max_ground_time) {
        }

        virtual void Reset(const GameState& initialState) override {
            ground_control_detected = false;
            ground_control_time = 0.0f;
        }

        virtual float GetReward(const Player& player, const GameState& state, bool isFinal) override {
            if (!state.prev)
                return 0.0f;

            // Phase 1: Détecter contrôle au sol
            if (player.isOnGround && player.ballTouchedStep && state.ball.pos.z < 150.0f) {
                ground_control_detected = true;
                ground_control_time = 0.0f;
                return 0.0f;
            }

            if (ground_control_detected) {
                ground_control_time += state.deltaTime;

                // Phase 2: Transition vers l'air
                if (!player.isOnGround && state.ball.pos.z > min_air_height && player.ballTouchedStep) {
                    if (ground_control_time <= max_ground_time) {
                        ground_control_detected = false;
                        return 1.0f;
                    }
                }

                // Timeout
                if (ground_control_time > max_ground_time) {
                    ground_control_detected = false;
                }
            }

            return 0.0f;
        }
    };

    // Récompense pour Bounce Dribble
    // Basé sur SimpleBounceDribbleReward Python
    class SimpleBounceDribbleReward : public Reward {
    private:
        int bounce_count = 0;
        float last_bounce_time = 0.0f;

    public:
        float max_time_between_bounces;
        int min_bounces;

        SimpleBounceDribbleReward(float max_time_between_bounces = 2.0f, int min_bounces = 3)
            : max_time_between_bounces(max_time_between_bounces), min_bounces(min_bounces) {
        }

        virtual void Reset(const GameState& initialState) override {
            bounce_count = 0;
            last_bounce_time = 0.0f;
        }

        virtual float GetReward(const Player& player, const GameState& state, bool isFinal) override {
            if (!state.prev)
                return 0.0f;

            last_bounce_time += state.deltaTime;

            // Détecter bounce : balle touche le sol puis remonte
            if (state.prev->ball.pos.z <= CommonValues::BALL_RADIUS * 1.1f &&
                state.ball.pos.z > CommonValues::BALL_RADIUS * 1.1f &&
                state.ball.vel.z > 0) {

                if (last_bounce_time <= max_time_between_bounces) {
                    bounce_count++;
                    last_bounce_time = 0.0f;

                    if (bounce_count >= min_bounces) {
                        float reward = RS_MIN(1.0f, (float)bounce_count / 10.0f);
                        bounce_count = 0; // Reset après récompense
                        return reward;
                    }
                }
                else {
                    bounce_count = 1; // Reset count
                    last_bounce_time = 0.0f;
                }
            }

            // Reset si trop de temps sans bounce
            if (last_bounce_time > max_time_between_bounces) {
                bounce_count = 0;
            }

            return 0.0f;
        }
    };

    // Récompense pour Pass
    // Basé sur PassReward Python
    class PassReward : public Reward {
    public:
        float min_pass_speed;
        float teammate_proximity;

        PassReward(float min_pass_speed = 800.0f, float teammate_proximity = 1000.0f)
            : min_pass_speed(min_pass_speed), teammate_proximity(teammate_proximity) {
        }

        virtual float GetReward(const Player& player, const GameState& state, bool isFinal) override {
            if (!player.ballTouchedStep)
                return 0.0f;

            float ball_speed = state.ball.vel.Length();
            if (ball_speed < min_pass_speed)
                return 0.0f;

            // Chercher des coéquipiers proches de la trajectoire de la balle
            for (const auto& other_player : state.players) {
                if (other_player.team == player.team && other_player.index != player.index) {
                    Vec ball_to_teammate = other_player.pos - state.ball.pos;
                    Vec ball_direction = state.ball.vel.Normalized();

                    float distance_along_trajectory = ball_direction.Dot(ball_to_teammate);
                    if (distance_along_trajectory > 0 && distance_along_trajectory < 2000.0f) {
                        Vec closest_point = state.ball.pos + ball_direction * distance_along_trajectory;
                        float distance_to_trajectory = (other_player.pos - closest_point).Length();

                        if (distance_to_trajectory < teammate_proximity) {
                            return RS_MIN(1.0f, ball_speed / CommonValues::BALL_MAX_SPEED);
                        }
                    }
                }
            }

            return 0.0f;
        }
    };

    class TeamSpacingReward : public Reward {
    public:
        float minSpacing;

        TeamSpacingReward(float minSpacing = 1000.0f)
            : minSpacing(RS_CLAMP(minSpacing, 0.0000001f, FLT_MAX)) {
        }

        virtual float GetReward(const Player& player, const GameState& state, bool isFinal) override {
            float reward = 0.0f;

            // Check spacing with all teammates
            for (const auto& p : state.players) {
                // Same team, different player, neither demoed
                if (p.team == player.team &&
                    p.carId != player.carId &&
                    !player.isDemoed &&
                    !p.isDemoed) {

                    float separation = (player.pos - p.pos).Length();

                    // Penalize if too close
                    if (separation < minSpacing) {
                        reward -= 1.0f - (separation / minSpacing);
                    }
                }
            }

            return reward;
        }
    };

    class HansiFlickReward : public Reward {
    public:
        float ballCarryHeightMin;
        float ballCarryHeightMax;
        float carryDistanceThreshold;
        float stableVelocityDiffThreshold;
        float stableCarryBonus;
        float jumpPreparationBonus;
        float flickSuccessReward;
        float ballSpeedAfterFlickBonus;
        float opponentPressureDistance;
        float goalDirectionBonus;

        // State tracking
        std::map<int, std::map<std::string, float>> playerStates;
        float ballSpeedBeforeTouch;
        Vec previousBallVelocity;

        HansiFlickReward(
            float ballCarryHeightMin = CommonValues::BALL_RADIUS + 20,
            float ballCarryHeightMax = CommonValues::BALL_RADIUS + 150,
            float carryDistanceThreshold = 180.0f,
            float stableVelocityDiffThreshold = 200.0f,
            float stableCarryBonus = 2.0f,
            float jumpPreparationBonus = 5.0f,
            float flickSuccessReward = 20.0f,
            float ballSpeedAfterFlickBonus = 10.0f,
            float opponentPressureDistance = 400.0f,
            float goalDirectionBonus = 3.0f
        ) : ballCarryHeightMin(ballCarryHeightMin), ballCarryHeightMax(ballCarryHeightMax),
            carryDistanceThreshold(carryDistanceThreshold), stableVelocityDiffThreshold(stableVelocityDiffThreshold),
            stableCarryBonus(stableCarryBonus), jumpPreparationBonus(jumpPreparationBonus),
            flickSuccessReward(flickSuccessReward), ballSpeedAfterFlickBonus(ballSpeedAfterFlickBonus),
            opponentPressureDistance(opponentPressureDistance), goalDirectionBonus(goalDirectionBonus),
            ballSpeedBeforeTouch(0) {
        }

        virtual void Reset(const GameState& initialState) override {
            playerStates.clear();
            previousBallVelocity = initialState.ball.vel;
            ballSpeedBeforeTouch = 0;
        }

        bool IsCarryingBall(const Player& player, const GameState& state) {
            Vec ballPos = state.ball.pos;
            Vec carPos = player.pos;

            // Height check
            if (ballPos.z < ballCarryHeightMin || ballPos.z > ballCarryHeightMax) {
                return false;
            }

            // Distance check
            float distanceToBall = (ballPos - carPos).Length();
            if (distanceToBall > carryDistanceThreshold) {
                return false;
            }

            // Player should be on ground and ball above car
            return player.isOnGround && ballPos.z > carPos.z;
        }

        bool IsStableCarry(const Player& player, const GameState& state) {
            if (!IsCarryingBall(player, state)) {
                return false;
            }

            float velDiff = (player.vel - state.ball.vel).Length();
            return velDiff <= stableVelocityDiffThreshold;
        }

        float GetGoalDirectionReward(const Player& player, const GameState& state) {
            Vec ballVel = state.ball.vel;
            Vec goalCenter = (player.team == Team::BLUE) ?
                Vec(0, CommonValues::BACK_WALL_Y, 0) :
                Vec(0, -CommonValues::BACK_WALL_Y, 0);

            Vec ballToGoal = (goalCenter - state.ball.pos).Normalized();
            float velocityTowardsGoal = ballVel.Dot(ballToGoal);
            return RS_MAX(0.0f, velocityTowardsGoal / CommonValues::CAR_MAX_SPEED);
        }

        virtual float GetReward(const Player& player, const GameState& state, bool isFinal) override {
            float totalReward = 0.0f;

            bool isCarrying = IsCarryingBall(player, state);
            bool isStable = IsStableCarry(player, state);

            // Phase 1: Basic ball control
            if (isCarrying) {
                float carryReward = 0.5f;

                // Height optimization
                float ballHeight = state.ball.pos.z;
                float optimalHeight = (ballCarryHeightMin + ballCarryHeightMax) / 2;
                float heightDeviation = fabsf(ballHeight - optimalHeight);
                float maxDeviation = (ballCarryHeightMax - ballCarryHeightMin) / 2;
                float heightReward = 0.5f * (1.0f - heightDeviation / maxDeviation);
                carryReward += heightReward;

                totalReward += carryReward;
            }

            // Phase 2: Stable dribbling
            if (isStable) {
                float stableReward = stableCarryBonus;

                // Speed matching bonus
                float playerSpeed = player.vel.Length();
                float ballSpeed = state.ball.vel.Length();
                float speedSync = 1.0f - fabsf(playerSpeed - ballSpeed) / CommonValues::CAR_MAX_SPEED;
                stableReward += speedSync * 0.8f;

                totalReward += stableReward;
            }

            // Phase 4: Flick execution
            if (player.ballTouchedStep) {
                ballSpeedBeforeTouch = previousBallVelocity.Length();
                float ballSpeedAfter = state.ball.vel.Length();
                float speedIncrease = ballSpeedAfter - ballSpeedBeforeTouch;

                if (speedIncrease > 300) { // Minimum speed increase for flick
                    float flickReward = flickSuccessReward;

                    // Speed bonus
                    float speedBonus = RS_MIN(speedIncrease / 1000.0f, 3.0f) * ballSpeedAfterFlickBonus;
                    flickReward += speedBonus;

                    // Direction bonus
                    float goalDirectionReward = GetGoalDirectionReward(player, state);
                    flickReward += goalDirectionReward * goalDirectionBonus;

                    totalReward += flickReward;
                }
            }

            previousBallVelocity = state.ball.vel;
            return totalReward;
        }
    };

    // ===== PICKUP BOOST REWARD - SIMPLE ET INTELLIGENT =====
    class FrenchPickupBoostReward : public Reward {
    private:
        std::map<uint32_t, float> lastBoost;

    public:
        float highBoostThreshold;     // Seuil boost élevé (défaut: 80%)
        float wastePenalty;           // Pénalité gaspillage (défaut: 0.7)

        FrenchPickupBoostReward(
            float highBoostThreshold = 80.0f,
            float wastePenalty = 0.7f
        ) : highBoostThreshold(highBoostThreshold), wastePenalty(wastePenalty) {
        }

        virtual void Reset(const GameState& initialState) override {
            lastBoost.clear();
            for (const auto& player : initialState.players) {
                lastBoost[player.carId] = player.boost;
            }
        }

        virtual float GetReward(const Player& player, const GameState& state, bool isFinal) override {
            if (!state.prev) return 0.0f;

            uint32_t carId = player.carId;
            float prevBoost = lastBoost[carId];
            float currentBoost = player.boost;

            // Détecter le pickup
            if (currentBoost <= prevBoost) {
                lastBoost[carId] = currentBoost;
                return 0.0f;
            }

            float boostGained = currentBoost - prevBoost;
            float reward = boostGained / 100.0f; // Récompense de base

            // Pénalité si gaspillage (boost déjà élevé)
            if (prevBoost > highBoostThreshold) {
                reward *= (1.0f - wastePenalty);
            }

            lastBoost[carId] = currentBoost;
            return reward;
        }
    };

    // Player state tracking for SaveBoostReward
    struct SaveBoostPlayerState {
        float prevBoost = 0.0f;
        float prevSpeed = 0.0f;
        bool prevOnGround = false;
    };

    // Converted from rlgymppo Python implementation
    class FrenchSaveBoostReward : public Reward {
    public:
        float highBoostBonus;
        float lowBoostPenalty;
        float touchBonus;
        float wastePenalty;
        float pickupBonus;
        float accelBonus;
        float minHigh;
        float maxLow;
        float minTouch;
        float speedThreshold;
        float minAccel;
        bool debug;

        // Player state tracking by carId
        std::unordered_map<uint32_t, SaveBoostPlayerState> playerStates;

        FrenchSaveBoostReward(
            float highBoostBonus = 1.0f,
            float lowBoostPenalty = -1.0f,
            float touchBonus = 2.0f,
            float wastePenalty = -0.5f,
            float pickupBonus = 1.5f,
            float accelBonus = 1.0f,
            float minHigh = 0.8f,
            float maxLow = 0.1f,
            float minTouch = 0.5f,
            float speedThreshold = 2100.0f,
            float minAccel = 400.0f,
            bool debug = false
        ) : highBoostBonus(highBoostBonus),
            lowBoostPenalty(lowBoostPenalty),
            touchBonus(touchBonus),
            wastePenalty(wastePenalty),
            pickupBonus(pickupBonus),
            accelBonus(accelBonus),
            minHigh(minHigh),
            maxLow(maxLow),
            minTouch(minTouch),
            speedThreshold(speedThreshold),
            minAccel(minAccel),
            debug(debug) {
        }

        virtual void Reset(const GameState& initialState) override {
            playerStates.clear();
            for (const Player& player : initialState.players) {
                SaveBoostPlayerState state;
                state.prevBoost = player.boost / 100.0f; // Convert to 0-1 range like Python
                state.prevSpeed = player.vel.Length();
                state.prevOnGround = player.isOnGround;
                playerStates[player.carId] = state;
            }
        }

        virtual float GetReward(const Player& player, const GameState& state, bool isFinal) override {
            uint32_t carId = player.carId;

            // Initialize player state if not exists
            if (playerStates.find(carId) == playerStates.end()) {
                SaveBoostPlayerState newState;
                newState.prevBoost = player.boost / 100.0f;
                newState.prevSpeed = player.vel.Length();
                newState.prevOnGround = player.isOnGround;
                playerStates[carId] = newState;
            }

            SaveBoostPlayerState& playerState = playerStates[carId];
            float reward = 0.0f;
            float boost = player.boost / 100.0f; // Convert to 0-1 range like Python
            float speed = player.vel.Length();

            // High boost bonus
            if (boost > minHigh) {
                reward += highBoostBonus;
            }

            // Low boost penalty (only if not touching ball)
            if (boost < maxLow && !player.ballTouchedStep) {
                reward += lowBoostPenalty;
            }

            // Ball touch bonus (only if sufficient boost)
            if (player.ballTouchedStep && boost > minTouch) {
                reward += touchBonus;
            }

            // Boost waste penalty (boosting at high speed)
            bool isBoosting = player.prevAction.boost > 0.5f;
            if (isBoosting && speed > speedThreshold) {
                reward += wastePenalty;
            }

            // Large boost pickup bonus (from low to high)
            if ((boost - playerState.prevBoost) > 0.9f && playerState.prevBoost < 0.2f) {
                reward += pickupBonus;
            }

            // Acceleration bonus (boost used effectively for speed gain while airborne)
            if (boost < playerState.prevBoost &&
                (speed - playerState.prevSpeed) > minAccel &&
                !playerState.prevOnGround &&
                !player.isOnGround) {
                reward += accelBonus;
            }

            // Debug logging
            if (debug && reward != 0.0f) {
                RG_LOG("[SaveBoostReward] car_id=" << carId << " boost=" << boost << " speed=" << speed << " reward=" << reward);
            }

            // Update previous states
            playerState.prevBoost = boost;
            playerState.prevSpeed = speed;
            playerState.prevOnGround = player.isOnGround;

            return reward;
        }
    };

    class EnhancedControlReward : public Reward {
    private:
        struct PlayerState {
            bool lastBallTouch = false;
            Vec lastPos = Vec();
        };
        std::unordered_map<uint32_t, PlayerState> playerStates;

        // Parameters
        float baseReward;
        float minBallHeight;
        float maxBallHeight;
        float maxControlDistance;
        float minPlayerSpeed;
        bool debug;

    public:
        EnhancedControlReward(
            float baseReward = 0.5f,
            float minBallHeight = CommonValues::BALL_RADIUS + 30,
            float maxBallHeight = CommonValues::BALL_RADIUS + 150,
            float maxControlDistance = 150.0f,
            float minPlayerSpeed = 200.0f,
            bool debug = false
        ) : baseReward(baseReward), minBallHeight(minBallHeight), maxBallHeight(maxBallHeight),
            maxControlDistance(maxControlDistance), minPlayerSpeed(minPlayerSpeed), debug(debug) {
        }

        virtual void Reset(const GameState& initialState) override {
            playerStates.clear();
            for (const auto& player : initialState.players) {
                PlayerState& state = playerStates[player.carId];
                state.lastBallTouch = player.ballTouchedStep;
                state.lastPos = player.pos;
            }
        }

        virtual float GetReward(const Player& player, const GameState& state, bool isFinal) override {
            uint32_t carId = player.carId;

            // Initialize player state if needed
            if (playerStates.find(carId) == playerStates.end()) {
                PlayerState& newState = playerStates[carId];
                newState.lastBallTouch = player.ballTouchedStep;
                newState.lastPos = player.pos;
            }

            PlayerState& playerState = playerStates[carId];

            // Check if controlling ball
            if (!IsControllingBall(player, state)) {
                return 0.0f;
            }

            // Check if control is useful
            if (!IsControlUseful(player, state)) {
                return 0.0f;
            }

            // Check if player touched ball recently (active control)
            bool ballTouched = player.ballTouchedStep && !playerState.lastBallTouch;
            playerState.lastBallTouch = player.ballTouchedStep;

            if (!ballTouched) {
                return 0.0f;
            }

            // Base reward for useful control
            float reward = baseReward;

            if (debug) {
                std::cout << "[EnhancedControl] car_id=" << carId
                    << " CONTRÔLE UTILE! Reward: " << reward << std::endl;
            }

            // Update last position
            playerState.lastPos = player.pos;

            return reward;
        }

    private:
        bool IsControllingBall(const Player& player, const GameState& state) const {
            Vec carPos = player.pos;
            Vec ballPos = state.ball.pos;
            float ballHeight = ballPos.z;
            float carHeight = carPos.z;

            // Player must be below ball
            if (carHeight >= ballHeight) {
                return false;
            }

            // Ball at correct height
            if (ballHeight <= minBallHeight || ballHeight >= maxBallHeight) {
                return false;
            }

            // Close to ball (2D distance)
            Vec horizontalDiff(carPos.x - ballPos.x, carPos.y - ballPos.y, 0.0f);
            float distance2D = horizontalDiff.Length();
            if (distance2D > maxControlDistance) {
                return false;
            }

            // Player must be moving (avoid static farming)
            Vec playerVel2D(player.vel.x, player.vel.y, 0.0f);
            float playerSpeed = playerVel2D.Length();
            if (playerSpeed < minPlayerSpeed) {
                return false;
            }

            // Check field bounds
            if (fabsf(ballPos.x) >= 3946 || fabsf(ballPos.y) >= 4970) {
                return false;
            }

            return true;
        }

        bool IsControlUseful(const Player& player, const GameState& state) const {
            Vec ballPos = state.ball.pos;

            // Check if opponent is nearby (control under pressure = more useful)
            for (const Player& otherPlayer : state.players) {
                if (otherPlayer.team != player.team) {
                    Vec diff = otherPlayer.pos - ballPos;
                    if (diff.Length() < 300.0f) {
                        return true;
                    }
                }
            }

            // Check if ball is heading towards our goal (defensive control = useful)
            Vec goalPos;
            if (player.team == Team::BLUE) {
                goalPos = Vec(0, 5120, 0);  // Orange goal
            }
            else {
                goalPos = Vec(0, -5120, 0); // Blue goal
            }

            Vec ballToGoal = goalPos - ballPos;
            Vec ballVel = state.ball.vel;

            float ballToGoalLength = ballToGoal.Length();
            if (ballToGoalLength > 0) {
                Vec ballToGoalNorm = ballToGoal / ballToGoalLength;
                float ballVelTowardsGoal = ballVel.Dot(ballToGoalNorm);

                if (ballVelTowardsGoal > 500.0f) {  // Ball heading towards our goal
                    return true;
                }
            }

            return false;
        }
    };

    class PerfectFinishingReward : public Reward {
    private:
        float finishingZoneDistance;
        float criticalZoneDistance;
        float goalWidth;
        float goalHeight;

        float optimalShotPowerMin;
        float optimalShotPowerMax;
        float precisionAngleThreshold;

        float openNetBonus;
        float beatingKeeperBonus;
        float pressureShotBonus;
        float clinicalFinishBonus;
        float opportunityRecognitionBonus;
        float shotSelectionBonus;
        float composureBonus;
        float techniqueBonus;
        float placementBonus;
        float powerControlBonus;

        struct PlayerState {
            bool inFinishingZone = false;
            int opportunityStartTick = -1;
            int totalTicks = 0;
            float lastShotQuality = 0.0f;
            int consecutiveGoodShots = 0;
            bool beatenDefender = false;
            float pressureLevel = 0.0f;
            int shotOpportunitiesTaken = 0;
            int shotOpportunitiesMissed = 0;
            int finishingZoneTime = 0;
            float bestAngleAchieved = 0.0f;
            float composureRating = 1.0f;
        };

        std::unordered_map<int, PlayerState> playerStates;
        std::unordered_map<int, Vec> previousBallVel;
        std::unordered_map<int, int> previousScores;

        struct DefensiveAnalysis {
            bool openNet = true;
            Vec keeperPosition = Vec(0, 0, 0);
            float keeperDistanceFromGoal = 999999.0f;
            float pressureLevel = 0.0f;
            bool beatenDefender = false;
            bool clearShot = true;
        };

        Vec GetGoalCenter(int teamNum) const {
            return (teamNum == 0) ?
                CommonValues::ORANGE_GOAL_BACK :
                CommonValues::BLUE_GOAL_BACK;
        }

        float GetGoalLine(int teamNum) const {
            return (teamNum == 0) ? 5120.f : -5120.f;
        }

        bool IsInFinishingZone(const Player& player, const GameState& state) const {
            Vec goalCenter = GetGoalCenter(player.team == Team::BLUE ? 0 : 1);
            float distanceToGoal = (state.ball.pos - goalCenter).Length();
            return distanceToGoal <= finishingZoneDistance;
        }

        bool IsInCriticalZone(const Player& player, const GameState& state) const {
            Vec goalCenter = GetGoalCenter(player.team == Team::BLUE ? 0 : 1);
            float distanceToGoal = (state.ball.pos - goalCenter).Length();
            return distanceToGoal <= criticalZoneDistance;
        }

        float CalculateShotAngle(Vec ballPos, Vec goalCenter) const {
            Vec ballToGoal = goalCenter - ballPos;
            float horizontalDistance = Vec(ballToGoal.x, ballToGoal.y, 0).Length();
            if (horizontalDistance < 1e-6f) return 0.0f;
            // FIXED: Manual radians to degrees conversion
            return atan2f(abs(ballToGoal.x), abs(ballToGoal.y)) * 57.2957795f;
        }

        DefensiveAnalysis AnalyzeDefensiveSetup(const Player& player, const GameState& state) const {
            DefensiveAnalysis analysis;

            int playerTeam = (player.team == Team::BLUE) ? 0 : 1;
            Vec goalCenter = GetGoalCenter(playerTeam);
            float goalLine = GetGoalLine(playerTeam);
            Vec ballPos = state.ball.pos;

            const Player* closestDefender = nullptr;
            float minDefenderDistance = 999999.0f;

            for (const auto& opponent : state.players) {
                int oppTeam = (opponent.team == Team::BLUE) ? 0 : 1;
                if (oppTeam == playerTeam) continue;

                Vec oppPos = opponent.pos;
                float ballDistance = (oppPos - ballPos).Length();

                Vec ballToGoal = goalCenter - ballPos;
                Vec ballToOpponent = oppPos - ballPos;

                float projectionLength = ballToOpponent.Dot(ballToGoal) / ballToGoal.Length();

                if (projectionLength > 0 && projectionLength < ballToGoal.Length()) {
                    analysis.clearShot = false;
                }

                float goalDistance = abs(oppPos.y - goalLine);

                if (goalDistance < analysis.keeperDistanceFromGoal) {
                    analysis.keeperPosition = oppPos;
                    analysis.keeperDistanceFromGoal = goalDistance;
                    closestDefender = &opponent;
                }

                if (ballDistance < minDefenderDistance) {
                    minDefenderDistance = ballDistance;
                }

                if (ballDistance < 300.f) {
                    analysis.pressureLevel += 1.0f;
                }
                else if (ballDistance < 600.f) {
                    analysis.pressureLevel += 0.5f;
                }
            }

            analysis.openNet = analysis.keeperDistanceFromGoal > 400.f;

            if (closestDefender && minDefenderDistance > 200.f) {
                float defenderToGoal = (closestDefender->pos - goalCenter).Length();
                float ballToGoalDist = (ballPos - goalCenter).Length();
                if (ballToGoalDist < defenderToGoal) {
                    analysis.beatenDefender = true;
                }
            }

            return analysis;
        }

        float CalculateShotQuality(const Player& player, const GameState& state, Vec ballVelocity) const {
            int teamNum = (player.team == Team::BLUE) ? 0 : 1;
            Vec goalCenter = GetGoalCenter(teamNum);
            Vec ballPos = state.ball.pos;

            float shotSpeed = ballVelocity.Length();
            float powerQuality = 0.0f;

            if (shotSpeed >= optimalShotPowerMin && shotSpeed <= optimalShotPowerMax) {
                powerQuality = 1.0f;
            }
            else if (shotSpeed < optimalShotPowerMin) {
                powerQuality = shotSpeed / optimalShotPowerMin;
            }
            else {
                float excess = shotSpeed - optimalShotPowerMax;
                powerQuality = RS_MAX(0.2f, 1.0f - excess / 1000.f);
            }

            Vec ballToGoal = goalCenter - ballPos;
            Vec ballToGoalNorm = ballToGoal.Normalized();
            Vec velocityNorm = ballVelocity.Normalized();
            float directionQuality = RS_MAX(0.0f, ballToGoalNorm.Dot(velocityNorm));

            float shotAngle = CalculateShotAngle(ballPos, goalCenter);
            float angleQuality = 1.0f;
            if (shotAngle > precisionAngleThreshold) {
                angleQuality = RS_MAX(0.3f, 1.0f - (shotAngle - precisionAngleThreshold) / 45.f);
            }

            float distanceFromCenter = abs(ballPos.x - goalCenter.x);
            float placementQuality = RS_MIN(1.0f, distanceFromCenter / (goalWidth / 3.f));

            float overallQuality = powerQuality * 0.3f +
                directionQuality * 0.4f +
                angleQuality * 0.2f +
                placementQuality * 0.1f;

            return overallQuality;
        }

    public:
        PerfectFinishingReward(
            float finishingZoneDistance = 1500.f,
            float criticalZoneDistance = 800.f,
            float goalWidth = 1786.f,
            float goalHeight = 642.775f,
            float optimalShotPowerMin = 800.f,
            float optimalShotPowerMax = 2200.f,
            float precisionAngleThreshold = 15.f,
            float openNetBonus = 15.0f,
            float beatingKeeperBonus = 20.0f,
            float pressureShotBonus = 10.0f,
            float clinicalFinishBonus = 25.0f,
            float opportunityRecognitionBonus = 5.0f,
            float shotSelectionBonus = 8.0f,
            float composureBonus = 12.0f,
            float techniqueBonus = 10.0f,
            float placementBonus = 15.0f,
            float powerControlBonus = 8.0f
        ) : finishingZoneDistance(finishingZoneDistance),
            criticalZoneDistance(criticalZoneDistance),
            goalWidth(goalWidth),
            goalHeight(goalHeight),
            optimalShotPowerMin(optimalShotPowerMin),
            optimalShotPowerMax(optimalShotPowerMax),
            precisionAngleThreshold(precisionAngleThreshold),
            openNetBonus(openNetBonus),
            beatingKeeperBonus(beatingKeeperBonus),
            pressureShotBonus(pressureShotBonus),
            clinicalFinishBonus(clinicalFinishBonus),
            opportunityRecognitionBonus(opportunityRecognitionBonus),
            shotSelectionBonus(shotSelectionBonus),
            composureBonus(composureBonus),
            techniqueBonus(techniqueBonus),
            placementBonus(placementBonus),
            powerControlBonus(powerControlBonus) {
        }

        virtual void Reset(const GameState& initialState) override {
            playerStates.clear();
            previousBallVel.clear();
            previousScores.clear();

            for (const auto& player : initialState.players) {
                playerStates[player.carId] = PlayerState();
                previousBallVel[player.carId] = initialState.ball.vel;
            }

            previousScores[0] = 0;
            previousScores[1] = 0;
        }

        virtual float GetReward(const Player& player, const GameState& state, bool isFinal) override {
            if (playerStates.find(player.carId) == playerStates.end()) {
                playerStates[player.carId] = PlayerState();
                previousBallVel[player.carId] = state.ball.vel;
                return 0.0f;
            }

            PlayerState& playerState = playerStates[player.carId];
            float totalReward = 0.0f;

            playerState.totalTicks++;

            int playerTeam = (player.team == Team::BLUE) ? 0 : 1;
            bool goalScoredByPlayer = false;

            bool inFinishingZone = IsInFinishingZone(player, state);
            bool inCriticalZone = IsInCriticalZone(player, state);
            DefensiveAnalysis defensiveAnalysis = AnalyzeDefensiveSetup(player, state);

            if (inFinishingZone) {
                playerState.finishingZoneTime++;
                if (!playerState.inFinishingZone) {
                    playerState.opportunityStartTick = playerState.totalTicks;
                }
                playerState.inFinishingZone = true;
            }
            else {
                playerState.inFinishingZone = false;
                playerState.finishingZoneTime = 0;
            }

            if (inFinishingZone) {
                totalReward += 0.5f;

                if (defensiveAnalysis.openNet) {
                    totalReward += opportunityRecognitionBonus * 0.3f;
                }

                if (defensiveAnalysis.beatenDefender) {
                    totalReward += opportunityRecognitionBonus * 0.5f;
                    playerState.beatenDefender = true;
                }

                Vec goalCenter = GetGoalCenter(playerTeam);
                float shotAngle = CalculateShotAngle(state.ball.pos, goalCenter);

                if (shotAngle < precisionAngleThreshold) {
                    float angleReward = techniqueBonus * 0.3f;
                    totalReward += angleReward;
                    playerState.bestAngleAchieved = RS_MAX(playerState.bestAngleAchieved,
                        1.0f - shotAngle / 90.f);
                }
            }

            float pressureLevel = defensiveAnalysis.pressureLevel;
            playerState.pressureLevel = pressureLevel;

            if (inFinishingZone && pressureLevel > 0.f) {
                float composureFactor = 1.0f / (1.0f + pressureLevel * 0.5f);
                playerState.composureRating = composureFactor;

                if (composureFactor > 0.7f) {
                    totalReward += composureBonus * 0.2f;
                }
            }

            if (player.ballTouchedStep) {
                Vec currentBallVelocity = state.ball.vel;
                Vec prevVel = previousBallVel[player.carId];
                float velocityChange = (currentBallVelocity - prevVel).Length();

                if (velocityChange > 300.f && inFinishingZone) {
                    float shotQuality = CalculateShotQuality(player, state, currentBallVelocity);
                    playerState.lastShotQuality = shotQuality;

                    float shotReward = shotSelectionBonus * shotQuality;

                    if (defensiveAnalysis.openNet) {
                        shotReward += openNetBonus * shotQuality;
                    }

                    if (defensiveAnalysis.beatenDefender) {
                        shotReward += beatingKeeperBonus * shotQuality;
                    }

                    if (pressureLevel > 0.5f) {
                        shotReward += pressureShotBonus * shotQuality;
                    }

                    if (inCriticalZone) {
                        shotReward *= 1.5f;
                    }

                    float shotSpeed = currentBallVelocity.Length();
                    if (shotSpeed >= optimalShotPowerMin && shotSpeed <= optimalShotPowerMax) {
                        shotReward += powerControlBonus;
                    }

                    totalReward += shotReward;
                    playerState.shotOpportunitiesTaken++;

                    if (shotQuality > 0.7f) {
                        playerState.consecutiveGoodShots++;
                    }
                    else {
                        playerState.consecutiveGoodShots = 0;
                    }
                }
            }

            if (goalScoredByPlayer) {
                float goalReward = 50.0f;

                if (playerState.lastShotQuality > 0.8f) {
                    goalReward += clinicalFinishBonus;
                }

                if (playerState.beatenDefender) {
                    goalReward += beatingKeeperBonus;
                }

                if (playerState.pressureLevel > 0.5f) {
                    goalReward += pressureShotBonus * 1.5f;
                }

                if (playerState.composureRating > 0.8f) {
                    goalReward += composureBonus;
                }

                if (playerState.bestAngleAchieved > 0.8f) {
                    goalReward += placementBonus;
                }

                totalReward += goalReward;

                playerState.beatenDefender = false;
                playerState.bestAngleAchieved = 0.0f;
            }

            if (playerState.inFinishingZone &&
                !inFinishingZone &&
                playerState.finishingZoneTime > 60 &&
                playerState.shotOpportunitiesTaken == 0) {

                totalReward -= 2.0f;
                playerState.shotOpportunitiesMissed++;
            }

            if (playerState.consecutiveGoodShots >= 3) {
                float consistencyBonus = RS_MIN(playerState.consecutiveGoodShots * 0.5f, 5.0f);
                totalReward += consistencyBonus;
            }

            previousBallVel[player.carId] = state.ball.vel;

            return totalReward;
        }
    };

    class WavedashKickoff50Reward : public Reward {
    private:
        float kickoffDetectionTime;
        float firstFlipBonus;
        float fiftyFiftyBonus;
        float wavedashBonus;
        float speedRecoveryBonus;
        float maxFlipDelay;
        float wavedashDelay;
        bool debug;

        struct PlayerKickoffState {
            bool kickoffDetected = false;
            bool firstFlipUsed = false;
            bool fiftyFiftyDone = false;
            bool wavedashDone = false;
            Vec lastPos = Vec(0, 0, 0);
            float lastTime = 0.0f;
            float initialSpeed = 0.0f;
            bool wasOnGround = true;
            bool hadDoubleJump = true;
            float kickoffStartTime = 0.0f;
            int airFrames = 0;
        };

        std::unordered_map<int, PlayerKickoffState> playerStates;
        float kickoffStartTime;
        int frameCount;

        const std::vector<Vec> KICKOFF_POSITIONS = {
            Vec(0, -4608, 17),
            Vec(-2048, -2560, 17),
            Vec(2048, -2560, 17),
            Vec(0, 4608, 17),
            Vec(-2048, 2560, 17),
            Vec(2048, 2560, 17)
        };

        bool IsKickoffPosition(Vec playerPos) const {
            for (const auto& kickoffPos : KICKOFF_POSITIONS) {
                float distance = (playerPos - kickoffPos).Length();
                if (distance < 100.f) {
                    return true;
                }
            }
            return false;
        }

        bool DetectFlip(const Player& player, PlayerKickoffState& state) const {
            bool nowInAir = !player.isOnGround;
            bool wasOnGround = state.wasOnGround;

            if (nowInAir && wasOnGround) {
                state.airFrames = 0;
                return false;
            }

            if (nowInAir) {
                state.airFrames++;
                if (state.airFrames < 10 && player.angVel.Length() > 3.0f) {
                    return true;
                }
            }

            return false;
        }

        bool DetectWavedash(const Player& player, const PlayerKickoffState& state) const {
            bool recentlyInAir = state.airFrames > 5 && state.airFrames < 30;
            bool nearOrOnGround = player.isOnGround || player.pos.z < 30.f;
            float forwardSpeed = abs(player.vel.y);
            bool hasForwardSpeed = forwardSpeed > 500.f;
            float horizontalSpeed = Vec(player.vel.x, player.vel.y, 0).Length();
            bool hasGoodSpeed = horizontalSpeed > 600.f;
            bool hadRecentRotation = player.angVel.Length() > 1.5f || state.airFrames < 20;

            return recentlyInAir && nearOrOnGround && hasForwardSpeed && hasGoodSpeed && hadRecentRotation;
        }

        bool DetectFiftyFifty(const Player& player, const GameState& state) const {
            Vec ballPos = state.ball.pos;
            Vec playerPos = player.pos;

            float ballDistFromCenter = Vec(ballPos.x, ballPos.y, 0).Length();
            bool ballAtCenter = ballDistFromCenter < 300.f;

            float playerDistToBall = (playerPos - ballPos).Length();
            bool playerNearBall = playerDistToBall < 200.f;

            bool touchedBall = player.ballTouchedStep;

            return ballAtCenter && playerNearBall && touchedBall;
        }

        float CalculateSpeedRecovery(Vec currentVel, Vec initialVel) const {
            float currentSpeed = currentVel.Length();
            float initialSpeed = initialVel.Length();
            return RS_MAX(0.f, currentSpeed - initialSpeed);
        }

    public:
        WavedashKickoff50Reward(
            float kickoffDetectionTime = 3.0f,
            float firstFlipBonus = 3.0f,
            float fiftyFiftyBonus = 5.0f,
            float wavedashBonus = 8.0f,
            float speedRecoveryBonus = 2.0f,
            float maxFlipDelay = 0.8f,
            float wavedashDelay = 0.5f,
            bool debug = false
        ) : kickoffDetectionTime(kickoffDetectionTime),
            firstFlipBonus(firstFlipBonus),
            fiftyFiftyBonus(fiftyFiftyBonus),
            wavedashBonus(wavedashBonus),
            speedRecoveryBonus(speedRecoveryBonus),
            maxFlipDelay(maxFlipDelay),
            wavedashDelay(wavedashDelay),
            debug(debug),
            kickoffStartTime(0.0f),
            frameCount(0) {
        }

        virtual void Reset(const GameState& initialState) override {
            playerStates.clear();
            kickoffStartTime = 0.0f;
            frameCount = 0;

            for (const auto& player : initialState.players) {
                PlayerKickoffState state;
                state.lastPos = player.pos;
                state.initialSpeed = player.vel.Length();
                state.wasOnGround = player.isOnGround;
                playerStates[player.carId] = state;
            }
        }

        virtual float GetReward(const Player& player, const GameState& state, bool isFinal) override {
            if (playerStates.find(player.carId) == playerStates.end()) {
                PlayerKickoffState newState;
                newState.lastPos = player.pos;
                newState.initialSpeed = player.vel.Length();
                newState.wasOnGround = player.isOnGround;
                playerStates[player.carId] = newState;
                return 0.0f;
            }

            PlayerKickoffState& pState = playerStates[player.carId];
            float reward = 0.0f;

            float currentTime = frameCount * (1.0f / 120.0f);
            frameCount++;

            if (!player.isOnGround) {
                pState.airFrames++;
            }
            else {
                pState.airFrames = 0;
            }

            if (!pState.kickoffDetected && IsKickoffPosition(player.pos)) {
                pState.kickoffDetected = true;
                pState.kickoffStartTime = currentTime;
                pState.initialSpeed = player.vel.Length();
            }

            if (pState.kickoffDetected &&
                !pState.firstFlipUsed) {

                if (DetectFlip(player, pState)) {
                    float flipDelay = currentTime - pState.kickoffStartTime;

                    if (flipDelay <= maxFlipDelay) {
                        pState.firstFlipUsed = true;
                        reward += firstFlipBonus;
                    }
                }
            }

            if (pState.firstFlipUsed && !pState.fiftyFiftyDone) {
                if (DetectFiftyFifty(player, state)) {
                    pState.fiftyFiftyDone = true;
                    reward += fiftyFiftyBonus;

                    if (!pState.wavedashDone) {
                        Vec ballPos = state.ball.pos;
                        Vec playerPos = player.pos;
                        float distanceToBall = (playerPos - ballPos).Length();

                        if (distanceToBall < 150.f && DetectWavedash(player, pState)) {
                            pState.wavedashDone = true;
                            reward += wavedashBonus;

                            float speedRecovery = CalculateSpeedRecovery(
                                player.vel,
                                Vec(0, 0, 0)
                            );

                            if (speedRecovery > 500.f) {
                                reward += speedRecoveryBonus;
                            }
                        }
                    }
                }
            }

            pState.wasOnGround = player.isOnGround;
            pState.lastPos = player.pos;
            pState.lastTime = currentTime;

            return reward;
        }
    };

    class DefensivePositioningReward : public Reward {
    public:
        float defensive_zone_threshold;

        DefensivePositioningReward(float defensive_zone_threshold = 2000.0f) : defensive_zone_threshold(defensive_zone_threshold) {}

        virtual float GetReward(const Player& player, const GameState& state, bool isFinal) override {
            bool player_is_blue = (player.team == Team::BLUE);
            Vec own_goal = player_is_blue ? CommonValues::BLUE_GOAL_CENTER : CommonValues::ORANGE_GOAL_CENTER;

            float distance_to_own_goal = (player.pos - own_goal).Length();

            if (distance_to_own_goal < defensive_zone_threshold) {
                float ball_distance_to_goal = (state.ball.pos - own_goal).Length();

                if (ball_distance_to_goal < 1500.0f) {
                    return 1.0f - (distance_to_own_goal / defensive_zone_threshold);
                }
            }

            return 0.0f;
        }
    };

    class AirRollTouchReward : public Reward {
    public:
        float min_angular_velocity;

        AirRollTouchReward(float min_angular_velocity = 2.0f) : min_angular_velocity(min_angular_velocity) {}

        virtual float GetReward(const Player& player, const GameState& state, bool isFinal) override {
            if (!player.ballTouchedStep || player.isOnGround)
                return 0.0f;

            float angular_speed = player.angVel.Length();
            if (angular_speed > min_angular_velocity) {
                return RS_MIN(1.0f, angular_speed / CommonValues::CAR_MAX_ANG_VEL);
            }

            return 0.0f;
        }
    };

    class FastAerialSimpleReward : public Reward {
    public:
        float min_height;
        float time_threshold;

        FastAerialSimpleReward(float min_height = 300.0f, float time_threshold = 2.0f)
            : min_height(min_height), time_threshold(time_threshold) {
        }

        virtual float GetReward(const Player& player, const GameState& state, bool isFinal) override {
            if (player.isOnGround || player.pos.z < min_height)
                return 0.0f;

            if (player.airTime > 0 && player.airTime < time_threshold) {
                if (player.hasDoubleJumped) {
                    return RS_MIN(1.0f, player.pos.z / 1000.0f);
                }
            }

            return 0.0f;
        }
    };

    class SimpleAirDribbleReward : public Reward {
    public:
        float min_height;
        float max_distance_to_ball;
        int min_touches;

        SimpleAirDribbleReward(float min_height = 200.0f, float max_distance_to_ball = 300.0f, int min_touches = 3)
            : min_height(min_height), max_distance_to_ball(max_distance_to_ball), min_touches(min_touches) {
        }

        virtual float GetReward(const Player& player, const GameState& state, bool isFinal) override {
            if (player.isOnGround || state.ball.pos.z < min_height)
                return 0.0f;

            float distance_to_ball = (player.pos - state.ball.pos).Length();
            if (distance_to_ball > max_distance_to_ball)
                return 0.0f;

            if (player.ballTouchedStep) {
                float height_reward = RS_MIN(1.0f, (player.pos.z - min_height) / (1000.0f - min_height));
                return height_reward * 0.5f;
            }

            return 0.0f;
        }
    };

    class WavedashRecoveryReward : public Reward {
    public:
        float speed_threshold;

        WavedashRecoveryReward(float speed_threshold = 500.0f) : speed_threshold(speed_threshold) {}

        virtual float GetReward(const Player& player, const GameState& state, bool isFinal) override {
            if (!player.prev)
                return 0.0f;

            if (player.isOnGround && !player.prev->isOnGround && player.prev->isFlipping) {
                float landing_speed = player.vel.Length();
                if (landing_speed > speed_threshold) {
                    return RS_MIN(1.0f, landing_speed / CommonValues::CAR_MAX_SPEED);
                }
            }

            return 0.0f;
        }
    };

    class KickoffSpeedFlipRewardv2 : public Reward {
    private:
        struct PlayerState {
            bool kickoff_detected;
            bool speedflip_started;
            bool speedflip_completed;
            bool ball_touched;
            bool had_flip;

            Vec kickoff_start_pos;
            Vec ball_start_pos;
            float kickoff_start_time;
            float speedflip_time;
            float ball_touch_time;
            float distance_to_ball;
            float speed_at_ball;
            float angle_to_ball;
        };

        std::unordered_map<int, PlayerState> player_states;

        float kickoff_detection_radius;
        float max_speedflip_time;
        float min_ball_speed;
        float max_angle_tolerance;
        bool debug;

    public:
        KickoffSpeedFlipRewardv2(
            float kickoff_detection_radius = 200.0f,
            float max_speedflip_time = 2.0f,
            float min_ball_speed = 800.0f,
            float max_angle_tolerance = 30.0f,
            bool debug = false
        ) : kickoff_detection_radius(kickoff_detection_radius), max_speedflip_time(max_speedflip_time),
            min_ball_speed(min_ball_speed), max_angle_tolerance(max_angle_tolerance), debug(debug) {
        }

        virtual void Reset(const GameState& initial_state) override {
            player_states.clear();
            for (const auto& player : initial_state.players) {
                player_states[player.carId] = {
                    false, false, false, false, player.HasFlipOrJump(),
                    player.pos, initial_state.ball.pos,
                    0.0f, 0.0f, 0.0f,
                    0.0f, 0.0f, 0.0f
                };
            }
        }

        bool _is_kickoff_position(const Vec& player_pos, const Vec& ball_pos) {
            float distance_to_center = player_pos.Length();
            float distance_to_ball = (player_pos - ball_pos).Length();

            return distance_to_center < kickoff_detection_radius && distance_to_ball < kickoff_detection_radius * 2.0f;
        }

        virtual float GetReward(const Player& player, const GameState& state, bool isFinal) override {
            if (!state.prev) return 0.0f;

            int car_id = player.carId;

            if (player_states.find(car_id) == player_states.end()) {
                player_states[car_id] = {
                    false, false, false, false, player.HasFlipOrJump(),
                    player.pos, state.ball.pos,
                    0.0f, 0.0f, 0.0f,
                    0.0f, 0.0f, 0.0f
                };
            }

            PlayerState& player_state = player_states[car_id];
            float reward = 0.0f;

            if (!player_state.kickoff_detected) {
                if (_is_kickoff_position(player.pos, state.ball.pos)) {
                    player_state.kickoff_detected = true;
                    player_state.kickoff_start_time = state.deltaTime;
                    player_state.kickoff_start_pos = player.pos;
                    player_state.ball_start_pos = state.ball.pos;
                }
            }

            if (player_state.kickoff_detected && !player_state.speedflip_started) {
                bool had_flip = player_state.had_flip;
                bool current_has_flip = player.HasFlipOrJump();

                if (had_flip && !current_has_flip) {
                    float kickoff_to_flip_delay = state.deltaTime - player_state.kickoff_start_time;

                    if (kickoff_to_flip_delay < 1.0f) {
                        player_state.speedflip_started = true;
                        player_state.speedflip_time = state.deltaTime;
                    }
                }
            }

            if (player_state.speedflip_started && !player_state.ball_touched) {
                if (player.ballTouchedStep) {
                    player_state.ball_touched = true;
                    player_state.ball_touch_time = state.deltaTime;

                    Vec to_ball = state.ball.pos - player_state.kickoff_start_pos;
                    player_state.distance_to_ball = to_ball.Length();
                    player_state.speed_at_ball = player.vel.Length();

                    Vec player_forward = player.rotMat.forward;
                    Vec ball_direction = to_ball.Normalized();
                    player_state.angle_to_ball = acosf(RS_MAX(-1.0f, RS_MIN(1.0f,
                        player_forward.Dot(ball_direction)))) * 57.2957795f;
                }
            }

            if (player_state.ball_touched && !player_state.speedflip_completed) {
                float total_time = player_state.ball_touch_time - player_state.kickoff_start_time;

                if (total_time <= max_speedflip_time) {
                    player_state.speedflip_completed = true;

                    float time_bonus = RS_MAX(0.0f, 1.0f - (total_time / max_speedflip_time));
                    float speed_bonus = RS_MIN(2.0f, player_state.speed_at_ball / 2000.0f);
                    float angle_bonus = RS_MAX(0.0f, 1.0f - (player_state.angle_to_ball / max_angle_tolerance));
                    float distance_bonus = RS_MAX(0.0f, 1.0f - (player_state.distance_to_ball / 2000.0f));

                    reward = 3.0f + time_bonus + speed_bonus + angle_bonus + distance_bonus;
                }
            }

            player_state.had_flip = player.HasFlipOrJump();

            return reward;
        }
    };

    // AdvancedSpeedFlipReward
    class AdvancedSpeedFlipReward : public Reward {
    private:
        struct PlayerState {
            bool had_flip;
            bool speedflip_started;
            bool dash_detected;
            bool flip_cancel_detected;
            bool recovery_started;
            bool speedflip_completed;

            Vec pos_before_jump;
            Vec vel_before_jump;
            Vec pos_after_flip;
            Vec vel_after_flip;
            Vec pos_at_landing;

            float jump_time;
            float flip_cancel_time;
            float landing_time;
            float total_speedflip_time;

            bool was_on_ground;
            Vec last_ground_pos;
            float dash_start_time;
        };

        std::unordered_map<int, PlayerState> player_states;

        float min_dash_speed;
        float max_jump_delay;
        float max_flip_cancel_delay;
        float min_speed_gain;
        float max_recovery_time;
        float min_landing_speed;
        bool debug;

    public:
        AdvancedSpeedFlipReward(
            float min_dash_speed = 800.0f,
            float max_jump_delay = 0.3f,
            float max_flip_cancel_delay = 0.2f,
            float min_speed_gain = 400.0f,
            float max_recovery_time = 1.5f,
            float min_landing_speed = 1000.0f,
            bool debug = false
        ) : min_dash_speed(min_dash_speed), max_jump_delay(max_jump_delay),
            max_flip_cancel_delay(max_flip_cancel_delay), min_speed_gain(min_speed_gain),
            max_recovery_time(max_recovery_time), min_landing_speed(min_landing_speed),
            debug(debug) {
        }

        virtual void Reset(const GameState& initial_state) override {
            player_states.clear();
            for (const auto& player : initial_state.players) {
                player_states[player.carId] = {
                    player.HasFlipOrJump(),
                    false, false, false, false, false,
                    player.pos, player.vel, Vec(0,0,0), Vec(0,0,0), Vec(0,0,0),
                    0.0f, 0.0f, 0.0f, 0.0f,
                    player.isOnGround, player.pos, 0.0f
                };
            }
        }

        bool _detect_diagonal_dash(const Player& player, const GameState& state, PlayerState& player_state) {
            if (!player_state.was_on_ground || player.isOnGround) return false;

            Vec displacement = player.pos - player_state.last_ground_pos;
            float distance = displacement.Length();
            float time_since_dash = state.deltaTime;

            if (time_since_dash > 0.0f) {
                float dash_speed = distance / time_since_dash;

                if (dash_speed > min_dash_speed) {
                    float horizontal_component = sqrtf(displacement.x * displacement.x + displacement.y * displacement.y);
                    float vertical_component = fabsf(displacement.z);

                    if (horizontal_component > 100.0f && vertical_component < 50.0f) {
                        return true;
                    }
                }
            }

            return false;
        }

        bool _detect_flip_cancel(const Player& player, const GameState& state, PlayerState& player_state) {
            if (!player_state.speedflip_started) return false;

            float time_since_jump = state.deltaTime - player_state.jump_time;
            if (time_since_jump > max_flip_cancel_delay) return false;

            bool low_vertical_vel = fabsf(player.vel.z) < 300.0f;
            bool good_orientation = player.rotMat.up.z > 0.6f;
            bool horizontal_acceleration = player.vel.Length() > player_state.vel_before_jump.Length();

            return low_vertical_vel && good_orientation && horizontal_acceleration;
        }

        bool _detect_speedflip_recovery(const Player& player, const GameState& state, PlayerState& player_state) {
            if (!player_state.flip_cancel_detected) return false;

            if (player.isOnGround && !player_state.recovery_started) {
                float landing_speed = player.vel.Length();
                bool good_landing_speed = landing_speed > min_landing_speed;
                bool good_orientation = player.rotMat.up.z > 0.8f;

                if (good_landing_speed && good_orientation) {
                    return true;
                }
            }

            return false;
        }

        virtual float GetReward(const Player& player, const GameState& state, bool isFinal) override {
            if (!state.prev) return 0.0f;

            int car_id = player.carId;

            if (player_states.find(car_id) == player_states.end()) {
                player_states[car_id] = {
                    player.HasFlipOrJump(),
                    false, false, false, false, false,
                    player.pos, player.vel, Vec(0,0,0), Vec(0,0,0), Vec(0,0,0),
                    0.0f, 0.0f, 0.0f, 0.0f,
                    player.isOnGround, player.pos, 0.0f
                };
            }

            PlayerState& player_state = player_states[car_id];
            float reward = 0.0f;

            if (!player_state.speedflip_started && !player_state.dash_detected) {
                if (_detect_diagonal_dash(player, state, player_state)) {
                    player_state.dash_detected = true;
                    player_state.dash_start_time = state.deltaTime;
                }
            }

            if (player_state.dash_detected && !player_state.speedflip_started) {
                bool had_flip = player_state.had_flip;
                bool current_has_flip = player.HasFlipOrJump();

                if (had_flip && !current_has_flip) {
                    float dash_to_jump_delay = state.deltaTime - player_state.dash_start_time;

                    if (dash_to_jump_delay <= max_jump_delay) {
                        player_state.speedflip_started = true;
                        player_state.jump_time = state.deltaTime;
                        player_state.pos_before_jump = player.pos;
                        player_state.vel_before_jump = player.vel;
                    }
                }
            }

            if (player_state.speedflip_started && !player_state.flip_cancel_detected) {
                if (_detect_flip_cancel(player, state, player_state)) {
                    player_state.flip_cancel_detected = true;
                    player_state.flip_cancel_time = state.deltaTime;
                    player_state.pos_after_flip = player.pos;
                    player_state.vel_after_flip = player.vel;
                }
            }

            if (player_state.flip_cancel_detected && !player_state.recovery_started) {
                if (_detect_speedflip_recovery(player, state, player_state)) {
                    player_state.recovery_started = true;
                    player_state.landing_time = state.deltaTime;
                    player_state.pos_at_landing = player.pos;
                    player_state.total_speedflip_time = player_state.landing_time - player_state.dash_start_time;
                }
            }

            if (player_state.recovery_started && !player_state.speedflip_completed) {
                player_state.speedflip_completed = true;

                float speed_gain = player_state.vel_after_flip.Length() - player_state.vel_before_jump.Length();
                float efficiency = speed_gain / player_state.total_speedflip_time;

                float base_reward = 5.0f;
                float speed_bonus = RS_MIN(2.0f, speed_gain / 1000.0f);
                float efficiency_bonus = RS_MIN(2.0f, efficiency / 1000.0f);
                float time_bonus = RS_MAX(0.0f, 1.0f - (player_state.total_speedflip_time / max_recovery_time));

                reward = base_reward + speed_bonus + efficiency_bonus + time_bonus;
            }

            player_state.had_flip = player.HasFlipOrJump();
            player_state.was_on_ground = player.isOnGround;
            if (player.isOnGround) {
                player_state.last_ground_pos = player.pos;
            }

            return reward;
        }
    };

    // BouncyAirDribbleRewardv8
    class BouncyAirDribbleRewardv8 : public Reward {
    private:
        struct PlayerState {
            Vec last_ball_pos;
            Vec last_player_pos;
            float last_ball_height;
            float last_player_height;
            bool air_dribble_active;
            int air_dribble_ticks;
        };

        std::unordered_map<int, PlayerState> player_states;

        float min_ball_distance;
        float min_relative_velocity_threshold;
        float min_ball_height;
        float max_player_ball_distance;
        float min_boost_multiplier;
        float min_direction_z_component;
        float min_towards_goal_speed;
        float max_direction_angle_diff;
        float max_goal_height_above_crossbar;
        float near_goal_shutoff_time;

    public:
        BouncyAirDribbleRewardv8(
            float min_ball_distance = 50.0f,
            float min_relative_velocity_threshold = 200.0f,
            float min_ball_height = 150.0f,
            float max_player_ball_distance = 300.0f,
            float min_boost_multiplier = 0.8f,
            float min_direction_z_component = 0.3f,
            float min_towards_goal_speed = 100.0f,
            float max_direction_angle_diff = 45.0f,
            float max_goal_height_above_crossbar = 200.0f,
            float near_goal_shutoff_time = 2.0f
        ) : min_ball_distance(min_ball_distance),
            min_relative_velocity_threshold(min_relative_velocity_threshold),
            min_ball_height(min_ball_height),
            max_player_ball_distance(max_player_ball_distance),
            min_boost_multiplier(min_boost_multiplier),
            min_direction_z_component(min_direction_z_component),
            min_towards_goal_speed(min_towards_goal_speed),
            max_direction_angle_diff(max_direction_angle_diff),
            max_goal_height_above_crossbar(max_goal_height_above_crossbar),
            near_goal_shutoff_time(near_goal_shutoff_time) {
        }

        virtual void Reset(const GameState& initial_state) override {
            player_states.clear();
            for (const auto& player : initial_state.players) {
                player_states[player.carId] = {
                    initial_state.ball.pos,
                    player.pos,
                    initial_state.ball.pos.z,
                    player.pos.z,
                    false,
                    0
                };
            }
        }

        bool _is_bouncy(const Player& player, const GameState& state) {
            float distance_to_ball = (player.pos - state.ball.pos).Length();
            bool distance_condition = distance_to_ball >= min_ball_distance;

            Vec relative_velocity = player.vel - state.ball.vel;
            float relative_velocity_magnitude = relative_velocity.Length();
            bool velocity_condition = relative_velocity_magnitude >= min_relative_velocity_threshold;

            return distance_condition || velocity_condition;
        }

        bool _is_air_dribble_towards_goal(const Player& player, const GameState& state) {
            bool height_condition = state.ball.pos.z >= min_ball_height;

            if (player_states.find(player.carId) != player_states.end()) {
                auto& player_state = player_states[player.carId];
                float ball_height_gain = state.ball.pos.z - player_state.last_ball_height;
                float player_height_gain = player.pos.z - player_state.last_player_height;
                bool gaining_height = (ball_height_gain > 10.0f) || (player_height_gain > 10.0f);
                height_condition = height_condition || gaining_height;
            }

            if (!height_condition) return false;

            float distance_to_ball = (player.pos - state.ball.pos).Length();
            if (distance_to_ball > max_player_ball_distance) return false;

            bool player_is_blue = (player.team == Team::BLUE);
            Vec enemy_goal = player_is_blue ? CommonValues::ORANGE_GOAL_CENTER : CommonValues::BLUE_GOAL_CENTER;
            Vec to_goal_2d = enemy_goal - player.pos;
            to_goal_2d.z = 0.0f;
            float dist_to_goal_2d = to_goal_2d.Length();

            float min_boost_required = 15.0f * powf(dist_to_goal_2d / 5000.0f, 1.75f);
            if (player.boost < min_boost_required * min_boost_multiplier) return false;

            Vec player_to_ball = state.ball.pos - player.pos;
            Vec player_to_ball_norm = player_to_ball.Normalized();
            if (player_to_ball_norm.z < min_direction_z_component) return false;

            Vec ball_to_goal = enemy_goal - state.ball.pos;
            ball_to_goal.z = 0.0f;
            Vec ball_to_goal_norm = ball_to_goal.Normalized();

            float ball_towards_goal_speed = state.ball.vel.Dot(ball_to_goal_norm);
            if (ball_towards_goal_speed < min_towards_goal_speed) return false;

            float player_towards_goal_speed = player.vel.Dot(ball_to_goal_norm);
            if (player_towards_goal_speed < min_towards_goal_speed) return false;

            Vec player_to_ball_2d = player_to_ball;
            player_to_ball_2d.z = 0.0f;
            Vec player_to_ball_2d_norm = player_to_ball_2d.Normalized();

            float direction_angle_diff = acosf(RS_MAX(-1.0f, RS_MIN(1.0f,
                player_to_ball_2d_norm.Dot(ball_to_goal_norm)))) * 57.2957795f;

            if (direction_angle_diff > max_direction_angle_diff) return false;

            float time_to_goal = dist_to_goal_2d / ball_towards_goal_speed;
            if (time_to_goal > 0.0f) {
                float ball_height_at_goal = state.ball.pos.z + state.ball.vel.z * time_to_goal;
                float crossbar_height = 642.775f;

                if (ball_height_at_goal > crossbar_height + max_goal_height_above_crossbar) {
                    return false;
                }
            }

            if (time_to_goal < near_goal_shutoff_time) return false;

            return true;
        }

        virtual float GetReward(const Player& player, const GameState& state, bool isFinal) override {
            if (!state.prev) return 0.0f;

            int car_id = player.carId;

            if (player_states.find(car_id) == player_states.end()) {
                player_states[car_id] = {
                    state.ball.pos,
                    player.pos,
                    state.ball.pos.z,
                    player.pos.z,
                    false,
                    0
                };
            }

            PlayerState& player_state = player_states[car_id];

            bool is_air_dribbling = _is_air_dribble_towards_goal(player, state);

            if (is_air_dribbling) {
                bool is_bouncy = _is_bouncy(player, state);

                if (is_bouncy) {
                    if (!player_state.air_dribble_active) {
                        player_state.air_dribble_active = true;
                        player_state.air_dribble_ticks = 0;
                    }

                    player_state.air_dribble_ticks++;

                    float base_reward = 0.1f;
                    float duration_bonus = RS_MIN(1.0f, player_state.air_dribble_ticks / 100.0f);

                    bool player_is_blue = (player.team == Team::BLUE);
                    Vec enemy_goal = player_is_blue ? CommonValues::ORANGE_GOAL_CENTER : CommonValues::BLUE_GOAL_CENTER;
                    Vec to_goal = enemy_goal - state.ball.pos;
                    to_goal.z = 0.0f;
                    float distance_to_goal = to_goal.Length();
                    float distance_bonus = RS_MAX(0.0f, 1.0f - (distance_to_goal / 10000.0f));

                    float height_bonus = RS_MIN(1.0f, state.ball.pos.z / 1000.0f);

                    float ball_speed = state.ball.vel.Length();
                    float speed_bonus = RS_MIN(1.0f, ball_speed / 2000.0f);

                    float total_reward = base_reward + duration_bonus + distance_bonus + height_bonus + speed_bonus;

                    return total_reward;
                }
                else {
                    player_state.air_dribble_active = false;
                    player_state.air_dribble_ticks = 0;
                }
            }
            else {
                player_state.air_dribble_active = false;
                player_state.air_dribble_ticks = 0;
            }

            player_state.last_ball_pos = state.ball.pos;
            player_state.last_player_pos = player.pos;
            player_state.last_ball_height = state.ball.pos.z;
            player_state.last_player_height = player.pos.z;

            return 0.0f;
        }
    };

    // Enhanced45DegreeFlickReward
    class Enhanced45DegreeFlickReward : public Reward {
    private:
        struct PlayerState {
            bool had_flip;
            bool ball_on_car;
            bool last_ball_touch;
            Vec last_ball_pos;
            Vec last_player_pos;
        };

        std::unordered_map<int, PlayerState> player_states;

        float min_flick_power;
        float angle_tolerance;
        float base_reward;
        float power_bonus;
        float control_bonus;
        bool debug;

    public:
        Enhanced45DegreeFlickReward(
            float min_flick_power = 800.0f,
            float angle_tolerance = 8.0f,
            float base_reward = 2.0f,
            float power_bonus = 1.0f,
            float control_bonus = 0.5f,
            bool debug = false
        ) : min_flick_power(min_flick_power), angle_tolerance(angle_tolerance),
            base_reward(base_reward), power_bonus(power_bonus),
            control_bonus(control_bonus), debug(debug) {
        }

        virtual void Reset(const GameState& initial_state) override {
            player_states.clear();
            for (const auto& player : initial_state.players) {
                player_states[player.carId] = {
                    player.HasFlipOrJump(),
                    false,
                    player.ballTouchedStep,
                    initial_state.ball.pos,
                    player.pos
                };
            }
        }

        bool _is_ball_on_car(const Player& player, const GameState& state) {
            Vec car_pos = player.pos;
            Vec ball_pos = state.ball.pos;

            float horizontal_dist = sqrtf(
                (car_pos.x - ball_pos.x) * (car_pos.x - ball_pos.x) +
                (car_pos.y - ball_pos.y) * (car_pos.y - ball_pos.y)
            );

            float height_diff = ball_pos.z - car_pos.z;

            return horizontal_dist < 100.0f && height_diff > 20.0f && height_diff < 80.0f;
        }

        bool _detect_flick_usage(const Player& player, PlayerState& player_state) {
            bool current_has_flip = player.HasFlipOrJump();
            bool had_flip = player_state.had_flip;

            bool flip_used = had_flip && !current_has_flip;

            player_state.had_flip = current_has_flip;

            return flip_used;
        }

        float _calculate_flick_angle(const Player& player, const GameState& state) {
            Vec car_forward = player.rotMat.forward;
            Vec ball_velocity = state.ball.vel;
            float ball_speed = ball_velocity.Length();

            if (ball_speed < 100.0f)
                return 0.0f;

            Vec ball_vel_dir = ball_velocity / ball_speed;
            float dot_product = car_forward.Dot(ball_vel_dir);
            dot_product = RS_MAX(-1.0f, RS_MIN(1.0f, dot_product));
            float angle = acosf(dot_product);
            return angle * 57.2957795f;
        }

        bool _is_air_rolling(const Player& player, const GameState& state) {
            if (player.isOnGround) return false;

            float roll_velocity = fabsf(player.angVel.z);

            return roll_velocity > 2.0f;
        }

        bool _is_air_rolling_advanced(const Player& player, const GameState& state) {
            if (player.isOnGround) return false;

            float roll_velocity = fabsf(player.angVel.z);
            float pitch_velocity = fabsf(player.angVel.x);
            float yaw_velocity = fabsf(player.angVel.y);

            bool strong_roll = roll_velocity > 3.0f;
            bool moderate_pitch = pitch_velocity > 1.0f;
            bool moderate_yaw = yaw_velocity > 1.0f;

            return strong_roll && (moderate_pitch || moderate_yaw);
        }

        bool _detect_advanced_flick_with_air_roll(const Player& player, const GameState& state, PlayerState& player_state) {
            if (!state.prev) return false;

            if (player.isOnGround) return false;

            bool flip_just_used = player_state.had_flip && !player.HasFlipOrJump();

            bool air_rolling = _is_air_rolling(player, state) || _is_air_rolling_advanced(player, state);

            bool ball_on_roof = _is_ball_on_car(player, state);

            float ball_speed = state.ball.vel.Length();
            bool ball_fast_enough = ball_speed > min_flick_power;

            float flick_angle = _calculate_flick_angle(player, state);
            bool angle_correct = fabsf(flick_angle - 45.0f) <= angle_tolerance;

            bool ball_touched = player.ballTouchedStep && !player_state.last_ball_touch;

            bool was_in_air = !player_state.last_player_pos.z < 50.0f;

            return flip_just_used && air_rolling && ball_on_roof && ball_fast_enough && angle_correct && ball_touched && was_in_air;
        }

        virtual float GetReward(const Player& player, const GameState& state, bool isFinal) override {
            if (!state.prev)
                return 0.0f;

            int car_id = player.carId;

            std::vector<int> current_player_ids;
            for (const auto& p : state.players) {
                current_player_ids.push_back(p.carId);
            }

            for (auto it = player_states.begin(); it != player_states.end();) {
                if (std::find(current_player_ids.begin(), current_player_ids.end(), it->first) == current_player_ids.end()) {
                    it = player_states.erase(it);
                }
                else {
                    ++it;
                }
            }

            if (player_states.find(car_id) == player_states.end()) {
                player_states[car_id] = {
                    player.HasFlipOrJump(),
                    false,
                    player.ballTouchedStep,
                    state.ball.pos,
                    player.pos
                };
            }

            PlayerState& player_state = player_states[car_id];

            bool ball_on_car = _is_ball_on_car(player, state);
            player_state.ball_on_car = ball_on_car;

            bool flick_used = _detect_flick_usage(player, player_state);

            bool ball_touched = player.ballTouchedStep && !player_state.last_ball_touch;
            player_state.last_ball_touch = player.ballTouchedStep;

            if (_detect_advanced_flick_with_air_roll(player, state, player_state)) {
                float ball_speed = state.ball.vel.Length();
                float flick_angle = _calculate_flick_angle(player, state);
                bool air_rolling_basic = _is_air_rolling(player, state);
                bool air_rolling_advanced = _is_air_rolling_advanced(player, state);

                float reward = base_reward;

                float power_bonus_value = RS_MIN(1.0f, (ball_speed - min_flick_power) / 500.0f) * power_bonus;
                reward += power_bonus_value;

                if (air_rolling_advanced) {
                    reward *= 2.0f;
                }
                else if (air_rolling_basic) {
                    reward *= 1.5f;
                }

                return reward;
            }

            if (flick_used && ball_on_car && ball_touched) {
                float flick_angle = _calculate_flick_angle(player, state);

                if (fabsf(flick_angle - 45.0f) <= angle_tolerance) {
                    float ball_speed = state.ball.vel.Length();

                    if (ball_speed > min_flick_power) {
                        float reward = base_reward * 0.7f;

                        float power_bonus_value = RS_MIN(1.0f, (ball_speed - min_flick_power) / 500.0f) * power_bonus;
                        reward += power_bonus_value;

                        return reward;
                    }
                }
            }

            player_state.last_ball_pos = state.ball.pos;
            player_state.last_player_pos = player.pos;

            return 0.0f;
        }
    };

    // DiagonalFlickReward
    class DiagonalFlickReward : public Reward {
    private:
        struct PlayerState {
            bool was_on_ground;
            bool was_facing_ball;
            Vec last_ball_pos;
            Vec last_player_pos;
            bool flick_started;
            int flick_tick;
            Vec ball_velocity_before;
            Vec ball_velocity_after;
        };

        std::unordered_map<int, PlayerState> player_states;

    public:
        float min_ball_speed;
        float max_ball_speed;
        float min_flick_distance;
        float max_flick_distance;

        DiagonalFlickReward(float min_ball_speed = 800.0f, float max_ball_speed = 2500.0f,
            float min_flick_distance = 100.0f, float max_flick_distance = 400.0f)
            : min_ball_speed(min_ball_speed), max_ball_speed(max_ball_speed),
            min_flick_distance(min_flick_distance), max_flick_distance(max_flick_distance) {
        }

        virtual void Reset(const GameState& initial_state) override {
            player_states.clear();
            for (const auto& player : initial_state.players) {
                player_states[player.carId] = { player.isOnGround, false, initial_state.ball.pos,
                                             player.pos, false, 0, Vec(0,0,0), Vec(0,0,0) };
            }
        }

        virtual float GetReward(const Player& player, const GameState& state, bool isFinal) override {
            if (!state.prev || !player.ballTouchedStep)
                return 0.0f;

            int car_id = player.carId;
            if (player_states.find(car_id) == player_states.end()) {
                player_states[car_id] = { player.isOnGround, false, state.ball.pos,
                                       player.pos, false, 0, Vec(0,0,0), Vec(0,0,0) };
            }

            PlayerState& st = player_states[car_id];
            float reward = 0.0f;

            float distance_to_ball = (player.pos - state.ball.pos).Length();
            if (distance_to_ball < min_flick_distance || distance_to_ball > max_flick_distance)
                return 0.0f;

            Vec ball_dir = (state.ball.pos - player.pos).Normalized();
            Vec forward = player.rotMat.forward;
            float facing_ball = forward.Dot(ball_dir);

            if (facing_ball < 0.7f)
                return 0.0f;

            if (!st.flick_started && player.ballTouchedStep) {
                st.flick_started = true;
                st.flick_tick = 0;
                st.ball_velocity_before = state.prev->ball.vel;
                st.ball_velocity_after = state.ball.vel;
                st.was_on_ground = player.isOnGround;
                st.was_facing_ball = facing_ball > 0.7f;
            }

            if (st.flick_started && st.flick_tick < 10) {
                st.flick_tick++;

                float ball_speed = state.ball.vel.Length();
                if (ball_speed < min_ball_speed)
                    return 0.0f;

                Vec ball_vel_norm = state.ball.vel.Normalized();

                float horizontal_component = sqrtf(ball_vel_norm.x * ball_vel_norm.x + ball_vel_norm.y * ball_vel_norm.y);
                float vertical_component = fabsf(ball_vel_norm.z);

                if (horizontal_component > 0.6f && vertical_component > 0.4f) {
                    float power_bonus = RS_MIN(1.0f, ball_speed / max_ball_speed);
                    float diagonal_precision = 1.0f - fabsf(horizontal_component - vertical_component);
                    float distance_bonus = 1.0f - fabsf(distance_to_ball - 200.0f) / 200.0f;

                    reward = power_bonus * diagonal_precision * distance_bonus;
                }
            }

            st.last_ball_pos = state.ball.pos;
            st.last_player_pos = player.pos;
            st.was_on_ground = player.isOnGround;
            st.was_facing_ball = facing_ball > 0.7f;

            return reward;
        }
    };

    class SpeedBallToGoalReward : public Reward {
    public:
        bool ownGoal;
        float exponent;
        float proximityBonus;
        float minDistanceThreshold;
        float minSpeedThreshold;

        SpeedBallToGoalReward(
            bool ownGoal = false,
            float exponent = 1.0f,
            float proximityBonus = 0.2f,
            float minDistanceThreshold = 2000.0f,
            float minSpeedThreshold = 100.0f
        ) : ownGoal(ownGoal),
            exponent(exponent),
            proximityBonus(proximityBonus),
            minDistanceThreshold(minDistanceThreshold),
            minSpeedThreshold(minSpeedThreshold) {
        }

        virtual float GetReward(const Player& player, const GameState& state, bool isFinal) override {
            bool targetOrangeGoal = (player.team == Team::BLUE);
            if (ownGoal) targetOrangeGoal = !targetOrangeGoal;

            Vec targetPos = targetOrangeGoal ? CommonValues::ORANGE_GOAL_BACK
                : CommonValues::BLUE_GOAL_BACK;

            Vec ballToGoal = targetPos - state.ball.pos;
            float distance = ballToGoal.Length();

            float ballSpeed = state.ball.vel.Length();
            if (distance > minDistanceThreshold && ballSpeed < minSpeedThreshold) {
                return 0.0f;
            }

            float towardFrac = ballToGoal.Dot(state.ball.vel) / (distance * CommonValues::BALL_MAX_SPEED);

            float positive = RS_MAX(towardFrac, 0.0f);

            float reward = (exponent == 1.0f) ? positive : powf(positive, exponent);

            if (proximityBonus > 0.0f && distance > 0.0f) {
                reward *= (1.0f + proximityBonus / distance);
            }

            return reward;
        }
    };

    // FlickReward45Degree - Flicks 45° puissants style Mawkzy depuis le toit
    class FlickReward45Degree : public Reward {
    private:
        struct PlayerState {
            bool wasOnGround;
            bool ballWasOnRoof;
            float timeSinceBallOnRoof;
            Vec lastCarPos;

            PlayerState() : wasOnGround(true), ballWasOnRoof(false),
                timeSinceBallOnRoof(0.0f), lastCarPos(0, 0, 0) {
            }
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

        FlickReward45Degree(
            float minBallSpeed = 1200.0f,
            float maxBallSpeed = 3500.0f,
            float angleTolerance = 20.0f,
            float roofDistanceMax = 150.0f,
            float maxTimeSinceRoof = 0.3f
        ) : minBallSpeed(minBallSpeed), maxBallSpeed(maxBallSpeed),
            angleTolerance(angleTolerance), roofDistanceMax(roofDistanceMax),
            maxTimeSinceRoof(maxTimeSinceRoof) {
        }

        virtual void Reset(const GameState& initialState) override {
            playerStates.clear();
            for (const auto& player : initialState.players) {
                PlayerState state;
                state.wasOnGround = player.isOnGround;
                state.ballWasOnRoof = false;
                state.timeSinceBallOnRoof = 999.0f;
                state.lastCarPos = player.pos;
                playerStates[player.carId] = state;
            }
        }

        virtual float GetReward(const Player& player, const GameState& state, bool isFinal) override {
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

            PlayerState& ps = playerStates[carId];

            // Check if ball is currently on car roof
            Vec carToBall = state.ball.pos - player.pos;
            float distToBall = carToBall.Length();
            Vec carUp = player.rotMat.up;

            // Ball is "on roof" if it's close, above the car, and aligned with car's up vector
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
            }
            else {
                ps.timeSinceBallOnRoof += DT;
            }

            float reward = 0.0f;

            // Detect flick: player touched ball recently after having it on roof
            if (player.ballTouchedStep && ps.ballWasOnRoof && ps.timeSinceBallOnRoof < maxTimeSinceRoof) {

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
                    float horizontalSpeed = sqrtf(ballVelNorm.x * ballVelNorm.x + ballVelNorm.y * ballVelNorm.y);
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
                        float timingBonus = 1.0f - (ps.timeSinceBallOnRoof / maxTimeSinceRoof);
                        timingBonus = RS_MAX(0.3f, timingBonus);

                        reward = powerBonus * anglePrecision * heightBonus * jumpBonus * timingBonus;

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
    class GoalDirectedTouchSpeedReward : public Reward {
    public:
        float minTouchSpeed;
        float speedPower;

        GoalDirectedTouchSpeedReward(float minTouchSpeed = 350.0f, float speedPower = 1.0f)
            : minTouchSpeed(minTouchSpeed), speedPower(speedPower) {
        }

        virtual float GetReward(const Player& player, const GameState& state, bool isFinal) override {
            if (!player.ballTouchedStep)
                return 0.0f;

            Vec target = player.team == Team::BLUE ? CommonValues::ORANGE_GOAL_BACK : CommonValues::BLUE_GOAL_BACK;
            Vec dirToGoal = (target - state.ball.pos).Normalized();
            float goalAlignment = RS_MAX(0.0f, dirToGoal.Dot(state.ball.vel.Normalized()));
            float speed = RS_CLAMP(
                (state.ball.vel.Length() - minTouchSpeed) /
                RS_MAX(CommonValues::BALL_MAX_SPEED - minTouchSpeed, 1.0f),
                0.0f, 1.0f
            );

            return goalAlignment * std::pow(speed, speedPower);
        }
    };

    class Kickoff2v2SimpleReward : public Reward {
    public:
        float approachWeight;
        float firstTouchReward;
        float closeDistance;
        float supportPenaltyDistance;
        float kickoffBallTolerance;
        float minSupportDistance;
        float maxUsefulDistance;

        Kickoff2v2SimpleReward(
            float approachWeight = 1.10f,
            float firstTouchReward = 1.0f,
            float closeDistance = 200.0f,
            float supportPenaltyDistance = 50.0f,
            float kickoffBallTolerance = 80.0f,
            float minSupportDistance = 750.0f,
            float maxUsefulDistance = 1500.0f)
            : approachWeight(approachWeight),
              firstTouchReward(firstTouchReward),
              closeDistance(closeDistance),
              supportPenaltyDistance(supportPenaltyDistance),
              kickoffBallTolerance(kickoffBallTolerance),
              minSupportDistance(minSupportDistance),
              maxUsefulDistance(maxUsefulDistance) {
        }

        virtual float GetReward(const Player& player, const GameState& state, bool isFinal) override {
            if (std::abs(state.ball.pos.x) > kickoffBallTolerance ||
                std::abs(state.ball.pos.y) > kickoffBallTolerance)
                return 0.0f;

            float playerBallDist = (player.pos - state.ball.pos).Length();
            float teammateBestDist = FLT_MAX;
            for (const Player& other : state.players) {
                if (other.carId == player.carId || other.team != player.team)
                    continue;
                teammateBestDist = RS_MIN(teammateBestDist, (other.pos - state.ball.pos).Length());
            }

            bool isPrimary = playerBallDist <= teammateBestDist;
            Vec dirToBall = (state.ball.pos - player.pos).Normalized();
            float speedToBall = RS_MAX(0.0f, player.vel.Dot(dirToBall)) / CommonValues::CAR_MAX_SPEED;
            float distanceFit = 1.0f - RS_CLAMP(playerBallDist / RS_MAX(maxUsefulDistance, 1.0f), 0.0f, 1.0f);

            float reward = 0.0f;
            if (isPrimary) {
                reward += approachWeight * speedToBall * distanceFit;
                if (player.ballTouchedStep)
                    reward += firstTouchReward;
                if (playerBallDist < closeDistance)
                    reward += 0.25f * (1.0f - playerBallDist / RS_MAX(closeDistance, 1.0f));
            } else {
                if (playerBallDist < minSupportDistance)
                    reward -= supportPenaltyDistance * 0.01f *
                        (1.0f - playerBallDist / RS_MAX(minSupportDistance, 1.0f));
            }

            return reward;
        }
    };

    class IsBallVelocityOnTargetReward : public Reward {
    public:
        float missSoftness;
        bool ownGoal;

        IsBallVelocityOnTargetReward(float missSoftness = 0.3f, bool ownGoal = false)
            : missSoftness(missSoftness), ownGoal(ownGoal) {
        }

        virtual float GetReward(const Player& player, const GameState& state, bool isFinal) override {
            Vec target = player.team == Team::BLUE ? CommonValues::ORANGE_GOAL_CENTER : CommonValues::BLUE_GOAL_CENTER;
            if (ownGoal)
                target = player.team == Team::BLUE ? CommonValues::BLUE_GOAL_CENTER : CommonValues::ORANGE_GOAL_CENTER;

            Vec dirToTarget = (target - state.ball.pos).Normalized();
            float alignment = RS_MAX(0.0f, dirToTarget.Dot(state.ball.vel.Normalized()));
            if (alignment <= 0.0f)
                return 0.0f;

            float targetY = target.y;
            float yVel = state.ball.vel.y;
            if (std::abs(yVel) < 1.0f)
                return 0.0f;

            float t = (targetY - state.ball.pos.y) / yVel;
            if (t <= 0.0f)
                return 0.0f;

            Vec projected = state.ball.pos + state.ball.vel * t;
            float missX = RS_MAX(0.0f, std::abs(projected.x) - CommonValues::GOAL_WIDTH_FROM_CENTER);
            float missZ = RS_MAX(0.0f, projected.z - CommonValues::GOAL_HEIGHT);
            float miss = std::sqrt(missX * missX + missZ * missZ);
            float softness = RS_MAX(1.0f, missSoftness * 1000.0f);
            float onTarget = 1.0f / (1.0f + miss / softness);
            float speed = RS_CLAMP(state.ball.vel.Length() / CommonValues::BALL_MAX_SPEED, 0.0f, 1.0f);

            return alignment * onTarget * speed;
        }
    };

    class BoostSpendPenaltyReward : public Reward {
    public:
        virtual float GetReward(const Player& player, const GameState& state, bool isFinal) override {
            if (!player.prev)
                return 0.0f;
            return -RS_MAX(0.0f, player.prev->boost - player.boost) / 100.0f;
        }
    };

    class FreeGoalEventPenaltyReward : public Reward {
    public:
        float penalty;
        float nearGoalDistance;

        FreeGoalEventPenaltyReward(float penalty = 1.0f, float nearGoalDistance = 1800.0f)
            : penalty(penalty), nearGoalDistance(nearGoalDistance) {
        }

        virtual float GetReward(const Player& player, const GameState& state, bool isFinal) override {
            if (!state.goalScored)
                return 0.0f;

            bool scoredByPlayerTeam = (player.team != RS_TEAM_FROM_Y(state.ball.pos.y));
            if (scoredByPlayerTeam)
                return 0.0f;

            Vec ownGoal = player.team == Team::BLUE ? CommonValues::BLUE_GOAL_CENTER : CommonValues::ORANGE_GOAL_CENTER;
            float closestDefender = FLT_MAX;
            for (const Player& other : state.players) {
                if (other.team == player.team && !other.isDemoed)
                    closestDefender = RS_MIN(closestDefender, (other.pos - ownGoal).Length());
            }

            float exposed = RS_CLAMP(closestDefender / RS_MAX(nearGoalDistance, 1.0f), 0.0f, 1.0f);
            return -penalty * exposed;
        }
    };

    class StagedFlickReward : public Reward {
    public:
        float carryReward;
        float popReward;
        float powerReward;

        StagedFlickReward(float carryReward = 0.25f, float popReward = 0.75f, float powerReward = 1.5f)
            : carryReward(carryReward), popReward(popReward), powerReward(powerReward) {
        }

        virtual float GetReward(const Player& player, const GameState& state, bool isFinal) override {
            Vec toBall = state.ball.pos - player.pos;
            float dist = toBall.Length();
            float roofAlignment = toBall.Normalized().Dot(player.rotMat.up);
            bool carrying =
                dist < 210.0f &&
                state.ball.pos.z > player.pos.z + 70.0f &&
                state.ball.pos.z < player.pos.z + 190.0f &&
                roofAlignment > 0.45f;

            float reward = carrying ? carryReward : 0.0f;
            if (player.ballTouchedStep && !player.isOnGround && state.ball.vel.z > 250.0f)
                reward += popReward * RS_CLAMP(state.ball.vel.z / 1200.0f, 0.0f, 1.0f);

            if (player.ballTouchedStep) {
                Vec target = player.team == Team::BLUE ? CommonValues::ORANGE_GOAL_BACK : CommonValues::BLUE_GOAL_BACK;
                Vec dirToGoal = (target - state.ball.pos).Normalized();
                float goalPower = RS_MAX(0.0f, state.ball.vel.Dot(dirToGoal)) / CommonValues::BALL_MAX_SPEED;
                reward += powerReward * RS_CLAMP(goalPower, 0.0f, 1.0f);
            }

            return reward;
        }
    };
} // namespace RLGC
