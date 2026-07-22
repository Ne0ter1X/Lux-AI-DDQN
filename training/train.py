# training/train.py
import numpy as np
from agent.model import get_model
from training.replay_buffer import ReplayBuffer


def train_step(replay_buffer, model, target_model, config):
    if len(replay_buffer) < config.min_memory_for_training:
        return None

    states, action_maps, rewards, next_states, dones, next_masks = replay_buffer.sample(config.batch_size)
    rewards = np.clip(rewards, -1.0, 1.0)  # stability

    q_vals = model(states, training=False).numpy()
    q_next_main = model(next_states, training=False).numpy()
    q_next_target = target_model(next_states, training=False).numpy()

    B, H, W = action_maps.shape

    dy = np.array([0, 1, -1, 0, 0, 0, 0, 0])
    dx = np.array([0, 0, 0, -1, 1, 0, 0, 0])

    batch_idx = np.arange(B)[:, None, None]
    y_idx = np.arange(H)[None, :, None]
    x_idx = np.arange(W)[None, None, :]

    a = action_maps.astype(np.int32)
    dest_y = np.clip(y_idx + dy[a], 0, H - 1)
    dest_x = np.clip(x_idx + dx[a], 0, W - 1)

    B_idx_flat = np.repeat(np.arange(B), H * W).reshape(B, H, W)

    mask_next_dest = next_masks[B_idx_flat, dest_y, dest_x, :]  # (B,H,W,8)

    # Get Q-values at destination from main network; mask invalid
    q_next_main_dest = q_next_main[B_idx_flat, dest_y, dest_x, :]  # (B,H,W,8)
    masked_q_dest = np.where(mask_next_dest == 1, q_next_main_dest, -1e9)

    # Best action at destination according to main network
    best_actions_dest = np.argmax(masked_q_dest, axis=-1)  # (B,H,W)

    # Get Q-value from target network for that best action at destination
    q_next_target_dest = q_next_target[B_idx_flat, dest_y, dest_x, :]  # (B,H,W,8)
    best_q_target_dest = np.take_along_axis(q_next_target_dest, best_actions_dest[..., None], axis=-1).squeeze(
        -1)  # (B,H,W)

    td_targets = rewards[:, None, None] + config.gamma * best_q_target_dest * (~dones[:, None, None])
    td_targets = np.clip(td_targets, -2.0, 2.0)  # safety

    # Current Q for the taken actions
    targets = q_vals.copy()
    current_q = targets[batch_idx, y_idx, x_idx, a]  # (B,H,W)

    player_units_presence = np.any(states[:, :, :, 6:11] > 0, axis=-1)  # (B,H,W)
    player_cities_presence = np.any(states[:, :, :, 14:17] > 0, axis=-1)
    active_mask = (player_units_presence | player_cities_presence)  # (B,H,W)

    targets[batch_idx, y_idx, x_idx, a] = np.where(
        active_mask,
        td_targets,
        current_q
    )

    loss = model.train_on_batch(states, targets)
    return loss

# training/train.py (continued)

class AgentTrainer:
    def __init__(self, model, target_model, replay_buffer, config):
        self.model = model
        self.target_model = target_model
        self.replay_buffer = replay_buffer
        self.config = config
        self.last_state = {}
        self.epsilon = config.epsilon_start

    def agent_fn(self, observation, configuration):
        """
        This function will be passed to env.run().
        It replaces the notebook's global 'agent' function.
        """
        from lux.game import Game
        import copy
        from agent.state_utils import get_inputs, get_action_mask_from_state, get_prediction_actions, calculate_shaped_reward

        if observation["step"] == 0:
            self.game_state = Game()
            self.game_state._initialize(observation["updates"])
            self.game_state._update(observation["updates"][2:])
            self.game_state.id = observation.player
            self.prev_game_state = None
        else:
            self.game_state._update(observation["updates"])

        player = self.game_state.players[observation.player]
        opponent = self.game_state.players[(observation.player + 1) % 2]

        shaped_rew = calculate_shaped_reward(self.prev_game_state, self.game_state, observation.player) if self.prev_game_state else 0.0
        rl_reward = shaped_rew
        self.prev_game_state = copy.deepcopy(self.game_state)

        state = get_inputs(self.game_state)
        q_values = self.model.predict(np.asarray([state]), verbose=0)[0]
        mask = get_action_mask_from_state(self.game_state, observation.player)

        # epsilon‑greedy
        if np.random.random() < self.epsilon:
            option = np.zeros((state.shape[0], state.shape[1]), dtype=np.int32)
            for y in range(state.shape[0]):
                for x in range(state.shape[1]):
                    valid = np.where(mask[y, x] == 1)[0]
                    if len(valid) > 0:
                        option[y, x] = np.random.choice(valid)
        else:
            masked_q = np.where(mask == 1, q_values, -np.inf)
            option = np.argmax(masked_q, axis=-1)

        actions = get_prediction_actions(q_values, player, self.game_state, option)

        # Store transition in replay buffer (same logic as notebook)
        pid = observation.player
        is_done = (observation["step"] == 359 or
                   (len(opponent.units) == 0 and len(opponent.cities) == 0) or
                   (len(player.units) == 0 and len(player.cities) == 0))

        if pid in self.last_state:
            prev_s, prev_opt = self.last_state[pid]
            if is_done:
                # terminal reward from outcome
                player_city_count = len(player.cities)
                opponent_city_count = len(opponent.cities)
                if player_city_count > opponent_city_count:
                    final_reward = 1.0
                elif player_city_count < opponent_city_count:
                    final_reward = -1.0
                else:
                    final_reward = 0.0
                self.replay_buffer.push(prev_s, prev_opt, final_reward, state, True, mask)
                del self.last_state[pid]
            else:
                self.replay_buffer.push(prev_s, prev_opt, rl_reward, state, False, mask)

        if not is_done:
            self.last_state[pid] = (state, option)

        # Decay epsilon per step? The notebook decays per episode; we'll move epsilon update to the episode loop.
        return actions
