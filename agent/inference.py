import numpy as np
from lux.game import Game
from .model import get_model
from .state_utils import get_inputs, get_action_mask_from_state, get_prediction_actions


class DQNAgent:
    def __init__(self, model_path, grid_size=12):
        self.model = get_model(grid_size)
        self.model.load_weights(model_path)
        self.grid_size = grid_size
        self.init_updates = None  # Store initialization commands

    def act(self, observation):
        game_state = Game()

        # print(f"Step: {observation['step']}, Player: {observation['player']}")
        # print(f"Number of updates: {len(observation['updates'])}")
        # print(f"First 3 updates: {observation['updates'][:3]}")

        if observation["step"] == 0:
            # game_state._initialize(observation["updates"][:2])
            game_state._initialize(observation["updates"])
            game_state._update(observation["updates"][2:])
            self.init_updates = observation["updates"][:2]
        else:
            full_updates = self.init_updates + observation["updates"]
            # game_state._initialize(observation["updates"][:2])
            game_state._initialize(full_updates)
            game_state._update(full_updates[2:])
            # game_state._initialize(observation["updates"])
            # game_state._update(observation["updates"][2:])
            # game_state._update(observation["updates"]) if we don't do a stateless Game() transfer
        game_state.id = observation['player']

        state = get_inputs(game_state)
        state = np.expand_dims(state, axis=0)
        q_values = self.model.predict(state, verbose=0)[0]   # shape (H,W,8)

        mask = get_action_mask_from_state(game_state, observation['player'])
        masked_q = np.where(mask == 1, q_values, -np.inf)
        option = np.argmax(masked_q, axis=-1)

        player = game_state.players[observation['player']]
        return get_prediction_actions(q_values, player, game_state, option)
