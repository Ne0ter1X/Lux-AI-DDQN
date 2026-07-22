import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
from kaggle_environments import make
from tensorflow import keras
from agent.model import get_model
from training.replay_buffer import ReplayBuffer
from training.config import TrainingConfig
from training.train import AgentTrainer, train_step
from agent.state_utils import get_final_outcome


def main():
    config = TrainingConfig()
    model = get_model(config.grid_size)
    target_model = get_model(config.grid_size)
    target_model.set_weights(model.get_weights())
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=config.learning_rate,
                                        clipnorm=config.clipnorm),
        loss='huber'
    )

    replay_buffer = ReplayBuffer(maxlen=config.memory_size)
    trainer = AgentTrainer(model, target_model, replay_buffer, config)

    epsilon = config.epsilon_start

    for episode in range(config.num_episodes):
        opponent = "random_agent" if episode < config.opponent_switch_episode else "simple_agent"
        env = make("lux_ai_2021", debug=False,
                   configuration={"width": config.grid_size, "height": config.grid_size})
        steps = env.run([trainer.agent_fn, opponent])

        # After episode, handle any leftover terminal transitions for both players
        for pid in (0, 1):
            if pid in trainer.last_state:
                prev_s, prev_opt = trainer.last_state[pid]
                outcome = get_final_outcome(steps, pid)
                final_reward = 1.0 if outcome == 1 else (-1.0 if outcome == -1 else 0.0)
                dummy_mask = np.zeros((config.grid_size, config.grid_size, 8))
                replay_buffer.push(prev_s, prev_opt, final_reward, prev_s, True, dummy_mask)
                del trainer.last_state[pid]

        epsilon = max(config.epsilon_final,
                      config.epsilon_start - (config.epsilon_start - config.epsilon_final) * episode / config.epsilon_decay_episodes)
        trainer.epsilon = epsilon

        if len(replay_buffer) >= config.min_memory_for_training:
            train_step(replay_buffer, model, target_model, config)

        if episode % config.target_update_freq == 0:
            target_model.set_weights(model.get_weights())

        # print(f"Episode {episode} finished.")

    model.save("models/model_final.h5")
    print("Training complete.")

if __name__ == "__main__":
    main()