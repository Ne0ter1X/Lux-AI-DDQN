# import numpy as np
# import requests, time
# from kaggle_environments import make
# from agent.state_utils import get_final_outcome
#
# API_URL = "http://localhost:8000/predict"
# EPISODES = 100
# wins = 0
# reward_sum = 0
# latencies = []
#
# for ep in range(EPISODES):
#
#     env = make("lux_ai_2021", configuration={"width":12, "height":12})
#     env.reset()
#     done = False
#     episode_reward = 0
#     opponent = "random_agent"
#     while not done:
#         obs = env.state[0]["observation"]
#         resp = requests.post(API_URL, json=obs)
#         print("Status:", resp.status_code)
#         print("Response JSON:", resp.json())  # will show the full dict
#         t0 = time.time()
#         resp = requests.post(API_URL, json=obs)
#         latencies.append(time.time() - t0)
#         actions = resp.json()["actions"]
#         states, info = env.step([actions, opponent])
#     # определение победителя
#     if get_final_outcome(env.steps, 0) == 1:
#         wins += 1
#     episode_reward = env.state[0].reward  # total reward for agent 0
#     reward_sum += episode_reward
#
# print(f"Win rate: {wins/EPISODES:.2f}")
# print(f"Avg reward: {reward_sum/EPISODES:.2f}")
# print(f"Avg latency: {np.mean(latencies)*1000:.1f} ms")

import numpy as np
import requests
import time
import json
from datetime import datetime
from kaggle_environments import make
from agent.state_utils import get_final_outcome

import numpy as np
import requests
import time
import json
from kaggle_environments import make
from agent.state_utils import get_final_outcome

API_URL = "http://localhost:8000/predict"
EPISODES = 100
LOG_FILE = "test_results.jsonl"

# --- Latency recording wrapper for http_agent ---
class LatencyRecorder:
    def __init__(self):
        self.latencies = []

    def agent_fn(self):
        recorder = self
        def agent(observation, configuration):
            t0 = time.perf_counter()
            resp = requests.post(API_URL, json=observation)
            lat = time.perf_counter() - t0
            recorder.latencies.append(lat)
            resp.raise_for_status()
            return resp.json()["actions"]
        return agent

recorder = LatencyRecorder()
agent_fn = recorder.agent_fn()

# --- Main loop ---
results = []
wins = 0
draws = 0
losses = 0
total_reward = 0.0      # use custom reward (1/-1/0)

for ep in range(1, EPISODES + 1):
    env = make("lux_ai_2021", configuration={"width": 12, "height": 12},
               debug=False)
    t_start = time.perf_counter()
    steps = env.run([agent_fn, "random_agent"])
    elapsed = time.perf_counter() - t_start

    # 1. Outcome (reliable)
    outcome = get_final_outcome(steps, player_id=0)
    if outcome == 1:
        wins += 1
        custom_reward = 1.0
    elif outcome == 0.5:
        draws += 1
        custom_reward = 0.0
    else:
        losses += 1
        custom_reward = -1.0
    total_reward += custom_reward

    # 2. Number of steps and latency
    n_steps = len(steps)
    avg_lat = (np.mean(recorder.latencies[-n_steps:]) * 1000
               if n_steps > 0 else 0.0)

    # 3. Per‑episode log entry
    ep_log = {
        "episode": ep,
        "outcome": outcome,                 # 1, 0.5, 0
        "reward": custom_reward,            # 1, 0, -1
        "steps": n_steps,
        "wall_time_s": round(elapsed, 2),
        "avg_latency_ms": round(avg_lat, 1),
    }
    results.append(ep_log)
    recorder.latencies.clear()   # reset for next episode

    if ep % 10 == 0:
        print(f"Episode {ep}/{EPISODES} | "
              f"W: {wins}, L: {losses}, D: {draws} | "
              f"Avg reward: {total_reward/ep:.2f}")

# --- Write logs ---
with open(LOG_FILE, "w") as f:
    for entry in results:
        f.write(json.dumps(entry) + "\n")

# --- Summary ---
print("\n=== FINAL RESULTS ===")
print(f"Played : {EPISODES}")
print(f"Wins   : {wins} ({wins/EPISODES:.2%})")
print(f"Draws  : {draws} ({draws/EPISODES:.2%})")
print(f"Losses : {losses} ({losses/EPISODES:.2%})")
print(f"Avg custom reward : {total_reward/EPISODES:.3f}")
print(f"Avg steps         : {np.mean([r['steps'] for r in results]):.1f}")
print(f"Avg latency (ms)  : {np.mean([r['avg_latency_ms'] for r in results]):.1f}")
print(f"Logs saved to {LOG_FILE}")