"""Quick retrain of user_00 PPO model using system Python with current SB3."""
import sys
import os
sys.path.insert(0, '.')

import logging
logging.basicConfig(level=logging.WARNING)  # suppress info spam

import numpy as np
from src.rl.environment import GraphMindEnv
from stable_baselines3 import PPO
from config import settings

print("Creating environment...")
env = GraphMindEnv("user_00")

print("Creating PPO model...")
model = PPO(
    "MlpPolicy",
    env,
    learning_rate=settings.PPO_LEARNING_RATE,
    n_steps=min(settings.PPO_N_STEPS, 512),
    batch_size=settings.PPO_BATCH_SIZE,
    n_epochs=settings.PPO_N_EPOCHS,
    gamma=settings.PPO_GAMMA,
    verbose=0,
    seed=settings.RANDOM_SEED
)

print("Training (3000 timesteps)...")
model.learn(total_timesteps=3000)

save_path = os.path.join(settings.RL_MODELS_DIR, "user_00_ppo")
model.save(save_path)
print(f"Saved to: {save_path}.zip")

# Verify it loads
model2 = PPO.load(save_path + ".zip")
obs = np.zeros((68,), dtype=np.float32)
action, _ = model2.predict(obs)
print(f"Predict OK, action={int(action)}")

env.close()
print("DONE")
