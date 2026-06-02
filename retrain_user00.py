"""Quick retrain of user_00 PPO model with current SB3 version."""
import sys
import os
sys.path.insert(0, '.')

import logging
logging.basicConfig(level=logging.INFO)

from src.rl.trainer import RLTrainer

trainer = RLTrainer()
print("Retraining user_00 with current SB3...")
path = trainer.train_user("user_00", total_timesteps=5000)  # quick train
print(f"Saved to: {path}")

# Verify it loads
model = trainer.load_policy("user_00")
import numpy as np
obs = np.zeros((68,), dtype=np.float32)
action, _ = model.predict(obs)
print(f"Predict OK, action={int(action)}")
print("DONE")
