"""
scripts/train_rl.py

Entry point: trains RL policies for one or all users.
Usage:
    python scripts/train_rl.py --user user_00 --timesteps 50000
    python scripts/train_rl.py --all --timesteps 200000
"""

import sys
import os
import argparse
import logging

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import settings
from src.rl.trainer import RLTrainer
from src.data.event_simulator import EventSimulator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


if __name__ == "__main__":
    """Train RL policies."""
    parser = argparse.ArgumentParser(description="Train GraphMind RL policies")
    parser.add_argument("--user", type=str, default=None)
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--timesteps", type=int, default=settings.PPO_TOTAL_TIMESTEPS)
    args = parser.parse_args()

    trainer = RLTrainer()
    if args.all:
        results = trainer.train_all_users()
        logger.info(f"Trained all users: {list(results.keys())}")
    elif args.user:
        path = trainer.train_user(args.user, total_timesteps=args.timesteps)
        logger.info(f"Saved to: {path}")
    else:
        parser.print_help()
