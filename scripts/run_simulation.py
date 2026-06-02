"""
scripts/run_simulation.py

Entry point: runs full orchestrated simulation for one user.
Usage:
    python scripts/run_simulation.py --user user_00
"""

import sys
import os
import argparse
import logging

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import settings
from src.agents.orchestrator import GraphMindOrchestrator
from src.data.event_simulator import EventSimulator
from src.core.event_bus import EventBus

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


if __name__ == "__main__":
    """Run orchestrated simulation for one user."""
    parser = argparse.ArgumentParser(description="Run GraphMind simulation")
    parser.add_argument("--user", type=str, default="user_00")
    args = parser.parse_args()

    logger.info(f"Starting simulation for {args.user}")
    EventBus.get_instance().clear_all()
    orch = GraphMindOrchestrator(args.user)
    states = orch.run_full_simulation()
    logger.info(f"Simulation complete for {args.user}. {len(states)} days processed.")
