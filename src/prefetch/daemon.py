"""
src/prefetch/daemon.py

Background daemon that proactively warms the HOT/WARM cache
based on predicted next nodes.
"""

import logging
import threading
from typing import Optional, List

from config import settings
from src.core.event_bus import (
    EventBus, TOPIC_APP_LAUNCHED, TOPIC_BATTERY_UPDATED,
    TOPIC_HEADPHONES_CONNECTED, TOPIC_CALENDAR_EVENT, TOPIC_PREFETCH_TRIGGERED
)
from src.core.memory_manager import MemoryManager
from src.core.graph_engine import BehaviouralGraph

logger = logging.getLogger(__name__)


class PrefetchDaemon:
    """
    Runs periodic pre-fetching of predicted next nodes into HOT tier.
    Triggered by time, events, and context signals.
    """

    def __init__(self, user_id: str, graph: BehaviouralGraph,
                 memory_manager: MemoryManager) -> None:
        """
        Store references. Do NOT start the scheduler here.
        Subscribe to EventBus:
            TOPIC_APP_LAUNCHED -> _on_app_launched()
            TOPIC_HEADPHONES_CONNECTED -> _on_headphones_connected()
            TOPIC_CALENDAR_EVENT -> _on_calendar_event()
            TOPIC_BATTERY_UPDATED -> _on_battery_updated()
        Set self.current_battery = 100.0
        Set self.current_node_id = None
        Set self.scheduler = None (initialized in start())
        """
        self.user_id = user_id
        self.graph = graph
        self.memory_manager = memory_manager
        self.current_battery: float = 100.0
        self.current_node_id: Optional[str] = None
        self.scheduler = None

        bus = EventBus.get_instance()
        bus.subscribe(TOPIC_APP_LAUNCHED, self._on_app_launched)
        bus.subscribe(TOPIC_BATTERY_UPDATED, self._on_battery_updated)
        bus.subscribe(TOPIC_HEADPHONES_CONNECTED, self._on_headphones_connected)
        bus.subscribe(TOPIC_CALENDAR_EVENT, self._on_calendar_event)

    def start(self) -> None:
        """
        Start the APScheduler background scheduler.
        Add a job: call run_prefetch_cycle() every PREFETCH_INTERVAL_MINUTES minutes.
        Start scheduler.
        Log: 'PrefetchDaemon started for user {user_id}'
        """
        try:
            from apscheduler.schedulers.background import BackgroundScheduler
            self.scheduler = BackgroundScheduler()
            self.scheduler.add_job(
                self.run_prefetch_cycle,
                "interval",
                minutes=settings.PREFETCH_INTERVAL_MINUTES
            )
            self.scheduler.start()
            logger.info(f"PrefetchDaemon started for user {self.user_id}")
        except Exception as e:
            logger.error(f"Could not start PrefetchDaemon: {e}")

    def stop(self) -> None:
        """Shutdown the scheduler gracefully."""
        if self.scheduler and self.scheduler.running:
            self.scheduler.shutdown(wait=False)
            logger.info(f"PrefetchDaemon stopped for user {self.user_id}")

    def run_prefetch_cycle(self) -> List[str]:
        """
        Main prefetch logic. Called every 15 minutes.
        1. If self.current_battery < BATTERY_SUPPRESS_THRESHOLD: k = 2, else k = PREFETCH_TOP_K
        2. If self.current_node_id is None: return []
        3. Call graph.get_top_k_next_nodes(self.current_node_id, k, self.current_battery)
        4. Call memory_manager.rebuild_warm_from_graph(predicted_ids)
        5. For top 2 predicted nodes: call memory_manager.promote_to_hot()
        6. Publish TOPIC_PREFETCH_TRIGGERED with {'user_id': user_id, 'prefetched_ids': list, 'battery': float}
        7. Returns list of prefetched node_ids.
        """
        if self.current_node_id is None:
            return []
        k = 2 if self.current_battery < settings.BATTERY_SUPPRESS_THRESHOLD else settings.PREFETCH_TOP_K
        predicted_ids = self.graph.get_top_k_next_nodes(
            self.current_node_id, k, self.current_battery
        )
        if predicted_ids:
            self.memory_manager.rebuild_warm_from_graph(predicted_ids)
            for nid in predicted_ids[:2]:
                self.memory_manager.promote_to_hot(nid)
        bus = EventBus.get_instance()
        bus.publish(TOPIC_PREFETCH_TRIGGERED, {
            "timestamp": 0.0,
            "user_id": self.user_id,
            "prefetched_ids": predicted_ids,
            "battery": self.current_battery
        })
        logger.debug(f"PrefetchDaemon: prefetched {len(predicted_ids)} nodes for {self.user_id}")
        return predicted_ids

    def _on_app_launched(self, payload: dict) -> None:
        """PRIVATE. Update self.current_node_id from the launched app's node."""
        if payload.get("user_id") != self.user_id:
            return
        app_id = payload.get("app_id", "unknown")
        time_bucket = int(payload.get("time_of_day_bucket", 0))
        battery = float(payload.get("battery", 100.0))
        battery_bucket = min(4, int(battery / 20))
        # Find matching node
        for nid in self.graph._graph.nodes():
            n = self.graph._graph.nodes[nid]["data"]
            if (n.app_id == app_id and n.time_bucket == time_bucket
                    and n.battery_bucket == battery_bucket):
                self.current_node_id = nid
                break

    def _on_battery_updated(self, payload: dict) -> None:
        """PRIVATE. Update self.current_battery."""
        if payload.get("user_id") == self.user_id:
            self.current_battery = float(payload.get("battery", self.current_battery))

    def _on_headphones_connected(self, payload: dict) -> None:
        """PRIVATE. Immediately promote music/entertainment nodes to HOT."""
        if payload.get("user_id") != self.user_id:
            return
        entertainment_apps = {"com.spotify.music", "com.google.youtube",
                               "com.netflix.mediaclient", "com.tiktok.android"}
        for nid in list(self.graph._graph.nodes()):
            n = self.graph._graph.nodes[nid]["data"]
            if n.app_id in entertainment_apps or n.category in ("entertainment", "social"):
                self.memory_manager.promote_to_hot(nid)
                break  # Promote top match only

    def _on_calendar_event(self, payload: dict) -> None:
        """
        PRIVATE. If event in <= 30 minutes:
        Identify nodes related to productivity/enterprise apps.
        Promote top-3 to HOT immediately.
        """
        if payload.get("user_id") != self.user_id:
            return
        minutes = payload.get("minutes_until_event", 60)
        if minutes <= 30:
            productivity_categories = {"productivity", "enterprise", "utility"}
            promoted = 0
            for nid in list(self.graph._graph.nodes()):
                if promoted >= 3:
                    break
                n = self.graph._graph.nodes[nid]["data"]
                if n.category in productivity_categories:
                    self.memory_manager.promote_to_hot(nid)
                    promoted += 1
