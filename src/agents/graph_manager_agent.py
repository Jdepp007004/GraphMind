"""
src/agents/graph_manager_agent.py

LangGraph agent node. Uses Gemma 2B to reason about which nodes to promote/demote.
Falls back to rule-based if Gemma not available.
"""

import logging
import os
from typing import Dict, Any, List

from config import settings
from src.core.graph_engine import BehaviouralGraph, GraphNode
from src.core.memory_manager import MemoryManager

logger = logging.getLogger(__name__)


class GraphManagerAgent:
    """
    LangGraph agent that manages graph decisions using Gemma 2B reasoning.
    Gemma is given current HOT tier contents and asked which nodes to keep/evict.
    """

    def __init__(self, graph: BehaviouralGraph, memory_manager: MemoryManager) -> None:
        """
        Store references. Load Gemma tokenizer + model or use fallback.
        If GEMMA_LOCAL_PATH exists: load model. Else set self.use_llm = False.
        """
        self.graph = graph
        self.memory_manager = memory_manager
        self.use_llm = False
        self.tokenizer = None
        self.model = None
        if os.path.isdir(settings.GEMMA_LOCAL_PATH):
            try:
                from transformers import AutoTokenizer, AutoModelForCausalLM
                import torch
                self.tokenizer = AutoTokenizer.from_pretrained(settings.GEMMA_LOCAL_PATH)
                self.model = AutoModelForCausalLM.from_pretrained(
                    settings.GEMMA_LOCAL_PATH, torch_dtype=torch.float32
                )
                self.model.eval()
                self.use_llm = True
                logger.info("GraphManagerAgent: using Gemma LLM")
            except Exception as e:
                logger.warning(f"Gemma load failed: {e}. Using rule-based fallback.")

    def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Main agent function called by LangGraph.
        1. Get HOT tier contents from memory_manager.
        2. If self.use_llm: prompt Gemma with HOT tier context and time_of_day -> get node priority decisions.
        3. Else: use rule-based priority (sort by access_count descending).
        4. Reorder HOT tier based on decisions.
        5. Run graph.prune_weak_edges() if current_day % 7 == 0.
        6. Append reasoning to state['messages'].
        7. Update state['last_agent'] = 'graph_manager'.
        8. Return updated state.
        """
        hot_ids = self.memory_manager.get_hot_node_ids()
        hot_nodes = [self.graph.get_node(nid) for nid in hot_ids if self.graph.get_node(nid)]
        current_day = state.get("current_day", 0)
        time_bucket = state.get("current_event", {}).get("time_bucket", 0) if state.get("current_event") else 0

        if self.use_llm:
            prompt = self._build_gemma_prompt(hot_nodes, time_bucket)
            priority_ids = self._query_gemma(prompt, hot_ids)
        else:
            # Rule-based: sort by access_count descending
            sorted_nodes = sorted(hot_nodes, key=lambda n: n.access_count, reverse=True)
            priority_ids = [n.node_id for n in sorted_nodes]

        # Re-promote in priority order (updates LRU ordering)
        for nid in priority_ids[:5]:
            self.memory_manager.promote_to_hot(nid)

        # Prune weekly
        pruned = 0
        if current_day % 7 == 0:
            pruned = self.graph.prune_weak_edges()

        state["last_agent"] = "graph_manager"
        state["messages"].append({
            "agent": "graph_manager",
            "hot_count": len(hot_ids),
            "pruned_edges": pruned,
            "llm_used": self.use_llm
        })
        return state

    def _build_gemma_prompt(self, hot_nodes: List[GraphNode], time_of_day: int) -> str:
        """
        PRIVATE. Build a short prompt for Gemma describing current HOT tier nodes.
        Ask Gemma: 'Given these apps in cache and time of day, which should be prioritized?'
        Return prompt string. Keep under 256 tokens for speed.
        """
        app_list = ", ".join(n.app_id for n in hot_nodes[:10])
        return (f"Time of day bucket: {time_of_day}/47. "
                f"Apps in cache: {app_list}. "
                f"Which 3 apps should be highest priority? List app IDs only.")

    def _query_gemma(self, prompt: str, fallback_ids: List[str]) -> List[str]:
        """Query Gemma and parse response, falling back to original order."""
        try:
            import torch
            inputs = self.tokenizer(prompt, return_tensors="pt", max_length=128, truncation=True)
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs["input_ids"],
                    max_new_tokens=64,
                    do_sample=False
                )
            text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            return self._parse_gemma_response(text) or fallback_ids
        except Exception:
            return fallback_ids

    def _parse_gemma_response(self, response: str) -> List[str]:
        """
        PRIVATE. Parse Gemma's response to extract app names or node_ids to prioritize.
        Return list of node_ids in priority order.
        Falls back to original order if parsing fails.
        """
        # Simple heuristic: find known app IDs in the response
        from src.data.context_encoder import APP_ID_VOCAB
        found = [app for app in APP_ID_VOCAB if app in response]
        return found if found else []
