"""Benchmark utilities for BPlusOmapHotNodesClient."""

import argparse
import json
import os
import random
import statistics
import sys
from typing import Any, Dict, List, Optional, Sequence, Tuple

# Support direct file execution: python daoram/omap/bplus_omap_hot_benchmark.py ...
if __package__ in (None, ""):
    ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    if ROOT_DIR not in sys.path:
        sys.path.insert(0, ROOT_DIR)

from daoram.dependency import InteractLocalServer
from daoram.omap.bplus_omap import BPlusOmap
from daoram.omap.bplus_omap_hot import BPlusOmapHotNodesClient
from daoram.omap.hot_cache_admission import (
    RejectAllHotCacheAdmissionLayer,
    ScoreBasedHotCacheAdmissionLayer,
)


DEFAULT_ORDER = 4
DEFAULT_KEY_SIZE = 10
DEFAULT_PLOT_BUCKET_SIZE = 16


class _InstrumentedBPlusOmapHotNodesClient(BPlusOmapHotNodesClient):
    """Benchmark-only hot client that records cache-full admission decisions."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._benchmark_cache_full_decisions: List[Dict[str, Any]] = []
        self._benchmark_actual_evictions: List[Dict[str, Any]] = []

    def _decide_hot_cache_admission(self, node):
        candidate = self._make_hot_cache_candidate(node)
        residents = [self._make_hot_cache_candidate(cached) for cached in self._hot_nodes_client.values()]

        decision_record = None
        if self._hot_nodes_client_size > 0 and len(residents) >= self._hot_nodes_client_size:
            resident_snapshot = [
                {
                    "key": resident.key,
                    "access_count": resident.access_count,
                    "secret_user_id": resident.secret_user_id,
                }
                for resident in residents
            ]
            min_resident_access_count = min((resident.access_count for resident in residents), default=None)
            decision_record = {
                "candidate_key": candidate.key,
                "candidate_access_count": candidate.access_count,
                "candidate_secret_user_id": candidate.secret_user_id,
                "resident_snapshot": resident_snapshot,
                "min_resident_access_count": min_resident_access_count,
            }
            self._benchmark_cache_full_decisions.append(decision_record)

        decision = super()._decide_hot_cache_admission(node=node)

        if decision_record is not None:
            decision_record["decision"] = {
                "admit": bool(decision.admit),
                "evict_key": decision.evict_key,
            }
            if decision.admit and decision.evict_key is not None:
                chosen_resident = next(
                    (resident for resident in residents if resident.key == decision.evict_key),
                    None,
                )
                if chosen_resident is not None:
                    self._benchmark_actual_evictions.append(
                        {
                            **decision_record,
                            "chosen_victim_access_count": chosen_resident.access_count,
                            "is_lowest_access_eviction": (
                                chosen_resident.access_count == decision_record["min_resident_access_count"]
                            ),
                        }
                    )

        return decision
    def get_benchmark_eviction_summary(self) -> Dict[str, Any]:
        """Return benchmark-local eviction observability."""
        total_actual_evictions = len(self._benchmark_actual_evictions)
        matching_lowest_access_evictions = sum(
            1 for record in self._benchmark_actual_evictions if record["is_lowest_access_eviction"]
        )
        return {
            "cache_full_decisions": list(self._benchmark_cache_full_decisions),
            "actual_eviction_records": list(self._benchmark_actual_evictions),
            "total_cache_full_decisions": len(self._benchmark_cache_full_decisions),
            "total_actual_evictions": total_actual_evictions,
            "matching_lowest_access_evictions": matching_lowest_access_evictions,
            "eviction_accuracy": (
                None
                if total_actual_evictions == 0
                else matching_lowest_access_evictions / total_actual_evictions
            ),
        }


class _UnpaddedBenchmarkBPlusOmap(BPlusOmap):
    """Benchmark-only plain B+ OMAP with search-padding removed."""

    def search(self, key: Any, value: Any = None) -> Any:
        if key is None:
            return None

        if self.root is None:
            return None

        if self._local:
            raise MemoryError("The local storage was not emptied before this operation.")

        self._find_leaf_to_local(key=key)
        leaf = self._local.get_leaf()
        search_value = None

        for index, each_key in enumerate(leaf.value.keys):
            if key == each_key:
                search_value = leaf.value.values[index]
                if value is not None:
                    leaf.value.values[index] = value
                break

        self._flush_local_to_stash()
        return search_value


class BPlusOmapHotNodesBenchmark:
    """Run the same seeded search workload against multiple B+ OMAP variants."""

    def __init__(self, order: int = DEFAULT_ORDER):
        self._order = order

    @staticmethod
    def _snapshot_client(client: InteractLocalServer) -> Dict[str, int]:
        bytes_read, bytes_written = client.get_bandwidth()
        rounds = client.get_rounds() if hasattr(client, "get_rounds") else 0
        return {
            "bytes_read": bytes_read,
            "bytes_written": bytes_written,
            "rounds": rounds,
        }

    @staticmethod
    def _build_payload_map(keys: Sequence[int], payload_size: int, rng: random.Random) -> Dict[int, bytes]:
        if payload_size < 0:
            raise ValueError("payload_size must be non-negative.")
        return {key: rng.randbytes(payload_size) for key in keys}

    @staticmethod
    def _build_hotset_request_trace(
        *,
        keys: Sequence[int],
        rng: random.Random,
        warmup_requests: int,
        num_queries: int,
        hotset_size: int,
        hot_query_probability: float,
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        if hotset_size <= 0 or hotset_size >= len(keys):
            raise ValueError("hotset_size must be in [1, insert_count - 1].")
        if hot_query_probability < 0.0 or hot_query_probability > 1.0:
            raise ValueError("hot_query_probability must be in [0.0, 1.0].")

        hotset_keys = sorted(rng.sample(list(keys), hotset_size))
        hotset = set(hotset_keys)
        coldset_keys = [key for key in keys if key not in hotset]
        if not coldset_keys:
            raise ValueError("hotset_size must leave at least one cold key.")

        request_trace: List[Dict[str, Any]] = []
        total_requests = warmup_requests + num_queries
        for request_index in range(total_requests):
            phase = "warmup" if request_index < warmup_requests else "measured"
            use_hot = rng.random() < hot_query_probability
            key = rng.choice(hotset_keys if use_hot else coldset_keys)
            request_trace.append(
                {
                    "request_index": request_index,
                    "phase": phase,
                    "key": key,
                    "label": "hot" if use_hot else "cold",
                }
            )

        metadata = {
            "workload": "hotset",
            "hotset_keys": hotset_keys,
            "coldset_size": len(coldset_keys),
            "hotset_size": len(hotset_keys),
            "hot_query_probability": hot_query_probability,
        }
        return request_trace, metadata

    @staticmethod
    def _build_zipf_request_trace(
        *,
        keys: Sequence[int],
        rng: random.Random,
        warmup_requests: int,
        num_queries: int,
        zipf_alpha: float,
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        if zipf_alpha <= 0.0:
            raise ValueError("zipf_alpha must be positive.")

        ranked_keys = list(keys)
        rng.shuffle(ranked_keys)
        weights = [1.0 / ((rank + 1) ** zipf_alpha) for rank in range(len(ranked_keys))]

        request_trace: List[Dict[str, Any]] = []
        total_requests = warmup_requests + num_queries
        for request_index in range(total_requests):
            phase = "warmup" if request_index < warmup_requests else "measured"
            key = rng.choices(ranked_keys, weights=weights, k=1)[0]
            request_trace.append(
                {
                    "request_index": request_index,
                    "phase": phase,
                    "key": key,
                    "label": "zipf",
                }
            )

        metadata = {
            "workload": "zipf",
            "zipf_alpha": zipf_alpha,
            "ranked_keys": ranked_keys,
        }
        return request_trace, metadata

    @classmethod
    def _rolling_average(cls, values: Sequence[float], window: int) -> List[float]:
        if window <= 0:
            raise ValueError("rolling window must be positive.")
        if not values:
            return []

        result: List[float] = []
        running_total = 0.0
        for index, value in enumerate(values):
            running_total += float(value)
            if index >= window:
                running_total -= float(values[index - window])
            current_window = min(index + 1, window)
            result.append(running_total / current_window)
        return result

    @staticmethod
    def _steady_state_warmup_index(
        values: Sequence[float],
        *,
        improving_when: str,
    ) -> Optional[int]:
        if not values:
            return None

        initial = float(values[0])
        final = float(values[-1])

        if improving_when == "decrease":
            improvement = initial - final
            if improvement <= 0.0:
                return None
            threshold = final + 0.1 * improvement
            comparator = lambda value: value <= threshold
        elif improving_when == "increase":
            gain = final - initial
            if gain <= 0.0:
                return None
            threshold = final - 0.1 * gain
            comparator = lambda value: value >= threshold
        else:
            raise ValueError("improving_when must be either 'decrease' or 'increase'.")

        for index in range(len(values)):
            if all(comparator(float(value)) for value in values[index:]):
                return index
        return None

    @staticmethod
    def _summarize_phase(requests: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
        if not requests:
            return {
                "count": 0,
                "avg_rounds": 0.0,
                "cache_hit_rate": 0.0,
                "avg_promotions": 0.0,
                "avg_evictions": 0.0,
                "correct_count": 0,
                "incorrect_count": 0,
            }

        return {
            "count": len(requests),
            "avg_rounds": statistics.mean(float(request["rounds"]) for request in requests),
            "cache_hit_rate": statistics.mean(1.0 if request["cache_hit"] else 0.0 for request in requests),
            "avg_promotions": statistics.mean(float(request["promotions_delta"]) for request in requests),
            "avg_evictions": statistics.mean(float(request["evictions_delta"]) for request in requests),
            "correct_count": sum(1 for request in requests if request["correct"]),
            "incorrect_count": sum(1 for request in requests if not request["correct"]),
        }

    def _build_hot_omap(
        self,
        *,
        num_data: int,
        payload_size: int,
        client: InteractLocalServer,
        cache_size: int,
        hot_threshold: int,
        admission_layer: Any,
    ) -> _InstrumentedBPlusOmapHotNodesClient:
        return _InstrumentedBPlusOmapHotNodesClient(
            order=self._order,
            num_data=num_data,
            key_size=DEFAULT_KEY_SIZE,
            data_size=max(DEFAULT_KEY_SIZE, payload_size),
            client=client,
            hot_nodes_client_size=cache_size,
            hot_access_threshold=hot_threshold,
            hot_admission_layer=admission_layer,
        )

    def _build_plain_omap(
        self,
        *,
        num_data: int,
        payload_size: int,
        client: InteractLocalServer,
    ) -> _UnpaddedBenchmarkBPlusOmap:
        return _UnpaddedBenchmarkBPlusOmap(
            order=self._order,
            num_data=num_data,
            key_size=DEFAULT_KEY_SIZE,
            data_size=max(DEFAULT_KEY_SIZE, payload_size),
            client=client,
        )

    @staticmethod
    def _populate_omap(omap: Any, keys: Sequence[int], payloads: Dict[int, bytes]) -> None:
        omap.init_server_storage()
        for key in keys:
            omap.insert(key=key, value=payloads[key])

    def _run_hot_workload(
        self,
        *,
        run_name: str,
        admission_layer: Any,
        num_data: int,
        payload_size: int,
        cache_size: int,
        hot_threshold: int,
        keys: Sequence[int],
        payloads: Dict[int, bytes],
        request_trace: Sequence[Dict[str, Any]],
        rolling_window: int,
    ) -> Dict[str, Any]:
        client = InteractLocalServer()
        omap = self._build_hot_omap(
            num_data=num_data,
            payload_size=payload_size,
            client=client,
            cache_size=cache_size,
            hot_threshold=hot_threshold,
            admission_layer=admission_layer,
        )
        self._populate_omap(omap=omap, keys=keys, payloads=payloads)

        requests: List[Dict[str, Any]] = []
        result_is_none_flags: List[bool] = []
        for request in request_trace:
            before_client = self._snapshot_client(client=client)
            before_cache = omap.get_hot_cache_stats()

            result = omap.search(key=request["key"])

            after_client = self._snapshot_client(client=client)
            after_cache = omap.get_hot_cache_stats()
            cache_hit = after_cache["hits"] > before_cache["hits"]
            promotions_delta = after_cache["promotions"] - before_cache["promotions"]
            evictions_delta = after_cache["evictions"] - before_cache["evictions"]

            requests.append(
                {
                    "request_index": request["request_index"],
                    "phase": request["phase"],
                    "key": request["key"],
                    "label": request["label"],
                    "rounds": after_client["rounds"] - before_client["rounds"],
                    "correct": result == payloads[request["key"]],
                    "cache_hit": cache_hit,
                    "promotions_delta": promotions_delta,
                    "evictions_delta": evictions_delta,
                    "cache_size_after": len(omap.hot_nodes_client),
                    "result_is_none": result is None,
                }
            )
            result_is_none_flags.append(result is None)

        rounds_series = [float(request["rounds"]) for request in requests]
        cache_hit_series = [1.0 if request["cache_hit"] else 0.0 for request in requests]
        promotions_series = [float(request["promotions_delta"]) for request in requests]
        evictions_series = [float(request["evictions_delta"]) for request in requests]
        rolling_avg_rounds = self._rolling_average(values=rounds_series, window=rolling_window)
        rolling_cache_hit_rate = self._rolling_average(values=cache_hit_series, window=rolling_window)

        phase_requests = {
            "warmup": [request for request in requests if request["phase"] == "warmup"],
            "measured": [request for request in requests if request["phase"] == "measured"],
        }
        summary = {
            "count": len(requests),
            "correct_count": sum(1 for request in requests if request["correct"]),
            "incorrect_count": sum(1 for request in requests if not request["correct"]),
            "total_rounds": sum(request["rounds"] for request in requests),
            "avg_rounds": statistics.mean(rounds_series) if rounds_series else 0.0,
            "cache_hit_rate": statistics.mean(cache_hit_series) if cache_hit_series else 0.0,
            "total_promotions": sum(request["promotions_delta"] for request in requests),
            "total_evictions": sum(request["evictions_delta"] for request in requests),
            "final_cache_size": len(omap.hot_nodes_client),
            "rounds_warmup_index_90pct": self._steady_state_warmup_index(
                rolling_avg_rounds,
                improving_when="decrease",
            ),
            "cache_hit_warmup_index_90pct": self._steady_state_warmup_index(
                rolling_cache_hit_rate,
                improving_when="increase",
            ),
            "phase_summary": {
                phase: self._summarize_phase(phase_request_list)
                for phase, phase_request_list in phase_requests.items()
            },
        }

        eviction_analysis = omap.get_benchmark_eviction_summary()

        return {
            "name": run_name,
            "implementation": "BPlusOmapHotNodesClient",
            "admission_layer": type(admission_layer).__name__,
            "requests": requests,
            "series": {
                "rounds": rounds_series,
                "rolling_avg_rounds": rolling_avg_rounds,
                "cache_hit": cache_hit_series,
                "rolling_cache_hit_rate": rolling_cache_hit_rate,
                "promotions_delta": promotions_series,
                "evictions_delta": evictions_series,
            },
            "summary": {
                **summary,
                "eviction_accuracy": eviction_analysis["eviction_accuracy"],
                "total_actual_evictions": eviction_analysis["total_actual_evictions"],
                "matching_lowest_access_evictions": eviction_analysis["matching_lowest_access_evictions"],
            },
            "final_cache_stats": {
                **omap.get_hot_cache_stats(),
                "cache_size": len(omap.hot_nodes_client),
            },
            "eviction_analysis": eviction_analysis,
            "client_observability": self._snapshot_client(client=client),
            "result_is_none_flags": result_is_none_flags,
        }

    def _run_plain_workload(
        self,
        *,
        run_name: str,
        num_data: int,
        payload_size: int,
        keys: Sequence[int],
        payloads: Dict[int, bytes],
        request_trace: Sequence[Dict[str, Any]],
        rolling_window: int,
    ) -> Dict[str, Any]:
        client = InteractLocalServer()
        omap = self._build_plain_omap(
            num_data=num_data,
            payload_size=payload_size,
            client=client,
        )
        self._populate_omap(omap=omap, keys=keys, payloads=payloads)

        requests: List[Dict[str, Any]] = []
        result_is_none_flags: List[bool] = []
        for request in request_trace:
            before_client = self._snapshot_client(client=client)
            result = omap.search(key=request["key"])
            after_client = self._snapshot_client(client=client)

            requests.append(
                {
                    "request_index": request["request_index"],
                    "phase": request["phase"],
                    "key": request["key"],
                    "label": request["label"],
                    "rounds": after_client["rounds"] - before_client["rounds"],
                    "correct": result == payloads[request["key"]],
                    "cache_hit": False,
                    "promotions_delta": 0,
                    "evictions_delta": 0,
                    "cache_size_after": 0,
                    "result_is_none": result is None,
                }
            )
            result_is_none_flags.append(result is None)

        rounds_series = [float(request["rounds"]) for request in requests]
        cache_hit_series = [0.0 for _ in requests]
        promotions_series = [0.0 for _ in requests]
        evictions_series = [0.0 for _ in requests]
        rolling_avg_rounds = self._rolling_average(values=rounds_series, window=rolling_window)
        rolling_cache_hit_rate = self._rolling_average(values=cache_hit_series, window=rolling_window)

        phase_requests = {
            "warmup": [request for request in requests if request["phase"] == "warmup"],
            "measured": [request for request in requests if request["phase"] == "measured"],
        }
        summary = {
            "count": len(requests),
            "correct_count": sum(1 for request in requests if request["correct"]),
            "incorrect_count": sum(1 for request in requests if not request["correct"]),
            "total_rounds": sum(request["rounds"] for request in requests),
            "avg_rounds": statistics.mean(rounds_series) if rounds_series else 0.0,
            "cache_hit_rate": 0.0,
            "total_promotions": 0,
            "total_evictions": 0,
            "final_cache_size": 0,
            "rounds_warmup_index_90pct": self._steady_state_warmup_index(
                rolling_avg_rounds,
                improving_when="decrease",
            ),
            "cache_hit_warmup_index_90pct": None,
            "phase_summary": {
                phase: self._summarize_phase(phase_request_list)
                for phase, phase_request_list in phase_requests.items()
            },
            "eviction_accuracy": None,
            "total_actual_evictions": 0,
            "matching_lowest_access_evictions": 0,
        }

        return {
            "name": run_name,
            "implementation": "BPlusOmap.search",
            "admission_layer": None,
            "requests": requests,
            "series": {
                "rounds": rounds_series,
                "rolling_avg_rounds": rolling_avg_rounds,
                "cache_hit": cache_hit_series,
                "rolling_cache_hit_rate": rolling_cache_hit_rate,
                "promotions_delta": promotions_series,
                "evictions_delta": evictions_series,
            },
            "summary": summary,
            "final_cache_stats": {
                "hits": 0,
                "misses": 0,
                "promotions": 0,
                "evictions": 0,
                "cache_size": 0,
            },
            "eviction_analysis": {
                "cache_full_decisions": [],
                "actual_eviction_records": [],
                "total_cache_full_decisions": 0,
                "total_actual_evictions": 0,
                "matching_lowest_access_evictions": 0,
                "eviction_accuracy": None,
            },
            "client_observability": self._snapshot_client(client=client),
            "result_is_none_flags": result_is_none_flags,
        }

    @staticmethod
    def _build_pairwise_comparison(
        reject_all_run: Dict[str, Any],
        plain_run: Dict[str, Any],
    ) -> Dict[str, Any]:
        reject_requests = reject_all_run["requests"]
        plain_requests = plain_run["requests"]
        request_count_equal = len(reject_requests) == len(plain_requests)

        round_mismatch_indices = [
            index
            for index, (reject_request, plain_request) in enumerate(zip(reject_requests, plain_requests))
            if reject_request["rounds"] != plain_request["rounds"]
        ]
        correctness_mismatch_indices = [
            index
            for index, (reject_request, plain_request) in enumerate(zip(reject_requests, plain_requests))
            if reject_request["correct"] != plain_request["correct"]
        ]
        result_none_mismatch_indices = [
            index
            for index, (reject_request, plain_request) in enumerate(zip(reject_requests, plain_requests))
            if reject_request["result_is_none"] != plain_request["result_is_none"]
        ]

        reject_summary = reject_all_run["summary"]
        plain_summary = plain_run["summary"]
        summary_deltas = {
            "avg_rounds": reject_summary["avg_rounds"] - plain_summary["avg_rounds"],
            "total_rounds": reject_summary["total_rounds"] - plain_summary["total_rounds"],
            "cache_hit_rate": reject_summary["cache_hit_rate"] - plain_summary["cache_hit_rate"],
            "total_promotions": reject_summary["total_promotions"] - plain_summary["total_promotions"],
            "total_evictions": reject_summary["total_evictions"] - plain_summary["total_evictions"],
            "correct_count": reject_summary["correct_count"] - plain_summary["correct_count"],
            "incorrect_count": reject_summary["incorrect_count"] - plain_summary["incorrect_count"],
        }
        reject_all_zero_cache_activity = (
            reject_summary["cache_hit_rate"] == 0.0
            and reject_summary["total_promotions"] == 0
            and reject_summary["total_evictions"] == 0
            and reject_all_run["final_cache_stats"]["cache_size"] == 0
        )

        return {
            "reject_all_vs_plain_search": {
                "request_count_equal": request_count_equal,
                "correctness_flags_equal": len(correctness_mismatch_indices) == 0,
                "all_requests_correct": (
                    reject_summary["incorrect_count"] == 0 and plain_summary["incorrect_count"] == 0
                ),
                "per_request_rounds_equal": len(round_mismatch_indices) == 0,
                "result_is_none_flags_equal": len(result_none_mismatch_indices) == 0,
                "reject_all_zero_cache_activity": reject_all_zero_cache_activity,
                "summary_deltas": summary_deltas,
                "round_mismatch_indices": round_mismatch_indices,
                "correctness_mismatch_indices": correctness_mismatch_indices,
                "result_is_none_mismatch_indices": result_none_mismatch_indices,
                "parity": (
                    request_count_equal
                    and len(reject_requests) == len(plain_requests)
                    and len(round_mismatch_indices) == 0
                    and len(correctness_mismatch_indices) == 0
                    and len(result_none_mismatch_indices) == 0
                    and reject_summary["incorrect_count"] == 0
                    and plain_summary["incorrect_count"] == 0
                    and reject_all_zero_cache_activity
                ),
            }
        }

    def run(
        self,
        *,
        num_data: int,
        insert_count: int,
        payload_size: int,
        seed: int,
        workload: str,
        warmup_requests: int,
        num_queries: int,
        cache_size: int,
        hot_threshold: int,
        rolling_window: int,
        hotset_size: Optional[int] = None,
        hot_query_probability: float = 0.7,
        zipf_alpha: float = 1.2,
    ) -> Dict[str, Any]:
        if self._order < 3:
            raise ValueError("order must be at least 3.")
        if num_data <= 0:
            raise ValueError("num_data must be positive.")
        if insert_count <= 0:
            raise ValueError("insert_count must be positive.")
        if insert_count > num_data:
            raise ValueError("insert_count must be <= num_data.")
        if warmup_requests < 0 or num_queries < 0:
            raise ValueError("warmup_requests and num_queries must be non-negative.")
        if cache_size < 0:
            raise ValueError("cache_size must be non-negative.")
        if hot_threshold < 0:
            raise ValueError("hot_threshold must be non-negative.")
        if rolling_window <= 0:
            raise ValueError("rolling_window must be positive.")

        keys = list(range(insert_count))
        payload_rng = random.Random(seed)
        workload_rng = random.Random(seed + 1)
        payloads = self._build_payload_map(keys=keys, payload_size=payload_size, rng=payload_rng)

        if workload == "hotset":
            if hotset_size is None:
                raise ValueError("hotset_size must be provided for hotset workloads.")
            request_trace, workload_metadata = self._build_hotset_request_trace(
                keys=keys,
                rng=workload_rng,
                warmup_requests=warmup_requests,
                num_queries=num_queries,
                hotset_size=hotset_size,
                hot_query_probability=hot_query_probability,
            )
        elif workload == "zipf":
            request_trace, workload_metadata = self._build_zipf_request_trace(
                keys=keys,
                rng=workload_rng,
                warmup_requests=warmup_requests,
                num_queries=num_queries,
                zipf_alpha=zipf_alpha,
            )
        else:
            raise ValueError(f"Unsupported workload {workload!r}.")

        score_run = self._run_hot_workload(
            run_name="score_based_hot_cache",
            admission_layer=ScoreBasedHotCacheAdmissionLayer(),
            num_data=num_data,
            payload_size=payload_size,
            cache_size=cache_size,
            hot_threshold=hot_threshold,
            keys=keys,
            payloads=payloads,
            request_trace=request_trace,
            rolling_window=rolling_window,
        )
        reject_run = self._run_hot_workload(
            run_name="reject_all_hot_cache",
            admission_layer=RejectAllHotCacheAdmissionLayer(),
            num_data=num_data,
            payload_size=payload_size,
            cache_size=cache_size,
            hot_threshold=hot_threshold,
            keys=keys,
            payloads=payloads,
            request_trace=request_trace,
            rolling_window=rolling_window,
        )
        plain_run = self._run_plain_workload(
            run_name="plain_bplus_search",
            num_data=num_data,
            payload_size=payload_size,
            keys=keys,
            payloads=payloads,
            request_trace=request_trace,
            rolling_window=rolling_window,
        )

        pairwise_comparison = self._build_pairwise_comparison(
            reject_all_run=reject_run,
            plain_run=plain_run,
        )

        return {
            "config": {
                "order": self._order,
                "num_data": num_data,
                "insert_count": insert_count,
                "payload_size": payload_size,
                "seed": seed,
                "workload": workload,
                "warmup_requests": warmup_requests,
                "num_queries": num_queries,
                "cache_size": cache_size,
                "hot_threshold": hot_threshold,
                "rolling_window": rolling_window,
                "default_plot_bucket_size": DEFAULT_PLOT_BUCKET_SIZE,
                "hotset_size": hotset_size,
                "hot_query_probability": hot_query_probability,
                "zipf_alpha": zipf_alpha,
            },
            "workload_metadata": workload_metadata,
            "request_trace": request_trace,
            "runs": {
                score_run["name"]: score_run,
                reject_run["name"]: reject_run,
                plain_run["name"]: plain_run,
            },
            "pairwise_comparison": pairwise_comparison,
        }


class BPlusOmapHotNodesBenchmarkMain:
    """CLI runner for BPlusOmapHotNodesBenchmark."""

    @staticmethod
    def build_parser() -> argparse.ArgumentParser:
        parser = argparse.ArgumentParser(
            description="Benchmark BPlusOmapHotNodesClient cache behavior against RejectAll and plain BPlusOmap.search()."
        )
        parser.add_argument("--order", type=int, default=DEFAULT_ORDER, help="B+ tree order.")
        parser.add_argument("--num-data", type=int, default=256, help="OMAP capacity.")
        parser.add_argument("--insert-count", type=int, default=64, help="How many keys [0..insert_count-1] to insert.")
        parser.add_argument("--payload-size", type=int, default=64, help="Random payload byte-size per value.")
        parser.add_argument("--seed", type=int, default=42, help="Random seed for payloads and workload generation.")
        parser.add_argument(
            "--workload",
            choices=("hotset", "zipf"),
            default="hotset",
            help="Seeded workload family to replay.",
        )
        parser.add_argument("--warmup-requests", type=int, default=128, help="Number of warmup search requests.")
        parser.add_argument("--num-queries", type=int, default=256, help="Number of measured search requests.")
        parser.add_argument("--cache-size", type=int, default=8, help="hot_nodes_client_size.")
        parser.add_argument("--hot-threshold", type=int, default=1, help="hot_access_threshold.")
        parser.add_argument("--rolling-window", type=int, default=32, help="Rolling window length for derived metrics.")
        parser.add_argument("--output-json", type=str, default=None, help="Optional path to write benchmark JSON.")
        parser.add_argument("--hotset-size", type=int, default=8, help="Exact hotset size for hotset workloads.")
        parser.add_argument(
            "--hot-query-probability",
            type=float,
            default=0.7,
            help="Probability that a hotset workload request comes from the hotset.",
        )
        parser.add_argument("--zipf-alpha", type=float, default=1.2, help="Zipf exponent for zipf workloads.")
        return parser

    @staticmethod
    def run(args: argparse.Namespace) -> Dict[str, Any]:
        default_fields = {
            "order": DEFAULT_ORDER,
            "workload": "hotset",
            "warmup_requests": 128,
            "num_queries": 256,
            "cache_size": 8,
            "hot_threshold": 1,
            "rolling_window": 32,
            "output_json": None,
            "hotset_size": 8,
            "hot_query_probability": 0.7,
            "zipf_alpha": 1.2,
        }
        for field, value in default_fields.items():
            if not hasattr(args, field):
                setattr(args, field, value)

        benchmark = BPlusOmapHotNodesBenchmark(order=args.order)
        return benchmark.run(
            num_data=args.num_data,
            insert_count=args.insert_count,
            payload_size=args.payload_size,
            seed=args.seed,
            workload=args.workload,
            warmup_requests=args.warmup_requests,
            num_queries=args.num_queries,
            cache_size=args.cache_size,
            hot_threshold=args.hot_threshold,
            rolling_window=args.rolling_window,
            hotset_size=args.hotset_size,
            hot_query_probability=args.hot_query_probability,
            zipf_alpha=args.zipf_alpha,
        )


def main() -> None:
    """Run the benchmark and either write JSON to a file or print it."""
    parser = BPlusOmapHotNodesBenchmarkMain.build_parser()
    args = parser.parse_args()
    output = BPlusOmapHotNodesBenchmarkMain.run(args=args)

    if args.output_json:
        with open(args.output_json, "w") as handle:
            json.dump(output, handle, indent=2, sort_keys=True, default=str)
    else:
        print(json.dumps(output, indent=2, sort_keys=True, default=str))


if __name__ == "__main__":
    main()
