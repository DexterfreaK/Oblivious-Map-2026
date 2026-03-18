"""Benchmark utilities for AVLOmapHotNodesClient."""

import argparse
import json
import math
import os
import random
import statistics
import sys
from typing import Any, Dict, Iterable, List, Optional

# Support direct file execution: python daoram/omap/avl_omap_hot_benchmark.py ...
if __package__ in (None, ""):
    ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    if ROOT_DIR not in sys.path:
        sys.path.insert(0, ROOT_DIR)

from daoram.dependency import InteractLocalServer
from daoram.omap.avl_omap import AVLOmap
from daoram.omap.avl_omap_hot import AVLOmapHotNodesClient


class AVLOmapHotNodesBenchmark:
    """
    Benchmark hot-vs-cold retrieval costs on AVLOmapHotNodesClient.

    Metrics are measured at client/server interface:
    - rounds (execute calls)
    - bytes_read
    - bytes_written
    """

    def __init__(self, omap: AVLOmapHotNodesClient):
        self._omap = omap
        self._client = omap.client
        # Aggregate cache observations collected after each measured search.
        self._cache_snapshot_count = 0
        self._cache_height_counts: Dict[int, int] = {}
        self._cache_presence_counts: Dict[Any, int] = {}
        self._cache_last_seen_height: Dict[Any, int] = {}

    def _snapshot_client(self) -> Dict[str, int]:
        bytes_read, bytes_written = self._client.get_bandwidth()
        rounds = self._client.get_rounds() if hasattr(self._client, "get_rounds") else 0
        return {
            "bytes_read": bytes_read,
            "bytes_written": bytes_written,
            "rounds": rounds,
        }

    @staticmethod
    def _delta(after: Dict[str, int], before: Dict[str, int]) -> Dict[str, int]:
        bytes_read = after["bytes_read"] - before["bytes_read"]
        bytes_written = after["bytes_written"] - before["bytes_written"]
        return {
            "bytes_read": bytes_read,
            "bytes_written": bytes_written,
            "bytes_total": bytes_read + bytes_written,
            "rounds": after["rounds"] - before["rounds"],
        }

    @staticmethod
    def _percentile(values: List[float], percentile: float) -> float:
        """Linear-interpolated percentile in [0, 1]."""
        if not values:
            return 0.0
        if len(values) == 1:
            return float(values[0])
        sorted_values = sorted(values)
        position = (len(sorted_values) - 1) * percentile
        low = int(math.floor(position))
        high = int(math.ceil(position))
        if low == high:
            return float(sorted_values[low])
        weight = position - low
        return float(sorted_values[low] * (1.0 - weight) + sorted_values[high] * weight)

    @staticmethod
    def _get_node_height(node: Any) -> int:
        """Best-effort subtree height extraction for cached AVL node data."""
        value = getattr(node, "value", None)
        l_height = getattr(value, "l_height", 0) if value is not None else 0
        r_height = getattr(value, "r_height", 0) if value is not None else 0
        return 1 + max(int(l_height), int(r_height))

    def _record_cache_snapshot(self) -> None:
        """Capture one snapshot of the client hot cache composition."""
        cache = getattr(self._omap, "_hot_nodes_client", None)
        if cache is None:
            return

        self._cache_snapshot_count += 1
        for key, node in cache.items():
            height = self._get_node_height(node)
            self._cache_height_counts[height] = self._cache_height_counts.get(height, 0) + 1
            self._cache_presence_counts[key] = self._cache_presence_counts.get(key, 0) + 1
            self._cache_last_seen_height[key] = height

    def _summarize_trials(self, trials: List[Dict[str, Any]]) -> Dict[str, float]:
        if not trials:
            return {
                "count": 0,
                "correct_count": 0,
                "incorrect_count": 0,
                "was_hot_before_true_count": 0,
                "was_hot_before_false_count": 0,
                "was_hot_before_true_fraction": 0.0,
                "avg_rounds": 0.0,
                "avg_bytes_read": 0.0,
                "avg_bytes_written": 0.0,
                "avg_bandwidth_bytes_total": 0.0,
                "min_rounds": 0.0,
                "max_rounds": 0.0,
                "lower_tail_rounds_p05": 0.0,
                "upper_tail_rounds_p95": 0.0,
                "min_bandwidth_bytes_total": 0.0,
                "max_bandwidth_bytes_total": 0.0,
                "lower_tail_bandwidth_bytes_total_p05": 0.0,
                "upper_tail_bandwidth_bytes_total_p95": 0.0,
            }

        rounds = [float(t["rounds"]) for t in trials]
        read_bw = [float(t["bytes_read"]) for t in trials]
        write_bw = [float(t["bytes_written"]) for t in trials]
        total_bw = [float(t["bytes_total"]) for t in trials]
        correct_count = sum(1 for t in trials if t["correct"])
        was_hot_true_count = sum(1 for t in trials if t.get("was_hot_before"))

        return {
            "count": len(trials),
            "correct_count": correct_count,
            "incorrect_count": len(trials) - correct_count,
            "was_hot_before_true_count": was_hot_true_count,
            "was_hot_before_false_count": len(trials) - was_hot_true_count,
            "was_hot_before_true_fraction": (was_hot_true_count / len(trials)),
            "avg_rounds": statistics.mean(rounds),
            "avg_bytes_read": statistics.mean(read_bw),
            "avg_bytes_written": statistics.mean(write_bw),
            "avg_bandwidth_bytes_total": statistics.mean(total_bw),
            "min_rounds": min(rounds),
            "max_rounds": max(rounds),
            "lower_tail_rounds_p05": self._percentile(rounds, 0.05),
            "upper_tail_rounds_p95": self._percentile(rounds, 0.95),
            "min_bandwidth_bytes_total": min(total_bw),
            "max_bandwidth_bytes_total": max(total_bw),
            "lower_tail_bandwidth_bytes_total_p05": self._percentile(total_bw, 0.05),
            "upper_tail_bandwidth_bytes_total_p95": self._percentile(total_bw, 0.95),
        }

    def _measure_search_once(self, key: Any, label: str, expected_value: Any = None) -> Dict[str, Any]:
        before = self._snapshot_client()
        was_hot_before = key in self._omap.hot_nodes_client
        result = self._omap.search(key=key)
        after = self._snapshot_client()
        self._record_cache_snapshot()
        delta = self._delta(after=after, before=before)
        correct = True if expected_value is None else (result == expected_value)
        if isinstance(result, (bytes, bytearray)):
            result_size = len(result)
        else:
            result_size = None if result is None else len(str(result))

        return {
            "label": label,
            "key": key,
            "was_hot_before": was_hot_before,
            "is_hot_after": key in self._omap.hot_nodes_client,
            "result_type": type(result).__name__,
            "result_size_bytes": result_size,
            "correct": correct,
            **delta,
        }

    def get_cache_aggregate(self, top_n: int = 10) -> Dict[str, Any]:
        """
        Return aggregate cache composition over the run.

        Percentages are computed over all observed cache-node entries across all
        measured operations (not just final cache state).
        """
        total_node_observations = sum(self._cache_height_counts.values())
        height_distribution = []
        for height in sorted(self._cache_height_counts.keys()):
            count = self._cache_height_counts[height]
            percentage = 0.0 if total_node_observations == 0 else (100.0 * count / total_node_observations)
            height_distribution.append(
                {
                    "height": int(height),
                    "observation_count": int(count),
                    "percentage_of_cache_node_observations": percentage,
                }
            )

        observed_nodes = []
        for key, presence_count in self._cache_presence_counts.items():
            observed_nodes.append(
                {
                    "key": key,
                    "access_count": int(self._omap.get_access_count(key)),
                    "last_seen_height": int(self._cache_last_seen_height.get(key, 0)),
                    "present_in_snapshots_count": int(presence_count),
                    "present_in_snapshots_fraction": (
                        0.0 if self._cache_snapshot_count == 0 else (presence_count / self._cache_snapshot_count)
                    ),
                }
            )
        observed_nodes.sort(
            key=lambda item: (
                -item["access_count"],
                -item["present_in_snapshots_count"],
                item["key"],
            )
        )

        return {
            "snapshot_count": int(self._cache_snapshot_count),
            "total_cache_node_observations": int(total_node_observations),
            "height_distribution": height_distribution,
            "top_accessed_cache_nodes": observed_nodes[: max(0, top_n)],
        }

    def ensure_hot(self, key: Any, expected_value: Any = None, max_attempts: int = 64) -> None:
        """Drive accesses until key is in hot cache, or raise if that fails."""
        if key in self._omap.hot_nodes_client:
            return

        for _ in range(max_attempts):
            trial = self._measure_search_once(key=key, label="ensure_hot", expected_value=expected_value)
            if not trial["correct"]:
                raise AssertionError(f"Correctness failure while warming key {key}.")
            if key in self._omap.hot_nodes_client:
                return

        raise RuntimeError(f"Unable to promote key {key} into hot cache after {max_attempts} attempts.")

    def benchmark_hot_vs_cold(
        self,
        hot_key: Any,
        cold_keys: Iterable[Any],
        hot_trials: int = 10,
        cold_trials: Optional[int] = None,
        rewarm_hot_between_cold: bool = True,
    ) -> Dict[str, Any]:
        """
        Legacy benchmark mode: fixed hot key vs explicit cold keys.
        """
        cold_candidates = [key for key in cold_keys if key != hot_key]
        if cold_trials is not None:
            cold_candidates = cold_candidates[:cold_trials]

        self.ensure_hot(key=hot_key)

        hot_results: List[Dict[str, Any]] = []
        for _ in range(max(0, hot_trials)):
            hot_results.append(self._measure_search_once(key=hot_key, label="hot"))

        cold_results: List[Dict[str, Any]] = []
        for cold_key in cold_candidates:
            if cold_key in self._omap.hot_nodes_client:
                self._omap.flush_hot_nodes_client_to_oram()
                if rewarm_hot_between_cold:
                    self.ensure_hot(key=hot_key)

            cold_results.append(self._measure_search_once(key=cold_key, label="cold"))

            if cold_key in self._omap.hot_nodes_client:
                self._omap.flush_hot_nodes_client_to_oram()
                if rewarm_hot_between_cold:
                    self.ensure_hot(key=hot_key)

        return {
            "hot_key": hot_key,
            "hot_trials": hot_results,
            "cold_trials": cold_results,
            "summary": {
                "hot": self._summarize_trials(hot_results),
                "cold": self._summarize_trials(cold_results),
            },
        }

    def benchmark_random_hotset_workload(
        self,
        hotset_keys: List[Any],
        coldset_keys: List[Any],
        expected_values: Dict[Any, Any],
        rng: random.Random,
        num_queries: int,
        hot_query_probability: float,
        warmup_accesses: int,
        strict_correctness: bool = True,
    ) -> Dict[str, Any]:
        """
        Benchmark a probabilistic hot/cold workload with hotset warmup.

        :param hotset_keys: Keys considered hot.
        :param coldset_keys: Keys considered cold.
        :param expected_values: Expected key->value map for correctness checks.
        :param rng: Random source.
        :param num_queries: Number of benchmark queries after warmup.
        :param hot_query_probability: Probability query key comes from hotset; should be >=0.5.
        :param warmup_accesses: Number of warmup searches from hotset.
        :param strict_correctness: Raise on any mismatch if True.
        """
        if not hotset_keys:
            raise ValueError("hotset_keys cannot be empty.")
        if not coldset_keys:
            raise ValueError("coldset_keys cannot be empty.")
        if hot_query_probability < 0.5 or hot_query_probability > 1.0:
            raise ValueError("hot_query_probability must be in [0.5, 1.0].")
        if num_queries < 0 or warmup_accesses < 0:
            raise ValueError("num_queries and warmup_accesses must be non-negative.")

        warmup_trials: List[Dict[str, Any]] = []
        for _ in range(warmup_accesses):
            key = rng.choice(hotset_keys)
            trial = self._measure_search_once(key=key, label="warmup", expected_value=expected_values[key])
            warmup_trials.append(trial)
            if strict_correctness and not trial["correct"]:
                raise AssertionError(f"Warmup correctness failure for key={key}.")

        hot_trials: List[Dict[str, Any]] = []
        cold_trials: List[Dict[str, Any]] = []
        for _ in range(num_queries):
            use_hot = rng.random() < hot_query_probability
            key = rng.choice(hotset_keys) if use_hot else rng.choice(coldset_keys)
            label = "hot" if use_hot else "cold"

            trial = self._measure_search_once(key=key, label=label, expected_value=expected_values[key])
            if strict_correctness and not trial["correct"]:
                raise AssertionError(f"Benchmark correctness failure for key={key}, label={label}.")

            if use_hot:
                hot_trials.append(trial)
            else:
                cold_trials.append(trial)

        return {
            "warmup_trials": warmup_trials,
            "hot_trials": hot_trials,
            "cold_trials": cold_trials,
            "summary": {
                "warmup": self._summarize_trials(warmup_trials),
                "hot": self._summarize_trials(hot_trials),
                "cold": self._summarize_trials(cold_trials),
            },
        }

    def benchmark_query_plan(
        self,
        warmup_keys: List[Any],
        query_plan: List[Dict[str, Any]],
        expected_values: Dict[Any, Any],
        strict_correctness: bool = True,
    ) -> Dict[str, Any]:
        """
        Benchmark a precomputed workload plan so multiple implementations can be compared fairly.

        :param warmup_keys: Warmup key sequence (always treated as warmup reads).
        :param query_plan: Query entries with fields: {"label": "hot"|"cold", "key": Any}.
        :param expected_values: Expected key->value map for correctness checks.
        :param strict_correctness: Raise on any mismatch if True.
        """
        warmup_trials: List[Dict[str, Any]] = []
        for key in warmup_keys:
            trial = self._measure_search_once(key=key, label="warmup", expected_value=expected_values[key])
            warmup_trials.append(trial)
            if strict_correctness and not trial["correct"]:
                raise AssertionError(f"Warmup correctness failure for key={key}.")

        hot_trials: List[Dict[str, Any]] = []
        cold_trials: List[Dict[str, Any]] = []
        for entry in query_plan:
            label = entry["label"]
            key = entry["key"]
            trial = self._measure_search_once(key=key, label=label, expected_value=expected_values[key])
            if strict_correctness and not trial["correct"]:
                raise AssertionError(f"Benchmark correctness failure for key={key}, label={label}.")

            if label == "hot":
                hot_trials.append(trial)
            elif label == "cold":
                cold_trials.append(trial)
            else:
                raise ValueError(f"Unsupported query label {label!r}. Expected 'hot' or 'cold'.")

        return {
            "warmup_trials": warmup_trials,
            "hot_trials": hot_trials,
            "cold_trials": cold_trials,
            "summary": {
                "warmup": self._summarize_trials(warmup_trials),
                "hot": self._summarize_trials(hot_trials),
                "cold": self._summarize_trials(cold_trials),
            },
        }


def _build_payload_map(keys: List[int], payload_size: int, rng: random.Random) -> Dict[int, bytes]:
    """Generate deterministic random payloads for inserted key-value pairs."""
    if payload_size < 0:
        raise ValueError("payload_size must be non-negative.")
    return {key: rng.randbytes(payload_size) for key in keys}


class AVLOmapHotNodesBenchmarkMain:
    """CLI runner for AVLOmapHotNodesBenchmark."""

    @staticmethod
    def build_parser() -> argparse.ArgumentParser:
        parser = argparse.ArgumentParser(description="Benchmark AVLOmapHotNodesClient with seeded random hotset workload")
        parser.add_argument("--num-data", type=int, default=256, help="OMAP capacity.")
        parser.add_argument("--insert-count", type=int, default=64, help="How many keys [0..insert_count-1] to insert.")
        parser.add_argument("--payload-size", type=int, default=64, help="Random payload byte-size per value.")
        parser.add_argument("--seed", type=int, default=42, help="Random seed for hotset/payload/query generation.")
        parser.add_argument(
            "--hotset-mode",
            choices=(
                "random",
                "range",
                "multi_range",
                "moving_range",
                "expanding_range",
                "contracting_range",
                "nested_ranges",
                "periodic_ranges",
                "boundary_hotspot",
            ),
            default="random",
            help="Hotset workload mode.",
        )
        parser.add_argument(
            "--hotset-range-start",
            type=int,
            default=0,
            help="Start index for range-derived hotset modes (when applicable).",
        )
        parser.add_argument(
            "--hotset-width",
            type=int,
            default=None,
            help="Optional explicit hot range width; defaults to rounded hotset fraction size.",
        )
        parser.add_argument(
            "--num-hot-ranges",
            type=int,
            default=3,
            help="Number of ranges for multi/periodic modes.",
        )
        parser.add_argument(
            "--temporal-window",
            type=int,
            default=32,
            help="Queries per phase before moving/periodic/expand/contract mode updates hotspot.",
        )
        parser.add_argument(
            "--range-step",
            type=int,
            default=0,
            help="Range shift step for moving mode; defaults to hotset width when unset/0.",
        )
        parser.add_argument(
            "--nested-inner-fraction",
            type=float,
            default=0.2,
            help="Inner hotspot width as a fraction of outer hotspot width for nested mode.",
        )
        parser.add_argument(
            "--nested-inner-hot-traffic",
            type=float,
            default=0.6,
            help="Within hot traffic, probability of picking the inner nested range.",
        )
        parser.add_argument(
            "--boundary-index",
            type=int,
            default=None,
            help="Boundary center index for boundary_hotspot mode (default: insert_count // 2).",
        )
        parser.add_argument(
            "--boundary-width",
            type=int,
            default=None,
            help="Boundary hotspot width around boundary-index; defaults to hotset width.",
        )
        parser.add_argument(
            "--hotset-fraction",
            type=float,
            default=0.1,
            help="Fraction of inserted items treated as hot (e.g., 0.1 => 10%%).",
        )
        parser.add_argument(
            "--hot-query-probability",
            type=float,
            default=0.7,
            help="Probability a benchmark query is drawn from hotset; must be >= 0.5.",
        )
        parser.add_argument("--warmup-accesses", type=int, default=128, help="Warmup searches sampled from hotset.")
        parser.add_argument("--num-queries", type=int, default=256, help="Number of benchmark queries after warmup.")
        parser.add_argument("--cache-size", type=int, default=8, help="hot_nodes_client_size.")
        parser.add_argument("--hot-threshold", type=int, default=1, help="hot_access_threshold.")
        parser.add_argument(
            "--search-padding",
            action="store_true",
            default=True,
            help="Enable padded search rounds (oblivious-style).",
        )
        parser.add_argument(
            "--always-dummy-after-search",
            action="store_true",
            default=False,
            help="When --search-padding is on, add one extra dummy round per search.",
        )
        return parser

    @staticmethod
    def _range_keys(start: int, width: int, insert_count: int) -> List[int]:
        """Return a valid contiguous range inside [0, insert_count)."""
        if insert_count <= 0:
            return []
        width = max(1, min(width, insert_count))
        max_start = max(0, insert_count - width)
        start = max(0, min(start, max_start))
        return list(range(start, start + width))

    @staticmethod
    def _build_hotset_mode_strategy(
        args: argparse.Namespace,
        keys: List[int],
        rng: random.Random,
        hot_count: int,
    ) -> Dict[str, Any]:
        """
        Build a mode-specific hotspot strategy.

        Returns:
          - hotset_at(step): List[int]
          - sample_hot_key(step, rng): int
          - metadata: mode-specific details
        """
        insert_count = len(keys)
        explicit_width = args.hotset_width
        target_hot_size = hot_count if explicit_width is None else explicit_width
        target_hot_size = max(1, min(target_hot_size, insert_count - 1))
        temporal_window = max(1, args.temporal_window)

        base_width = target_hot_size
        max_base_start = max(0, insert_count - base_width)
        base_start = max(0, min(args.hotset_range_start, max_base_start))
        mode = args.hotset_mode

        def _build_multi_ranges(width: int, num_ranges: int, start_offset: int = 0) -> List[List[int]]:
            num_ranges = max(1, min(num_ranges, width))
            max_start = max(0, insert_count - width)
            num_ranges = min(num_ranges, max_start + 1)
            if num_ranges == 1:
                starts = [max(0, min(start_offset, max_start))]
            else:
                stride = max(1, (max_start + 1) // num_ranges)
                raw_starts = [(start_offset + idx * stride) % (max_start + 1) for idx in range(num_ranges)]
                # Preserve order and remove duplicates when keyspace is tight.
                starts = list(dict.fromkeys(raw_starts))
                while len(starts) < num_ranges:
                    starts.append((starts[-1] + 1) % (max_start + 1))
                    starts = list(dict.fromkeys(starts))
            per_width = max(1, width // max(1, len(starts)))
            return [AVLOmapHotNodesBenchmarkMain._range_keys(s, per_width, insert_count) for s in starts]

        def _flatten_ranges_to_target(ranges: List[List[int]], target_size: int) -> List[int]:
            ordered: List[int] = []
            seen = set()
            for group in ranges:
                for key in group:
                    if key not in seen:
                        seen.add(key)
                        ordered.append(key)
            if len(ordered) < target_size:
                for key in keys:
                    if key not in seen:
                        ordered.append(key)
                        seen.add(key)
                    if len(ordered) >= target_size:
                        break
            return sorted(ordered[:target_size])

        if mode == "random":
            hotset = sorted(rng.sample(keys, target_hot_size))

            def hotset_at(_step: int) -> List[int]:
                return hotset

            def sample_hot_key(_step: int, local_rng: random.Random) -> int:
                return local_rng.choice(hotset)

            metadata = {"mode": mode}
            return {"hotset_at": hotset_at, "sample_hot_key": sample_hot_key, "metadata": metadata}

        if mode == "range":
            hotset = AVLOmapHotNodesBenchmarkMain._range_keys(base_start, base_width, insert_count)

            def hotset_at(_step: int) -> List[int]:
                return hotset

            def sample_hot_key(_step: int, local_rng: random.Random) -> int:
                return local_rng.choice(hotset)

            metadata = {"mode": mode, "range_start": hotset[0], "range_end_exclusive": hotset[-1] + 1}
            return {"hotset_at": hotset_at, "sample_hot_key": sample_hot_key, "metadata": metadata}

        if mode == "multi_range":
            ranges = _build_multi_ranges(width=base_width, num_ranges=max(2, args.num_hot_ranges), start_offset=base_start)
            hotset = _flatten_ranges_to_target(ranges=ranges, target_size=base_width)

            def hotset_at(_step: int) -> List[int]:
                return hotset

            def sample_hot_key(_step: int, local_rng: random.Random) -> int:
                return local_rng.choice(hotset)

            metadata = {
                "mode": mode,
                "ranges": [(group[0], group[-1] + 1) for group in ranges if group],
                "num_hot_ranges": len(ranges),
            }
            return {"hotset_at": hotset_at, "sample_hot_key": sample_hot_key, "metadata": metadata}

        if mode == "nested_ranges":
            outer = AVLOmapHotNodesBenchmarkMain._range_keys(base_start, base_width, insert_count)
            inner_width = max(1, int(round(base_width * args.nested_inner_fraction)))
            inner_width = min(inner_width, len(outer))
            inner_start = outer[0] + max(0, (len(outer) - inner_width) // 2)
            inner = AVLOmapHotNodesBenchmarkMain._range_keys(inner_start, inner_width, insert_count)
            inner_set = set(inner)
            outer_only = [key for key in outer if key not in inner_set]

            def hotset_at(_step: int) -> List[int]:
                return outer

            def sample_hot_key(_step: int, local_rng: random.Random) -> int:
                if inner and local_rng.random() < args.nested_inner_hot_traffic:
                    return local_rng.choice(inner)
                if outer_only:
                    return local_rng.choice(outer_only)
                return local_rng.choice(outer)

            metadata = {
                "mode": mode,
                "outer_range": (outer[0], outer[-1] + 1),
                "inner_range": (inner[0], inner[-1] + 1) if inner else None,
            }
            return {"hotset_at": hotset_at, "sample_hot_key": sample_hot_key, "metadata": metadata}

        if mode == "boundary_hotspot":
            center = (insert_count // 2) if args.boundary_index is None else args.boundary_index
            boundary_width = base_width if args.boundary_width is None else args.boundary_width
            boundary_width = max(1, min(boundary_width, insert_count - 1))
            start = center - (boundary_width // 2)
            hotset = AVLOmapHotNodesBenchmarkMain._range_keys(start, boundary_width, insert_count)

            def hotset_at(_step: int) -> List[int]:
                return hotset

            def sample_hot_key(_step: int, local_rng: random.Random) -> int:
                return local_rng.choice(hotset)

            metadata = {
                "mode": mode,
                "boundary_center": center,
                "boundary_range": (hotset[0], hotset[-1] + 1),
            }
            return {"hotset_at": hotset_at, "sample_hot_key": sample_hot_key, "metadata": metadata}

        move_step = args.range_step if args.range_step > 0 else base_width
        max_start = max(0, insert_count - base_width)

        if mode == "moving_range":

            def hotset_at(step: int) -> List[int]:
                phase = step // temporal_window
                start = (base_start + phase * move_step) % (max_start + 1)
                return AVLOmapHotNodesBenchmarkMain._range_keys(start, base_width, insert_count)

            def sample_hot_key(step: int, local_rng: random.Random) -> int:
                return local_rng.choice(hotset_at(step))

            metadata = {"mode": mode, "range_step": move_step, "temporal_window": temporal_window}
            return {"hotset_at": hotset_at, "sample_hot_key": sample_hot_key, "metadata": metadata}

        if mode == "periodic_ranges":
            periodic_ranges = _build_multi_ranges(
                width=base_width,
                num_ranges=max(2, args.num_hot_ranges),
                start_offset=base_start,
            )
            if not periodic_ranges:
                periodic_ranges = [AVLOmapHotNodesBenchmarkMain._range_keys(base_start, base_width, insert_count)]

            def hotset_at(step: int) -> List[int]:
                phase = step // temporal_window
                return periodic_ranges[phase % len(periodic_ranges)]

            def sample_hot_key(step: int, local_rng: random.Random) -> int:
                return local_rng.choice(hotset_at(step))

            metadata = {
                "mode": mode,
                "ranges": [(group[0], group[-1] + 1) for group in periodic_ranges if group],
                "temporal_window": temporal_window,
            }
            return {"hotset_at": hotset_at, "sample_hot_key": sample_hot_key, "metadata": metadata}

        # expanding_range / contracting_range
        min_width = max(1, base_width // 4)
        width_step = args.range_step if args.range_step > 0 else max(1, base_width // 4)

        if mode == "expanding_range":

            def hotset_at(step: int) -> List[int]:
                phase = step // temporal_window
                width = min(base_width, min_width + phase * width_step)
                width = max(1, min(width, insert_count - 1))
                start = max(0, min(base_start, insert_count - width))
                return AVLOmapHotNodesBenchmarkMain._range_keys(start, width, insert_count)

            def sample_hot_key(step: int, local_rng: random.Random) -> int:
                return local_rng.choice(hotset_at(step))

            metadata = {
                "mode": mode,
                "min_width": min_width,
                "max_width": base_width,
                "width_step": width_step,
                "temporal_window": temporal_window,
            }
            return {"hotset_at": hotset_at, "sample_hot_key": sample_hot_key, "metadata": metadata}

        if mode == "contracting_range":

            def hotset_at(step: int) -> List[int]:
                phase = step // temporal_window
                width = max(min_width, base_width - phase * width_step)
                width = max(1, min(width, insert_count - 1))
                start = max(0, min(base_start, insert_count - width))
                return AVLOmapHotNodesBenchmarkMain._range_keys(start, width, insert_count)

            def sample_hot_key(step: int, local_rng: random.Random) -> int:
                return local_rng.choice(hotset_at(step))

            metadata = {
                "mode": mode,
                "min_width": min_width,
                "start_width": base_width,
                "width_step": width_step,
                "temporal_window": temporal_window,
            }
            return {"hotset_at": hotset_at, "sample_hot_key": sample_hot_key, "metadata": metadata}

        raise ValueError(f"Unsupported hotset mode {mode!r}.")

    @staticmethod
    def _build_workload_plan(
        args: argparse.Namespace,
        keys: List[int],
        rng: random.Random,
        hot_count: int,
    ) -> Dict[str, Any]:
        """Build warmup and benchmark query plans for the selected hotset mode."""
        strategy = AVLOmapHotNodesBenchmarkMain._build_hotset_mode_strategy(
            args=args,
            keys=keys,
            rng=rng,
            hot_count=hot_count,
        )
        hotset_at = strategy["hotset_at"]
        sample_hot_key = strategy["sample_hot_key"]

        warmup_keys: List[int] = []
        query_plan: List[Dict[str, Any]] = []
        observed_hotset_sizes: List[int] = []

        for step in range(args.warmup_accesses):
            current_hotset = hotset_at(step)
            observed_hotset_sizes.append(len(current_hotset))
            warmup_keys.append(sample_hot_key(step, rng))

        for query_index in range(args.num_queries):
            step = args.warmup_accesses + query_index
            current_hotset = hotset_at(step)
            current_hotset_set = set(current_hotset)
            observed_hotset_sizes.append(len(current_hotset))

            use_hot = rng.random() < args.hot_query_probability
            if use_hot:
                key = sample_hot_key(step, rng)
                label = "hot"
            else:
                cold_candidates = [key for key in keys if key not in current_hotset_set]
                if not cold_candidates:
                    key = sample_hot_key(step, rng)
                    label = "hot"
                else:
                    key = rng.choice(cold_candidates)
                    label = "cold"

            query_plan.append({"label": label, "key": key})

        initial_hotset = hotset_at(0) if keys else []
        hotset_size_min = min(observed_hotset_sizes) if observed_hotset_sizes else 0
        hotset_size_max = max(observed_hotset_sizes) if observed_hotset_sizes else 0
        hotset_size_avg = statistics.mean(observed_hotset_sizes) if observed_hotset_sizes else 0.0

        return {
            "warmup_keys": warmup_keys,
            "query_plan": query_plan,
            "initial_hotset_keys": list(initial_hotset),
            "mode_metadata": strategy["metadata"],
            "hotset_size_min": hotset_size_min,
            "hotset_size_max": hotset_size_max,
            "hotset_size_avg": hotset_size_avg,
        }

    @staticmethod
    def _snapshot_client(client: InteractLocalServer) -> Dict[str, int]:
        """Snapshot bandwidth and round counters from a local benchmark client."""
        bytes_read, bytes_written = client.get_bandwidth()
        rounds = client.get_rounds() if hasattr(client, "get_rounds") else 0
        return {
            "bytes_read": bytes_read,
            "bytes_written": bytes_written,
            "rounds": rounds,
        }

    @staticmethod
    def _measure_avl_fast_search_once(
        omap: AVLOmap,
        client: InteractLocalServer,
        key: Any,
        label: str,
        expected_value: Any = None,
    ) -> Dict[str, Any]:
        """Measure one plain-AVLOmap fast_search trial for a key."""
        before = AVLOmapHotNodesBenchmarkMain._snapshot_client(client=client)
        result = omap.fast_search(key=key)
        after = AVLOmapHotNodesBenchmarkMain._snapshot_client(client=client)
        delta = AVLOmapHotNodesBenchmark._delta(after=after, before=before)
        correct = True if expected_value is None else (result == expected_value)

        if isinstance(result, (bytes, bytearray)):
            result_size = len(result)
        else:
            result_size = None if result is None else len(str(result))

        return {
            "label": label,
            "key": key,
            "was_hot_before": False,
            "is_hot_after": False,
            "result_type": type(result).__name__,
            "result_size_bytes": result_size,
            "correct": correct,
            **delta,
        }

    @staticmethod
    def _benchmark_avl_fast_search_workload(
        args: argparse.Namespace,
        keys: List[int],
        payloads: Dict[int, bytes],
        warmup_keys: List[int],
        query_plan: List[Dict[str, Any]],
        bench: AVLOmapHotNodesBenchmark,
    ) -> Dict[str, Any]:
        """Run a precomputed warmup/query workload on plain AVLOmap.fast_search."""
        client = InteractLocalServer()
        omap = AVLOmap(
            num_data=args.num_data,
            key_size=10,
            data_size=max(10, args.payload_size),
            client=client,
        )
        omap.init_server_storage()

        for key in keys:
            omap.insert(key=key, value=payloads[key])

        warmup_trials: List[Dict[str, Any]] = []
        for key in warmup_keys:
            trial = AVLOmapHotNodesBenchmarkMain._measure_avl_fast_search_once(
                omap=omap,
                client=client,
                key=key,
                label="warmup",
                expected_value=payloads[key],
            )
            warmup_trials.append(trial)
            if not trial["correct"]:
                raise AssertionError(f"AVLOmap.fast_search warmup correctness failure for key={key}.")

        hot_trials: List[Dict[str, Any]] = []
        cold_trials: List[Dict[str, Any]] = []
        for entry in query_plan:
            key = entry["key"]
            label = entry["label"]

            trial = AVLOmapHotNodesBenchmarkMain._measure_avl_fast_search_once(
                omap=omap,
                client=client,
                key=key,
                label=label,
                expected_value=payloads[key],
            )
            if not trial["correct"]:
                raise AssertionError(f"AVLOmap.fast_search benchmark correctness failure for key={key}, label={label}.")

            if label == "hot":
                hot_trials.append(trial)
            elif label == "cold":
                cold_trials.append(trial)
            else:
                raise ValueError(f"Unsupported query label {label!r}. Expected 'hot' or 'cold'.")

        return {
            "summary": {
                "warmup": bench._summarize_trials(warmup_trials),
                "hot": bench._summarize_trials(hot_trials),
                "cold": bench._summarize_trials(cold_trials),
            },
            "client_observability": AVLOmapHotNodesBenchmarkMain._snapshot_client(client=client),
        }

    @staticmethod
    def _build_summary_comparison(
        hot_summary: Dict[str, Dict[str, float]],
        avl_fast_summary: Dict[str, Dict[str, float]],
    ) -> Dict[str, Dict[str, Optional[float]]]:
        """Build side-by-side deltas and ratios against plain AVLOmap.fast_search."""

        def _ratio(numerator: float, denominator: float) -> Optional[float]:
            if denominator == 0:
                return None
            return numerator / denominator

        comparison: Dict[str, Dict[str, Optional[float]]] = {}
        for section in ("warmup", "hot", "cold"):
            hot_metrics = hot_summary.get(section, {})
            avl_metrics = avl_fast_summary.get(section, {})

            hot_rounds = float(hot_metrics.get("avg_rounds", 0.0))
            avl_rounds = float(avl_metrics.get("avg_rounds", 0.0))
            hot_bw = float(hot_metrics.get("avg_bandwidth_bytes_total", 0.0))
            avl_bw = float(avl_metrics.get("avg_bandwidth_bytes_total", 0.0))

            comparison[section] = {
                "hot_avg_rounds": hot_rounds,
                "avl_fast_search_avg_rounds": avl_rounds,
                "delta_avg_rounds_hot_minus_avl_fast_search": hot_rounds - avl_rounds,
                "1/ratio_avg_rounds_hot_over_avl_fast_search": _ratio(avl_rounds,hot_rounds),
                "hot_avg_bandwidth_bytes_total": hot_bw,
                "avl_fast_search_avg_bandwidth_bytes_total": avl_bw,
                "delta_avg_bandwidth_hot_minus_avl_fast_search": hot_bw - avl_bw,
                "1/ratio_avg_bandwidth_hot_over_avl_fast_search": _ratio(avl_bw, hot_bw),
            }

        return comparison

    @staticmethod
    def run(args: argparse.Namespace) -> Dict[str, Any]:
        """Run benchmark and return full stats payload."""
        # Backward-compatible defaults for callers constructing Namespace manually.
        default_fields = {
            "hotset_width": None,
            "num_hot_ranges": 3,
            "temporal_window": 32,
            "range_step": 0,
            "nested_inner_fraction": 0.2,
            "nested_inner_hot_traffic": 0.6,
            "boundary_index": None,
            "boundary_width": None,
        }
        for field, value in default_fields.items():
            if not hasattr(args, field):
                setattr(args, field, value)

        if args.insert_count <= 1:
            raise ValueError("--insert-count must be > 1.")
        if args.insert_count > args.num_data:
            raise ValueError("--insert-count must be <= --num-data.")
        if args.hotset_fraction <= 0.0 or args.hotset_fraction >= 1.0:
            raise ValueError("--hotset-fraction must be in (0, 1) so hot and cold sets are both non-empty.")
        if args.hot_query_probability < 0.5 or args.hot_query_probability > 1.0:
            raise ValueError("--hot-query-probability must be in [0.5, 1.0].")
        if args.warmup_accesses < 0 or args.num_queries < 0:
            raise ValueError("--warmup-accesses and --num-queries must be non-negative.")
        if args.payload_size < 0:
            raise ValueError("--payload-size must be non-negative.")
        if args.hotset_range_start < 0:
            raise ValueError("--hotset-range-start must be non-negative.")
        if args.hotset_width is not None and args.hotset_width <= 0:
            raise ValueError("--hotset-width must be positive when provided.")
        if args.num_hot_ranges <= 0:
            raise ValueError("--num-hot-ranges must be positive.")
        if args.temporal_window <= 0:
            raise ValueError("--temporal-window must be positive.")
        if args.range_step < 0:
            raise ValueError("--range-step must be non-negative.")
        if args.nested_inner_fraction <= 0.0 or args.nested_inner_fraction > 1.0:
            raise ValueError("--nested-inner-fraction must be in (0, 1].")
        if args.nested_inner_hot_traffic < 0.0 or args.nested_inner_hot_traffic > 1.0:
            raise ValueError("--nested-inner-hot-traffic must be in [0, 1].")
        if args.boundary_index is not None and (args.boundary_index < 0 or args.boundary_index >= args.insert_count):
            raise ValueError("--boundary-index must be in [0, insert_count).")
        if args.boundary_width is not None and args.boundary_width <= 0:
            raise ValueError("--boundary-width must be positive when provided.")

        rng = random.Random(args.seed)

        keys = list(range(args.insert_count))
        hot_count = max(1, int(round(args.insert_count * args.hotset_fraction)))
        hot_count = min(hot_count, args.insert_count - 1)
        effective_hot_width = hot_count if args.hotset_width is None else min(max(1, args.hotset_width), args.insert_count - 1)
        if args.hotset_mode == "range":
            hotset_end = args.hotset_range_start + effective_hot_width
            if hotset_end > args.insert_count:
                raise ValueError(
                    "--hotset-range-start + hotset_size exceeds --insert-count; "
                    "pick a smaller start or hotset-fraction."
                )
        workload_plan = AVLOmapHotNodesBenchmarkMain._build_workload_plan(
            args=args,
            keys=keys,
            rng=rng,
            hot_count=hot_count,
        )
        hotset_keys = workload_plan["initial_hotset_keys"]
        coldset_keys = [key for key in keys if key not in set(hotset_keys)]
        warmup_keys = workload_plan["warmup_keys"]
        query_plan = workload_plan["query_plan"]

        payloads = _build_payload_map(keys=keys, payload_size=args.payload_size, rng=rng)

        client = InteractLocalServer()
        omap = AVLOmapHotNodesClient(
            num_data=args.num_data,
            key_size=10,
            data_size=max(10, args.payload_size),
            client=client,
            hot_nodes_client_size=args.cache_size,
            hot_access_threshold=args.hot_threshold,
            search_padding=args.search_padding,
            always_dummy_after_search=args.always_dummy_after_search,
        )
        omap.init_server_storage()

        for key in keys:
            omap.insert(key=key, value=payloads[key])

        bench = AVLOmapHotNodesBenchmark(omap=omap)
        benchmark_result = bench.benchmark_query_plan(
            warmup_keys=warmup_keys,
            query_plan=query_plan,
            expected_values=payloads,
            strict_correctness=True,
        )
        benchmark_summary = benchmark_result["summary"]
        avl_fast_baseline = AVLOmapHotNodesBenchmarkMain._benchmark_avl_fast_search_workload(
            args=args,
            keys=keys,
            payloads=payloads,
            warmup_keys=warmup_keys,
            query_plan=query_plan,
            bench=bench,
        )
        comparison_summary = AVLOmapHotNodesBenchmarkMain._build_summary_comparison(
            hot_summary=benchmark_summary,
            avl_fast_summary=avl_fast_baseline["summary"],
        )
        cache_aggregate = bench.get_cache_aggregate(top_n=10)

        return {
            "config": {
                "num_data": args.num_data,
                "insert_count": args.insert_count,
                "payload_size": args.payload_size,
                "seed": args.seed,
                "hotset_mode": args.hotset_mode,
                "hotset_range_start": args.hotset_range_start,
                "hotset_width": args.hotset_width,
                "num_hot_ranges": args.num_hot_ranges,
                "temporal_window": args.temporal_window,
                "range_step": args.range_step,
                "nested_inner_fraction": args.nested_inner_fraction,
                "nested_inner_hot_traffic": args.nested_inner_hot_traffic,
                "boundary_index": args.boundary_index,
                "boundary_width": args.boundary_width,
                "hotset_fraction": args.hotset_fraction,
                "hotset_size": len(hotset_keys),
                "coldset_size": len(coldset_keys),
                "hotset_size_min_over_time": workload_plan["hotset_size_min"],
                "hotset_size_max_over_time": workload_plan["hotset_size_max"],
                "hotset_size_avg_over_time": workload_plan["hotset_size_avg"],
                "hot_query_probability": args.hot_query_probability,
                "warmup_accesses": args.warmup_accesses,
                "num_queries": args.num_queries,
                "cache_size": args.cache_size,
                "hot_threshold": args.hot_threshold,
                "search_padding": args.search_padding,
                "always_dummy_after_search": args.always_dummy_after_search,
                "mode_metadata": workload_plan["mode_metadata"],
            },
            # "hotset_keys": hotset_keys,
            # "coldset_keys": coldset_keys,
            "benchmark": {
                "summary": benchmark_summary,
                "avl_fast_search_summary": avl_fast_baseline["summary"],
                "comparison_hot_search_vs_avl_fast_search": comparison_summary,
            },
            "hot_cache_stats": omap.get_hot_cache_stats(),
            "cache_aggregate": cache_aggregate,
            "operation_observability": omap.get_operation_observability(),
            "client_observability": omap.get_client_observability(),
            "avl_fast_search_client_observability": avl_fast_baseline["client_observability"],
        }


def main() -> None:
    """Run seeded hotset benchmark and print full stats as JSON."""
    parser = AVLOmapHotNodesBenchmarkMain.build_parser()
    args = parser.parse_args()
    output = AVLOmapHotNodesBenchmarkMain.run(args=args)
    with open("output_2.json", "w") as f:
        json.dump(output, f, indent=2, sort_keys=True, default=str)
    # print(json.dumps(output, indent=2, sort_keys=True, default=str))


if __name__ == "__main__":
    main()
