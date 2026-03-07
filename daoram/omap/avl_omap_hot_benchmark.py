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
            choices=("random", "range"),
            default="random",
            help="How to choose hotset keys: random sample or contiguous range.",
        )
        parser.add_argument(
            "--hotset-range-start",
            type=int,
            default=0,
            help="Start index for range hotset mode; ignored for random mode.",
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
            default=False,
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
    def run(args: argparse.Namespace) -> Dict[str, Any]:
        """Run benchmark and return full stats payload."""
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

        rng = random.Random(args.seed)

        keys = list(range(args.insert_count))
        hot_count = max(1, int(round(args.insert_count * args.hotset_fraction)))
        hot_count = min(hot_count, args.insert_count - 1)
        if args.hotset_mode == "range":
            hotset_end = args.hotset_range_start + hot_count
            if hotset_end > args.insert_count:
                raise ValueError(
                    "--hotset-range-start + hotset_size exceeds --insert-count; "
                    "pick a smaller start or hotset-fraction."
                )
            hotset_keys = list(range(args.hotset_range_start, hotset_end))
        else:
            hotset_keys = sorted(rng.sample(keys, hot_count))
        hotset_key_set = set(hotset_keys)
        coldset_keys = [key for key in keys if key not in hotset_key_set]

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
        benchmark_result = bench.benchmark_random_hotset_workload(
            hotset_keys=hotset_keys,
            coldset_keys=coldset_keys,
            expected_values=payloads,
            rng=rng,
            num_queries=args.num_queries,
            hot_query_probability=args.hot_query_probability,
            warmup_accesses=args.warmup_accesses,
            strict_correctness=True,
        )
        benchmark_summary = benchmark_result["summary"]
        cache_aggregate = bench.get_cache_aggregate(top_n=10)

        return {
            "config": {
                "num_data": args.num_data,
                "insert_count": args.insert_count,
                "payload_size": args.payload_size,
                "seed": args.seed,
                "hotset_mode": args.hotset_mode,
                "hotset_range_start": args.hotset_range_start,
                "hotset_fraction": args.hotset_fraction,
                "hotset_size": len(hotset_keys),
                "coldset_size": len(coldset_keys),
                "hot_query_probability": args.hot_query_probability,
                "warmup_accesses": args.warmup_accesses,
                "num_queries": args.num_queries,
                "cache_size": args.cache_size,
                "hot_threshold": args.hot_threshold,
                "search_padding": args.search_padding,
                "always_dummy_after_search": args.always_dummy_after_search,
            },
            "hotset_keys": hotset_keys,
            "coldset_keys": coldset_keys,
            "benchmark": {
                "summary": benchmark_summary,
            },
            "hot_cache_stats": omap.get_hot_cache_stats(),
            "cache_aggregate": cache_aggregate,
            "operation_observability": omap.get_operation_observability(),
            "client_observability": omap.get_client_observability(),
        }


def main() -> None:
    """Run seeded hotset benchmark and print full stats as JSON."""
    parser = AVLOmapHotNodesBenchmarkMain.build_parser()
    args = parser.parse_args()
    output = AVLOmapHotNodesBenchmarkMain.run(args=args)
    with open("output.json", "w") as f:
        json.dump(output, f, indent=2, sort_keys=True, default=str)
    # print(json.dumps(output, indent=2, sort_keys=True, default=str))


if __name__ == "__main__":
    main()
