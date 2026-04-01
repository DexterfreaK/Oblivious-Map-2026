import argparse
import random
from collections import Counter

from daoram.omap import BPlusOmapHotNodesBenchmark
from daoram.omap.bplus_omap_hot_benchmark import BPlusOmapHotNodesBenchmarkMain


def _make_args(**overrides):
    defaults = {
        "order": 4,
        "num_data": 64,
        "insert_count": 24,
        "payload_size": 16,
        "seed": 17,
        "workload": "hotset",
        "warmup_requests": 12,
        "num_queries": 24,
        "cache_size": 2,
        "hot_threshold": 0,
        "rolling_window": 8,
        "output_json": None,
        "hotset_size": 8,
        "hot_query_probability": 0.7,
        "zipf_alpha": 1.2,
    }
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


class TestBPlusOmapHotNodesBenchmark:
    def test_hotset_request_trace_is_seeded_and_matches_hotset_membership(self):
        benchmark = BPlusOmapHotNodesBenchmark(order=4)
        keys = list(range(20))

        trace_one, metadata_one = benchmark._build_hotset_request_trace(
            keys=keys,
            rng=random.Random(123),
            warmup_requests=10,
            num_queries=20,
            hotset_size=5,
            hot_query_probability=0.7,
        )
        trace_two, metadata_two = benchmark._build_hotset_request_trace(
            keys=keys,
            rng=random.Random(123),
            warmup_requests=10,
            num_queries=20,
            hotset_size=5,
            hot_query_probability=0.7,
        )

        assert trace_one == trace_two
        assert metadata_one == metadata_two
        assert metadata_one["hotset_size"] == 5
        assert len(metadata_one["hotset_keys"]) == 5

        hotset = set(metadata_one["hotset_keys"])
        assert any(entry["label"] == "hot" for entry in trace_one)
        assert any(entry["label"] == "cold" for entry in trace_one)
        for entry in trace_one:
            if entry["label"] == "hot":
                assert entry["key"] in hotset
            else:
                assert entry["key"] not in hotset

    def test_zipf_request_trace_is_seeded_and_skewed(self):
        benchmark = BPlusOmapHotNodesBenchmark(order=4)
        keys = list(range(30))

        trace_one, metadata_one = benchmark._build_zipf_request_trace(
            keys=keys,
            rng=random.Random(321),
            warmup_requests=50,
            num_queries=150,
            zipf_alpha=1.2,
        )
        trace_two, metadata_two = benchmark._build_zipf_request_trace(
            keys=keys,
            rng=random.Random(321),
            warmup_requests=50,
            num_queries=150,
            zipf_alpha=1.2,
        )

        assert trace_one == trace_two
        assert metadata_one == metadata_two
        assert all(entry["label"] == "zipf" for entry in trace_one)

        counts = Counter(entry["key"] for entry in trace_one)
        assert counts[metadata_one["ranked_keys"][0]] >= counts[metadata_one["ranked_keys"][-1]]

    def test_end_to_end_run_has_expected_json_shape_and_non_negative_rounds(self):
        output = BPlusOmapHotNodesBenchmarkMain.run(
            args=_make_args(
                workload="hotset",
                warmup_requests=8,
                num_queries=16,
                hotset_size=6,
                hot_query_probability=0.75,
                cache_size=2,
            )
        )

        assert set(output.keys()) == {"config", "workload_metadata", "request_trace", "runs", "pairwise_comparison"}
        assert len(output["request_trace"]) == output["config"]["warmup_requests"] + output["config"]["num_queries"]

        for run_name in ("score_based_hot_cache", "reject_all_hot_cache", "plain_bplus_search"):
            run = output["runs"][run_name]
            assert len(run["requests"]) == len(output["request_trace"])
            assert set(run["series"].keys()) == {
                "rounds",
                "rolling_avg_rounds",
                "cache_hit",
                "rolling_cache_hit_rate",
                "promotions_delta",
                "evictions_delta",
            }
            assert len(run["series"]["rounds"]) == len(output["request_trace"])
            assert len(run["series"]["rolling_avg_rounds"]) == len(output["request_trace"])
            assert len(run["series"]["rolling_cache_hit_rate"]) == len(output["request_trace"])
            assert run["summary"]["incorrect_count"] == 0

            for request in run["requests"]:
                assert request["correct"] is True
                assert request["rounds"] >= 0
                assert request["promotions_delta"] >= 0
                assert request["evictions_delta"] >= 0

    def test_score_based_eviction_accuracy_is_one_when_evictions_happen(self):
        output = BPlusOmapHotNodesBenchmarkMain.run(
            args=_make_args(
                workload="hotset",
                seed=5,
                warmup_requests=20,
                num_queries=40,
                hotset_size=12,
                hot_query_probability=1.0,
                cache_size=1,
                hot_threshold=0,
            )
        )

        score_run = output["runs"]["score_based_hot_cache"]
        assert score_run["summary"]["total_actual_evictions"] > 0
        assert score_run["summary"]["eviction_accuracy"] == 1.0

    def test_reject_all_matches_plain_bplus_search(self):
        output = BPlusOmapHotNodesBenchmarkMain.run(
            args=_make_args(
                workload="hotset",
                seed=13,
                warmup_requests=10,
                num_queries=20,
                hotset_size=7,
                hot_query_probability=0.8,
                cache_size=3,
                hot_threshold=0,
            )
        )

        reject_run = output["runs"]["reject_all_hot_cache"]
        comparison = output["pairwise_comparison"]["reject_all_vs_plain_search"]

        assert comparison["parity"] is True
        assert comparison["per_request_rounds_equal"] is True
        assert comparison["correctness_flags_equal"] is True
        assert comparison["all_requests_correct"] is True
        assert comparison["summary_deltas"]["avg_rounds"] == 0.0
        assert comparison["summary_deltas"]["total_rounds"] == 0
        assert reject_run["summary"]["cache_hit_rate"] == 0.0
        assert reject_run["summary"]["total_promotions"] == 0
        assert reject_run["summary"]["total_evictions"] == 0
        assert reject_run["final_cache_stats"]["cache_size"] == 0

    def test_warmup_index_is_none_when_series_never_improves(self):
        assert BPlusOmapHotNodesBenchmark._steady_state_warmup_index(
            [5.0, 5.0, 5.0],
            improving_when="decrease",
        ) is None
        assert BPlusOmapHotNodesBenchmark._steady_state_warmup_index(
            [0.0, 0.0, 0.0],
            improving_when="increase",
        ) is None
