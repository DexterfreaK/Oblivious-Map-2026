import argparse
import random

from daoram.omap import AVLOmapHotNodesBenchmark, AVLOmapHotNodesClient
from daoram.omap.avl_omap_hot_benchmark import AVLOmapHotNodesBenchmarkMain


class TestAVLOmapHotNodesBenchmark:
    def test_random_hotset_workload_metrics_and_correctness(self, client):
        omap = AVLOmapHotNodesClient(
            num_data=128,
            key_size=10,
            data_size=32,
            client=client,
            hot_nodes_client_size=2,
            hot_access_threshold=1,
        )
        omap.init_server_storage()

        insert_count = 32
        payload_rng = random.Random(111)
        expected = {i: payload_rng.randbytes(24) for i in range(insert_count)}
        for i in range(insert_count):
            omap.insert(key=i, value=expected[i])

        key_rng = random.Random(222)
        all_keys = list(range(insert_count))
        hotset_keys = sorted(key_rng.sample(all_keys, 8))
        hotset = set(hotset_keys)
        coldset_keys = [key for key in all_keys if key not in hotset]

        bench = AVLOmapHotNodesBenchmark(omap=omap)
        result = bench.benchmark_random_hotset_workload(
            hotset_keys=hotset_keys,
            coldset_keys=coldset_keys,
            expected_values=expected,
            rng=random.Random(333),
            num_queries=40,
            hot_query_probability=0.65,
            warmup_accesses=16,
        )

        assert len(result["warmup_trials"]) == 16
        assert len(result["hot_trials"]) + len(result["cold_trials"]) == 40
        assert len(result["hot_trials"]) > 0
        assert len(result["cold_trials"]) > 0

        for section in ("warmup_trials", "hot_trials", "cold_trials"):
            for trial in result[section]:
                assert trial["correct"] is True
                assert trial["rounds"] >= 0
                assert trial["bytes_read"] >= 0
                assert trial["bytes_written"] >= 0
                assert trial["bytes_total"] == trial["bytes_read"] + trial["bytes_written"]

        summary = result["summary"]
        for section, expected_count in (("warmup", 16), ("hot", len(result["hot_trials"])), ("cold", len(result["cold_trials"]))):
            assert summary[section]["count"] == expected_count
            assert summary[section]["correct_count"] == expected_count
            assert summary[section]["incorrect_count"] == 0
            assert summary[section]["was_hot_before_true_count"] >= 0
            assert summary[section]["was_hot_before_false_count"] >= 0
            assert 0.0 <= summary[section]["was_hot_before_true_fraction"] <= 1.0
            assert summary[section]["avg_bandwidth_bytes_total"] >= 0
            assert summary[section]["avg_rounds"] >= 0
            assert summary[section]["lower_tail_rounds_p05"] <= summary[section]["upper_tail_rounds_p95"]
            assert summary[section]["min_rounds"] <= summary[section]["max_rounds"]
            assert summary[section]["lower_tail_bandwidth_bytes_total_p05"] <= summary[section]["upper_tail_bandwidth_bytes_total_p95"]
            assert summary[section]["min_bandwidth_bytes_total"] <= summary[section]["max_bandwidth_bytes_total"]

    def test_cli_run_no_keyerror_with_small_cache(self):
        args = argparse.Namespace(
            num_data=64,
            insert_count=16,
            payload_size=32,
            seed=17,
            hotset_mode="random",
            hotset_range_start=0,
            hotset_fraction=0.25,
            hot_query_probability=0.7,
            warmup_accesses=40,
            num_queries=80,
            cache_size=2,
            hot_threshold=1,
            search_padding=False,
            always_dummy_after_search=False,
        )

        output = AVLOmapHotNodesBenchmarkMain.run(args=args)
        summary = output["benchmark"]["summary"]
        assert "warmup_trials" not in output["benchmark"]
        assert "hot_trials" not in output["benchmark"]
        assert "cold_trials" not in output["benchmark"]

        assert summary["warmup"]["incorrect_count"] == 0
        assert summary["hot"]["incorrect_count"] == 0
        assert summary["cold"]["incorrect_count"] == 0
        assert "cache_aggregate" in output
        assert output["cache_aggregate"]["snapshot_count"] == (args.warmup_accesses + args.num_queries)
        assert isinstance(output["cache_aggregate"]["height_distribution"], list)
        assert isinstance(output["cache_aggregate"]["top_accessed_cache_nodes"], list)

    def test_cli_range_hotset_mode(self):
        args = argparse.Namespace(
            num_data=64,
            insert_count=20,
            payload_size=16,
            seed=11,
            hotset_mode="range",
            hotset_range_start=3,
            hotset_fraction=0.25,  # hot_count = round(20*0.25) = 5
            hot_query_probability=0.7,
            warmup_accesses=10,
            num_queries=20,
            cache_size=4,
            hot_threshold=1,
            search_padding=False,
            always_dummy_after_search=False,
        )

        output = AVLOmapHotNodesBenchmarkMain.run(args=args)
        assert output["config"]["hotset_mode"] == "range"
        assert output["config"]["hotset_range_start"] == 3
        assert output["hotset_keys"] == [3, 4, 5, 6, 7]

    def test_cli_range_hotset_mode_out_of_bounds(self):
        args = argparse.Namespace(
            num_data=64,
            insert_count=10,
            payload_size=16,
            seed=11,
            hotset_mode="range",
            hotset_range_start=8,
            hotset_fraction=0.4,  # hot_count = 4, range [8..11] invalid for insert_count=10
            hot_query_probability=0.7,
            warmup_accesses=10,
            num_queries=20,
            cache_size=4,
            hot_threshold=1,
            search_padding=False,
            always_dummy_after_search=False,
        )

        try:
            AVLOmapHotNodesBenchmarkMain.run(args=args)
            assert False, "Expected ValueError for out-of-bounds range hotset mode."
        except ValueError as err:
            assert "hotset-range-start + hotset_size exceeds --insert-count" in str(err)
