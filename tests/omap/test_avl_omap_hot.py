import copy

from daoram.omap import AVLOmapHotNodesClient


class TestAVLOmapHotNodesClient:
    def test_search_correctness_with_hot_cache(self, client):
        omap = AVLOmapHotNodesClient(
            num_data=64,
            key_size=10,
            data_size=10,
            client=client,
            hot_nodes_client_size=2,
            hot_access_threshold=1,
        )
        omap.init_server_storage()

        for i in range(16):
            omap.insert(key=i, value=i)

        for _ in range(3):
            assert omap.search(key=7) == 7

        assert 0 < len(omap.hot_nodes_client) <= 2

        hits_before = omap.hot_cache_hits
        assert omap.search(key=7) == 7
        assert omap.hot_cache_hits > hits_before

    def test_eviction_and_reinsertion(self, client):
        omap = AVLOmapHotNodesClient(
            num_data=64,
            key_size=10,
            data_size=10,
            client=client,
            hot_nodes_client_size=1,
            hot_access_threshold=1,
        )
        omap.init_server_storage()

        for i in range(16):
            omap.insert(key=i, value=i)

        # Diverse traversals to force cache replacement at size=1.
        for key in (2, 12, 3, 14, 4, 10):
            assert omap.search(key=key) == key

        assert omap.hot_cache_evictions > 0
        assert len(omap.hot_nodes_client) <= 1

        # Ensure data remains retrievable after evictions.
        for key in (2, 3, 4, 10, 12, 14):
            assert omap.search(key=key) == key

    def test_search_update_with_hot_cache(self, client):
        omap = AVLOmapHotNodesClient(
            num_data=64,
            key_size=10,
            data_size=10,
            client=client,
            hot_nodes_client_size=2,
            hot_access_threshold=1,
        )
        omap.init_server_storage()

        for i in range(16):
            omap.insert(key=i, value=i)

        assert omap.search(key=5) == 5
        assert omap.search(key=5, value=500) == 5
        assert omap.search(key=5) == 500

    def test_operation_observability_counters(self, client):
        omap = AVLOmapHotNodesClient(
            num_data=64,
            key_size=10,
            data_size=10,
            client=client,
            hot_nodes_client_size=2,
            hot_access_threshold=1,
        )
        omap.init_server_storage()

        for i in range(8):
            omap.insert(key=i, value=i)

        assert omap.search(key=3) == 3
        assert omap.search(key=3, value=33) == 3
        assert omap.fast_search(key=3) == 33
        assert omap.fast_search(key=3, value=333) == 33
        assert omap.delete(key=3) == 333

        totals = omap.get_operation_observability()
        for op_name in ("insert", "search", "update", "fast_search", "fast_update", "delete"):
            assert op_name in totals
            assert totals[op_name]["count"] > 0
            # In no-padding mode, fully hot searches may incur zero ORAM rounds.
            assert totals[op_name]["rounds"] >= 0
            assert totals[op_name]["bytes_read"] >= 0
            assert totals[op_name]["bytes_written"] >= 0

        assert totals["insert"]["rounds"] > 0
        assert totals["delete"]["rounds"] > 0

        last = omap.get_last_operation_observability()
        assert last is not None
        assert last["operation"] == "delete"
        assert last["success"] is True
        assert last["rounds"] > 0

        client_obs = omap.get_client_observability()
        assert client_obs["rounds"] == client.get_rounds()
        assert client_obs["rounds"] > 0

    def test_insert_after_hot_cache_rotation_safety(self, client):
        omap = AVLOmapHotNodesClient(
            num_data=64,
            key_size=10,
            data_size=10,
            client=client,
            hot_nodes_client_size=2,
            hot_access_threshold=1,
        )
        omap.init_server_storage()

        # Build a tree where a later insert triggers balancing/rotation.
        for key in (30, 20, 40, 10):
            omap.insert(key=key, value=key)

        # Warm cache through searches.
        assert omap.search(key=10) == 10
        assert len(omap.hot_nodes_client) > 0

        # Structural op should remain correct even with prior hot caching.
        omap.insert(key=5, value=5)

        for key in (5, 10, 20, 30, 40):
            assert omap.search(key=key) == key

    def test_search_padding_runs_random_walk_tail(self, client):
        omap = AVLOmapHotNodesClient(
            num_data=64,
            key_size=10,
            data_size=10,
            client=client,
            hot_nodes_client_size=64,
            hot_access_threshold=1000,
            search_padding=True,
            always_dummy_after_search=False,
        )
        omap.init_server_storage()

        for i in range(16):
            omap.insert(key=i, value=i)

        call_count = {"n": 0}
        original = omap._random_walk_to_avl_leaf

        def wrapped(start_key, start_leaf):
            call_count["n"] += 1
            return original(start_key=start_key, start_leaf=start_leaf)

        omap._random_walk_to_avl_leaf = wrapped
        assert omap.search(key=7) == 7
        assert call_count["n"] == 1

        omap2 = AVLOmapHotNodesClient(
            num_data=64,
            key_size=10,
            data_size=10,
            client=client,
            hot_nodes_client_size=64,
            hot_access_threshold=1000,
            search_padding=True,
            always_dummy_after_search=True,
        )
        omap2.init_server_storage()
        for i in range(16):
            omap2.insert(key=i, value=i)

        call_count2 = {"n": 0}
        original2 = omap2._random_walk_to_avl_leaf

        def wrapped2(start_key, start_leaf):
            call_count2["n"] += 1
            return original2(start_key=start_key, start_leaf=start_leaf)

        omap2._random_walk_to_avl_leaf = wrapped2
        assert omap2.search(key=7) == 7
        assert call_count2["n"] == 2

    def test_search_padding_false_skips_search_random_walk_tail(self, client):
        omap = AVLOmapHotNodesClient(
            num_data=64,
            key_size=10,
            data_size=10,
            client=client,
            hot_nodes_client_size=64,
            hot_access_threshold=1000,
            search_padding=False,
            always_dummy_after_search=False,
        )
        omap.init_server_storage()

        for i in range(16):
            omap.insert(key=i, value=i)

        call_count = {"n": 0}
        original = omap._random_walk_to_avl_leaf

        def wrapped(start_key, start_leaf):
            call_count["n"] += 1
            return original(start_key=start_key, start_leaf=start_leaf)

        omap._random_walk_to_avl_leaf = wrapped
        assert omap.search(key=7) == 7
        assert call_count["n"] == 0

    def test_reinsertion_uses_random_walk_tail(self, client):
        omap = AVLOmapHotNodesClient(
            num_data=64,
            key_size=10,
            data_size=10,
            client=client,
            hot_nodes_client_size=1,
            hot_access_threshold=1,
            search_padding=False,
            always_dummy_after_search=False,
        )
        omap.init_server_storage()

        for i in range(16):
            omap.insert(key=i, value=i)

        random_walk_calls = {"n": 0}
        original_walk = omap._random_walk_to_avl_leaf

        def wrapped_walk(start_key, start_leaf):
            random_walk_calls["n"] += 1
            return original_walk(start_key=start_key, start_leaf=start_leaf)

        omap._random_walk_to_avl_leaf = wrapped_walk

        original_dummy = omap._perform_dummy_operation

        def fail_dummy(*args, **kwargs):
            raise AssertionError("Fixed-round dummy operation should not be used in reinsertion.")

        omap._perform_dummy_operation = fail_dummy
        try:
            for key in (2, 12, 3, 14, 4, 10):
                assert omap.search(key=key) == key
        finally:
            omap._perform_dummy_operation = original_dummy

        assert omap.hot_cache_evictions > 0
        assert random_walk_calls["n"] > 0

    def test_dummy_walk_freezes_hot_metadata(self, client):
        omap = AVLOmapHotNodesClient(
            num_data=64,
            key_size=10,
            data_size=10,
            client=client,
            hot_nodes_client_size=8,
            hot_access_threshold=1,
            search_padding=False,
        )
        omap.init_server_storage()

        for i in range(16):
            omap.insert(key=i, value=i)

        for key in (7, 8, 9, 10):
            assert omap.search(key=key) == key

        access_before = copy.deepcopy(omap._access_counts)
        hot_order_before = list(omap._hot_nodes_client.keys())
        temp_before = list(omap._temp_hot_nodes.keys())
        pending_before = copy.deepcopy(omap._pending_reinsert_nodes)
        stats_before = omap.get_hot_cache_stats()

        start_key, start_leaf = omap.root
        omap._random_walk_to_avl_leaf(start_key=start_key, start_leaf=start_leaf)

        assert omap._access_counts == access_before
        assert list(omap._hot_nodes_client.keys()) == hot_order_before
        assert list(omap._temp_hot_nodes.keys()) == temp_before
        assert omap._pending_reinsert_nodes == pending_before
        assert omap.get_hot_cache_stats() == stats_before

    def test_dummy_walk_cache_hit_avoids_oram_round(self, client):
        omap = AVLOmapHotNodesClient(
            num_data=16,
            key_size=10,
            data_size=10,
            client=client,
            hot_nodes_client_size=4,
            hot_access_threshold=1,
            search_padding=False,
        )
        omap.init_server_storage()

        omap.insert(key=1, value=1)
        omap.insert(key=2, value=2)

        # Warm both nodes so root and child are in hot cache.
        for _ in range(4):
            assert omap.search(key=1) == 1
            assert omap.search(key=2) == 2

        root_key, root_leaf = omap.root
        assert root_key in omap.hot_nodes_client
        root_cached = omap._hot_nodes_client[root_key]
        child_key = root_cached.value.l_key if root_cached.value.l_key is not None else root_cached.value.r_key
        assert child_key is not None
        assert child_key in omap.hot_nodes_client

        rounds_before = client.get_rounds()
        omap._random_walk_to_avl_leaf(start_key=root_key, start_leaf=root_leaf)
        rounds_after = client.get_rounds()

        assert rounds_after == rounds_before

    def test_precise_rounds_search_no_padding_two_oram_nodes(self, client):
        omap = AVLOmapHotNodesClient(
            num_data=64,
            key_size=10,
            data_size=10,
            client=client,
            hot_nodes_client_size=0,
            hot_access_threshold=1,
            search_padding=False,
        )
        omap.init_server_storage()

        # AVL after inserts [2, 1, 3]:
        #      2
        #     / \
        #    1   3
        for key in (2, 1, 3):
            omap.insert(key=key, value=key)

        client.reset_rounds()
        assert omap.search(key=1) == 1

        # root(2) and child(1) are both ORAM fetches:
        # each _move_node_to_local => 1 read execute + 1 write execute => 2 rounds.
        # total = 2 nodes * 2 rounds = 4.
        assert client.get_rounds() == 4

    def test_precise_rounds_search_padding_one_oram_tail_hop(self, client):
        omap = AVLOmapHotNodesClient(
            num_data=64,
            key_size=10,
            data_size=10,
            client=client,
            hot_nodes_client_size=1,
            hot_access_threshold=1,
            search_padding=False,
        )
        omap.init_server_storage()

        # AVL after inserts [2, 1, 3]:
        #      2
        #     / \
        #    1   3
        for key in (2, 1, 3):
            omap.insert(key=key, value=key)

        # Warm key 2 so only root is cached.
        for _ in range(3):
            assert omap.search(key=2) == 2
        assert omap.hot_nodes_client == [2]

        # Enable random-tail padding for measured operation.
        omap._search_padding = True

        before_access = copy.deepcopy(omap._access_counts)
        client.reset_rounds()
        assert omap.search(key=2) == 2
        rounds = client.get_rounds()

        # User traversal root->2 is cached => 0 rounds.
        # Random tail from 2 picks one existing child (1 or 3), both non-cached leaves:
        # one ORAM fetch => 2 rounds total.
        assert rounds == 2

        # Dummy tail is frozen: only root(2) from user traversal increments.
        assert omap.get_access_count(2) == before_access.get(2, 0) + 1
        assert omap.get_access_count(1) == before_access.get(1, 0)
        assert omap.get_access_count(3) == before_access.get(3, 0)

    def test_precise_rounds_eviction_reinsertion_with_leaf_tail(self, client):
        omap = AVLOmapHotNodesClient(
            num_data=64,
            key_size=10,
            data_size=10,
            client=client,
            hot_nodes_client_size=1,
            hot_access_threshold=1,
            search_padding=False,
        )
        omap.init_server_storage()

        # AVL after inserts [2, 1, 3, 4]:
        #      2
        #     / \
        #    1   3
        #         \
        #          4
        for key in (2, 1, 3, 4):
            omap.insert(key=key, value=key)

        # Make 4 the sole cached key.
        assert omap.search(key=4) == 4
        assert omap.hot_nodes_client == [4]

        client.reset_rounds()
        assert omap.search(key=1) == 1
        rounds = client.get_rounds()

        # Search path 2->1 are ORAM (cache only has 4) => 2 nodes * 2 rounds = 4.
        # Evicting 4 triggers reinsertion:
        # parent update for child=4 traverses root->3, both ORAM => 2 nodes * 2 rounds = 4.
        # reinsert read+write path for 4 => 2 rounds.
        # random tail from 4: 4 is leaf, starts from pending/local => 0 rounds.
        # total expected = search(4) + parent-update(4) + reinsert(2) + tail(0) = 10.
        assert rounds == 10

    def test_precise_rounds_double_tail_all_cached_zero_rounds(self, client):
        omap = AVLOmapHotNodesClient(
            num_data=32,
            key_size=10,
            data_size=10,
            client=client,
            hot_nodes_client_size=2,
            hot_access_threshold=1,
            search_padding=False,
            always_dummy_after_search=True,
        )
        omap.init_server_storage()

        # AVL after inserts [1, 2]:
        #    1
        #     \
        #      2
        for key in (1, 2):
            omap.insert(key=key, value=key)

        # Warm both nodes into cache.
        for _ in range(4):
            assert omap.search(key=2) == 2
        assert set((1, 2)).issubset(set(omap.hot_nodes_client))

        # Turn on padded search (which now means random-tail walk), with extra tail.
        omap._search_padding = True

        client.reset_rounds()
        assert omap.search(key=2) == 2

        # user traversal root->2 cached => 0 rounds.
        # tail-1 starts at leaf 2 (cached) => 0 rounds.
        # tail-2 same => 0 rounds.
        assert client.get_rounds() == 0
