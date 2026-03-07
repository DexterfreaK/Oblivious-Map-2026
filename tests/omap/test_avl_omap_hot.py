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
