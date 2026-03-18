from daoram.omap import BPlusOmapHotNodesClient


def _build_two_level_hot_bplus(client, cache_size=2, threshold=0):
    """
    Build a small two-level B+ tree.

    With `order=4`, inserting 1..4 produces:
    - an internal root
    - two leaf children
    """
    omap = BPlusOmapHotNodesClient(
        order=4,
        num_data=64,
        key_size=10,
        data_size=10,
        client=client,
        hot_nodes_client_size=cache_size,
        hot_access_threshold=threshold,
    )
    omap.init_server_storage()
    for key in (1, 2, 3, 4):
        omap.insert(key=key, value=key)
    return omap


class TestBPlusOmapHotNodesClient:
    def test_repeated_search_promotes_leaf_and_reduces_rounds(self, client):
        omap = _build_two_level_hot_bplus(client=client, cache_size=2, threshold=0)

        client.reset_rounds()
        assert omap.search(key=4) == 4
        first_rounds = client.get_rounds()

        assert len(omap.hot_nodes_client) == 1

        hits_before = omap.hot_cache_hits
        client.reset_rounds()
        assert omap.search(key=4) == 4
        second_rounds = client.get_rounds()

        assert omap.hot_cache_hits > hits_before
        assert first_rounds == 4
        assert second_rounds == 2

    def test_evicted_leaf_keeps_parent_pointer_hot_until_revisit(self, client):
        omap = _build_two_level_hot_bplus(client=client, cache_size=1, threshold=0)

        assert omap.search(key=4) == 4
        right_leaf_key = omap.hot_nodes_client[0]

        assert omap.search(key=1) == 1
        assert omap.hot_cache_evictions > 0
        assert len(omap.hot_nodes_client) == 1
        assert right_leaf_key not in omap.hot_nodes_client

        root_index = omap._find_in_stash(omap.root[0])
        assert root_index >= 0
        root_node = omap._stash[root_index]
        child_index = omap._find_child_index(root_node, 4)
        pointer = root_node.value.values[child_index]

        assert pointer["node_id"] == right_leaf_key
        assert pointer["location"] == omap.HOT_CLI_CACHE

        assert omap.search(key=4) == 4

    def test_search_update_works_with_hot_leaf(self, client):
        omap = _build_two_level_hot_bplus(client=client, cache_size=2, threshold=0)

        assert omap.search(key=4) == 4
        assert omap.search(key=4, value=400) == 4
        assert omap.search(key=4) == 400

    def test_fast_search_flushes_hot_cache(self, client):
        omap = _build_two_level_hot_bplus(client=client, cache_size=2, threshold=0)

        assert omap.search(key=4) == 4
        assert omap.hot_nodes_client

        assert omap.fast_search(key=4) == 4
        assert omap.hot_nodes_client == []

        assert omap.fast_search(key=4, value=44) == 4
        assert omap.fast_search(key=4) == 44
        assert omap.hot_nodes_client == []

    def test_insert_and_delete_remain_correct_after_hot_cache_warmup(self, client):
        omap = BPlusOmapHotNodesClient(
            order=4,
            num_data=64,
            key_size=10,
            data_size=10,
            client=client,
            hot_nodes_client_size=2,
            hot_access_threshold=0,
        )
        omap.init_server_storage()

        for key in range(1, 9):
            omap.insert(key=key, value=key)

        assert omap.search(key=6) == 6
        assert omap.hot_nodes_client

        omap.insert(key=9, value=9)
        assert omap.search(key=9) == 9

        assert omap.delete(key=2) == 2
        assert omap.search(key=2) is None

        for key in (1, 3, 4, 5, 6, 7, 8, 9):
            assert omap.search(key=key) == key
