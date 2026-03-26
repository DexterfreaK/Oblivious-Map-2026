import math
import random

from daoram.dependency import InteractLocalServer
from daoram.omap import (
    BPlusOmap,
    BPlusOmapHotNodesClient,
    ExponentialMechanismHotCacheAdmissionLayer,
    HotCacheAdmissionCandidate,
    RejectAllHotCacheAdmissionLayer,
    ScoreBasedHotCacheAdmissionLayer,
    secret_user_id_access_utility,
)


def _build_two_level_hot_bplus(client, cache_size=2, threshold=0, hot_admission_layer=None):
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
        hot_admission_layer=hot_admission_layer or ScoreBasedHotCacheAdmissionLayer(),
    )
    omap.init_server_storage()
    for key in (1, 2, 3, 4):
        omap.insert(key=key, value=key)
    return omap


def _build_regular_bplus(client, order=4, num_data=64):
    return BPlusOmap(
        order=order,
        num_data=num_data,
        key_size=10,
        data_size=10,
        client=client,
    )


def _build_three_level_hot_bplus(client, cache_size=0, threshold=100, hot_admission_layer=None):
    """Build a deeper B+ tree so subtree-height secret_user_id assignment can target a mid-level internal node."""
    omap = BPlusOmapHotNodesClient(
        order=4,
        num_data=128,
        key_size=10,
        data_size=10,
        client=client,
        hot_nodes_client_size=cache_size,
        hot_access_threshold=threshold,
        hot_admission_layer=hot_admission_layer or ScoreBasedHotCacheAdmissionLayer(),
    )
    omap.init_server_storage()
    for key in range(1, 17):
        omap.insert(key=key, value=key)
    return omap


def _build_pictured_hot_bplus(client, cache_size=2, threshold=0, hot_admission_layer=None):
    """
    Build the exact tree shape from the user-supplied diagram.

    Root: [4, 21]
    - left internal [3] -> leaves [1, 2], [3]
    - middle internal [5, 6] -> leaves [4], [5], [6, 10]
    - right internal [23] -> leaves [21], [23, 30]
    """
    omap = BPlusOmapHotNodesClient(
        order=4,
        num_data=64,
        key_size=10,
        data_size=10,
        client=client,
        hot_nodes_client_size=cache_size,
        hot_access_threshold=threshold,
        hot_admission_layer=hot_admission_layer or ScoreBasedHotCacheAdmissionLayer(),
    )
    tree = omap._init_ods_storage(data=None)

    def make_node(keys, values, leaf):
        node = omap._get_bplus_data(keys=keys, values=values)
        node.leaf = leaf
        return node

    leaf_12 = make_node(keys=[1, 2], values=[1, 2], leaf=0)
    leaf_3 = make_node(keys=[3], values=[3], leaf=1)
    leaf_4 = make_node(keys=[4], values=[4], leaf=2)
    leaf_5 = make_node(keys=[5], values=[5], leaf=3)
    leaf_610 = make_node(keys=[6, 10], values=[6, 10], leaf=4)
    leaf_21 = make_node(keys=[21], values=[21], leaf=5)
    leaf_2330 = make_node(keys=[23, 30], values=[23, 30], leaf=6)

    left = make_node(
        keys=[3],
        values=[
            omap._make_pointer(node_id=leaf_12.key, leaf_p=leaf_12.leaf, location=omap.ORAM),
            omap._make_pointer(node_id=leaf_3.key, leaf_p=leaf_3.leaf, location=omap.ORAM),
        ],
        leaf=7,
    )
    middle = make_node(
        keys=[5, 6],
        values=[
            omap._make_pointer(node_id=leaf_4.key, leaf_p=leaf_4.leaf, location=omap.ORAM),
            omap._make_pointer(node_id=leaf_5.key, leaf_p=leaf_5.leaf, location=omap.ORAM),
            omap._make_pointer(node_id=leaf_610.key, leaf_p=leaf_610.leaf, location=omap.ORAM),
        ],
        leaf=8,
    )
    right = make_node(
        keys=[23],
        values=[
            omap._make_pointer(node_id=leaf_21.key, leaf_p=leaf_21.leaf, location=omap.ORAM),
            omap._make_pointer(node_id=leaf_2330.key, leaf_p=leaf_2330.leaf, location=omap.ORAM),
        ],
        leaf=9,
    )
    root = make_node(
        keys=[4, 21],
        values=[
            omap._make_pointer(node_id=left.key, leaf_p=left.leaf, location=omap.ORAM),
            omap._make_pointer(node_id=middle.key, leaf_p=middle.leaf, location=omap.ORAM),
            omap._make_pointer(node_id=right.key, leaf_p=right.leaf, location=omap.ORAM),
        ],
        leaf=10,
    )

    nodes = [leaf_12, leaf_3, leaf_4, leaf_5, leaf_610, leaf_21, leaf_2330, left, middle, right, root]
    for node in nodes:
        assert tree.fill_data_to_storage_leaf(node)

    omap.root = (root.key, root.leaf)
    client.init_storage(storage={omap._name: tree})
    return omap, {
        "right_internal": right.key,
        "leaf_21": leaf_21.key,
        "leaf_2330": leaf_2330.key,
    }


def _resident_node(omap, node_id):
    cached = omap._hot_nodes_client.get(node_id)
    if cached is not None:
        return cached

    index = omap._find_in_stash(node_id)
    if index >= 0:
        return omap._stash[index]

    tree = omap.client._storage[omap._name]
    for bucket in tree.storage._internal_data:
        for node in bucket:
            if node.key == node_id:
                return node

    raise AssertionError(f"Node {node_id} is not resident in cache, stash, or server storage.")


def _set_access_count_everywhere(omap, node_id, access_count):
    cached = omap._hot_nodes_client.get(node_id)
    if cached is not None:
        cached.value.metadata["access_count"] = access_count

    for node in omap._stash:
        if node.key == node_id:
            node.value.metadata["access_count"] = access_count

    tree = omap.client._storage[omap._name]
    for bucket in tree.storage._internal_data:
        for node in bucket:
            if node.key == node_id:
                node.value.metadata["access_count"] = access_count


def _expected_insert_probability(candidate_access, right_access, resident_access, epsilon, utility_sensitivity=1.0):
    candidate = HotCacheAdmissionCandidate(
        key="candidate",
        metadata={"access_count": candidate_access, "secret_user_id": 0},
    )
    right_internal = HotCacheAdmissionCandidate(
        key="right_internal",
        metadata={"access_count": right_access, "secret_user_id": 0},
    )
    resident_leaf = HotCacheAdmissionCandidate(
        key="resident_leaf",
        metadata={"access_count": resident_access, "secret_user_id": 0},
    )

    right_weight = math.exp(
        (epsilon * secret_user_id_access_utility(candidate, right_internal)) / (2.0 * utility_sensitivity)
    )
    resident_weight = math.exp(
        (epsilon * secret_user_id_access_utility(candidate, resident_leaf)) / (2.0 * utility_sensitivity)
    )
    return (right_weight + resident_weight) / (1.0 + right_weight + resident_weight)


def _simulate_pictured_exponential_insert_probability(
    *,
    trials,
    epsilon,
    candidate_pre_access,
    right_internal_pre_access,
    resident_leaf_access,
    utility_sensitivity=1.0,
):
    inserted = 0

    for trial_seed in range(trials):
        client = InteractLocalServer()
        omap, ids = _build_pictured_hot_bplus(
            client=client,
            cache_size=2,
            threshold=0,
            hot_admission_layer=ScoreBasedHotCacheAdmissionLayer(),
        )

        # Fill the cache through a real search first, but choose the initial hot leaf randomly.
        warm_key = random.Random(trial_seed).choice([21, 30])
        assert omap.search(key=warm_key) == warm_key

        # Normalize the resident leaf so the trial starts from the same cached pair.
        if ids["leaf_21"] not in omap.hot_nodes_client:
            assert omap.search(key=21) == 21
            assert omap.search(key=21) == 21

        assert set(omap.hot_nodes_client) == {ids["right_internal"], ids["leaf_21"]}

        _set_access_count_everywhere(omap, ids["right_internal"], right_internal_pre_access)
        _set_access_count_everywhere(omap, ids["leaf_21"], resident_leaf_access)
        _set_access_count_everywhere(omap, ids["leaf_2330"], candidate_pre_access)

        omap._hot_admission_layer = ExponentialMechanismHotCacheAdmissionLayer(
            epsilon=epsilon,
            utility_sensitivity=utility_sensitivity,
            rng=random.Random(trial_seed),
        )

        assert omap.search(key=30) == 30
        inserted += int(ids["leaf_2330"] in omap.hot_nodes_client)

    expected_probability = _expected_insert_probability(
        candidate_access=candidate_pre_access + 1,
        right_access=right_internal_pre_access + 1,
        resident_access=resident_leaf_access,
        epsilon=epsilon,
        utility_sensitivity=utility_sensitivity,
    )
    observed_probability = inserted / float(trials)
    return observed_probability, expected_probability


def _assert_probability_matches(observed_probability, expected_probability, trials):
    sigma = math.sqrt(expected_probability * (1.0 - expected_probability) / float(trials))
    tolerance = max(0.03, 5.0 * sigma)
    assert abs(observed_probability - expected_probability) <= tolerance


class TestBPlusOmapHotNodesClient:
    def test_reject_all_layer_matches_base_bplus_omap_behavior(self):
        base_client = InteractLocalServer()
        hot_client = InteractLocalServer()
        base = _build_regular_bplus(client=base_client)
        hot = BPlusOmapHotNodesClient(
            order=4,
            num_data=64,
            key_size=10,
            data_size=10,
            client=hot_client,
            hot_nodes_client_size=2,
            hot_access_threshold=0,
            hot_admission_layer=RejectAllHotCacheAdmissionLayer(),
        )

        operations = [
            ("init_server_storage", {}),
            ("search", {"key": None}),
            ("search", {"key": 1}),
            ("insert", {"key": None, "value": None}),
            ("insert", {"key": 1, "value": 10}),
            ("insert", {"key": 4, "value": 40}),
            ("insert", {"key": 2, "value": 20}),
            ("insert", {"key": 3, "value": 30}),
            ("search", {"key": 2}),
            ("search", {"key": 9}),
            ("search", {"key": 3, "value": 300}),
            ("fast_search", {"key": 3}),
            ("fast_search", {"key": 7}),
            ("delete", {"key": 4}),
            ("delete", {"key": 99}),
            ("search", {"key": 4}),
            ("search", {"key": 3}),
            ("delete", {"key": 1}),
            ("delete", {"key": 2}),
            ("delete", {"key": 3}),
            ("search", {"key": 3}),
        ]

        for method_name, kwargs in operations:
            base_before = base_client.get_rounds()
            hot_before = hot_client.get_rounds()

            base_result = getattr(base, method_name)(**kwargs)
            hot_result = getattr(hot, method_name)(**kwargs)

            assert hot_result == base_result
            assert hot_client.get_rounds() - hot_before == base_client.get_rounds() - base_before
            assert hot.hot_nodes_client == []

        for key in (1, 2, 3, 4, 9, 99):
            base_before = base_client.get_rounds()
            hot_before = hot_client.get_rounds()

            assert hot.search(key=key) == base.search(key=key)
            assert hot_client.get_rounds() - hot_before == base_client.get_rounds() - base_before
            assert hot.hot_nodes_client == []

        assert (hot.root is None) == (base.root is None)

    def test_reject_all_admission_layer_disables_hot_cache_promotions(self, client):
        omap = _build_two_level_hot_bplus(
            client=client,
            cache_size=2,
            threshold=0,
            hot_admission_layer=RejectAllHotCacheAdmissionLayer(),
        )

        client.reset_rounds()
        assert omap.search(key=4) == 4
        first_rounds = client.get_rounds()

        client.reset_rounds()
        assert omap.search(key=4) == 4
        second_rounds = client.get_rounds()

        assert omap.hot_nodes_client == []
        assert omap.hot_cache_hits == 0
        assert omap.hot_cache_promotions == 0
        assert omap.hot_cache_evictions == 0
        assert first_rounds == second_rounds == 4

    def test_pictured_tree_exponential_mechanism_candidate_is_very_likely_inserted(self):
        trials = 800
        observed_probability, expected_probability = _simulate_pictured_exponential_insert_probability(
            trials=trials,
            epsilon=5.0,
            candidate_pre_access=29,
            right_internal_pre_access=0,
            resident_leaf_access=1,
        )

        assert expected_probability > 0.9
        _assert_probability_matches(observed_probability, expected_probability, trials)

    def test_pictured_tree_exponential_mechanism_candidate_is_very_unlikely_inserted(self):
        trials = 800
        observed_probability, expected_probability = _simulate_pictured_exponential_insert_probability(
            trials=trials,
            epsilon=6.0,
            candidate_pre_access=0,
            right_internal_pre_access=29,
            resident_leaf_access=30,
        )

        assert expected_probability < 0.15
        _assert_probability_matches(observed_probability, expected_probability, trials)

    def test_pictured_tree_exponential_mechanism_candidate_is_near_fifty_fifty(self):
        trials = 800
        observed_probability, expected_probability = _simulate_pictured_exponential_insert_probability(
            trials=trials,
            epsilon=3.0,
            candidate_pre_access=8,
            right_internal_pre_access=11,
            resident_leaf_access=40,
        )

        assert 0.45 < expected_probability < 0.55
        _assert_probability_matches(observed_probability, expected_probability, trials)

    def test_exponential_admission_layer_can_reject_arriving_candidate(self, client):
        layer = ExponentialMechanismHotCacheAdmissionLayer(
            epsilon=1.0,
            distance_fn=lambda candidate, resident: 0.0,
            rng=random.Random(1),
        )
        omap = _build_two_level_hot_bplus(
            client=client,
            cache_size=1,
            threshold=0,
            hot_admission_layer=layer,
        )

        assert omap.search(key=4) == 4
        hot_before = list(omap.hot_nodes_client)

        assert omap.search(key=1) == 1
        assert omap.hot_nodes_client == hot_before
        assert omap.hot_cache_evictions == 0

    def test_exponential_admission_layer_can_evict_cached_resident(self, client):
        layer = ExponentialMechanismHotCacheAdmissionLayer(
            epsilon=1.0,
            distance_fn=lambda candidate, resident: 0.0,
            rng=random.Random(0),
        )
        omap = _build_two_level_hot_bplus(
            client=client,
            cache_size=1,
            threshold=0,
            hot_admission_layer=layer,
        )

        assert omap.search(key=4) == 4
        hot_before = list(omap.hot_nodes_client)

        assert omap.search(key=1) == 1
        assert omap.hot_nodes_client != hot_before
        assert omap.hot_cache_evictions == 1

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
        assert omap.hot_cache_evictions == 0
        assert omap.hot_nodes_client == [right_leaf_key]

        assert omap.search(key=1) == 1
        assert omap.hot_cache_evictions > 0
        assert len(omap.hot_nodes_client) == 1
        assert right_leaf_key not in omap.hot_nodes_client

        root_node = _resident_node(omap, omap.root[0])
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

    # refer to bplus_cache_test.png
    def test_precise_rounds_on_pictured_tree_with_cache_size_two(self, client):
        omap, ids = _build_pictured_hot_bplus(client=client, cache_size=2, threshold=0)

        client.reset_rounds()
        assert omap.search(key=30) == 30
        assert client.get_rounds() == 6
        assert omap.hot_nodes_client == [ids["right_internal"], ids["leaf_2330"]]

        right_node = omap._hot_nodes_client[ids["right_internal"]]
        child_index = omap._find_child_index(right_node, 30)
        stale_pointer = dict(right_node.value.values[child_index])

        client.reset_rounds()
        assert omap.search(key=30) == 30
        assert client.get_rounds() == 2
        assert omap.hot_nodes_client == [ids["right_internal"], ids["leaf_2330"]]

        client.reset_rounds()
        assert omap.search(key=21) == 21
        assert client.get_rounds() == 4
        assert omap.hot_nodes_client == [ids["leaf_2330"], ids["right_internal"]]

        client.reset_rounds()
        assert omap.search(key=21) == 21
        assert client.get_rounds() == 4
        assert omap.hot_nodes_client == [ids["leaf_2330"], ids["right_internal"]]

        client.reset_rounds()
        assert omap.search(key=21) == 21
        assert client.get_rounds() == 4
        assert omap.hot_nodes_client == [ids["right_internal"], ids["leaf_21"]]

        client.reset_rounds()
        assert omap.search(key=21) == 21
        assert client.get_rounds() == 2
        assert omap.hot_nodes_client == [ids["right_internal"], ids["leaf_21"]]

        client.reset_rounds()
        assert omap.search(key=21) == 21
        assert client.get_rounds() == 2
        assert omap.hot_nodes_client == [ids["right_internal"], ids["leaf_21"]]

        client.reset_rounds()
        assert omap.search(key=21) == 21
        assert client.get_rounds() == 2
        assert omap.hot_nodes_client == [ids["right_internal"], ids["leaf_21"]]

        client.reset_rounds()
        assert omap.search(key=21) == 21
        assert client.get_rounds() == 2
        assert omap.hot_nodes_client == [ids["right_internal"], ids["leaf_21"]]
        

        right_node = omap._hot_nodes_client[ids["right_internal"]]
        child_index = omap._find_child_index(right_node, 30)
        pointer_after_eviction = right_node.value.values[child_index]
        assert pointer_after_eviction["node_id"] == ids["leaf_2330"]
        assert pointer_after_eviction["location"] == omap.HOT_CLI_CACHE
        assert pointer_after_eviction["leaf_p"] == stale_pointer["leaf_p"]

        client.reset_rounds()
        assert omap.search(key=30) == 30
        assert client.get_rounds() == 4
        assert omap.hot_nodes_client == [ids["leaf_21"], ids["right_internal"]]

        right_node = omap._hot_nodes_client[ids["right_internal"]]
        child_index = omap._find_child_index(right_node, 30)
        pointer_after_revisit = right_node.value.values[child_index]
        assert pointer_after_revisit["node_id"] == ids["leaf_2330"]
        assert pointer_after_revisit["location"] == omap.ORAM

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
            hot_admission_layer=ScoreBasedHotCacheAdmissionLayer(),
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

    def test_set_secret_user_id_updates_mid_subtree_and_current_path(self, client):
        omap = _build_three_level_hot_bplus(client=client, cache_size=0, threshold=100)

        assert omap.set_secret_user_id(key=12, subtree_height=2, secret_user_id=10) == 12

        root_node = _resident_node(omap, omap.root[0])
        middle_ptr = root_node.value.values[omap._find_child_index(root_node, 12)]
        middle_node = _resident_node(omap, middle_ptr["node_id"])
        leaf_ptr = middle_node.value.values[omap._find_child_index(middle_node, 12)]
        leaf_node = _resident_node(omap, leaf_ptr["node_id"])

        assert root_node.value.metadata["secret_user_id"] == 0
        assert root_node.value.metadata["lazy_secret_user_id"] is None

        assert middle_node.value.metadata["secret_user_id"] == 10
        assert middle_node.value.metadata["lazy_secret_user_id"] == 10
        assert middle_node.value.metadata["lazy_height"] == 1

        assert leaf_node.value.keys == [11, 12]
        assert leaf_node.value.metadata["secret_user_id"] == 10
        assert leaf_node.value.metadata["lazy_secret_user_id"] is None

    def test_later_access_lazily_propagates_secret_user_id_to_sibling_leaf(self, client):
        omap = _build_three_level_hot_bplus(client=client, cache_size=0, threshold=100)

        assert omap.set_secret_user_id(key=12, subtree_height=2, secret_user_id=10) == 12
        assert omap.search(key=9) == 9

        root_node = _resident_node(omap, omap.root[0])
        middle_ptr = root_node.value.values[omap._find_child_index(root_node, 12)]
        middle_node = _resident_node(omap, middle_ptr["node_id"])
        sibling_leaf = _resident_node(omap, middle_node.value.values[1]["node_id"])

        assert sibling_leaf.value.keys == [9, 10]
        assert sibling_leaf.value.metadata["secret_user_id"] == 10
        assert sibling_leaf.value.metadata["secret_user_id_version"] == 1
