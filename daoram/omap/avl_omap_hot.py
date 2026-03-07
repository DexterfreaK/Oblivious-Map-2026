"""AVL OMAP wrapper with a client-only hot-node cache."""

import copy
from collections import OrderedDict
from typing import Any, Dict, List, Optional, Set

from daoram.dependency import Data, Encryptor, InteractServer
from daoram.omap.avl_omap import AVLOmap


class AVLOmapHotNodesClient(AVLOmap):
    """
    AVL OMAP with a strict client-side hot cache for frequently accessed nodes.

    The cache lives entirely on the client and is never stored on the server.
    During search traversal, cached nodes are loaded locally without ORAM path
    access. Hot-node promotions are staged in a temporary cache and committed
    after traversal completes.
    """

    def __init__(
        self,
        num_data: int,
        key_size: int,
        data_size: int,
        client: InteractServer,
        name: str = "avl_hot_nodes_client",
        filename: str = None,
        bucket_size: int = 4,
        stash_scale: int = 7,
        encryptor: Encryptor = None,
        hot_nodes_client_size: int = 2,
        hot_access_threshold: int = 2,
        search_padding: bool = False,
        always_dummy_after_search: bool = False,
    ):
        super().__init__(
            num_data=num_data,
            key_size=key_size,
            data_size=data_size,
            client=client,
            name=name,
            filename=filename,
            bucket_size=bucket_size,
            stash_scale=stash_scale,
            encryptor=encryptor,
        )

        self._hot_nodes_client_size = max(0, hot_nodes_client_size)
        self._hot_access_threshold = max(1, hot_access_threshold)
        self._search_padding = search_padding
        self._always_dummy_after_search = always_dummy_after_search

        # Permanent client cache and per-search temporary staging cache.
        self._hot_nodes_client: "OrderedDict[Any, Data]" = OrderedDict()
        self._temp_hot_nodes: "OrderedDict[Any, Data]" = OrderedDict()

        # Client-side metadata.
        self._access_counts: Dict[Any, int] = {}
        self._hot_parent_links: Dict[Any, Set[Any]] = {}

        # Runtime flags and observability counters.
        self._hot_cache_active = False
        self.hot_cache_hits = 0
        self.hot_cache_misses = 0
        self.hot_cache_promotions = 0
        self.hot_cache_evictions = 0
        self._pending_reinsert_nodes: Dict[Any, Data] = {}

        # Per-operation observability.
        self._operation_totals: Dict[str, Dict[str, int]] = {}
        self._operation_last: Optional[Dict[str, Any]] = None

    @property
    def hot_nodes_client(self) -> List[Any]:
        """Return hot cache keys in current eviction-order."""
        return list(self._hot_nodes_client.keys())

    def get_access_count(self, key: Any) -> int:
        """Return the client-side access count for a key."""
        return self._access_counts.get(key, 0)

    def get_hot_cache_stats(self) -> Dict[str, int]:
        """Return basic counters for cache behavior."""
        return {
            "hits": self.hot_cache_hits,
            "misses": self.hot_cache_misses,
            "promotions": self.hot_cache_promotions,
            "evictions": self.hot_cache_evictions,
        }

    def _snapshot_observability(self) -> Dict[str, int]:
        """Capture current client counters and cache counters."""
        bytes_read, bytes_written = self._client.get_bandwidth()
        rounds = self._client.get_rounds() if hasattr(self._client, "get_rounds") else 0
        return {
            "bytes_read": bytes_read,
            "bytes_written": bytes_written,
            "rounds": rounds,
            "hot_hits": self.hot_cache_hits,
            "hot_misses": self.hot_cache_misses,
            "hot_promotions": self.hot_cache_promotions,
            "hot_evictions": self.hot_cache_evictions,
        }

    def _record_operation_observability(
        self,
        operation: str,
        key: Any,
        before: Dict[str, int],
        success: bool,
    ) -> None:
        """Record per-operation deltas for bandwidth, rounds, and hot-cache effects."""
        after = self._snapshot_observability()
        delta = {
            "bytes_read": after["bytes_read"] - before["bytes_read"],
            "bytes_written": after["bytes_written"] - before["bytes_written"],
            "rounds": after["rounds"] - before["rounds"],
            "hot_hits": after["hot_hits"] - before["hot_hits"],
            "hot_misses": after["hot_misses"] - before["hot_misses"],
            "hot_promotions": after["hot_promotions"] - before["hot_promotions"],
            "hot_evictions": after["hot_evictions"] - before["hot_evictions"],
        }

        total = self._operation_totals.setdefault(
            operation,
            {
                "count": 0,
                "success": 0,
                "failure": 0,
                "bytes_read": 0,
                "bytes_written": 0,
                "rounds": 0,
                "hot_hits": 0,
                "hot_misses": 0,
                "hot_promotions": 0,
                "hot_evictions": 0,
            },
        )
        total["count"] += 1
        if success:
            total["success"] += 1
        else:
            total["failure"] += 1
        total["bytes_read"] += delta["bytes_read"]
        total["bytes_written"] += delta["bytes_written"]
        total["rounds"] += delta["rounds"]
        total["hot_hits"] += delta["hot_hits"]
        total["hot_misses"] += delta["hot_misses"]
        total["hot_promotions"] += delta["hot_promotions"]
        total["hot_evictions"] += delta["hot_evictions"]

        self._operation_last = {
            "operation": operation,
            "key": key,
            "success": success,
            **delta,
        }

    def get_last_operation_observability(self) -> Optional[Dict[str, Any]]:
        """Return metrics for the most recent operation."""
        return copy.deepcopy(self._operation_last)

    def get_operation_observability(self) -> Dict[str, Dict[str, int]]:
        """Return cumulative observability counters grouped by operation."""
        return copy.deepcopy(self._operation_totals)

    def get_client_observability(self) -> Dict[str, int]:
        """Return current client counters (not delta)."""
        bytes_read, bytes_written = self._client.get_bandwidth()
        rounds = self._client.get_rounds() if hasattr(self._client, "get_rounds") else 0
        return {
            "bytes_read": bytes_read,
            "bytes_written": bytes_written,
            "rounds": rounds,
        }

    def reset_operation_observability(self, reset_client_counters: bool = False) -> None:
        """Reset operation observability counters, optionally client counters too."""
        self._operation_totals.clear()
        self._operation_last = None
        if reset_client_counters:
            self._client.reset_bandwidth()
            if hasattr(self._client, "reset_rounds"):
                self._client.reset_rounds()

    def _record_access(self, key: Any, parent_key: Any) -> None:
        """Update access metadata and stage candidates for deferred promotion."""
        self._access_counts[key] = self._access_counts.get(key, 0) + 1

        if key in self._hot_nodes_client:
            self._hot_nodes_client.move_to_end(key)
            if parent_key is not None:
                self._hot_parent_links.setdefault(parent_key, set()).add(key)

        if self._should_stage_hot_node(key):
            node = self._local.get(key)
            if node is not None:
                self._temp_hot_nodes[key] = copy.deepcopy(node)

    def _should_stage_hot_node(self, key: Any) -> bool:
        """Decide whether a traversed key should be staged for hot promotion."""
        if self._hot_nodes_client_size == 0:
            return False
        if key in self._hot_nodes_client:
            return False
        if self._access_counts.get(key, 0) < self._hot_access_threshold:
            return False
        if len(self._hot_nodes_client) < self._hot_nodes_client_size:
            return True

        min_hot_access = min(self._access_counts.get(hot_key, 0) for hot_key in self._hot_nodes_client)
        # Use >= so ties can still rotate through the bounded cache.
        return self._access_counts.get(key, 0) >= min_hot_access

    def _select_eviction_candidate(self) -> Any:
        """Pick a victim key from hot cache by least access count, then LRU order."""
        victim_key = None
        victim_score = None
        for lru_rank, key in enumerate(self._hot_nodes_client.keys()):
            score = (self._access_counts.get(key, 0), lru_rank)
            if victim_score is None or score < victim_score:
                victim_key = key
                victim_score = score
        return victim_key

    def _drop_hot_parent_links(self, child_key: Any) -> None:
        """Remove stale 'child in hot cache' hints for a given child key."""
        for parent_key in list(self._hot_parent_links.keys()):
            linked_children = self._hot_parent_links[parent_key]
            linked_children.discard(child_key)
            if not linked_children:
                del self._hot_parent_links[parent_key]

    def _remove_stash_key(self, key: Any) -> None:
        """Remove any stale copies of a key from stash before reinsertion."""
        self._stash = [node for node in self._stash if node.key != key]

    def _fetch_node_for_maintenance(self, key: Any, leaf: int, parent_key: Any = None) -> None:
        """
        Fetch a node for maintenance operations.

        This can read from client hot cache or ORAM without touching hot-access
        counters/promotions, and preserves LocalNodes parent tracking.
        """
        if self._local.get(key) is not None:
            return

        cached = self._hot_nodes_client.get(key)
        if cached is not None:
            self._local.add(node=copy.deepcopy(cached), parent_key=parent_key)
            return

        pending = self._pending_reinsert_nodes.get(key)
        if pending is not None:
            self._local.add(node=copy.deepcopy(pending), parent_key=parent_key)
            return

        super()._move_node_to_local(key=key, leaf=leaf, parent_key=parent_key)

    def _update_parent_leaf_for_key(self, child_key: Any, child_new_leaf: int) -> None:
        """
        Traverse from root to locate child's parent and update the child's leaf pointer.

        The parent update is done in whichever tier currently holds it:
        - hot cache (if parent is hot), or
        - ORAM/stash write-back (if parent was fetched from ORAM).
        """
        if self.root is None:
            return

        # Root has no parent pointer to update.
        if self.root[0] == child_key:
            self.root = (child_key, child_new_leaf)
            return

        self._fetch_node_for_maintenance(key=self.root[0], leaf=self.root[1], parent_key=None)
        current_key = self.root[0]

        while True:
            current = self._local.get(current_key)
            go_right = current.key < child_key
            next_key = current.value.r_key if go_right else current.value.l_key
            next_leaf = current.value.r_leaf if go_right else current.value.l_leaf

            if next_key is None:
                raise KeyError(f"Failed to locate parent for key {child_key} during hot-node reinsertion.")

            if next_key == child_key:
                if go_right:
                    current.value.r_leaf = child_new_leaf
                else:
                    current.value.l_leaf = child_new_leaf
                break

            self._fetch_node_for_maintenance(key=next_key, leaf=next_leaf, parent_key=current_key)
            current_key = next_key

    def _flush_maintenance_local(self) -> None:
        """Flush local nodes touched by maintenance back to hot cache or stash."""
        if not self._local:
            return

        local_by_key = {node.key: node for node in self._local.to_list()}
        for node_key, local_node in local_by_key.items():
            if node_key in self._hot_nodes_client:
                self._hot_nodes_client[node_key] = copy.deepcopy(local_node)
                self._hot_nodes_client.move_to_end(node_key)
            elif node_key in self._pending_reinsert_nodes:
                self._pending_reinsert_nodes[node_key] = copy.deepcopy(local_node)
            else:
                self._stash.append(local_node)

        self._local.clear()

    def _reinsert_evicted_hot_node(self, node: Data) -> None:
        """
        Reinsert an evicted hot node into ORAM.

        This performs a concrete path read/write so the evicted node can move
        from client-only state back toward server storage.
        """
        # Re-map to a fresh leaf and update parent pointer root->...->parent(child).
        pending = self._pending_reinsert_nodes.get(node.key)
        remapped_node = copy.deepcopy(pending if pending is not None else node)
        remapped_node.leaf = self._get_new_leaf()
        self._update_parent_leaf_for_key(child_key=remapped_node.key, child_new_leaf=remapped_node.leaf)
        self._flush_maintenance_local()

        self._remove_stash_key(remapped_node.key)
        self._stash.append(remapped_node)

        # Reinsert on the remapped leaf path.
        self._client.add_read_path(label=self._name, leaves=[remapped_node.leaf])
        result = self._client.execute()
        path_data = result.results[self._name]
        path = self._decrypt_path_data(path=path_data)

        for bucket in path.values():
            for data in bucket:
                if data.key != remapped_node.key:
                    self._stash.append(data)

        if len(self._stash) > self._stash_size:
            raise MemoryError("Stash overflow while reinserting evicted hot node.")

        self._client.add_write_path(label=self._name, data=self._evict_stash(leaves=[remapped_node.leaf]))
        self._client.execute()

        # Add a random tail traversal after reinsertion to reduce distinguishability.
        self._perform_dummy_operation(num_round=self._max_height)
        self._pending_reinsert_nodes.pop(remapped_node.key, None)

    def _flush_hot_nodes_to_oram(self) -> None:
        """Persist all currently hot client-side nodes back into ORAM."""
        if not self._hot_nodes_client:
            return

        hot_nodes = list(self._hot_nodes_client.values())
        self._hot_nodes_client.clear()
        self._temp_hot_nodes.clear()
        self._hot_parent_links.clear()
        self._pending_reinsert_nodes.clear()

        for node in hot_nodes:
            self._reinsert_evicted_hot_node(node=copy.deepcopy(node))

    def flush_hot_nodes_client_to_oram(self) -> None:
        """Public helper for benchmarks/tests to flush client hot cache to ORAM."""
        self._flush_hot_nodes_to_oram()

    def _commit_hot_nodes(self, local_by_key: Dict[Any, Data]) -> List[Data]:
        """Commit staged candidates to hot cache and return evicted nodes."""
        evicted_nodes: List[Data] = []

        # Refresh cache copies for touched hot nodes.
        for key, node in local_by_key.items():
            if key in self._hot_nodes_client:
                self._hot_nodes_client[key] = copy.deepcopy(node)
                self._hot_nodes_client.move_to_end(key)

        # Commit staged candidates after traversal completes.
        for key in list(self._temp_hot_nodes.keys()):
            node = local_by_key.get(key)
            if node is None:
                continue

            if key in self._hot_nodes_client:
                self._hot_nodes_client[key] = copy.deepcopy(node)
                self._hot_nodes_client.move_to_end(key)
                continue

            if self._hot_nodes_client_size == 0:
                break

            if len(self._hot_nodes_client) >= self._hot_nodes_client_size:
                victim_key = self._select_eviction_candidate()
                victim_node = self._hot_nodes_client.pop(victim_key)
                self.hot_cache_evictions += 1
                self._drop_hot_parent_links(victim_key)
                if victim_key not in local_by_key:
                    evicted_nodes.append(victim_node)

            self._hot_nodes_client[key] = copy.deepcopy(node)
            self.hot_cache_promotions += 1

        self._temp_hot_nodes.clear()
        return evicted_nodes

    def _move_node_to_local(self, key: Any, leaf: int, parent_key: Any = None) -> None:
        """Hot-aware node fetch: use client cache first, otherwise fallback to ORAM."""
        if not self._hot_cache_active:
            super()._move_node_to_local(key=key, leaf=leaf, parent_key=parent_key)
            return

        existing = self._local.get(key)
        if existing is not None:
            self._record_access(key=key, parent_key=parent_key)
            return

        cached = self._hot_nodes_client.get(key)
        if cached is not None:
            self._local.add(node=copy.deepcopy(cached), parent_key=parent_key)
            self.hot_cache_hits += 1
            self._record_access(key=key, parent_key=parent_key)
            return

        self.hot_cache_misses += 1
        super()._move_node_to_local(key=key, leaf=leaf, parent_key=parent_key)
        self._record_access(key=key, parent_key=parent_key)

    def _move_node_to_local_without_eviction(self, key: Any, leaf: int, parent_key: Any = None) -> None:
        """Hot-aware fast-path variant without immediate eviction."""
        if not self._hot_cache_active:
            super()._move_node_to_local_without_eviction(key=key, leaf=leaf, parent_key=parent_key)
            return

        existing = self._local.get(key)
        if existing is not None:
            self._record_access(key=key, parent_key=parent_key)
            return

        cached = self._hot_nodes_client.get(key)
        if cached is not None:
            self._local.add(node=copy.deepcopy(cached), parent_key=parent_key)
            self.hot_cache_hits += 1
            self._record_access(key=key, parent_key=parent_key)
            return

        self.hot_cache_misses += 1
        super()._move_node_to_local_without_eviction(key=key, leaf=leaf, parent_key=parent_key)
        self._record_access(key=key, parent_key=parent_key)

    def insert(self, key: Any, value: Any = None) -> None:
        """Insert while recording per-operation observability counters."""
        before = self._snapshot_observability()
        success = False
        try:
            self._flush_hot_nodes_to_oram()
            super().insert(key=key, value=value)
            success = True
        finally:
            self._record_operation_observability(
                operation="insert",
                key=key,
                before=before,
                success=success,
            )

    def fast_search(self, key: Any, value: Any = None) -> Any:
        """Run fast_search while recording per-operation observability counters."""
        operation = "fast_update" if value is not None else "fast_search"
        before = self._snapshot_observability()
        success = False
        try:
            self._flush_hot_nodes_to_oram()
            result = super().fast_search(key=key, value=value)
            success = True
            return result
        finally:
            self._record_operation_observability(
                operation=operation,
                key=key,
                before=before,
                success=success,
            )

    def delete(self, key: Any) -> Any:
        """Delete while recording per-operation observability counters."""
        before = self._snapshot_observability()
        success = False
        try:
            self._flush_hot_nodes_to_oram()
            result = super().delete(key=key)
            success = True
            return result
        finally:
            self._record_operation_observability(
                operation="delete",
                key=key,
                before=before,
                success=success,
            )

    def search(self, key: Any, value: Any = None) -> Any:
        """
        Given a search key, return its corresponding value with client hot caching.

        If ``value`` is provided, update the value in-place and return the old one.
        """
        if self.root is None:
            raise ValueError("It seems the tree is empty and can't perform search.")

        operation = "update" if value is not None else "search"
        before = self._snapshot_observability()
        success = False
        self._hot_cache_active = True
        self._temp_hot_nodes.clear()
        self._pending_reinsert_nodes.clear()

        try:
            # Traverse from root to desired key (or terminal miss node).
            self._move_node_to_local(key=self._root[0], leaf=self._root[1], parent_key=None)
            current_key = self._root[0]

            node = self._local.get(current_key)
            while node.key != key:
                if node.key < key:
                    if node.value.r_key is not None:
                        self._move_node_to_local(
                            key=node.value.r_key, leaf=node.value.r_leaf, parent_key=current_key
                        )
                        current_key = node.value.r_key
                    else:
                        break
                else:
                    if node.value.l_key is not None:
                        self._move_node_to_local(
                            key=node.value.l_key, leaf=node.value.l_leaf, parent_key=current_key
                        )
                        current_key = node.value.l_key
                    else:
                        break
                node = self._local.get(current_key)

            search_value = node.value.value if node.key == key else None
            if value is not None:
                node.value.value = value

            # Keep the leaf-update behavior compatible with AVLOmap.
            self._local.update_all_leaves(self._get_new_leaf)

            root_node = self._local.get_root()
            self.root = (root_node.key, root_node.leaf)

            num_retrieved_nodes = len(self._local)
            local_by_key = {node.key: node for node in self._local.to_list()}

            # Deferred promotion + bounded eviction.
            evicted_nodes = self._commit_hot_nodes(local_by_key=local_by_key)

            # Keep hot nodes client-side; only non-hot locals return to stash/ORAM.
            for node_key, local_node in local_by_key.items():
                if node_key in self._hot_nodes_client:
                    self._hot_nodes_client[node_key] = copy.deepcopy(local_node)
                    self._hot_nodes_client.move_to_end(node_key)
                else:
                    self._stash.append(local_node)
            self._local.clear()

            # Reinsert evicted hot nodes back to ORAM.
            self._pending_reinsert_nodes = {node.key: copy.deepcopy(node) for node in evicted_nodes}
            for evicted in evicted_nodes:
                self._reinsert_evicted_hot_node(node=evicted)

            if self._search_padding:
                dummy_rounds = 3 * self._max_height + 1 - num_retrieved_nodes
                if self._always_dummy_after_search:
                    dummy_rounds += 1
                self._perform_dummy_operation(num_round=dummy_rounds)

            success = True
            return search_value

        finally:
            self._temp_hot_nodes.clear()
            self._pending_reinsert_nodes.clear()
            self._hot_cache_active = False
            self._record_operation_observability(
                operation=operation,
                key=key,
                before=before,
                success=success,
            )
