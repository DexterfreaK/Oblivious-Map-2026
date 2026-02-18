"""AVL OMAP variant that uses TrueVoram as the underlying storage backend."""

from __future__ import annotations

import logging
from pickle import TRUE
import secrets
from typing import Any, Dict, List, Optional

from daoram.dependency import AVLData, AVLTree, Data, Encryptor, InteractServer, KVPair, UNSET
from daoram.omap.avl_omap import AVLOmap
from daoram.omap.oblivious_search_tree import KV_LIST, ROOT
from daoram.oram import TrueVoram

logger = logging.getLogger(__name__)


class AVLOmapVoram(AVLOmap):
    """
    AVLOmap implementation backed by TrueVoram.

    Pointer model:
    - Node key is the vORAM key where that node is stored.
    - Parent stores children via l_key/r_key pointers (and l_leaf/r_leaf metadata).
    - Traversal fetches child directly by child key.
    """

    _EMPTY_SLOT_VALUE = ("__avl_omap_voram_empty__",)

    def __init__(
            self,
            num_data: int,
            key_size: int,
            data_size: int,
            client: InteractServer,
            name: str = "avl_voram",
            filename: str = None,
            bucket_size: int = 4,
            stash_scale: int = 7,
            encryptor: Encryptor = None,
            voram_name: str | None = None,
            voram_Z: int | None = None,
            voram_optimize: bool = True,
            voram_keylen: int = 32,
            voram_idlen: int | None = None,
            voram_compress: bool = True,
            voram_suggested_params: bool = False,
            voram_suggested_nblobs: int = 4,
            voram_debug: bool = False,
            trace: bool = True,
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

        backend_name = voram_name if voram_name else f"{name}_voram"
        self._voram = TrueVoram(
            num_data=self._num_data,
            data_size=self._max_block_size,
            client=self._client,
            name=backend_name,
            filename=filename,
            stash_scale=stash_scale,
            Z=voram_Z,
            optimize=voram_optimize,
            keylen=voram_keylen,
            idlen=voram_idlen,
            compress=voram_compress,
            encryptor=encryptor,
            suggested_params=voram_suggested_params,
            suggested_nblobs=voram_suggested_nblobs,
            allow_dynamic_keys=True,
            debug=voram_debug,
        )
        self._trace_enabled: bool = bool(trace or voram_debug)
        self._node_visit_count: int = 0

    def set_trace(self, enabled: bool = True) -> None:
        """Enable/disable verbose traversal logs at runtime."""
        self._trace_enabled = bool(enabled)

    def get_voram_round_counters(self) -> Dict[str, int]:
        """Expose vORAM path/client counters for diagnostics."""
        return self._voram.get_round_counters()

    def reset_voram_round_counters(self) -> None:
        """Reset vORAM path/client counters."""
        self._voram.reset_round_counters()

    def _trace(self, msg: str, *args: Any) -> None:
        if self._trace_enabled:
            logger.info("[AVLOmapVoram] " + msg, *args)

    @staticmethod
    def _summarize_value(value: Any, max_len: int = 80) -> str:
        """Compact value formatter for traversal logs."""
        if isinstance(value, (bytes, bytearray)):
            return f"<{type(value).__name__}:{len(value)}B>"
        text = repr(value)
        return text if len(text) <= max_len else text[: max_len - 3] + "..."

    def _log_current_node(self, node: Data, source: str) -> None:
        """Print current AVL node and child pointer metadata."""
        self._node_visit_count += 1
        self._trace(
            (
                "CURRENTLY AT AVL NODE #%d "
                "(key=%s, value=%s) "
                "(left_child: key=%s path=%s h=%s) "
                "(right_child: key=%s path=%s h=%s) "
                "[source=%s]"
            ),
            self._node_visit_count,
            node.key,
            self._summarize_value(node.value.value),
            node.value.l_key,
            node.value.l_leaf,
            node.value.l_height,
            node.value.r_key,
            node.value.r_leaf,
            node.value.r_height,
            source,
        )

    def _log_fetch_request(self, key: Any, leaf: int, parent_key: Any) -> None:
        """Print parent->child fetch direction and path hint."""
        if parent_key is None:
            self._trace("FETCHING ROOT NODE (key=%s) using path hint=%s", key, leaf)
            return

        parent = self._local.get(parent_key)
        if parent is None:
            self._trace("FETCHING CHILD (key=%s) from parent=%s using path hint=%s", key, parent_key, leaf)
            return

        if parent.value.l_key == key:
            side = "LEFT"
            edge_leaf = parent.value.l_leaf
        elif parent.value.r_key == key:
            side = "RIGHT"
            edge_leaf = parent.value.r_leaf
        else:
            side = "UNKNOWN"
            edge_leaf = leaf

        self._trace(
            "FETCHING %s CHILD (key=%s) from parent=%s (parent path pointer=%s)",
            side,
            key,
            parent_key,
            edge_leaf,
        )

    def _log_fetch_result(self, key: Any) -> None:
        """Print vORAM leaf/path movement and round counters for last fetch."""
        access = self._voram.get_last_access()
        counters = self._voram.get_round_counters()
        if access is None:
            self._trace("vORAM FETCH for key=%s completed (no access metadata).", key)
            return

        self._trace(
            (
                "vORAM FETCHED PATH %s for key=%s -> remapped to path %s "
                "[logical_ops=%d path_reads=%d path_writes=%d client_rounds=%d]"
            ),
            access.get("old_leaf"),
            access.get("key", key),
            access.get("new_leaf"),
            counters["logical_accesses"],
            counters["path_reads"],
            counters["path_writes"],
            counters["client_rounds"],
        )

    def _log_operation_start(self, op: str, key: Any, value: Any = None) -> Dict[str, int]:
        """Log operation start and snapshot round counters."""
        if value is UNSET:
            self._trace("%s START key=%s", op, key)
        else:
            self._trace("%s START key=%s value=%s", op, key, self._summarize_value(value))
        self._node_visit_count = 0
        return self._voram.get_round_counters()

    def _log_operation_end(
            self,
            op: str,
            key: Any,
            before: Dict[str, int],
            result: Any = UNSET,
            error: Exception | None = None,
    ) -> None:
        """Log per-operation round deltas and final status."""
        after = self._voram.get_round_counters()
        delta = {name: after[name] - before[name] for name in before.keys()}

        if error is not None:
            self._trace(
                (
                    "%s FAILED key=%s error=%s "
                    "[delta logical_ops=%d path_reads=%d path_writes=%d client_rounds=%d visits=%d]"
                ),
                op,
                key,
                error,
                delta["logical_accesses"],
                delta["path_reads"],
                delta["path_writes"],
                delta["client_rounds"],
                self._node_visit_count,
            )
            return

        if result is UNSET:
            self._trace(
                (
                    "%s DONE key=%s "
                    "[delta logical_ops=%d path_reads=%d path_writes=%d client_rounds=%d visits=%d]"
                ),
                op,
                key,
                delta["logical_accesses"],
                delta["path_reads"],
                delta["path_writes"],
                delta["client_rounds"],
                self._node_visit_count,
            )
        else:
            self._trace(
                (
                    "%s DONE key=%s result=%s "
                    "[delta logical_ops=%d path_reads=%d path_writes=%d client_rounds=%d visits=%d]"
                ),
                op,
                key,
                self._summarize_value(result),
                delta["logical_accesses"],
                delta["path_reads"],
                delta["path_writes"],
                delta["client_rounds"],
                self._node_visit_count,
            )

    def _validate_voram_key(self, key: Any) -> int:
        """Validate that key is a legal vORAM index for direct pointer access."""
        if not isinstance(key, int):
            raise TypeError(
                "AVLOmapVoram pointer mode requires integer keys so nodes can be fetched directly from vORAM."
            )
        if key < 0:
            raise ValueError("AVLOmapVoram pointer mode requires non-negative integer keys.")
        return key

    def _write_node_to_voram(self, node: Data) -> None:
        """Persist one AVL node to vORAM at its pointer key."""
        if node.key is None:
            raise ValueError("Cannot persist a node with an empty key.")
        key = self._validate_voram_key(key=node.key)
        self._voram.operate_on_key(key=key, value=node)

    def _read_node_from_voram(self, key: Any) -> Data:
        """Load one AVL node from vORAM by pointer key."""
        key = self._validate_voram_key(key=key)
        node = self._voram.operate_on_key(key=key)
        if node == self._EMPTY_SLOT_VALUE:
            raise KeyError(f"The search key {key} is not found.")
        if not isinstance(node, Data):
            raise ValueError(f"Malformed vORAM payload for key {key}.")
        if node.key != key:
            raise ValueError(f"vORAM key mismatch: expected key {key}, got {node.key}.")
        return node

    def _clear_node_from_voram(self, key: Any) -> None:
        """Overwrite one node slot with empty sentinel."""
        key = self._validate_voram_key(key=key)
        self._voram.operate_on_key(key=key, value=self._EMPTY_SLOT_VALUE)

    def _flush_stash_to_voram(self) -> None:
        """Persist all stashed nodes to vORAM and clear stash."""
        if not self._stash:
            return

        # Keep only the latest version per key in this flush.
        latest: Dict[Any, Data] = {}
        for node in self._stash:
            if node.key is not None:
                latest[node.key] = node

        for node in latest.values():
            self._write_node_to_voram(node=node)

        self._stash = []

    def _build_avl_data_nodes(self, data: KV_LIST) -> tuple[List[Data], Optional[ROOT]]:
        """Build AVL nodes from KV pairs using AVLOmap's initialization path."""
        if not data:
            return [], None

        root = None
        avl_tree = AVLTree(leaf_range=self._leaf_range)

        for kv_pair in data:
            if isinstance(kv_pair, tuple):
                kv_pair = KVPair(key=kv_pair[0], value=kv_pair[1])
            root = avl_tree.recursive_insert(root=root, kv_pair=kv_pair)

        data_nodes = avl_tree.get_data_list(root=root, encryption=False)
        for node in data_nodes:
            if isinstance(node.value, bytes):
                node.value = AVLData.from_pickle(data=node.value)
            self._validate_voram_key(node.key)

        return data_nodes, (root.key, root.leaf)

    def init_server_storage(self, data: KV_LIST = None) -> None:
        """Initialize storage by creating an empty vORAM and optionally preloading one AVL tree."""
        self.root = None
        self._stash = []
        self._local.clear()

        self._voram.init_server_storage(data_map={})

        if not data:
            return

        data_nodes, root = self._build_avl_data_nodes(data=data)
        for node in data_nodes:
            self._write_node_to_voram(node=node)
        self.root = root

    def init_mul_tree_server_storage(self, data_list: List[KV_LIST] = None) -> List[ROOT]:
        """Initialize storage with multiple AVL trees and return each tree root."""
        if data_list is None:
            data_list = []

        self.root = None
        self._stash = []
        self._local.clear()

        self._voram.init_server_storage(data_map={})

        root_list = []
        for data in data_list:
            if data:
                data_nodes, root = self._build_avl_data_nodes(data=data)
                for node in data_nodes:
                    self._write_node_to_voram(node=node)
                root_list.append(root)
            else:
                root_list.append(None)

        return root_list

    def _move_node_to_local(self, key: Any, leaf: int, parent_key: Any = None) -> None:
        """Move node from stash/vORAM to local; no PathORAM path writeback needed."""
        self._move_node_to_local_without_eviction(key=key, leaf=leaf, parent_key=parent_key)

    def _move_node_to_local_without_eviction(self, key: Any, leaf: int, parent_key: Any = None) -> None:
        """Move one node to local, preferring stash and falling back to direct vORAM fetch."""
        if self._local.get(key) is not None:
            self._log_current_node(node=self._local.get(key), source="local-cache")
            return

        stash_idx = self._find_in_stash(key)
        if stash_idx >= 0:
            node = self._stash.pop(stash_idx)
            self._local.add(node=node, parent_key=parent_key)
            self._trace("FETCH HIT in AVL stash for key=%s (no vORAM round).", key)
            self._log_current_node(node=node, source="avl-stash")
            return

        self._log_fetch_request(key=key, leaf=leaf, parent_key=parent_key)
        node = self._read_node_from_voram(key=key)
        self._log_fetch_result(key=key)
        self._local.add(node=node, parent_key=parent_key)
        self._log_current_node(node=node, source="voram")

    def _perform_dummy_operation(self, num_round: int) -> None:
        """Persist pending nodes, then issue dummy vORAM accesses."""
        if num_round < 0:
            raise ValueError("The height is not enough, as the number of dummy operation required is negative.")

        before = self._voram.get_round_counters()
        self._flush_stash_to_voram()

        existing_keys = list(self._voram._pos_map.keys())
        if not existing_keys:
            # Bootstrap one inert key so we can still issue fixed-round dummy accesses.
            self._voram.operate_on_key(key=0, value=self._EMPTY_SLOT_VALUE)
            existing_keys = [0]

        self._trace("DUMMY ROUND START rounds=%d candidate_keys=%d", num_round, len(existing_keys))
        for i in range(num_round):
            dummy_key = secrets.choice(existing_keys)
            self._voram.operate_on_key(key=dummy_key)
            access = self._voram.get_last_access()
            if access is not None and self._trace_enabled:
                if num_round <= 6 or i < 3 or i == num_round - 1:
                    self._trace(
                        "DUMMY ACCESS[%d/%d] key=%s fetched_path=%s new_path=%s",
                        i + 1,
                        num_round,
                        dummy_key,
                        access.get("old_leaf"),
                        access.get("new_leaf"),
                    )
                elif i == 3:
                    self._trace("... skipping %d intermediate dummy accesses ...", num_round - 4)

        after = self._voram.get_round_counters()
        self._trace(
            (
                "DUMMY ROUND DONE "
                "[delta logical_ops=%d path_reads=%d path_writes=%d client_rounds=%d]"
            ),
            after["logical_accesses"] - before["logical_accesses"],
            after["path_reads"] - before["path_reads"],
            after["path_writes"] - before["path_writes"],
            after["client_rounds"] - before["client_rounds"],
        )

    def insert(self, key: Any, value: Any = None) -> None:
        before = self._log_operation_start(op="INSERT", key=key, value=value)
        try:
            super().insert(key=key, value=value)
        except Exception as exc:
            self._log_operation_end(op="INSERT", key=key, before=before, error=exc)
            raise
        self._log_operation_end(op="INSERT", key=key, before=before)

    def search(self, key: Any, value: Any = None) -> Any:
        before = self._log_operation_start(op="SEARCH", key=key, value=value if value is not None else UNSET)
        try:
            result = super().search(key=key, value=value)
        except Exception as exc:
            self._log_operation_end(op="SEARCH", key=key, before=before, error=exc)
            raise
        self._log_operation_end(op="SEARCH", key=key, before=before, result=result)
        return result

    def fast_search(self, key: Any, value: Any = None) -> Any:
        """Functional-parity fast search implemented via the vORAM-safe search path."""
        before = self._log_operation_start(op="FAST_SEARCH", key=key, value=value if value is not None else UNSET)
        try:
            result = super().search(key=key, value=value)
        except Exception as exc:
            self._log_operation_end(op="FAST_SEARCH", key=key, before=before, error=exc)
            raise
        self._log_operation_end(op="FAST_SEARCH", key=key, before=before, result=result)
        return result

    def delete(self, key: Any) -> Any:
        """Delete key using AVL logic, then clear the deleted key slot in vORAM."""
        before = self._log_operation_start(op="DELETE", key=key)
        if key is not None:
            self._validate_voram_key(key=key)

        try:
            deleted_value = super().delete(key=key)
        except Exception as exc:
            self._log_operation_end(op="DELETE", key=key, before=before, error=exc)
            raise
        if deleted_value is not None:
            self._trace("DELETE clearing vORAM slot for deleted key=%s", key)
            self._clear_node_from_voram(key=key)
        self._log_operation_end(op="DELETE", key=key, before=before, result=deleted_value)
        return deleted_value
