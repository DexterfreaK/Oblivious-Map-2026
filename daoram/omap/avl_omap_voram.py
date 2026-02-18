"""AVL OMAP variant that uses TrueVoram as the underlying storage backend."""

from __future__ import annotations

import secrets
from typing import Any, Dict, List, Optional

from daoram.dependency import AVLData, AVLTree, Data, Encryptor, InteractServer, KVPair
from daoram.omap.avl_omap import AVLOmap
from daoram.omap.oblivious_search_tree import KV_LIST, ROOT
from daoram.oram import TrueVoram


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
            return

        stash_idx = self._find_in_stash(key)
        if stash_idx >= 0:
            self._local.add(node=self._stash.pop(stash_idx), parent_key=parent_key)
            return

        self._local.add(node=self._read_node_from_voram(key=key), parent_key=parent_key)

    def _perform_dummy_operation(self, num_round: int) -> None:
        """Persist pending nodes, then issue dummy vORAM accesses."""
        if num_round < 0:
            raise ValueError("The height is not enough, as the number of dummy operation required is negative.")

        self._flush_stash_to_voram()

        existing_keys = list(self._voram._pos_map.keys())
        if not existing_keys:
            # Bootstrap one inert key so we can still issue fixed-round dummy accesses.
            self._voram.operate_on_key(key=0, value=self._EMPTY_SLOT_VALUE)
            existing_keys = [0]

        for _ in range(num_round):
            self._voram.operate_on_key(key=secrets.choice(existing_keys))

    def fast_search(self, key: Any, value: Any = None) -> Any:
        """Functional-parity fast search implemented via the vORAM-safe search path."""
        return self.search(key=key, value=value)

    def delete(self, key: Any) -> Any:
        """Delete key using AVL logic, then clear the deleted key slot in vORAM."""
        if key is not None:
            self._validate_voram_key(key=key)

        deleted_value = super().delete(key=key)
        if deleted_value is not None:
            self._clear_node_from_voram(key=key)
        return deleted_value
