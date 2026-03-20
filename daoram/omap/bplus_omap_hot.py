"""B+ tree OMAP with a client-side hot-node cache."""

import copy
import os
import pickle
from collections import OrderedDict
from dataclasses import astuple, dataclass
from functools import cached_property
from typing import Any, Dict, List, Optional, Set, Tuple

from daoram.dependency import (
    BinaryTree,
    BPlusTree,
    BPlusTreeNode,
    Data,
    Encryptor,
    Helper,
    InteractServer,
    KVPair,
    PathData,
)
from daoram.omap.bplus_omap import BPlusOmap


@dataclass
class BPlusHotData:
    """Serialized payload for a B+ node with hot-cache metadata."""

    keys: Optional[List[Any]] = None
    values: Optional[List[Any]] = None
    metadata: Optional[Dict[str, Any]] = None

    @classmethod
    def from_pickle(cls, data: bytes) -> "BPlusHotData":
        return cls(*pickle.loads(data))

    def dump(self) -> bytes:
        return pickle.dumps(astuple(self))  # type: ignore[arg-type]


class BPlusOmapHotNodesClient(BPlusOmap):
    """B+ OMAP variant with a bounded client-only hot-node cache."""

    HOT_CLI_CACHE = "HOT_CLI_CACHE"
    ORAM = "ORAM"
    _ACCESS_COUNT_CAP = (1 << 63) - 1

    def __init__(
        self,
        order: int,
        num_data: int,
        key_size: int,
        data_size: int,
        client: InteractServer,
        name: str = "bplus_hot_nodes_client",
        filename: str = None,
        bucket_size: int = 4,
        stash_scale: int = 7,
        encryptor: Encryptor = None,
        hot_nodes_client_size: int = 64,
        hot_access_threshold: int = 2,
        default_sensitivity: Any = 0,
        max_sensitivity_value: Any = None,
    ):
        self._default_sensitivity = copy.deepcopy(default_sensitivity)
        self._max_sensitivity_value = (
            copy.deepcopy(default_sensitivity)
            if max_sensitivity_value is None
            else copy.deepcopy(max_sensitivity_value)
        )
        self._max_sensitivity_pickle_size = len(pickle.dumps(self._max_sensitivity_value))
        self._validate_sensitivity_size(self._default_sensitivity)

        super().__init__(
            order=order,
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
        self._hot_access_threshold = max(0, hot_access_threshold)
        self._hot_nodes_client: "OrderedDict[Any, Data]" = OrderedDict()
        self._stash_size += self._hot_nodes_client_size
        self._sensitivity_version = 0

        self.hot_cache_hits = 0
        self.hot_cache_misses = 0
        self.hot_cache_promotions = 0
        self.hot_cache_evictions = 0

    @property
    def hot_nodes_client(self) -> List[Any]:
        """Return cached node ids in current eviction order."""
        return list(self._hot_nodes_client.keys())

    def get_hot_cache_stats(self) -> Dict[str, int]:
        """Return aggregate hot-cache counters."""
        return {
            "hits": self.hot_cache_hits,
            "misses": self.hot_cache_misses,
            "promotions": self.hot_cache_promotions,
            "evictions": self.hot_cache_evictions,
        }

    def _validate_sensitivity_size(self, sensitivity: Any) -> None:
        """Reject sensitivities that would exceed the configured block budget."""
        if len(pickle.dumps(sensitivity)) > self._max_sensitivity_pickle_size:
            raise ValueError(
                "Sensitivity value exceeds the configured sizing bound. "
                "Pass a larger max_sensitivity_value when constructing the OMAP."
            )

    def _make_metadata(
        self,
        access_count: int = 0,
        sensitivity: Any = None,
        pinned_leaf: Optional[int] = None,
        sensitivity_version: int = 0,
        lazy_sensitivity: Any = None,
        lazy_height: int = 0,
        lazy_version: int = -1,
    ) -> Dict[str, Any]:
        """Create a metadata block for a node."""
        sensitivity_value = self._default_sensitivity if sensitivity is None else sensitivity
        self._validate_sensitivity_size(sensitivity_value)
        if lazy_sensitivity is not None:
            self._validate_sensitivity_size(lazy_sensitivity)
        return {
            "access_count": min(int(access_count), self._ACCESS_COUNT_CAP),
            "sensitivity": copy.deepcopy(sensitivity_value),
            "pinned_leaf": pinned_leaf,
            "sensitivity_version": int(sensitivity_version),
            "lazy_sensitivity": copy.deepcopy(lazy_sensitivity),
            "lazy_height": max(0, int(lazy_height)),
            "lazy_version": int(lazy_version),
        }

    def _ensure_node_metadata(self, node: Data) -> Dict[str, Any]:
        """Ensure a node carries the expected metadata shape."""
        metadata = getattr(node.value, "metadata", None)
        if metadata is None:
            metadata = self._make_metadata()
            node.value.metadata = metadata

        if "access_count" not in metadata:
            metadata["access_count"] = 0
        if "sensitivity" not in metadata:
            metadata["sensitivity"] = copy.deepcopy(self._default_sensitivity)
        if "pinned_leaf" not in metadata:
            metadata["pinned_leaf"] = None
        if "sensitivity_version" not in metadata:
            metadata["sensitivity_version"] = 0
        if "lazy_sensitivity" not in metadata:
            metadata["lazy_sensitivity"] = None
        if "lazy_height" not in metadata:
            metadata["lazy_height"] = 0
        if "lazy_version" not in metadata:
            metadata["lazy_version"] = -1

        self._validate_sensitivity_size(metadata["sensitivity"])
        if metadata["lazy_sensitivity"] is not None:
            self._validate_sensitivity_size(metadata["lazy_sensitivity"])
        metadata["access_count"] = min(int(metadata["access_count"]), self._ACCESS_COUNT_CAP)
        metadata["sensitivity_version"] = int(metadata["sensitivity_version"])
        metadata["lazy_height"] = max(0, int(metadata["lazy_height"]))
        metadata["lazy_version"] = int(metadata["lazy_version"])
        return metadata

    def _next_sensitivity_version(self) -> int:
        """Return a monotonically increasing version for sensitivity assignments."""
        self._sensitivity_version += 1
        return self._sensitivity_version

    def _apply_sensitivity_value(self, node: Data, sensitivity: Any, version: int) -> None:
        """Apply a sensitivity value if it is newer than the node's current one."""
        metadata = self._ensure_node_metadata(node)
        if version < metadata["sensitivity_version"]:
            return

        self._validate_sensitivity_size(sensitivity)
        metadata["sensitivity"] = copy.deepcopy(sensitivity)
        metadata["sensitivity_version"] = int(version)

    def _install_lazy_sensitivity(self, node: Data, sensitivity: Any, height: int, version: int) -> None:
        """Install or extend a pending lazy sensitivity propagation marker."""
        if height <= 0:
            return

        metadata = self._ensure_node_metadata(node)
        if version < metadata["lazy_version"]:
            return
        if version == metadata["lazy_version"] and height <= metadata["lazy_height"]:
            return

        self._validate_sensitivity_size(sensitivity)
        metadata["lazy_sensitivity"] = copy.deepcopy(sensitivity)
        metadata["lazy_height"] = int(height)
        metadata["lazy_version"] = int(version)

    def _propagate_lazy_sensitivity_to_child(self, parent: Data, child: Data) -> None:
        """Propagate pending subtree sensitivity from a parent into a visited child."""
        parent_metadata = self._ensure_node_metadata(parent)
        pending_sensitivity = parent_metadata["lazy_sensitivity"]
        pending_height = parent_metadata["lazy_height"]
        pending_version = parent_metadata["lazy_version"]

        if pending_sensitivity is None or pending_height <= 0:
            return

        self._apply_sensitivity_value(node=child, sensitivity=pending_sensitivity, version=pending_version)
        self._install_lazy_sensitivity(
            node=child,
            sensitivity=pending_sensitivity,
            height=pending_height - 1,
            version=pending_version,
        )

    def _assign_sensitivity_to_current_subtree(
        self,
        subtree_height: int,
        sensitivity: Any,
    ) -> None:
        """Apply sensitivity to the current search path and lazily mark the remaining subtree."""
        if subtree_height is None and sensitivity is None:
            return
        if subtree_height is None or sensitivity is None:
            raise ValueError("subtree_height and sensitivity must be provided together.")
        if subtree_height < 1:
            raise ValueError("subtree_height must be at least 1.")
        if subtree_height > len(self._local.path):
            raise ValueError("subtree_height exceeds the current root-to-leaf path length.")

        version = self._next_sensitivity_version()
        start_index = len(self._local.path) - subtree_height

        for offset, node_key in enumerate(self._local.path[start_index:]):
            node = self._local.get(node_key)
            remaining_height = subtree_height - 1 - offset
            self._apply_sensitivity_value(node=node, sensitivity=sensitivity, version=version)
            self._install_lazy_sensitivity(
                node=node,
                sensitivity=sensitivity,
                height=remaining_height,
                version=version,
            )

    @staticmethod
    def _is_pointer(value: Any) -> bool:
        return isinstance(value, dict) and "node_id" in value and "leaf_p" in value

    def _normalize_pointer(self, value: Any) -> Dict[str, Any]:
        """Normalize legacy tuple pointers into the hot pointer shape."""
        if self._is_pointer(value):
            return {
                "node_id": value["node_id"],
                "location": value.get("location", self.ORAM),
                "leaf_p": value["leaf_p"],
            }
        if isinstance(value, tuple) and len(value) == 2:
            return {"node_id": value[0], "location": self.ORAM, "leaf_p": value[1]}
        raise TypeError(f"Unsupported child pointer format: {type(value)!r}")

    def _make_pointer(self, node_id: Any, leaf_p: int, location: str = ORAM) -> Dict[str, Any]:
        return {"node_id": node_id, "location": location, "leaf_p": leaf_p}

    def _set_pointer(
        self,
        parent: Data,
        child_index: int,
        node_id: Any,
        leaf_p: int,
        location: str,
    ) -> None:
        parent.value.values[child_index] = self._make_pointer(node_id=node_id, leaf_p=leaf_p, location=location)

    @staticmethod
    def _is_leaf_node(node: Data) -> bool:
        return len(node.value.keys) == len(node.value.values)

    def _find_child_index(self, node: Data, key: Any) -> int:
        """Locate the child slot used to descend for a search key."""
        child_index = len(node.value.keys)
        for index, each_key in enumerate(node.value.keys):
            if key == each_key:
                child_index = index + 1
                break
            if key < each_key:
                child_index = index
                break
        return child_index

    def _numeric_sensitivity(self, node: Data) -> float:
        """Convert sensitivity to a numeric score when possible."""
        sensitivity = self._ensure_node_metadata(node)["sensitivity"]
        return float(sensitivity) if isinstance(sensitivity, (int, float)) else 0.0

    def _hot_score(self, node: Data) -> Tuple[int, float]:
        metadata = self._ensure_node_metadata(node)
        return metadata["access_count"], self._numeric_sensitivity(node)

    def _select_eviction_candidate(self) -> Any:
        """Evict lowest-score node, breaking ties by current cache order."""
        victim_key = None
        victim_score = None
        for lru_rank, (key, node) in enumerate(self._hot_nodes_client.items()):
            score = (*self._hot_score(node), lru_rank)
            if victim_score is None or score < victim_score:
                victim_key = key
                victim_score = score
        return victim_key

    def _check_stash_capacity(self) -> None:
        if len(self._stash) > self._stash_size:
            raise MemoryError("Stash overflow!")

    def _remove_stash_key(self, key: Any) -> None:
        """Remove stale copies of a key from stash before reinsertion."""
        self._stash = [node for node in self._stash if node.key != key]

    @cached_property
    def _max_block_size(self) -> int:
        """Size ORAM blocks against the actual hot-node wire format."""
        max_key = self._num_data - 1
        max_leaf = self._leaf_range - 1
        metadata = self._make_metadata(
            access_count=self._ACCESS_COUNT_CAP,
            sensitivity=self._max_sensitivity_value,
            pinned_leaf=max_leaf,
            sensitivity_version=self._ACCESS_COUNT_CAP,
            lazy_sensitivity=self._max_sensitivity_value,
            lazy_height=self._max_height,
            lazy_version=self._ACCESS_COUNT_CAP,
        )

        internal = Data(
            key=max_key,
            leaf=max_leaf,
            value=BPlusHotData(
                keys=[os.urandom(self._key_size) for _ in range(self._order - 1)],
                values=[
                    self._make_pointer(
                        node_id=max_key,
                        location=self.HOT_CLI_CACHE,
                        leaf_p=max_leaf,
                    )
                    for _ in range(self._order)
                ],
                metadata=copy.deepcopy(metadata),
            ).dump(),
        )
        leaf = Data(
            key=max_key,
            leaf=max_leaf,
            value=BPlusHotData(
                keys=[os.urandom(self._key_size) for _ in range(self._order - 1)],
                values=[os.urandom(self._data_size) for _ in range(self._order - 1)],
                metadata=copy.deepcopy(metadata),
            ).dump(),
        )

        return max(len(internal.dump()), len(leaf.dump())) + Helper.LENGTH_HEADER_SIZE

    def _encrypt_path_data(self, path: PathData) -> PathData:
        """Encrypt hot B+ nodes stored in a path."""

        def _enc_bucket(bucket: List[Data]) -> List[bytes]:
            for data in bucket:
                data.value = data.value.dump()

            enc_bucket = [
                self._encryptor.enc(plaintext=Helper.pad_pickle(data=data.dump(), length=self._max_block_size))
                for data in bucket
            ]

            dummy_needed = self._bucket_size - len(bucket)
            if dummy_needed > 0:
                enc_bucket.extend([
                    self._encryptor.enc(plaintext=Helper.pad_pickle(data=Data().dump(), length=self._max_block_size))
                    for _ in range(dummy_needed)
                ])

            return enc_bucket

        return {idx: _enc_bucket(bucket) for idx, bucket in path.items()} if self._encryptor else path

    def _decrypt_path_data(self, path: PathData) -> PathData:
        """Decrypt hot B+ nodes stored in a path."""

        def _dec_bucket(bucket: List[bytes]) -> List[Data]:
            dec_bucket = [
                dec for data in bucket
                if (dec := Data.load_unpad(self._encryptor.dec(ciphertext=data))).key is not None
            ]

            for data in dec_bucket:
                data.value = BPlusHotData.from_pickle(data=data.value)
                self._ensure_node_metadata(data)

            return dec_bucket

        return {idx: _dec_bucket(bucket) for idx, bucket in path.items()} if self._encryptor else path

    def _get_bplus_data(
        self,
        keys: Any = None,
        values: Any = None,
        sensitivity: Any = None,
        metadata: Dict[str, Any] = None,
    ) -> Data:
        """Create a new hot-aware B+ node."""
        node_metadata = copy.deepcopy(metadata) if metadata is not None else self._make_metadata(sensitivity=sensitivity)
        data_block = Data(
            key=self._block_id,
            leaf=self._get_new_leaf(),
            value=BPlusHotData(keys=keys, values=values, metadata=node_metadata),
        )
        self._block_id += 1
        return data_block

    def _get_hot_data_list(self, root: BPlusTreeNode, encryption: bool = False) -> List[Data]:
        """Expand a plain B+ tree into hot-node `Data` records."""
        root.id = self._block_id
        root.leaf = self._get_new_leaf()
        self._block_id += 1

        stack = [root]
        result = []

        while stack:
            node = stack.pop()

            if not node.is_leaf:
                for child in node.values:
                    child.id = self._block_id
                    child.leaf = self._get_new_leaf()
                    self._block_id += 1

                values = [
                    self._make_pointer(node_id=child.id, leaf_p=child.leaf, location=self.ORAM)
                    for child in node.values
                ]
                stack.extend(list(node.values))
            else:
                values = list(node.values)

            hot_value = BPlusHotData(
                keys=list(node.keys),
                values=copy.deepcopy(values),
                metadata=self._make_metadata(),
            )

            if encryption:
                result.append(Data(key=node.id, leaf=node.leaf, value=hot_value.dump()))
            else:
                result.append(Data(key=node.id, leaf=node.leaf, value=hot_value))

        return result

    def _init_ods_storage(self, data: List[Tuple[Any, Any]]) -> BinaryTree:
        """Initialize ORAM storage for a single hot-aware B+ tree."""
        tree = BinaryTree(
            filename=self._filename,
            num_data=self._num_data,
            data_size=self._max_block_size,
            bucket_size=self._bucket_size,
            disk_size=self._disk_size,
            encryption=True if self._encryptor else False,
        )

        if data:
            root = BPlusTreeNode()
            bplus_tree = BPlusTree(order=self._order, leaf_range=self._leaf_range)

            for kv_pair in data:
                if isinstance(kv_pair, tuple):
                    kv_pair = KVPair(key=kv_pair[0], value=kv_pair[1])
                root = bplus_tree.insert(root=root, kv_pair=kv_pair)

            data_list = self._get_hot_data_list(root=root, encryption=self._encryptor is not None)
            for bplus_data in data_list:
                tree.fill_data_to_storage_leaf(data=bplus_data)

            self.root = (root.id, root.leaf)

        if self._encryptor:
            tree.storage.encrypt(encryptor=self._encryptor)

        return tree

    def _init_mul_tree_ods_storage(
        self,
        data_list: List[List[Tuple[Any, Any]]],
    ) -> Tuple[BinaryTree, List[Tuple[Any, int]]]:
        """Initialize ORAM storage for multiple hot-aware B+ trees."""
        tree = BinaryTree(
            filename=self._filename,
            num_data=self._num_data,
            disk_size=self._disk_size,
            bucket_size=self._bucket_size,
            data_size=self._max_block_size,
            encryption=True if self._encryptor else False,
        )

        root_list = []

        for data in data_list:
            if data:
                root = BPlusTreeNode()
                bplus_tree = BPlusTree(order=self._order, leaf_range=self._leaf_range)

                for kv_pair in data:
                    if isinstance(kv_pair, tuple):
                        kv_pair = KVPair(key=kv_pair[0], value=kv_pair[1])
                    root = bplus_tree.insert(root=root, kv_pair=kv_pair)

                node_data_list = self._get_hot_data_list(root=root, encryption=self._encryptor is not None)
                for bplus_data in node_data_list:
                    tree.fill_data_to_storage_leaf(data=bplus_data)

                root_list.append((root.id, root.leaf))
            else:
                root_list.append(None)

        if self._encryptor:
            tree.storage.encrypt(encryptor=self._encryptor)

        return tree, root_list

    def _prepare_cold_node(self, node: Data) -> None:
        """Clear any pinned leaf before normal remapping resumes."""
        self._ensure_node_metadata(node)["pinned_leaf"] = None

    def _promote_node_to_hot_cache(
        self,
        node: Data,
        evicted_active_keys: Set[Any],
    ) -> bool:
        """Insert a node into the hot cache immediately, evicting a victim if needed."""
        if self._hot_nodes_client_size == 0:
            return False

        if node.key in self._hot_nodes_client:
            self._hot_nodes_client[node.key] = copy.deepcopy(node)
            self._hot_nodes_client.move_to_end(node.key)
            return True

        while len(self._hot_nodes_client) >= self._hot_nodes_client_size:
            victim_key = self._select_eviction_candidate()
            if victim_key is None:
                return False

            victim_cached = self._hot_nodes_client.pop(victim_key)
            victim_live = self._local.get(victim_key)
            victim_node = copy.deepcopy(victim_live if victim_live is not None else victim_cached)
            self._evict_hot_node_to_stash(victim_node)
            if victim_live is not None:
                evicted_active_keys.add(victim_key)
            self.hot_cache_evictions += 1

        self._hot_nodes_client[node.key] = copy.deepcopy(node)
        self.hot_cache_promotions += 1
        return True

    def _record_access_and_place_node(
        self,
        node: Data,
        parent: Data = None,
        child_index: int = None,
        from_oram: bool = False,
        evicted_active_keys: Set[Any] = None,
    ) -> bool:
        """Increment access metadata and immediately decide whether the node stays hot."""
        metadata = self._ensure_node_metadata(node)
        metadata["access_count"] = min(metadata["access_count"] + 1, self._ACCESS_COUNT_CAP)

        if node.key in self._hot_nodes_client:
            self._hot_nodes_client[node.key] = copy.deepcopy(node)
            self._hot_nodes_client.move_to_end(node.key)
            return True

        if parent is None or child_index is None:
            return False

        if from_oram:
            self._prepare_cold_node(node)

        if metadata["access_count"] > self._hot_access_threshold:
            leaf_p = self._get_new_leaf()
            metadata["pinned_leaf"] = leaf_p
            node.leaf = leaf_p
            self._set_pointer(
                parent=parent,
                child_index=child_index,
                node_id=node.key,
                leaf_p=leaf_p,
                location=self.HOT_CLI_CACHE,
            )
            if self._promote_node_to_hot_cache(
                node=node,
                evicted_active_keys=evicted_active_keys if evicted_active_keys is not None else set(),
            ):
                self._hot_nodes_client[node.key] = copy.deepcopy(node)
                self._hot_nodes_client.move_to_end(node.key)
                return True

        if from_oram:
            new_leaf = self._get_new_leaf()
            metadata["pinned_leaf"] = None
            node.leaf = new_leaf
            self._set_pointer(
                parent=parent,
                child_index=child_index,
                node_id=node.key,
                leaf_p=new_leaf,
                location=self.ORAM,
            )
        return False

    def _evict_hot_node_to_stash(self, node: Data) -> None:
        """Place an evicted hot node into the ORAM stash at its committed leaf."""
        metadata = self._ensure_node_metadata(node)
        if metadata["pinned_leaf"] is not None:
            node.leaf = metadata["pinned_leaf"]

        self._remove_stash_key(node.key)
        self._stash.append(copy.deepcopy(node))
        self._check_stash_capacity()

    def _finalize_search_node(self, node_key: Any, evicted_active_keys: Set[Any]) -> None:
        """Remove a traversal node from local and place it in hot cache or stash."""
        node = self._local.remove(node_key)
        if node is None:
            return

        if node.key in self._hot_nodes_client:
            self._hot_nodes_client[node.key] = copy.deepcopy(node)
            self._hot_nodes_client.move_to_end(node.key)
            return

        if node.key not in evicted_active_keys:
            self._stash.append(node)
            self._check_stash_capacity()

    def _writeback_path(self, leaf: Optional[int]) -> None:
        """Evict the stash onto a path that was just accessed from ORAM."""
        if leaf is None:
            return
        self._client.add_write_path(label=self._name, data=self._evict_stash(leaves=[leaf]))
        self._client.execute()

    def _move_pointer_target_to_local(
        self,
        pointer: Any,
        parent_key: Any = None,
        child_index: int = None,
        without_eviction: bool = False,
        use_hot_cache: bool = False,
    ) -> Tuple[Data, bool]:
        """
        Move the child addressed by a pointer into local storage.

        Returns `(node, from_oram)` so callers can decide whether to perform a
        normal ORAM remap or keep the pinned hot identity intact.
        """
        pointer = self._normalize_pointer(pointer)
        child_key = pointer["node_id"]
        child_leaf = pointer["leaf_p"]

        existing = self._local.get(child_key)
        if existing is not None:
            return existing, False

        if pointer["location"] == self.HOT_CLI_CACHE:
            cached = self._hot_nodes_client.get(child_key)
            if cached is not None:
                self._local.add(node=copy.deepcopy(cached), parent_key=parent_key, child_index=child_index)
                self.hot_cache_hits += 1
                return self._local.get(child_key), False
            if use_hot_cache:
                self.hot_cache_misses += 1
        elif use_hot_cache:
            self.hot_cache_misses += 1

        if without_eviction:
            super()._move_node_to_local_without_eviction(
                key=child_key,
                leaf=child_leaf,
                parent_key=parent_key,
                child_index=child_index,
            )
        else:
            super()._move_node_to_local(
                key=child_key,
                leaf=child_leaf,
                parent_key=parent_key,
                child_index=child_index,
            )

        return self._local.get(child_key), True

    def _flush_hot_nodes_client_to_oram(self) -> None:
        """Move all cached hot nodes into the stash on their committed leaves."""
        if not self._hot_nodes_client:
            return

        for node in self._hot_nodes_client.values():
            self._evict_hot_node_to_stash(copy.deepcopy(node))

        self._hot_nodes_client.clear()

    def flush_hot_nodes_client_to_oram(self) -> None:
        """Public helper for tests and benchmarks."""
        self._flush_hot_nodes_client_to_oram()

    def _find_leaf(self, key: Any) -> Tuple[int, int]:
        """Fast-search traversal for the hot pointer format."""
        if self._local:
            raise MemoryError("The local storage was not emptied before this operation.")

        super()._move_node_to_local_without_eviction(
            key=self.root[0],
            leaf=self.root[1],
            parent_key=None,
            child_index=None,
        )

        node = self._local.get_root()
        old_leaf = node.leaf
        node.leaf = self._get_new_leaf()
        self.root = (node.key, node.leaf)
        num_retrieved_node = 1

        while not self._is_leaf_node(node):
            new_leaf = self._get_new_leaf()
            child_index = self._find_child_index(node=node, key=key)
            pointer = self._normalize_pointer(node.value.values[child_index])
            child_key = pointer["node_id"]
            child_leaf = pointer["leaf_p"]

            self._set_pointer(parent=node, child_index=child_index, node_id=child_key, leaf_p=new_leaf, location=self.ORAM)
            self._stash.append(self._local.remove(node.key))
            self._client.add_write_path(label=self._name, data=self._evict_stash(leaves=[old_leaf]))
            self._client.execute()

            child, _ = self._move_pointer_target_to_local(
                pointer=pointer,
                parent_key=None,
                child_index=None,
                without_eviction=True,
                use_hot_cache=False,
            )

            self._propagate_lazy_sensitivity_to_child(parent=node, child=child)
            node = child
            old_leaf = node.leaf
            self._prepare_cold_node(node)
            node.leaf = new_leaf
            num_retrieved_node += 1

        return old_leaf, num_retrieved_node

    def _find_leaf_to_local(self, key: Any) -> None:
        """Load the full root-to-leaf path into local storage."""
        if self._local:
            raise MemoryError("The local storage was not emptied before this operation.")

        super()._move_node_to_local(key=self.root[0], leaf=self.root[1], parent_key=None, child_index=None)
        node = self._local.get_root()
        node.leaf = self._get_new_leaf()
        self.root = (node.key, node.leaf)

        while not self._is_leaf_node(node):
            new_leaf = self._get_new_leaf()
            child_index = self._find_child_index(node=node, key=key)
            pointer = self._normalize_pointer(node.value.values[child_index])

            child, _ = self._move_pointer_target_to_local(
                pointer=pointer,
                parent_key=node.key,
                child_index=child_index,
                without_eviction=False,
                use_hot_cache=False,
            )

            self._propagate_lazy_sensitivity_to_child(parent=node, child=child)
            self._prepare_cold_node(child)
            self._set_pointer(parent=node, child_index=child_index, node_id=child.key, leaf_p=new_leaf, location=self.ORAM)
            child.leaf = new_leaf
            node = child

    def _split_node(self, node: Data) -> Tuple[int, int]:
        """Split an overflowing node and add the new right sibling to stash."""
        right_metadata = copy.deepcopy(self._ensure_node_metadata(node))
        right_metadata["pinned_leaf"] = None
        right_node = self._get_bplus_data(metadata=right_metadata)

        if self._is_leaf_node(node):
            right_node.value.keys = node.value.keys[self._mid:]
            right_node.value.values = node.value.values[self._mid:]
            node.value.keys = node.value.keys[:self._mid]
            node.value.values = node.value.values[:self._mid]
        else:
            right_node.value.keys = node.value.keys[self._mid + 1:]
            right_node.value.values = node.value.values[self._mid + 1:]
            node.value.keys = node.value.keys[:self._mid]
            node.value.values = node.value.values[:self._mid + 1]

        self._stash.append(right_node)
        self._check_stash_capacity()
        return right_node.key, right_node.leaf

    def _insert_in_parent(self, child_node: Data, parent_node: Data) -> None:
        """Insert a split child into its parent."""
        insert_key = child_node.value.keys[self._mid]
        right_node_key, right_node_leaf = self._split_node(node=child_node)
        right_pointer = self._make_pointer(node_id=right_node_key, leaf_p=right_node_leaf, location=self.ORAM)

        for index, each_key in enumerate(parent_node.value.keys):
            if insert_key < each_key:
                parent_node.value.keys = parent_node.value.keys[:index] + [insert_key] + parent_node.value.keys[index:]
                parent_node.value.values = (
                    parent_node.value.values[:index + 1] + [right_pointer] + parent_node.value.values[index + 1:]
                )
                return
            if index + 1 == len(parent_node.value.keys):
                parent_node.value.keys.append(insert_key)
                parent_node.value.values.append(right_pointer)
                return

    def _create_parent(self, child_node: Data) -> None:
        """Create a new root after splitting the old root."""
        insert_key = child_node.value.keys[self._mid]
        right_node_key, right_node_leaf = self._split_node(node=child_node)
        values = [
            self._make_pointer(node_id=child_node.key, leaf_p=child_node.leaf, location=self.ORAM),
            self._make_pointer(node_id=right_node_key, leaf_p=right_node_leaf, location=self.ORAM),
        ]

        parent_node = self._get_bplus_data(keys=[insert_key], values=values)
        self._stash.append(parent_node)
        self._check_stash_capacity()
        self.root = (parent_node.key, parent_node.leaf)

    def _perform_insertion(self) -> None:
        """Perform split propagation after inserting into the target leaf."""
        path = self._local.path
        index = len(path) - 1

        while index >= 0:
            node_key = path[index]
            node = self._local.get(node_key)

            if len(node.value.keys) >= self._order:
                if index > 0:
                    parent_key = path[index - 1]
                    parent_node = self._local.get(parent_key)
                    self._insert_in_parent(child_node=node, parent_node=parent_node)
                    index -= 1
                else:
                    self._create_parent(child_node=node)
                    break
            else:
                break

    def insert(self, key: Any, value: Any = None) -> None:
        """Insert a key/value pair into the hot-pointer B+ tree."""
        self._flush_hot_nodes_client_to_oram()

        if key is None:
            self._perform_dummy_operation(num_round=3 * self._max_height)
            return

        if self.root is None:
            data_block = self._get_bplus_data(keys=[key], values=[value])
            self._stash.append(data_block)
            self._check_stash_capacity()
            self.root = (data_block.key, data_block.leaf)
            self._perform_dummy_operation(num_round=3 * self._max_height)
            return

        if self._local:
            raise MemoryError("The local storage was not emptied before this operation.")

        self._find_leaf_to_local(key=key)
        leaf = self._local.get_leaf()

        for index, each_key in enumerate(leaf.value.keys):
            if key < each_key:
                leaf.value.keys = leaf.value.keys[:index] + [key] + leaf.value.keys[index:]
                leaf.value.values = leaf.value.values[:index] + [value] + leaf.value.values[index:]
                break
            if index + 1 == len(leaf.value.keys):
                leaf.value.keys.append(key)
                leaf.value.values.append(value)
                break

        num_retrieved_nodes = len(self._local)
        self._perform_insertion()
        self._flush_local_to_stash()
        self._check_stash_capacity()
        self._perform_dummy_operation(num_round=2 * self._max_height - num_retrieved_nodes)

    def _search_with_buffered_path(
        self,
        key: Any,
        value: Any = None,
        subtree_height: int = None,
        sensitivity: Any = None,
    ) -> Any:
        """Search while keeping the full root-to-leaf path local."""
        evicted_active_keys: Set[Any] = set()

        super()._move_node_to_local(key=self.root[0], leaf=self.root[1], parent_key=None, child_index=None)
        node = self._local.get_root()
        node.leaf = self._get_new_leaf()
        self.root = (node.key, node.leaf)
        self._record_access_and_place_node(
            node=node,
            parent=None,
            child_index=None,
            from_oram=True,
            evicted_active_keys=evicted_active_keys,
        )

        while not self._is_leaf_node(node):
            child_index = self._find_child_index(node=node, key=key)
            child, from_oram = self._move_pointer_target_to_local(
                pointer=node.value.values[child_index],
                parent_key=node.key,
                child_index=child_index,
                without_eviction=False,
                use_hot_cache=True,
            )

            self._propagate_lazy_sensitivity_to_child(parent=node, child=child)
            self._record_access_and_place_node(
                node=child,
                parent=node,
                child_index=child_index,
                from_oram=from_oram,
                evicted_active_keys=evicted_active_keys,
            )
            node = child

        search_value = None
        for index, each_key in enumerate(node.value.keys):
            if key == each_key:
                search_value = node.value.values[index]
                if value is not None:
                    node.value.values[index] = value
                break

        self._assign_sensitivity_to_current_subtree(
            subtree_height=subtree_height,
            sensitivity=sensitivity,
        )

        for local_node in self._local.to_list():
            if local_node.key in self._hot_nodes_client:
                self._hot_nodes_client[local_node.key] = copy.deepcopy(local_node)
                self._hot_nodes_client.move_to_end(local_node.key)
            elif local_node.key not in evicted_active_keys:
                self._stash.append(local_node)

        self._local.clear()
        self._check_stash_capacity()
        return search_value

    def _search_with_immediate_writeback(
        self,
        key: Any,
        value: Any = None,
    ) -> Any:
        """Search while writing back cold path nodes as soon as their child choice is known."""
        evicted_active_keys: Set[Any] = set()

        super()._move_node_to_local_without_eviction(
            key=self.root[0],
            leaf=self.root[1],
            parent_key=None,
            child_index=None,
        )

        node = self._local.get_root()
        pending_write_leaf = node.leaf
        node.leaf = self._get_new_leaf()
        self.root = (node.key, node.leaf)
        self._record_access_and_place_node(
            node=node,
            parent=None,
            child_index=None,
            from_oram=True,
            evicted_active_keys=evicted_active_keys,
        )

        while not self._is_leaf_node(node):
            self._writeback_path(leaf=pending_write_leaf)

            child_index = self._find_child_index(node=node, key=key)
            pointer = self._normalize_pointer(node.value.values[child_index])
            child, from_oram = self._move_pointer_target_to_local(
                pointer=pointer,
                parent_key=node.key,
                child_index=child_index,
                without_eviction=True,
                use_hot_cache=True,
            )

            self._propagate_lazy_sensitivity_to_child(parent=node, child=child)
            self._record_access_and_place_node(
                node=child,
                parent=node,
                child_index=child_index,
                from_oram=from_oram,
                evicted_active_keys=evicted_active_keys,
            )

            self._finalize_search_node(node_key=node.key, evicted_active_keys=evicted_active_keys)
            pending_write_leaf = pointer["leaf_p"] if from_oram else None
            node = child

        search_value = None
        for index, each_key in enumerate(node.value.keys):
            if key == each_key:
                search_value = node.value.values[index]
                if value is not None:
                    node.value.values[index] = value
                break

        self._finalize_search_node(node_key=node.key, evicted_active_keys=evicted_active_keys)
        self._writeback_path(leaf=pending_write_leaf)
        self._check_stash_capacity()
        return search_value

    def search(
        self,
        key: Any,
        value: Any = None,
        subtree_height: int = None,
        sensitivity: Any = None,
    ) -> Any:
        """Search using the client hot-node cache on internal traversals."""
        if key is None or self.root is None:
            return None

        if self._local:
            raise MemoryError("The local storage was not emptied before this operation.")

        if subtree_height is not None or sensitivity is not None:
            return self._search_with_buffered_path(
                key=key,
                value=value,
                subtree_height=subtree_height,
                sensitivity=sensitivity,
            )

        return self._search_with_immediate_writeback(key=key, value=value)

    def fast_search(self, key: Any, value: Any = None) -> Any:
        """Fast search on the hot-pointer format, after flushing client hot nodes."""
        self._flush_hot_nodes_client_to_oram()

        if key is None:
            self._perform_dummy_operation(num_round=self._max_height)
            return None

        if self.root is None:
            self._perform_dummy_operation(num_round=self._max_height)
            return None

        if self._local:
            raise MemoryError("The local storage was not emptied before this operation.")

        old_leaf, num_retrieved_nodes = self._find_leaf(key=key)
        leaf = self._local.get_leaf()
        search_value = None

        for index, each_key in enumerate(leaf.value.keys):
            if key == each_key:
                search_value = leaf.value.values[index]
                if value is not None:
                    leaf.value.values[index] = value
                break

        self._flush_local_to_stash()
        self._check_stash_capacity()
        self._client.add_write_path(label=self._name, data=self._evict_stash(leaves=[old_leaf]))
        self._client.execute()
        self._perform_dummy_operation(num_round=self._max_height - num_retrieved_nodes)

        return search_value

    def set_sensitivity(self, key: Any, subtree_height: int, sensitivity: Any) -> Any:
        """Traverse to `key` and assign `sensitivity` to the subtree of the given height."""
        return self.search(key=key, subtree_height=subtree_height, sensitivity=sensitivity)

    def _find_path_for_delete(self, key: Any) -> Tuple[Dict[int, Data], List[int]]:
        """Find the delete path under the hot pointer representation."""
        path_nodes: Dict[int, Data] = {}
        child_indices: List[int] = []
        level = 0

        super()._move_node_to_local(key=self.root[0], leaf=self.root[1], parent_key=None, child_index=None)
        node = self._local.remove(self._local.root_key)
        node.leaf = self._get_new_leaf()
        self.root = (node.key, node.leaf)
        path_nodes[level] = node

        while not self._is_leaf_node(node):
            new_leaf = self._get_new_leaf()
            child_index = self._find_child_index(node=node, key=key)
            child_indices.append(child_index)
            pointer = self._normalize_pointer(node.value.values[child_index])

            child, _ = self._move_pointer_target_to_local(
                pointer=pointer,
                parent_key=None,
                child_index=None,
                without_eviction=False,
                use_hot_cache=False,
            )
            child_node = self._local.remove(self._local.root_key)

            self._propagate_lazy_sensitivity_to_child(parent=node, child=child_node)
            self._prepare_cold_node(child_node)
            self._set_pointer(parent=node, child_index=child_index, node_id=child_node.key, leaf_p=new_leaf, location=self.ORAM)
            child_node.leaf = new_leaf

            level += 1
            path_nodes[level] = child_node
            node = child_node

        return path_nodes, child_indices

    def _fetch_sibling_for_delete(self, parent: Data, sibling_index: int) -> Data:
        """Fetch a sibling node during delete rebalancing."""
        pointer = self._normalize_pointer(parent.value.values[sibling_index])
        new_leaf = self._get_new_leaf()

        sibling, _ = self._move_pointer_target_to_local(
            pointer=pointer,
            parent_key=None,
            child_index=None,
            without_eviction=False,
            use_hot_cache=False,
        )
        sibling = self._local.remove(self._local.root_key)

        self._propagate_lazy_sensitivity_to_child(parent=parent, child=sibling)
        self._prepare_cold_node(sibling)
        sibling.leaf = new_leaf
        self._set_pointer(parent=parent, child_index=sibling_index, node_id=sibling.key, leaf_p=new_leaf, location=self.ORAM)
        return sibling

    def delete(self, key: Any) -> Any:
        """Delete a key from the hot-pointer B+ tree."""
        self._flush_hot_nodes_client_to_oram()

        if key is None:
            self._perform_dummy_operation(num_round=3 * self._max_height)
            return None

        if self.root is None:
            self._perform_dummy_operation(num_round=3 * self._max_height)
            return None

        min_keys = (self._order - 1) // 2
        path_nodes, child_indices = self._find_path_for_delete(key=key)
        leaf_level = len(path_nodes) - 1
        leaf = path_nodes[leaf_level]
        fetched_siblings: List[Data] = []

        key_index = None
        deleted_value = None
        for index, leaf_key in enumerate(leaf.value.keys):
            if leaf_key == key:
                key_index = index
                deleted_value = leaf.value.values[index]
                break

        if key_index is None:
            for node in path_nodes.values():
                self._stash.append(node)
            self._check_stash_capacity()
            self._perform_dummy_operation(num_round=3 * self._max_height - len(path_nodes))
            return None

        leaf.value.keys.pop(key_index)
        leaf.value.values.pop(key_index)

        if not child_indices:
            if len(leaf.value.keys) == 0:
                self.root = None
            else:
                self._stash.append(leaf)
                self._check_stash_capacity()
            self._perform_dummy_operation(num_round=3 * self._max_height - 1)
            return deleted_value

        node_level = leaf_level
        node = leaf

        for level in range(len(child_indices) - 1, -1, -1):
            parent = path_nodes[level]
            child_index = child_indices[level]

            if len(node.value.keys) >= min_keys:
                break

            has_left = child_index > 0
            has_right = child_index < len(parent.value.values) - 1

            left_sib = None
            right_sib = None

            if has_left:
                left_sib = self._fetch_sibling_for_delete(parent, child_index - 1)
                fetched_siblings.append(left_sib)
                if len(left_sib.value.keys) > min_keys:
                    if self._is_leaf_node(node):
                        node.value.keys.insert(0, left_sib.value.keys.pop())
                        node.value.values.insert(0, left_sib.value.values.pop())
                        parent.value.keys[child_index - 1] = node.value.keys[0]
                    else:
                        node.value.keys.insert(0, parent.value.keys[child_index - 1])
                        node.value.values.insert(0, left_sib.value.values.pop())
                        parent.value.keys[child_index - 1] = left_sib.value.keys.pop()
                    break

            if has_right:
                right_sib = self._fetch_sibling_for_delete(parent, child_index + 1)
                fetched_siblings.append(right_sib)
                if len(right_sib.value.keys) > min_keys:
                    if self._is_leaf_node(node):
                        node.value.keys.append(right_sib.value.keys.pop(0))
                        node.value.values.append(right_sib.value.values.pop(0))
                        parent.value.keys[child_index] = right_sib.value.keys[0]
                    else:
                        node.value.keys.append(parent.value.keys[child_index])
                        node.value.values.append(right_sib.value.values.pop(0))
                        parent.value.keys[child_index] = right_sib.value.keys.pop(0)
                    break

            if has_left and left_sib is not None:
                if self._is_leaf_node(node):
                    left_sib.value.keys.extend(node.value.keys)
                    left_sib.value.values.extend(node.value.values)
                else:
                    left_sib.value.keys.append(parent.value.keys[child_index - 1])
                    left_sib.value.keys.extend(node.value.keys)
                    left_sib.value.values.extend(node.value.values)

                parent.value.keys.pop(child_index - 1)
                parent.value.values.pop(child_index)
                del path_nodes[node_level]

            elif has_right and right_sib is not None:
                if self._is_leaf_node(node):
                    node.value.keys.extend(right_sib.value.keys)
                    node.value.values.extend(right_sib.value.values)
                else:
                    node.value.keys.append(parent.value.keys[child_index])
                    node.value.keys.extend(right_sib.value.keys)
                    node.value.values.extend(right_sib.value.values)

                parent.value.keys.pop(child_index)
                parent.value.values.pop(child_index + 1)
                fetched_siblings.remove(right_sib)

            node_level = level
            node = parent

        root_node = path_nodes.get(0)
        if root_node is not None and len(root_node.value.keys) == 0:
            if self._is_leaf_node(root_node):
                self.root = None
            else:
                child_pointer = self._normalize_pointer(root_node.value.values[0])
                self.root = (child_pointer["node_id"], child_pointer["leaf_p"])
            del path_nodes[0]

        if self.root is not None and 0 in path_nodes:
            root_node = path_nodes[0]
            self.root = (root_node.key, root_node.leaf)

        total_nodes = 0
        for path_node in path_nodes.values():
            self._stash.append(path_node)
            total_nodes += 1
        for sibling in fetched_siblings:
            self._stash.append(sibling)
            total_nodes += 1

        self._check_stash_capacity()
        self._perform_dummy_operation(num_round=2 * self._max_height - total_nodes)
        return deleted_value
