"""Native variable-size ORAM (vORAM) implementation for DAORAM."""

from __future__ import annotations

import logging
import os
import pickle
import zlib
from typing import Any, Dict, List, Optional, Tuple

from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes

from daoram.dependency import BinaryTree, Encryptor, InteractServer, PathData, UNSET
from daoram.oram.tree_base_oram import TreeBaseOram

logger = logging.getLogger(__name__)

# Global debug gate for this module. Set to True for verbose internal tracing.
DEBUG = pickle.TRUE


class TrueVoram(TreeBaseOram):
    """Path-style vORAM with variable-size values and bucket_size fixed to 1."""

    _AES_BLOCK_SIZE = 16

    def __init__(
            self,
            num_data: int,
            data_size: int,
            client: InteractServer,
            name: str = "voram_true",
            filename: str = None,
            bucket_size: int = 1,
            stash_scale: int = 7,
            Z: Optional[int] = None,
            optimize: bool = True,
            keylen: int = 32,
            idlen: Optional[int] = None,
            compress: bool = True,
            encryptor: Encryptor = None,
            debug: bool = False,
    ):
        if bucket_size != 1:
            raise ValueError("TrueVoram requires bucket_size to be exactly 1.")
        if keylen not in (16, 24, 32):
            raise ValueError("AES key length must be one of {16, 24, 32} bytes.")

        # Keep TreeBaseOram initialization for shared attributes / conventions.
        super().__init__(
            name=name,
            num_data=num_data,
            data_size=data_size,
            client=client,
            filename=filename,
            bucket_size=bucket_size,
            stash_scale=stash_scale,
            encryptor=encryptor,  # accepted for API consistency, not used for node crypto
        )

        if Z is None:
            if optimize:
                Z = 4096
            else:
                raise ValueError("Z must be provided when optimize=False.")
        if Z % self._AES_BLOCK_SIZE != 0:
            raise ValueError(f"Z={Z} must be a multiple of AES block size {self._AES_BLOCK_SIZE}.")

        self._Z: int = int(Z)
        self._keylen: int = keylen
        self._idlen: int = idlen if idlen is not None else self._default_idlen(num_data=self._num_data)
        self._compress: bool = bool(compress)
        self._node_data_bytes: int = self._Z - 2 * self._keylen
        self._lenlen: int = self._compute_lenlen()
        self._chunk_header: int = self._idlen + self._lenlen

        if self._node_data_bytes <= self._chunk_header:
            raise ValueError("Node metadata is too large for the chosen Z/keylen/idlen.")

        # Byte-based stash cap (scaled by tree height).
        self._stash_byte_cap: int = self._stash_scale * max(1, self._level - 1) * self._Z

        # Runtime state (set/reset by init_server_storage).
        self._tree: Optional[BinaryTree] = None
        self._stash: Dict[int, bytearray] = {}
        self._node_keys: Dict[int, bytes] = {}
        self._root_key: Optional[bytes] = None
        self._iv: Optional[bytes] = None
        self._tmp_leaf: Optional[int] = None
        self._initialized: bool = False
        self._debug_enabled: bool = bool(DEBUG or debug)

        # Initialize a position map right away; init_server_storage resets it.
        self._init_pos_map()
        self._debug(
            "Initialized TrueVoram config: num_data=%d level=%d Z=%d idlen=%d keylen=%d compress=%s",
            self._num_data,
            self._level,
            self._Z,
            self._idlen,
            self._keylen,
            self._compress,
        )

    # --------------------------- Public API ---------------------------------

    def init_server_storage(self, data_map: dict = None) -> None:
        """Initialize server storage and preload all keys."""
        self._debug("init_server_storage start: data_map_provided=%s", data_map is not None)
        self._init_pos_map()
        self._tmp_leaf = None
        self._stash = {}
        self._tree = BinaryTree(
            num_data=self._num_data,
            bucket_size=1,
            filename=self._filename,
            data_size=self._Z,
            disk_size=self._Z,
            encryption=True,
        )

        self._iv = os.urandom(self._AES_BLOCK_SIZE)
        self._root_key = os.urandom(self._keylen)
        self._node_keys = {idx: os.urandom(self._keylen) for idx in range(self._tree.size)}
        self._node_keys[0] = self._root_key

        # Pre-populate all keys, following TreeBase semantics.
        for key in range(self._num_data):
            if data_map is None:
                value = os.urandom(self._data_size)
            else:
                if key not in data_map:
                    raise KeyError(f"Key {key} missing from data_map during initialization.")
                value = data_map[key]
            self._stash[key] = bytearray(self._encode_value(value))
        self._debug("Stash preloaded with %d keys (%d bytes).", len(self._stash), self._stash_bytes())

        # Bottom-up packing so lower nodes are filled first.
        for idx in reversed(range(self._tree.size)):
            plaintext = self._pack_node_plaintext(idx=idx)
            ciphertext = self._aes_encrypt(key=self._node_keys[idx], plaintext=plaintext)
            self._tree.storage[idx] = [ciphertext]

        self._enforce_stash_cap()
        self._initialized = True
        self.client.init_storage({self._name: self._tree})
        self._debug(
            "init_server_storage done: tree_size=%d stash_remaining=%d bytes",
            self._tree.size,
            self._stash_bytes(),
        )

    def operate_on_key(self, key: int, value: Any = UNSET) -> Any:
        """Read/write one key and immediately evict the accessed path."""
        self._ensure_ready()
        if self._tmp_leaf is not None:
            raise ValueError("A deferred access is pending; call eviction_with_update_stash first.")

        old_leaf = self._look_up_pos_map(key=key)
        new_leaf = self._get_new_leaf()
        self._pos_map[key] = new_leaf
        self._debug(
            "operate_on_key: key=%d old_leaf=%d new_leaf=%d write=%s",
            key,
            old_leaf,
            new_leaf,
            value is not UNSET,
        )

        self._read_path_into_stash(leaf=old_leaf)

        old_value = self._decode_key_from_stash(key=key)
        if value is not UNSET:
            self._stash[key] = bytearray(self._encode_value(value))
            self._debug("operate_on_key: key=%d updated payload bytes=%d", key, len(self._stash[key]))

        self._enforce_stash_cap()
        self._write_path_from_stash(leaf=old_leaf, execute=True)
        self._debug("operate_on_key complete: key=%d stash=%d bytes", key, self._stash_bytes())
        return old_value

    def operate_on_key_without_eviction(self, key: int, value: Any = UNSET) -> Any:
        """Read/write one key but defer eviction to eviction_with_update_stash."""
        self._ensure_ready()
        if self._tmp_leaf is not None:
            raise ValueError("A deferred access is already pending.")

        old_leaf = self._look_up_pos_map(key=key)
        new_leaf = self._get_new_leaf()
        self._pos_map[key] = new_leaf
        self._debug(
            "operate_on_key_without_eviction: key=%d old_leaf=%d new_leaf=%d write=%s",
            key,
            old_leaf,
            new_leaf,
            value is not UNSET,
        )

        self._read_path_into_stash(leaf=old_leaf)

        old_value = self._decode_key_from_stash(key=key)
        if value is not UNSET:
            self._stash[key] = bytearray(self._encode_value(value))
            self._debug("deferred update staged: key=%d payload bytes=%d", key, len(self._stash[key]))

        self._enforce_stash_cap()
        self._tmp_leaf = old_leaf
        self._debug("deferred access pending on leaf=%d", self._tmp_leaf)
        return old_value

    def eviction_with_update_stash(self, key: int, value: Any, execute: bool = True) -> None:
        """Update one stashed key, then evict the deferred path."""
        self._ensure_ready()
        if self._tmp_leaf is None:
            raise ValueError("No deferred access pending.")
        if key not in self._stash:
            raise KeyError(f"Key {key} not found in stash.")

        self._stash[key] = bytearray(self._encode_value(value))
        self._debug(
            "eviction_with_update_stash: key=%d payload bytes=%d execute=%s",
            key,
            len(self._stash[key]),
            execute,
        )
        self._enforce_stash_cap()
        self._write_path_from_stash(leaf=self._tmp_leaf, execute=execute)
        self._tmp_leaf = None
        self._debug("eviction_with_update_stash complete")

    # --------------------------- Internal helpers ---------------------------

    @staticmethod
    def _default_idlen(num_data: int) -> int:
        """Smallest id length such that key+1 is always representable."""
        return max(1, (int(num_data).bit_length() + 7) // 8)

    def _compute_lenlen(self) -> int:
        """Compute byte-length used for chunk length fields."""
        # Must be able to encode any possible chunk length in one node.
        max_chunk_len = self._node_data_bytes - self._idlen
        if max_chunk_len <= 0:
            raise ValueError("Invalid configuration: node has no room for data chunks.")
        return max(1, (max_chunk_len.bit_length() + 7) // 8)

    def _ensure_ready(self) -> None:
        if not self._initialized or self._tree is None or self._root_key is None or self._iv is None:
            raise ValueError("Storage not initialized. Call init_server_storage() first.")

    def _debug(self, msg: str, *args: Any) -> None:
        """Emit debug logs only when DEBUG mode is enabled."""
        if self._debug_enabled:
            logger.debug("[TrueVoram] " + msg, *args)

    def _stash_bytes(self) -> int:
        return sum(len(value) for value in self._stash.values())

    def _encode_ident(self, key: int) -> bytes:
        return (key + 1).to_bytes(self._idlen, byteorder="big", signed=False)

    def _decode_ident(self, encoded: bytes) -> int:
        key = int.from_bytes(encoded, byteorder="big", signed=False) - 1
        if key < 0:
            raise ValueError("Encountered invalid vORAM identifier.")
        return key

    def _encode_value(self, value: Any) -> bytes:
        raw = pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL)
        return zlib.compress(raw) if self._compress else raw

    def _decode_value(self, raw: bytes) -> Any:
        data = zlib.decompress(raw) if self._compress else raw
        return pickle.loads(data)

    def _aes_encrypt(self, key: bytes, plaintext: bytes) -> bytes:
        cipher = Cipher(algorithms.AES(key), modes.CBC(self._iv))
        encryptor = cipher.encryptor()
        return encryptor.update(plaintext) + encryptor.finalize()

    def _aes_decrypt(self, key: bytes, ciphertext: bytes) -> bytes:
        cipher = Cipher(algorithms.AES(key), modes.CBC(self._iv))
        decryptor = cipher.decryptor()
        return decryptor.update(ciphertext) + decryptor.finalize()

    def _node_leaf_range(self, idx: int) -> Tuple[int, int]:
        """Return [min_leaf, max_leaf] covered by node index idx."""
        start_leaf = self._tree.start_leaf
        left = idx
        right = idx
        while left < start_leaf:
            left = 2 * left + 1
        while right < start_leaf:
            right = 2 * right + 2
        return left - start_leaf, right - start_leaf

    def _eligible_for_node(self, idx: int, leaf: int) -> bool:
        lo, hi = self._node_leaf_range(idx=idx)
        return lo <= leaf <= hi

    def _read_path_into_stash(self, leaf: int) -> None:
        """Read one path, decrypt root->leaf, and append chunks to stash."""
        self._debug("read_path start: leaf=%d", leaf)
        self.client.add_read_path(label=self._name, leaves=[leaf])
        result = self.client.execute()
        if not result.success:
            raise RuntimeError(result.error or "Failed to read path from server.")
        path_data: PathData = result.results[self._name]

        # Path keys for this operation are rebuilt from root key.
        self._node_keys = {0: self._root_key}

        for idx in reversed(self._tree.get_leaf_path(leaf)):
            bucket = path_data[idx]
            if len(bucket) != 1:
                raise ValueError(f"Node {idx} must contain exactly one ciphertext block.")
            ciphertext = bucket[0]
            plaintext = self._aes_decrypt(key=self._node_keys[idx], ciphertext=ciphertext)
            self._parse_node_plaintext(idx=idx, plaintext=plaintext)
        self._debug("read_path done: leaf=%d stash=%d bytes", leaf, self._stash_bytes())

    def _parse_node_plaintext(self, idx: int, plaintext: bytes) -> None:
        """Parse node plaintext: load child keys and append chunk records to stash."""
        if len(plaintext) != self._Z:
            raise ValueError("Malformed node plaintext length.")

        # Child keys.
        offset = 0
        left_key = plaintext[offset: offset + self._keylen]
        offset += self._keylen
        right_key = plaintext[offset: offset + self._keylen]
        offset += self._keylen

        left_idx = 2 * idx + 1
        right_idx = 2 * idx + 2
        if left_idx < self._tree.size:
            self._node_keys[left_idx] = left_key
        if right_idx < self._tree.size:
            self._node_keys[right_idx] = right_key

        # Chunk records.
        parsed_chunks = 0
        parsed_bytes = 0
        while offset + self._chunk_header <= self._Z:
            ident = plaintext[offset: offset + self._idlen]
            offset += self._idlen

            if ident == b"\x00" * self._idlen:
                break

            chunk_len = int.from_bytes(
                plaintext[offset: offset + self._lenlen], byteorder="big", signed=False
            )
            offset += self._lenlen

            if offset + chunk_len > self._Z:
                raise ValueError("Malformed chunk length while parsing node.")

            chunk = plaintext[offset: offset + chunk_len]
            offset += chunk_len

            key = self._decode_ident(encoded=ident)
            self._stash.setdefault(key, bytearray()).extend(chunk)
            parsed_chunks += 1
            parsed_bytes += chunk_len
        self._debug(
            "parsed node idx=%d chunks=%d bytes=%d stash=%d bytes",
            idx,
            parsed_chunks,
            parsed_bytes,
            self._stash_bytes(),
        )

    def _pack_node_plaintext(self, idx: int) -> bytes:
        """Build one node plaintext from child keys + eligible stash chunks."""
        buffer = bytearray(self._Z)
        offset = 0

        left_idx = 2 * idx + 1
        right_idx = 2 * idx + 2

        if left_idx < self._tree.size:
            child_key = self._node_keys.get(left_idx)
            if child_key is None:
                raise ValueError(f"Missing key for child node {left_idx}.")
            buffer[offset: offset + self._keylen] = child_key
        else:
            buffer[offset: offset + self._keylen] = os.urandom(self._keylen)
        offset += self._keylen

        if right_idx < self._tree.size:
            child_key = self._node_keys.get(right_idx)
            if child_key is None:
                raise ValueError(f"Missing key for child node {right_idx}.")
            buffer[offset: offset + self._keylen] = child_key
        else:
            buffer[offset: offset + self._keylen] = os.urandom(self._keylen)
        offset += self._keylen

        # Write variable chunks greedily.
        written_chunks = 0
        written_bytes = 0
        for key in sorted(list(self._stash.keys())):
            if offset + self._chunk_header > self._Z:
                break
            if not self._eligible_for_node(idx=idx, leaf=self._pos_map[key]):
                continue

            data = self._stash[key]
            if not data:
                del self._stash[key]
                continue

            max_payload = self._Z - offset - self._chunk_header
            if max_payload <= 0:
                break

            chunk_len = min(max_payload, len(data))
            ident = self._encode_ident(key=key)

            buffer[offset: offset + self._idlen] = ident
            offset += self._idlen
            buffer[offset: offset + self._lenlen] = chunk_len.to_bytes(self._lenlen, byteorder="big")
            offset += self._lenlen
            buffer[offset: offset + chunk_len] = data[-chunk_len:]
            offset += chunk_len
            written_chunks += 1
            written_bytes += chunk_len

            del data[-chunk_len:]
            if not data:
                del self._stash[key]

        self._debug(
            "packed node idx=%d chunks=%d bytes=%d stash=%d bytes",
            idx,
            written_chunks,
            written_bytes,
            self._stash_bytes(),
        )
        return bytes(buffer)

    def _decode_key_from_stash(self, key: int) -> Any:
        if key not in self._stash:
            raise KeyError(f"Key {key} not found in stash.")
        raw = bytes(self._stash[key])
        try:
            return self._decode_value(raw=raw)
        except Exception as exc:
            raise ValueError(f"Could not decode value for key {key}.") from exc

    def _write_path_from_stash(self, leaf: int, execute: bool) -> None:
        """Evict stash to one accessed path (leaf->root), rotate keys, and write."""
        self._debug("write_path start: leaf=%d execute=%s", leaf, execute)
        out_path: PathData = {}
        for idx in self._tree.get_leaf_path(leaf):
            # Re-key node on every write, as in parent-held-key vORAM.
            new_key = os.urandom(self._keylen)
            self._node_keys[idx] = new_key
            if idx == 0:
                self._root_key = new_key

            plaintext = self._pack_node_plaintext(idx=idx)
            ciphertext = self._aes_encrypt(key=new_key, plaintext=plaintext)
            out_path[idx] = [ciphertext]

        self.client.add_write_path(label=self._name, data=out_path)
        if execute:
            result = self.client.execute()
            if not result.success:
                raise RuntimeError(result.error or "Failed to write path to server.")
        self._debug("write_path done: leaf=%d nodes=%d stash=%d bytes", leaf, len(out_path), self._stash_bytes())

    def _enforce_stash_cap(self) -> None:
        stash_bytes = self._stash_bytes()
        self._debug("stash cap check: bytes=%d cap=%d", stash_bytes, self._stash_byte_cap)
        if stash_bytes > self._stash_byte_cap:
            raise MemoryError(
                f"Stash overflow: {stash_bytes} bytes > {self._stash_byte_cap} bytes."
            )
