import os
import random

import pytest

from daoram.oram import TrueVoram


def _collect_chunk_nodes_for_key(oram: TrueVoram, key: int) -> set:
    """Decrypt all nodes and return indices where this key has at least one chunk."""
    chunk_nodes = set()
    pending = [(0, oram._root_key)]
    sentinel = b"\x00" * oram._idlen

    while pending:
        idx, node_key = pending.pop()
        bucket = oram._tree.storage[idx]
        assert len(bucket) == 1

        plaintext = oram._aes_decrypt(key=node_key, ciphertext=bucket[0])

        offset = 0
        left_key = plaintext[offset: offset + oram._keylen]
        offset += oram._keylen
        right_key = plaintext[offset: offset + oram._keylen]
        offset += oram._keylen

        left_idx = 2 * idx + 1
        right_idx = 2 * idx + 2
        if right_idx < oram._tree.size:
            pending.append((right_idx, right_key))
        if left_idx < oram._tree.size:
            pending.append((left_idx, left_key))

        while offset + oram._chunk_header <= oram._Z:
            ident = plaintext[offset: offset + oram._idlen]
            offset += oram._idlen

            if ident == sentinel:
                break

            chunk_len = int.from_bytes(
                plaintext[offset: offset + oram._lenlen], byteorder="big", signed=False
            )
            offset += oram._lenlen
            key_in_node = int.from_bytes(ident, byteorder="big", signed=False) - 1

            if key_in_node == key:
                chunk_nodes.add(idx)

            offset += chunk_len

    return chunk_nodes


class TestTrueVoram:
    def test_basic_path_style(self, client):
        num_data = 64
        oram = TrueVoram(
            num_data=num_data,
            data_size=16,
            client=client,
            optimize=False,
            Z=2048,
            compress=False,
        )
        oram.init_server_storage()

        for i in range(num_data):
            oram.operate_on_key(key=i, value=i)

        for i in range(num_data):
            assert oram.operate_on_key(key=i) == i

        for _ in range(num_data * 3):
            k = random.randint(0, num_data - 1)
            oram.operate_on_key(key=k, value=k * 5)
            assert oram.operate_on_key(key=k) == k * 5

    def test_operate_then_evict(self, client):
        num_data = 32
        oram = TrueVoram(
            num_data=num_data,
            data_size=12,
            client=client,
            optimize=False,
            Z=2048,
            compress=True,
        )
        oram.init_server_storage()

        old_value = oram.operate_on_key_without_eviction(key=7)
        assert isinstance(old_value, bytes)

        new_obj = {"msg": "updated", "v": [1, 2, 3]}
        oram.eviction_with_update_stash(key=7, value=new_obj, execute=True)
        assert oram.operate_on_key(key=7) == new_obj

    def test_large_item_chunked_across_multiple_nodes(self, client):
        num_data = 32
        oram = TrueVoram(
            num_data=num_data,
            data_size=8,
            client=client,
            optimize=False,
            Z=1024,
            compress=False,
        )
        oram.init_server_storage()

        # Big enough that one node cannot hold it.
        payload = os.urandom(9000)
        max_attempts = 32
        observed_multi_node = False
        for _ in range(max_attempts):
            oram.operate_on_key(key=0, value=payload)
            chunk_nodes = _collect_chunk_nodes_for_key(oram=oram, key=0)
            if len(chunk_nodes) >= 2:
                observed_multi_node = True
                break

        assert observed_multi_node, "Did not observe multi-node chunk placement within bounded attempts."
        assert oram.operate_on_key(key=0) == payload

    def test_requires_init_before_access(self, client):
        oram = TrueVoram(
            num_data=8,
            data_size=8,
            client=client,
            optimize=False,
            Z=1024,
        )

        with pytest.raises(ValueError):
            oram.operate_on_key(key=0)

        with pytest.raises(ValueError):
            oram.operate_on_key_without_eviction(key=0)

        with pytest.raises(ValueError):
            oram.eviction_with_update_stash(key=0, value=b"x")

    def test_constructor_parameter_validation(self, client):
        with pytest.raises(ValueError):
            TrueVoram(
                num_data=8,
                data_size=8,
                client=client,
                bucket_size=2,
                optimize=False,
                Z=1024,
            )

        with pytest.raises(ValueError):
            TrueVoram(
                num_data=8,
                data_size=8,
                client=client,
                optimize=False,
                Z=None,
            )

        with pytest.raises(ValueError):
            TrueVoram(
                num_data=8,
                data_size=8,
                client=client,
                optimize=False,
                Z=1030,
            )

    def test_init_with_incomplete_data_map_raises(self, client):
        oram = TrueVoram(
            num_data=8,
            data_size=8,
            client=client,
            optimize=False,
            Z=1024,
        )

        with pytest.raises(KeyError):
            oram.init_server_storage(data_map={0: 100})

    def test_out_of_range_key_raises(self, client):
        num_data = 16
        oram = TrueVoram(
            num_data=num_data,
            data_size=8,
            client=client,
            optimize=False,
            Z=1024,
        )
        oram.init_server_storage()

        with pytest.raises(KeyError):
            oram.operate_on_key(key=-1)

        with pytest.raises(KeyError):
            oram.operate_on_key(key=num_data)

    def test_pending_deferred_access_guards(self, client):
        oram = TrueVoram(
            num_data=16,
            data_size=8,
            client=client,
            optimize=False,
            Z=1024,
        )
        oram.init_server_storage()

        _ = oram.operate_on_key_without_eviction(key=1)

        with pytest.raises(ValueError):
            oram.operate_on_key_without_eviction(key=2)

        with pytest.raises(ValueError):
            oram.operate_on_key(key=2)

        oram.eviction_with_update_stash(key=1, value=123, execute=True)
        assert oram.operate_on_key(key=1) == 123

    def test_eviction_with_missing_stash_key_raises(self, client):
        num_data = 16
        oram = TrueVoram(
            num_data=num_data,
            data_size=8,
            client=client,
            optimize=False,
            Z=1024,
        )
        oram.init_server_storage()
        _ = oram.operate_on_key_without_eviction(key=0)

        with pytest.raises(KeyError):
            oram.eviction_with_update_stash(key=num_data + 1, value=b"bad")

    def test_data_map_initialization_roundtrip(self, client):
        num_data = 16
        data_map = {
            i: {"k": i, "blob": bytes([i % 256]) * (i + 1), "hello": "world"}
            for i in range(num_data)
        }
        oram = TrueVoram(
            num_data=num_data,
            data_size=8,
            client=client,
            optimize=False,
            Z=2048,
            compress=True,
        )
        oram.init_server_storage(data_map=data_map)

        for i in range(num_data):
            a = oram.operate_on_key(key=i)
            assert a["hello"] == "world"
            assert a["k"] == i
            assert a["blob"] == bytes([i % 256]) * (i + 1)

    def test_deferred_eviction_execute_false_flushes_on_next_execute(self, client):
        oram = TrueVoram(
            num_data=16,
            data_size=8,
            client=client,
            optimize=False,
            Z=1024,
            compress=True,
        )
        oram.init_server_storage()

        _ = oram.operate_on_key_without_eviction(key=3)
        oram.eviction_with_update_stash(key=3, value={"lazy": True}, execute=False)

        # This next call executes pending write first, then performs its own read/write cycle.
        assert oram.operate_on_key(key=3) == {"lazy": True}

    def test_stash_overflow_raises(self, client):
        oram = TrueVoram(
            num_data=4,
            data_size=8,
            client=client,
            optimize=False,
            Z=1024,
            stash_scale=1,
            compress=False,
        )
        oram.init_server_storage()

        with pytest.raises(MemoryError):
            oram.operate_on_key(key=0, value=os.urandom(50000))
