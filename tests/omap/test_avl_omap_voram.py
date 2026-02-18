import pytest

from daoram.omap import AVLOmapVoram


class TestAVLOmapVoram:
    def test_int_key_roundtrip(self, client):
        num_data = 32
        omap = AVLOmapVoram(num_data=num_data, key_size=10, data_size=10, client=client)
        omap.init_server_storage()

        for i in range(num_data):
            omap.insert(key=i * 2, value=i)

        for i in range(num_data):
            omap.search(key=i * 2, value=i * 2)

        for i in range(num_data):
            assert omap.search(key=i * 2) == i * 2

        for i in range(num_data):
            omap.fast_search(key=i * 2, value=i * 3)

        for i in range(num_data):
            assert omap.fast_search(key=i * 2) == i * 3

    def test_complex_values_roundtrip(self, client):
        num_data = 24
        omap = AVLOmapVoram(num_data=num_data, key_size=10, data_size=10, client=client)
        omap.init_server_storage()

        for i in range(num_data):
            omap.insert(key=i, value={"id": i, "tags": [i, i + 1], "payload": bytes([i % 256]) * (i + 1)})

        for i in range(num_data):
            value = omap.search(key=i)
            assert value["id"] == i
            assert value["tags"] == [i, i + 1]
            assert value["payload"] == bytes([i % 256]) * (i + 1)

    def test_data_init(self, client):
        num_data = 40
        init_count = 16
        omap = AVLOmapVoram(num_data=num_data, key_size=10, data_size=10, client=client)
        omap.init_server_storage(data=[(i, f"v-{i}") for i in range(init_count)])

        for i in range(init_count, num_data):
            omap.insert(key=i, value=f"v-{i}")

        for i in range(num_data):
            assert omap.search(key=i) == f"v-{i}"

    def test_mul_data_init(self, client):
        num_data = 48
        group_count = 3
        group_size = 8
        extra = 2

        init_data = [
            [(g * 100 + i, g * 100 + i) for i in range(group_size)]
            for g in range(group_count)
        ]

        omap = AVLOmapVoram(num_data=num_data, key_size=10, data_size=10, client=client)
        roots = omap.init_mul_tree_server_storage(data_list=init_data)

        for g, root in enumerate(roots):
            omap.root = root
            for i in range(extra):
                key = g * 100 + group_size + i
                omap.insert(key=key, value=key)
            roots[g] = omap.root

        for g, root in enumerate(roots):
            omap.root = root
            for i in range(group_size + extra):
                key = g * 100 + i
                assert omap.search(key=key) == key
                assert omap.fast_search(key=key) == key

    def test_delete_and_delete_all(self, client):
        num_data = 32
        omap = AVLOmapVoram(num_data=num_data, key_size=10, data_size=10, client=client)
        omap.init_server_storage()

        for i in range(num_data):
            omap.insert(key=i, value=i)

        for i in range(0, num_data, 2):
            assert omap.delete(key=i) == i

        for i in range(num_data):
            if i % 2 == 0:
                assert omap.search(key=i) is None
            else:
                assert omap.search(key=i) == i

        omap.init_server_storage()
        for i in range(num_data):
            omap.insert(key=i, value=i)
        for i in range(num_data):
            assert omap.delete(key=i) == i
        assert omap.root is None

    def test_constructor_voram_options(self, client):
        omap = AVLOmapVoram(
            num_data=16,
            key_size=10,
            data_size=10,
            client=client,
            voram_optimize=False,
            voram_Z=1024,
            voram_compress=False,
            voram_keylen=32,
            voram_suggested_params=False,
        )

        omap.init_server_storage()
        omap.insert(key=1, value="value-1")
        assert omap.search(key=1) == "value-1"

    def test_with_encryptor_and_file(self, client, encryptor, test_file):
        omap = AVLOmapVoram(
            num_data=16,
            key_size=10,
            data_size=10,
            client=client,
            filename=str(test_file),
            encryptor=encryptor,
        )

        omap.init_server_storage()
        for i in range(8):
            omap.insert(key=i, value=i)
        for i in range(8):
            assert omap.search(key=i) == i

    def test_non_int_keys_rejected(self, client):
        omap = AVLOmapVoram(num_data=16, key_size=10, data_size=10, client=client)
        omap.init_server_storage()

        with pytest.raises(TypeError):
            omap.insert(key="k1", value="v1")
