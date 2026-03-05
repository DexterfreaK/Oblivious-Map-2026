import math

from daoram.dependency import InteractLocalServer
from daoram.omap import BPlusOmap


class RoundCountingInteractLocalServer(InteractLocalServer):
    """Local client/server adapter that counts execute() calls as communication rounds."""

    def __init__(self):
        super().__init__()
        self.total_rounds = 0

    def execute(self):
        self.total_rounds += 1
        return super().execute()


class TestBPlusOmap:
    def test_int_key(self, num_data, client):
        # Create the omap instance.
        omap = BPlusOmap(order=5, num_data=num_data, key_size=10, data_size=10, client=client)

        # Initialize an empty storage.
        omap.init_server_storage()

        # Issue some insert queries.
        for i in range(num_data):
            omap.insert(key=i, value=i)

        # Issue some search update queries.
        for i in range(num_data):
            omap.search(key=i, value=i * 2)

        # Issue some search queries.
        for i in range(num_data):
            assert omap.search(key=i) == i * 2

        # Issue some fast update queries.
        for i in range(num_data):
            omap.fast_search(key=i, value=i * 3)

        # Issue some fast search queries.
        for i in range(num_data):
            assert omap.fast_search(key=i) == i * 3

    def test_str_key(self, num_data, client):
        # Create the omap instance.
        omap = BPlusOmap(order=10, num_data=num_data, key_size=10, data_size=10, client=client)

        # Initialize an empty storage.
        omap.init_server_storage()

        # Issue some insert queries.
        for i in range(num_data):
            omap.insert(key=f"{i}", value=f"{i}")

        # Issue some search queries.
        for i in range(num_data):
            assert omap.search(key=f"{i}") == f"{i}"

        # Issue some fast search queries.
        for i in range(num_data):
            assert omap.fast_search(key=f"{i}") == f"{i}"

    def test_str_key_with_enc(self, num_data, client, encryptor):
        # Create the omap instance.
        omap = BPlusOmap(
            order=10, num_data=num_data, key_size=10, data_size=10, client=client, encryptor=encryptor
        )

        # Initialize an empty storage.
        omap.init_server_storage()

        # Issue some insert queries.
        for i in range(num_data // 10):
            omap.insert(key=f"{i}", value=f"{i}")

        # Issue some search queries.
        for i in range(num_data // 10):
            assert omap.search(key=f"{i}") == f"{i}"

        # Issue some fast search queries.
        for i in range(num_data // 10):
            assert omap.fast_search(key=f"{i}") == f"{i}"

    def test_str_key_with_enc_file(self, num_data, client, encryptor, test_file):
        # Create the omap instance.
        omap = BPlusOmap(
            order=10,
            num_data=num_data,
            key_size=10,
            data_size=10,
            client=client,
            filename=str(test_file),
            encryptor=encryptor
        )

        # Initialize an empty storage.
        omap.init_server_storage()

        # Issue some insert queries.
        for i in range(num_data // 10):
            omap.insert(key=f"{i}", value=f"{i}")

        # Issue some search queries.
        for i in range(num_data // 10):
            assert omap.search(key=f"{i}") == f"{i}"

        # Issue some fast search queries.
        for i in range(num_data // 10):
            assert omap.fast_search(key=f"{i}") == f"{i}"

    def test_data_init(self, num_data, client):
        # Set the init number of data.
        num_init = pow(2, 8)

        # Create the omap instance.
        omap = BPlusOmap(order=20, num_data=num_data, key_size=10, data_size=10, client=client)

        # Initialize storage with a list of key-value pairs.
        omap.init_server_storage(data=[(f"{i}", f"{i}") for i in range(num_init)])

        # Issue some insert queries.
        for i in range(num_init, num_data):
            omap.insert(key=f"{i}", value=f"{i}")

        # Issue some search queries.
        for i in range(num_data):
            assert omap.search(key=f"{i}") == f"{i}"

        # Issue some fast search queries.
        for i in range(num_data):
            assert omap.fast_search(key=f"{i}") == f"{i}"

    def test_mul_data_init(self, num_data, client):
        # Set extra data to insert.
        extra = 3
        size_group = math.floor(math.log2(num_data))
        num_group = num_data // size_group

        # Set the init number of data.
        init_data = [[(j, j) for j in range(i * 2 * size_group, (i * 2 + 1) * size_group)] for i in range(num_group)]

        # Create the omap instance.
        omap = BPlusOmap(order=30, num_data=num_data, key_size=10, data_size=10, client=client)

        # Initialize storage with lists of key-value pairs.
        roots = omap.init_mul_tree_server_storage(data_list=init_data)

        # Insert some data to each subgroup.
        for i, root in enumerate(roots):
            # Update root.
            omap.root = root
            # Insert some new data to this root.
            for j in range(extra):
                omap.insert(key=(i * 2 + 1) * size_group + j, value=(i * 2 + 1) * size_group + j)
            # Update the stored root.
            roots[i] = omap.root

        # For each root, search the key stored in this root.
        for i, root in enumerate(roots):
            # Update root.
            omap.root = root
            # Search for data in this root.
            for j in range(i * 2 * size_group, (i * 2 + 1) * size_group + extra):
                assert omap.search(key=j) == j

            # Fast search for data in this root.
            for j in range(i * 2 * size_group, (i * 2 + 1) * size_group + extra):
                assert omap.fast_search(key=j) == j

    def test_delete_int_key(self, num_data, client):
        # Create the omap instance with lower order to test underflow handling.
        omap = BPlusOmap(order=5, num_data=num_data, key_size=10, data_size=10, client=client)

        # Initialize an empty storage.
        omap.init_server_storage()

        # Insert enough data to create a multi-level tree.
        num_insert = 30
        for i in range(num_insert):
            omap.insert(key=i, value=i)

        # Verify all inserted.
        for i in range(num_insert):
            assert omap.search(key=i) == i

        # Delete every other element (triggers underflow handling).
        for i in range(0, num_insert, 2):
            deleted = omap.delete(key=i)
            assert deleted == i

        # Verify deleted elements return None.
        for i in range(0, num_insert, 2):
            assert omap.search(key=i) is None

        # Verify remaining elements still exist.
        for i in range(1, num_insert, 2):
            assert omap.search(key=i) == i

    def test_delete_all(self, num_data, client):
        # Create the omap instance with lower order to test underflow.
        omap = BPlusOmap(order=5, num_data=num_data, key_size=10, data_size=10, client=client)

        # Initialize an empty storage.
        omap.init_server_storage()

        # Insert enough data to create a multi-level tree.
        num_insert = 20
        for i in range(num_insert):
            omap.insert(key=i, value=i)

        # Delete all in reverse order (triggers cascading underflows).
        for i in range(num_insert - 1, -1, -1):
            deleted = omap.delete(key=i)
            assert deleted == i

        # Verify tree is empty.
        assert omap.root is None

    def test_delete_nonexistent(self, num_data, client):
        # Create the omap instance.
        omap = BPlusOmap(order=5, num_data=num_data, key_size=10, data_size=10, client=client)

        # Initialize an empty storage.
        omap.init_server_storage()

        # Insert some data.
        for i in range(10):
            omap.insert(key=i, value=i)

        # Delete nonexistent key.
        result = omap.delete(key=999)
        assert result is None

        # Verify existing data still intact.
        for i in range(10):
            assert omap.search(key=i) == i

    def test_total_rounds_per_operation(self):
        # Keep this test compact while still covering all requested operations.
        num_data = 64
        num_ops = 8

        client = RoundCountingInteractLocalServer()
        omap = BPlusOmap(order=5, num_data=num_data, key_size=10, data_size=10, client=client)
        omap.init_server_storage()

        round_totals = {
            "insert": 0,
            "search": 0,
            "update": 0,
            "delete": 0,
        }

        for i in range(num_ops):
            before = client.total_rounds
            omap.insert(key=i, value=i)
            round_totals["insert"] += client.total_rounds - before

        for i in range(num_ops):
            before = client.total_rounds
            assert omap.search(key=i) == i
            round_totals["search"] += client.total_rounds - before

        for i in range(num_ops):
            before = client.total_rounds
            assert omap.search(key=i, value=i * 10) == i
            round_totals["update"] += client.total_rounds - before

        for i in range(num_ops):
            before = client.total_rounds
            assert omap.delete(key=i) == i * 10
            round_totals["delete"] += client.total_rounds - before

        print(
            "Total client/server rounds "
            f"(insert={round_totals['insert']}, "
            f"search={round_totals['search']}, "
            f"update={round_totals['update']}, "
            f"delete={round_totals['delete']})"
        )

        for operation, total in round_totals.items():
            assert total > 0
