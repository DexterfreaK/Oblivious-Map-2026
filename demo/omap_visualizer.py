#!/usr/bin/env python3
"""
OMAP Visualizer

Generates an HTML file that visualizes:
  - Logical OMAP tree (left)
  - Physical ORAM tree(s) (right)

Runs a small scenario: init -> insert -> search.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from daoram.dependency import Data, InteractLocalServer  # noqa: E402
from daoram.dependency.binary_tree import BinaryTree  # noqa: E402
from daoram.omap import (  # noqa: E402
    AVLOmap,
    AVLOmapCached,
    BPlusOmap,
    BPlusOmapCached,
    GroupOmap,
    OramOstOmap,
)
from daoram.oram import DAOram, PathOram  # noqa: E402


MAX_BUCKET_LINES = 4
MAX_LIST_ITEMS = 4
MAX_TEXT_LEN = 18


@dataclass
class Graph:
    nodes: Dict[Any, str]
    edges: List[Tuple[Any, Any]]
    root: Optional[Any]


def _short(value: Any, max_len: int = MAX_TEXT_LEN) -> str:
    if value is None:
        return "None"
    if isinstance(value, bytes):
        text = value.hex()
    else:
        text = str(value)
    return text if len(text) <= max_len else text[: max_len - 3] + "..."


def _short_list(values: Iterable[Any], max_items: int = MAX_LIST_ITEMS) -> str:
    items = list(values)
    shown = items[:max_items]
    body = ", ".join(_short(v) for v in shown)
    if len(items) > max_items:
        body += ", ..."
    return body


def _is_int_like(value: str) -> bool:
    if value is None:
        return False
    value = value.strip()
    if value.startswith("-"):
        return value[1:].isdigit()
    return value.isdigit()


def _parse_keys(raw_keys: List[str]) -> List[Any]:
    if raw_keys and all(_is_int_like(k) for k in raw_keys):
        return [int(k) for k in raw_keys]
    return raw_keys


def _parse_pairs(raw_pairs: List[Tuple[str, str]]) -> List[Tuple[Any, Any]]:
    keys = _parse_keys([k for k, _ in raw_pairs])
    parsed = []
    for (key, value), parsed_key in zip(raw_pairs, keys):
        parsed.append((parsed_key, value))
    return parsed


def _collect_tree_nodes(tree: BinaryTree) -> Dict[Any, Data]:
    nodes: Dict[Any, Data] = {}
    for idx in range(tree.size):
        bucket = tree.storage[idx]
        for block in bucket:
            if isinstance(block, Data) and block.key is not None:
                nodes[block.key] = block
    return nodes


def _collect_stash_nodes(ost: Any) -> Dict[Any, Data]:
    nodes: Dict[Any, Data] = {}
    if hasattr(ost, "_stash"):
        for block in getattr(ost, "_stash"):
            if isinstance(block, Data) and block.key is not None:
                nodes[block.key] = block
    if hasattr(ost, "_local"):
        local = getattr(ost, "_local")
        if isinstance(local, list):
            for block in local:
                if isinstance(block, Data) and block.key is not None:
                    nodes[block.key] = block
        elif hasattr(local, "to_list"):
            for block in local.to_list():
                if isinstance(block, Data) and block.key is not None:
                    nodes[block.key] = block
    return nodes


def _build_avl_graph(nodes_by_key: Dict[Any, Data], root_key: Any) -> Graph:
    edges: List[Tuple[Any, Any]] = []
    nodes: Dict[Any, str] = {}
    for key, data in nodes_by_key.items():
        val = data.value
        if not hasattr(val, "l_key"):
            continue
        label_lines = [
            f"key: {_short(key)}",
            f"leaf: {_short(data.leaf)}",
            f"val: {_short(getattr(val, 'value', None))}",
        ]
        if getattr(val, "l_key", None) is not None:
            label_lines.append(f"L: {_short(val.l_key)}")
            edges.append((key, val.l_key))
        if getattr(val, "r_key", None) is not None:
            label_lines.append(f"R: {_short(val.r_key)}")
            edges.append((key, val.r_key))
        nodes[key] = "\n".join(label_lines)
    return Graph(nodes=nodes, edges=edges, root=root_key)


def _build_bplus_graph(nodes_by_key: Dict[Any, Data], root_key: Any) -> Graph:
    edges: List[Tuple[Any, Any]] = []
    nodes: Dict[Any, str] = {}
    for key, data in nodes_by_key.items():
        val = data.value
        if not hasattr(val, "keys") or not hasattr(val, "values"):
            continue
        keys = val.keys or []
        values = val.values or []
        is_leaf = len(keys) == len(values)
        label_lines = [
            f"id: {_short(key)}",
            f"leaf: {_short(data.leaf)}",
            f"keys: [{_short_list(keys)}]",
        ]
        if is_leaf:
            label_lines.append(f"vals: [{_short_list(values)}]")
        else:
            child_keys = []
            for child in values:
                if isinstance(child, tuple) and len(child) == 2:
                    child_keys.append(child[0])
            label_lines.append(f"children: [{_short_list(child_keys)}]")
            for child_key in child_keys:
                edges.append((key, child_key))
        nodes[key] = "\n".join(label_lines)
    return Graph(nodes=nodes, edges=edges, root=root_key)


def _build_group_graph(group: GroupOmap, client: InteractLocalServer) -> Graph:
    # Logical view: root -> bucket metadata nodes
    root_id = "buckets"
    nodes: Dict[Any, str] = {root_id: "GroupOmap\nbuckets"}
    edges: List[Tuple[Any, Any]] = []

    upper_label = group._upper_oram._name  # pylint: disable=protected-access
    tree = client._storage.get(upper_label)
    if not isinstance(tree, BinaryTree):
        return Graph(nodes=nodes, edges=edges, root=root_id)

    buckets: Dict[int, Tuple[int, bytes, List[Any]]] = {}
    for block in _collect_tree_nodes(tree).values():
        try:
            count, seed, keys = group._decode_metadata(block.value)
            buckets[block.key] = (count, seed, keys)
        except Exception:
            continue

    for bucket_id in sorted(buckets.keys()):
        count, seed, keys = buckets[bucket_id]
        node_id = f"bucket:{bucket_id}"
        label = "\n".join([
            f"bucket {bucket_id}",
            f"count: {count}",
            f"seed: {_short(seed)}",
            f"keys: [{_short_list(keys)}]",
        ])
        nodes[node_id] = label
        edges.append((root_id, node_id))

    return Graph(nodes=nodes, edges=edges, root=root_id)


def _build_logical_graph(omap: Any, client: InteractLocalServer) -> Graph:
    if isinstance(omap, OramOstOmap):
        return _build_logical_graph(omap._ost, client)  # pylint: disable=protected-access
    if isinstance(omap, GroupOmap):
        return _build_group_graph(omap, client)

    if not hasattr(omap, "root"):
        return Graph(nodes={}, edges=[], root=None)

    label = omap._name  # pylint: disable=protected-access
    tree = client._storage.get(label)
    if not isinstance(tree, BinaryTree):
        return Graph(nodes={}, edges=[], root=None)

    nodes_by_key = _collect_tree_nodes(tree)
    nodes_by_key.update(_collect_stash_nodes(omap))

    if omap.root is None:
        return Graph(nodes={}, edges=[], root=None)

    root_key = omap.root[0]
    root_node = nodes_by_key.get(root_key)
    if root_node is None:
        return Graph(nodes={}, edges=[], root=None)

    value = root_node.value
    if hasattr(value, "l_key"):
        return _build_avl_graph(nodes_by_key, root_key)
    if hasattr(value, "keys"):
        return _build_bplus_graph(nodes_by_key, root_key)
    return Graph(nodes={}, edges=[], root=None)


def _build_oram_graph(tree: BinaryTree) -> Graph:
    nodes: Dict[Any, str] = {}
    edges: List[Tuple[Any, Any]] = []
    for idx in range(tree.size):
        bucket = tree.storage[idx]
        lines = [f"idx: {idx}", f"blocks: {len(bucket)}"]
        if bucket:
            for block in bucket[:MAX_BUCKET_LINES]:
                if isinstance(block, Data):
                    lines.append(f"{_short(block.key)} -> {_short(block.leaf)}")
                else:
                    lines.append("enc")
            if len(bucket) > MAX_BUCKET_LINES:
                lines.append("...")
        else:
            lines.append("(empty)")
        nodes[idx] = "\n".join(lines)
        if idx > 0:
            parent = BinaryTree.get_parent_index(idx)
            edges.append((parent, idx))
    return Graph(nodes=nodes, edges=edges, root=0 if tree.size > 0 else None)


def _layout_from_edges(graph: Graph) -> Tuple[Dict[Any, Tuple[float, int]], int]:
    children: Dict[Any, List[Any]] = {}
    for parent, child in graph.edges:
        children.setdefault(parent, []).append(child)

    positions: Dict[Any, Tuple[float, int]] = {}
    max_depth = 0
    x_counter = 0

    def dfs(node: Any, depth: int) -> float:
        nonlocal x_counter, max_depth
        max_depth = max(max_depth, depth)
        child_nodes = [c for c in children.get(node, []) if c in graph.nodes]
        if not child_nodes:
            x = float(x_counter)
            x_counter += 1
        else:
            child_xs = [dfs(child, depth + 1) for child in child_nodes]
            x = sum(child_xs) / len(child_xs)
        positions[node] = (x, depth)
        return x

    if graph.root is None or graph.root not in graph.nodes:
        return {}, 0
    dfs(graph.root, 0)
    return positions, max_depth


def _layout_binary_tree(tree: BinaryTree) -> Tuple[Dict[int, Tuple[float, int]], int]:
    positions: Dict[int, Tuple[float, int]] = {}
    if tree.size == 0:
        return positions, 0

    max_depth = int(math.floor(math.log2(tree.size)))
    for idx in range(tree.size):
        depth = int(math.floor(math.log2(idx + 1)))
        first = (2 ** depth) - 1
        pos_in_level = idx - first
        # Center each node above its subtree width to keep levels aligned.
        span = 2 ** (max_depth - depth)
        x = (pos_in_level + 0.5) * span
        positions[idx] = (float(x), depth)
    return positions, max_depth


def _render_svg(
    graph: Graph,
    positions: Dict[Any, Tuple[float, int]],
    max_depth: int,
    title: str,
    node_width: int,
    node_height: int,
    level_height: int,
    h_margin: int = 40,
    v_margin: int = 40,
) -> str:
    if not graph.nodes or graph.root is None:
        return (
            '<svg width="600" height="200" xmlns="http://www.w3.org/2000/svg">'
            '<rect x="0" y="0" width="100%" height="100%" rx="12" fill="#f5f2ed" stroke="#b9aa9a"/>'
            '<text x="50%" y="50%" text-anchor="middle" font-size="16" fill="#6a5d52">Empty</text>'
            "</svg>"
        )

    x_values = [pos[0] for pos in positions.values()]
    min_x, max_x = min(x_values), max(x_values)
    span = max_x - min_x
    width = max(800, int((span + 1) * node_width * 1.2))
    height = v_margin * 2 + (max_depth + 1) * level_height

    def to_xy(node_id: Any) -> Tuple[float, float]:
        x_unit, depth = positions[node_id]
        if span == 0:
            x = width / 2
        else:
            x = h_margin + (x_unit - min_x) / span * (width - h_margin * 2)
        y = v_margin + depth * level_height
        return x, y

    lines = [
        f'<svg width="{width}" height="{height}" viewBox="0 0 {width} {height}" '
        'xmlns="http://www.w3.org/2000/svg">',
        f'<rect x="0" y="0" width="{width}" height="{height}" rx="16" fill="#fefaf4" stroke="#c8bfb3"/>',
        f'<text x="{h_margin}" y="24" font-size="14" fill="#6a5d52">{title}</text>',
    ]

    # Draw edges
    for parent, child in graph.edges:
        if parent not in positions or child not in positions:
            continue
        x1, y1 = to_xy(parent)
        x2, y2 = to_xy(child)
        lines.append(
            f'<line x1="{x1}" y1="{y1 + node_height / 2}" x2="{x2}" y2="{y2 - node_height / 2}" '
            'stroke="#b9aa9a" stroke-width="1.2" />'
        )

    # Draw nodes
    for node_id, label in graph.nodes.items():
        if node_id not in positions:
            continue
        x, y = to_xy(node_id)
        x0 = x - node_width / 2
        y0 = y - node_height / 2
        lines.append(
            f'<rect x="{x0}" y="{y0}" width="{node_width}" height="{node_height}" rx="10" '
            'fill="#ffffff" stroke="#8f7f73" stroke-width="1.2" />'
        )
        label_lines = label.split("\n")
        text_y = y0 + 18
        for line in label_lines[:6]:
            lines.append(
                f'<text x="{x}" y="{text_y}" text-anchor="middle" font-size="11" fill="#4f433a">'
                f'{line}</text>'
            )
            text_y += 14
        if len(label_lines) > 6:
            lines.append(
                f'<text x="{x}" y="{text_y}" text-anchor="middle" font-size="11" fill="#4f433a">...</text>'
            )

    lines.append("</svg>")
    return "".join(lines)


def _capture_snapshot(title: str, omap: Any, client: InteractLocalServer) -> Dict[str, Any]:
    logical_graph = _build_logical_graph(omap, client)
    logical_positions, logical_depth = _layout_from_edges(logical_graph)
    logical_svg = _render_svg(
        logical_graph,
        logical_positions,
        logical_depth,
        title="Logical OMAP",
        node_width=160,
        node_height=86,
        level_height=110,
    )

    physical_svgs: Dict[str, str] = {}
    for label, storage in client._storage.items():  # pylint: disable=protected-access
        if not isinstance(storage, BinaryTree):
            continue
        oram_graph = _build_oram_graph(storage)
        oram_positions, oram_depth = _layout_binary_tree(storage)
        physical_svgs[label] = _render_svg(
            oram_graph,
            oram_positions,
            oram_depth,
            title=f"Physical ORAM: {label}",
            node_width=140,
            node_height=68,
            level_height=90,
        )

    return {
        "title": title,
        "logical_svg": logical_svg,
        "physical_svgs": physical_svgs,
    }


def _build_html(title: str, snapshots: List[Dict[str, Any]], physical_labels: List[str]) -> str:
    data = {
        "snapshots": snapshots,
        "physical_labels": physical_labels,
    }
    json_data = json.dumps(data, ensure_ascii=True)

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{title}</title>
  <style>
    :root {{
      --paper: #f7f0e8;
      --ink: #3d332b;
      --muted: #6e6055;
      --accent: #2e6f5a;
      --panel: #fffaf3;
      --border: #d2c7bb;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: "Avenir Next", "Avenir", "Helvetica Neue", "Segoe UI", sans-serif;
      color: var(--ink);
      background: linear-gradient(160deg, #f7f0e8 0%, #f3e7da 45%, #eadfce 100%);
    }}
    header {{
      padding: 24px 32px 8px 32px;
    }}
    h1 {{
      margin: 0 0 6px 0;
      font-size: 22px;
      letter-spacing: 0.3px;
    }}
    .subtitle {{
      color: var(--muted);
      font-size: 13px;
    }}
    .controls {{
      display: flex;
      gap: 16px;
      flex-wrap: wrap;
      padding: 12px 32px 18px 32px;
    }}
    label {{
      font-size: 12px;
      color: var(--muted);
      text-transform: uppercase;
      letter-spacing: 0.08em;
    }}
    select {{
      margin-top: 6px;
      padding: 6px 10px;
      border-radius: 8px;
      border: 1px solid var(--border);
      background: #fff;
      font-size: 14px;
    }}
    .panel {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));
      gap: 16px;
      padding: 0 32px 32px 32px;
    }}
    .pane {{
      background: var(--panel);
      border: 1px solid var(--border);
      border-radius: 16px;
      padding: 14px;
      box-shadow: 0 10px 24px rgba(60, 42, 26, 0.08);
      min-height: 360px;
      overflow: auto;
      position: relative;
      animation: fadeIn 300ms ease-in;
    }}
    .pane h2 {{
      margin: 0 0 10px 0;
      font-size: 16px;
      color: var(--accent);
    }}
    .pane svg {{
      display: block;
    }}
    @keyframes fadeIn {{
      from {{ opacity: 0; transform: translateY(6px); }}
      to {{ opacity: 1; transform: translateY(0); }}
    }}
  </style>
</head>
<body>
  <header>
    <h1>{title}</h1>
    <div class="subtitle">Left: logical OMAP tree · Right: physical ORAM tree</div>
  </header>

  <div class="controls">
    <div>
      <label for="snapshotSelect">Snapshot</label><br>
      <select id="snapshotSelect"></select>
    </div>
    <div id="physicalLabelWrap" style="display:none;">
      <label for="physicalSelect">Physical ORAM</label><br>
      <select id="physicalSelect"></select>
    </div>
  </div>

  <div class="panel">
    <div class="pane">
      <h2>Logical OMAP</h2>
      <div id="logicalPane"></div>
    </div>
    <div class="pane">
      <h2>Physical ORAM</h2>
      <div id="physicalPane"></div>
    </div>
  </div>

  <script>
    const data = {json_data};
    const snapshotSelect = document.getElementById("snapshotSelect");
    const physicalSelect = document.getElementById("physicalSelect");
    const physicalWrap = document.getElementById("physicalLabelWrap");
    const logicalPane = document.getElementById("logicalPane");
    const physicalPane = document.getElementById("physicalPane");

    data.snapshots.forEach((snap, idx) => {{
      const opt = document.createElement("option");
      opt.value = idx;
      opt.textContent = snap.title;
      snapshotSelect.appendChild(opt);
    }});

    if (data.physical_labels.length > 1) {{
      physicalWrap.style.display = "block";
      data.physical_labels.forEach(label => {{
        const opt = document.createElement("option");
        opt.value = label;
        opt.textContent = label;
        physicalSelect.appendChild(opt);
      }});
    }}

    function render() {{
      const snap = data.snapshots[Number(snapshotSelect.value)];
      logicalPane.innerHTML = snap.logical_svg;
      const label = data.physical_labels.length > 1 ? physicalSelect.value : data.physical_labels[0];
      physicalPane.innerHTML = snap.physical_svgs[label] || "<div>No ORAM view available.</div>";
    }}

    snapshotSelect.addEventListener("change", render);
    physicalSelect.addEventListener("change", render);
    snapshotSelect.value = "0";
    if (data.physical_labels.length > 0) {{
      physicalSelect.value = data.physical_labels[0];
    }}
    render();
  </script>
</body>
</html>
"""


def _create_omap(
    omap_type: str,
    num_data: int,
    order: int,
    key_size: int,
    data_size: int,
    client: InteractLocalServer,
) -> Any:
    if omap_type == "avl":
        return AVLOmap(num_data=num_data, key_size=key_size, data_size=data_size, client=client)
    if omap_type == "avl-cache":
        return AVLOmapCached(num_data=num_data, key_size=key_size, data_size=data_size, client=client)
    if omap_type == "bplus":
        return BPlusOmap(order=order, num_data=num_data, key_size=key_size, data_size=data_size, client=client)
    if omap_type == "bplus-cache":
        return BPlusOmapCached(order=order, num_data=num_data, key_size=key_size, data_size=data_size, client=client)
    if omap_type == "daoram-avl":
        ost = AVLOmap(num_data=num_data, key_size=key_size, data_size=data_size, client=client, name="avl")
        oram = DAOram(num_data=num_data, data_size=data_size, client=client, name="da")
        return OramOstOmap(num_data=num_data, ost=ost, oram=oram)
    if omap_type == "daoram-bplus":
        ost = BPlusOmap(order=order, num_data=num_data, key_size=key_size, data_size=data_size, client=client, name="bplus")
        oram = DAOram(num_data=num_data, data_size=data_size, client=client, name="da")
        return OramOstOmap(num_data=num_data, ost=ost, oram=oram)
    if omap_type == "group":
        upper = PathOram(num_data=num_data, data_size=data_size, client=client, name="group_upper")
        return GroupOmap(
            num_data=num_data,
            key_size=key_size,
            data_size=data_size,
            upper_oram=upper,
            client=client,
            name="group_omap",
        )
    raise ValueError(f"Unknown omap type: {omap_type}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate an OMAP visualizer HTML")
    parser.add_argument(
        "--type",
        default="avl",
        choices=[
            "avl",
            "avl-cache",
            "bplus",
            "bplus-cache",
            "daoram-avl",
            "daoram-bplus",
            "group",
        ],
        help="OMAP type",
    )
    parser.add_argument("--num-data", type=int, default=16, help="Number of entries (small is better for viz)")
    parser.add_argument("--order", type=int, default=4, help="B+ tree order (for bplus variants)")
    parser.add_argument(
        "--kv",
        action="append",
        nargs=2,
        metavar=("KEY", "VALUE"),
        help="Insert key/value pair (repeatable). Example: --kv 1 alpha --kv 2 beta",
    )
    parser.add_argument(
        "--auto",
        type=int,
        default=0,
        help="Insert keys 0..N-1 with value equal to key (overrides --kv/--key)",
    )
    parser.add_argument("--key", default="1", help="Key to insert/search (single-pair fallback)")
    parser.add_argument("--value", default="value", help="Value to insert (single-pair fallback)")
    parser.add_argument(
        "--search-all",
        action="store_true",
        help="Capture a search snapshot for every inserted key (can be many)",
    )
    parser.add_argument("--key-size", type=int, default=8, help="Dummy key size in bytes")
    parser.add_argument("--data-size", type=int, default=8, help="Dummy data size in bytes")
    parser.add_argument("--out", default="demo/omap_visualizer.html", help="Output HTML file")
    args = parser.parse_args()

    client = InteractLocalServer()
    omap = _create_omap(
        omap_type=args.type,
        num_data=args.num_data,
        order=args.order,
        key_size=args.key_size,
        data_size=args.data_size,
        client=client,
    )

    # Build insert pairs
    pairs: List[Tuple[Any, Any]] = []
    if args.auto and args.auto > 0:
        pairs = [(i, i) for i in range(args.auto)]
    elif args.kv:
        pairs = _parse_pairs(args.kv)
    else:
        try:
            parsed_key: Any = int(args.key)
        except (TypeError, ValueError):
            parsed_key = args.key
        pairs = [(parsed_key, args.value)]

    # Initialize storage
    omap.init_server_storage()

    snapshots = []
    snapshots.append(_capture_snapshot("init storage", omap, client))

    # Insert
    last_key: Any = None
    for key, value in pairs:
        omap.insert(key=key, value=value)
        snapshots.append(_capture_snapshot(f"insert key={key}", omap, client))
        last_key = key

    # Search
    if args.search_all:
        for key, _ in pairs:
            omap.search(key=key)
            snapshots.append(_capture_snapshot(f"search key={key}", omap, client))
    elif last_key is not None:
        omap.search(key=last_key)
        snapshots.append(_capture_snapshot(f"search key={last_key}", omap, client))

    # Resolve physical labels
    physical_labels = [label for label, storage in client._storage.items() if isinstance(storage, BinaryTree)]
    if not physical_labels:
        physical_labels = ["(none)"]

    html = _build_html(f"OMAP Visualizer ({args.type})", snapshots, physical_labels)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write(html)

    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
