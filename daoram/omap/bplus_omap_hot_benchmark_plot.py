"""Render PNG plots for B+ hot-cache benchmark JSON output."""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

DEFAULT_PLOT_BUCKET_SIZE = 16

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError as exc:  # pragma: no cover - exercised only when plotting deps are installed.
    raise SystemExit(
        "matplotlib is required to render benchmark plots. Install it before running this plotter."
    ) from exc


def _load_benchmark_json(path: Path) -> Dict[str, Any]:
    with path.open("r") as handle:
        return json.load(handle)


def _bucket_average(values: Sequence[float], bucket_size: int) -> Tuple[List[float], List[float]]:
    if bucket_size <= 0:
        raise ValueError("plot_bucket_size must be positive.")

    x_values: List[float] = []
    y_values: List[float] = []
    for start in range(0, len(values), bucket_size):
        bucket = values[start:start + bucket_size]
        if not bucket:
            continue
        x_values.append(start + (len(bucket) - 1) / 2.0)
        y_values.append(sum(bucket) / len(bucket))
    return x_values, y_values


def _warmup_divider_x(config: Dict[str, Any]) -> float:
    warmup_requests = int(config.get("warmup_requests", 0))
    return warmup_requests - 0.5


def _run_specs(payload: Dict[str, Any]) -> Iterable[Tuple[str, str, str]]:
    return (
        ("score_based_hot_cache", "ScoreBased", "#0b6e4f"),
        ("reject_all_hot_cache", "RejectAll", "#8d3b72"),
        ("plain_bplus_search", "Plain BPlus.search", "#1f4e79"),
    )


def _plot_rounds_timeline(payload: Dict[str, Any], output_dir: Path, bucket_size: int) -> Path:
    fig, ax = plt.subplots(figsize=(12, 6))
    for run_key, label, color in _run_specs(payload):
        run = payload["runs"][run_key]
        rounds = [float(value) for value in run["series"]["rounds"]]
        rolling = [float(value) for value in run["series"]["rolling_avg_rounds"]]
        bucket_x, bucket_y = _bucket_average(rounds, bucket_size)

        ax.plot(bucket_x, bucket_y, label=f"{label} bucketed rounds", color=color, alpha=0.35, linewidth=2)
        ax.plot(range(len(rolling)), rolling, label=f"{label} rolling avg", color=color, linewidth=2)

    ax.axvline(_warmup_divider_x(payload["config"]), color="#444444", linestyle="--", linewidth=1)
    ax.set_title("Request Timeline vs ORAM Rounds")
    ax.set_xlabel("Request index")
    ax.set_ylabel("ORAM rounds")
    ax.legend()
    ax.grid(alpha=0.2)

    output_path = output_dir / "request_rounds_timeline.png"
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return output_path


def _plot_hit_rate_timeline(payload: Dict[str, Any], output_dir: Path) -> Path:
    fig, ax = plt.subplots(figsize=(12, 6))
    for run_key, label, color in _run_specs(payload):
        run = payload["runs"][run_key]
        rolling = [float(value) for value in run["series"]["rolling_cache_hit_rate"]]
        ax.plot(range(len(rolling)), rolling, label=label, color=color, linewidth=2)

    ax.axvline(_warmup_divider_x(payload["config"]), color="#444444", linestyle="--", linewidth=1)
    ax.set_title("Rolling Cache Hit Rate")
    ax.set_xlabel("Request index")
    ax.set_ylabel("Rolling cache hit rate")
    ax.legend()
    ax.grid(alpha=0.2)

    output_path = output_dir / "cache_hit_rate_timeline.png"
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return output_path


def _plot_cache_events_timeline(payload: Dict[str, Any], output_dir: Path, bucket_size: int) -> Path:
    fig, ax = plt.subplots(figsize=(12, 6))
    event_specs = (
        ("score_based_hot_cache", "Score promotions", "promotions_delta", "#0b6e4f", "-"),
        ("score_based_hot_cache", "Score evictions", "evictions_delta", "#0b6e4f", "--"),
        ("reject_all_hot_cache", "RejectAll promotions", "promotions_delta", "#8d3b72", "-"),
        ("reject_all_hot_cache", "RejectAll evictions", "evictions_delta", "#8d3b72", "--"),
    )
    for run_key, label, series_name, color, linestyle in event_specs:
        run = payload["runs"][run_key]
        series = [float(value) for value in run["series"][series_name]]
        bucket_x, bucket_y = _bucket_average(series, bucket_size)
        ax.plot(bucket_x, bucket_y, label=label, color=color, linestyle=linestyle, linewidth=2)

    ax.axvline(_warmup_divider_x(payload["config"]), color="#444444", linestyle="--", linewidth=1)
    ax.set_title("Average Promotions and Evictions per Request Bucket")
    ax.set_xlabel("Request index")
    ax.set_ylabel("Average events per request")
    ax.legend()
    ax.grid(alpha=0.2)

    output_path = output_dir / "cache_event_timeline.png"
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return output_path


def render_plots(input_json: Path, output_dir: Path, plot_bucket_size: int) -> List[Path]:
    payload = _load_benchmark_json(input_json)
    output_dir.mkdir(parents=True, exist_ok=True)

    return [
        _plot_rounds_timeline(payload=payload, output_dir=output_dir, bucket_size=plot_bucket_size),
        _plot_hit_rate_timeline(payload=payload, output_dir=output_dir),
        _plot_cache_events_timeline(payload=payload, output_dir=output_dir, bucket_size=plot_bucket_size),
    ]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Render PNG plots for B+ hot-cache benchmark output JSON.")
    parser.add_argument("input_json", type=Path, help="Benchmark JSON file produced by bplus_omap_hot_benchmark.py.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for generated PNG files. Defaults to <input_json_stem>_plots.",
    )
    parser.add_argument(
        "--plot-bucket-size",
        type=int,
        default=DEFAULT_PLOT_BUCKET_SIZE,
        help="Number of requests to average together when plotting bucketed series.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    output_dir = args.output_dir or args.input_json.with_name(f"{args.input_json.stem}_plots")
    render_plots(input_json=args.input_json, output_dir=output_dir, plot_bucket_size=args.plot_bucket_size)


if __name__ == "__main__":
    main()
