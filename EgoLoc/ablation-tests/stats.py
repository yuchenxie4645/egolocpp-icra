#!/usr/bin/env python3
"""Aggregate N=3 trials and compute paired OccluBench statistics."""

import argparse
import hashlib
import json
import math
from itertools import combinations
from pathlib import Path

from manifest import load_manifest
from results_store import read_jsonl_tolerant, result_key

HERE = Path(__file__).resolve().parent
DEFAULT_MANIFEST = HERE / "data" / "manifest.json"
DEFAULT_RESULTS = HERE / "results" / "trials.jsonl"
DEFAULT_OUTPUT = HERE / "results"
TASKS = ("contact", "separation")
MODES = ("speed", "pinch", "speed_pinch")
TRIALS = (0, 1, 2)


def rounded_mean_prediction(values):
    """Locked policy: ``int(round(np.mean(valid)))`` (banker's ties-to-even)."""
    import numpy as np

    valid = [int(value) for value in values if value is not None and int(value) >= 0]
    if not valid:
        return None
    return int(round(float(np.mean(valid))))


def bootstrap_mean_ci(values, seed=20260908, n_boot=10000, alpha=0.05):
    import numpy as np

    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return (None, None)
    if values.size == 1:
        value = float(values[0])
        return (value, value)
    rng = np.random.default_rng(int(seed))
    indices = rng.integers(0, values.size, size=(int(n_boot), values.size))
    means = values[indices].mean(axis=1)
    low, high = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    return float(low), float(high)


def paired_bootstrap_difference(
    values_a,
    values_b,
    seed=20260908,
    n_boot=10000,
    alpha=0.05,
):
    import numpy as np

    a = np.asarray(values_a, dtype=float)
    b = np.asarray(values_b, dtype=float)
    if a.shape != b.shape:
        raise ValueError("paired bootstrap arrays must have equal shape")
    finite = np.isfinite(a) & np.isfinite(b)
    differences = a[finite] - b[finite]
    return bootstrap_mean_ci(
        differences, seed=seed, n_boot=n_boot, alpha=alpha
    )


def holm_adjust(p_values):
    """Holm family-wise adjusted p-values; None/NaN remain None."""
    adjusted = [None] * len(p_values)
    finite = []
    for index, value in enumerate(p_values):
        if value is not None and math.isfinite(float(value)):
            finite.append((index, float(value)))
    finite.sort(key=lambda item: item[1])
    m = len(finite)
    running = 0.0
    for rank, (index, value) in enumerate(finite):
        candidate = min(1.0, (m - rank) * value)
        running = max(running, candidate)
        adjusted[index] = running
    return adjusted


def matched_rank_biserial(differences):
    """Signed matched rank-biserial effect; positive means B has lower error."""
    import numpy as np
    from scipy.stats import rankdata

    values = np.asarray(differences, dtype=float)
    values = values[np.isfinite(values) & (values != 0)]
    if values.size == 0:
        return 0.0
    ranks = rankdata(np.abs(values), method="average")
    positive = float(ranks[values > 0].sum())
    negative = float(ranks[values < 0].sum())
    denominator = positive + negative
    return (positive - negative) / denominator if denominator else 0.0


def robust_wilcoxon(differences):
    import numpy as np
    from scipy.stats import wilcoxon

    values = np.asarray(differences, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return None, None
    if np.all(values == 0):
        return 0.0, 1.0
    try:
        result = wilcoxon(
            values,
            zero_method="wilcox",
            alternative="two-sided",
            method="auto",
        )
        return float(result.statistic), float(result.pvalue)
    except ValueError:
        return 0.0, 1.0


def _seed_for(base, *parts):
    digest = hashlib.sha256(
        "|".join([str(base), *map(str, parts)]).encode("utf-8")
    ).hexdigest()
    return int(digest[:8], 16)


def _expected_keys(manifest):
    return {
        result_key(record["episode"], task, mode, trial)
        for record in manifest["episodes"]
        for task in TASKS
        if record["labels"].get(
            "contact_local" if task == "contact" else "separate_local"
        )
        is not None
        for mode in MODES
        for trial in TRIALS
    }


def _finite_or_none(value):
    if value is None:
        return None
    value = float(value)
    return value if math.isfinite(value) else None


def _metric_summary(rows, bootstrap_seed, n_boot):
    import numpy as np

    total = len(rows)
    successful = [row for row in rows if row["prediction_frame"] is not None]
    mae = np.asarray([row["absolute_error"] for row in successful], dtype=float)
    nmae = np.asarray(
        [row["normalized_absolute_error"] for row in successful], dtype=float
    )
    mae_ci = bootstrap_mean_ci(mae, seed=bootstrap_seed, n_boot=n_boot)
    nmae_ci = bootstrap_mean_ci(
        nmae, seed=bootstrap_seed + 1, n_boot=n_boot
    )
    recall_values = [
        float(row["candidate_grid_recall"])
        for row in rows
        if row.get("candidate_grid_recall") is not None
    ]
    return {
        "total_events": total,
        "successful_events": len(successful),
        "failed_events": total - len(successful),
        "failure_rate": (total - len(successful)) / total if total else None,
        "mean_frame_mae": float(mae.mean()) if mae.size else None,
        "median_frame_mae": float(np.median(mae)) if mae.size else None,
        "frame_mae_ci95": list(mae_ci),
        "mean_normalized_mae": float(nmae.mean()) if nmae.size else None,
        "median_normalized_mae": float(np.median(nmae)) if nmae.size else None,
        "normalized_mae_ci95": list(nmae_ci),
        "candidate_grid_recall": (
            float(np.mean(recall_values)) if recall_values else None
        ),
    }


def _build_rows(manifest, records, allow_partial):
    by_episode = {record["episode"]: record for record in manifest["episodes"]}
    trial_rows = []
    grouped = {}
    for record in records:
        episode = record["episode"]
        task = record["task"]
        mode = record["mode"]
        trial = int(record["trial"])
        prediction = (
            int(record["prediction_frame"])
            if record.get("status") == "success"
            and record.get("prediction_frame") is not None
            else None
        )
        error = record.get("error") or {}
        trial_row = {
            "key": record["key"],
            "episode": episode,
            "task": task,
            "mode": mode,
            "trial": trial,
            "seed": record.get("seed"),
            "status": record.get("status"),
            "prediction_frame": prediction,
            "ground_truth_local": record.get("ground_truth_local"),
            "num_frames": record.get("num_frames"),
            "error_class": error.get("class"),
            "error_message": error.get("message"),
            "latency_s": record.get("latency_s"),
            "request_count": len(record.get("request_traces", [])),
            "speed_cues": record.get("cue_coverage", {}).get("speed_n"),
            "pinch_cues": record.get("cue_coverage", {}).get("pinch_n"),
            "neutral_fallback": record.get("cue_coverage", {}).get("neutral_n"),
            "candidate_grid_recall": record.get("candidate_grid_recall"),
            "candidate_grid_distance_to_ground_truth": record.get(
                "candidate_grid_distance_to_ground_truth"
            ),
        }
        trial_rows.append(trial_row)
        grouped.setdefault((episode, task, mode), []).append(trial_row)

    event_rows = []
    for episode_record in manifest["episodes"]:
        episode = episode_record["episode"]
        for task in TASKS:
            raw_ground_truth = episode_record["labels"].get(
                "contact_local" if task == "contact" else "separate_local"
            )
            if raw_ground_truth is None:
                continue
            ground_truth = int(raw_ground_truth)
            for mode in MODES:
                rows = grouped.get((episode, task, mode), [])
                valid_predictions = [
                    row["prediction_frame"]
                    for row in rows
                    if row["prediction_frame"] is not None
                ]
                prediction = rounded_mean_prediction(valid_predictions)
                recall_values = [
                    row["candidate_grid_recall"]
                    for row in rows
                    if row.get("candidate_grid_recall") is not None
                ]
                candidate_recall = (
                    bool(recall_values[0]) if recall_values else None
                )
                if recall_values and any(
                    bool(value) != candidate_recall
                    for value in recall_values
                ):
                    raise RuntimeError(
                        f"candidate grid changed across paired trials for "
                        f"{episode}/{task}/{mode}"
                    )
                absolute_error = (
                    abs(prediction - ground_truth)
                    if prediction is not None
                    else None
                )
                denominator = max(1, int(episode_record["num_frames"]) - 1)
                normalized = (
                    absolute_error / denominator
                    if absolute_error is not None
                    else None
                )
                event_rows.append(
                    {
                        "episode": episode,
                        "task": task,
                        "mode": mode,
                        "ground_truth_local": ground_truth,
                        "prediction_frame": prediction,
                        "absolute_error": absolute_error,
                        "normalized_absolute_error": normalized,
                        "valid_trials": len(valid_predictions),
                        "failed_trials": len(rows) - len(valid_predictions),
                        "recorded_trials": len(rows),
                        "expected_trials": 3,
                        "event_success": prediction is not None,
                        "candidate_grid_recall": candidate_recall,
                        "partial": bool(allow_partial and len(rows) != 3),
                    }
                )
    return trial_rows, event_rows


def _pairwise(event_rows, bootstrap_seed, n_boot):
    import pandas as pd

    frame = pd.DataFrame(event_rows)
    rows = []
    for scope in (*TASKS, "combined"):
        scoped = frame if scope == "combined" else frame[frame["task"] == scope]
        for mode_a, mode_b in combinations(MODES, 2):
            a = scoped[scoped["mode"] == mode_a][
                ["episode", "task", "absolute_error"]
            ].rename(columns={"absolute_error": "error_a"})
            b = scoped[scoped["mode"] == mode_b][
                ["episode", "task", "absolute_error"]
            ].rename(columns={"absolute_error": "error_b"})
            common = a.merge(b, on=["episode", "task"], how="inner").dropna()
            values_a = common["error_a"].to_numpy(dtype=float)
            values_b = common["error_b"].to_numpy(dtype=float)
            differences = values_a - values_b
            ci = paired_bootstrap_difference(
                values_a,
                values_b,
                seed=_seed_for(bootstrap_seed, scope, mode_a, mode_b),
                n_boot=n_boot,
            )
            statistic, p_value = robust_wilcoxon(differences)
            rows.append(
                {
                    "scope": scope,
                    "mode_a": mode_a,
                    "mode_b": mode_b,
                    "common_successful_events": len(common),
                    "mean_mae_difference_a_minus_b": (
                        float(differences.mean())
                        if len(differences)
                        else None
                    ),
                    "difference_ci95_low": ci[0],
                    "difference_ci95_high": ci[1],
                    "wilcoxon_statistic": statistic,
                    "wilcoxon_p_raw": p_value,
                    "matched_rank_biserial": matched_rank_biserial(differences),
                }
            )
    adjusted = holm_adjust([row["wilcoxon_p_raw"] for row in rows])
    for row, p_adjusted in zip(rows, adjusted):
        row["wilcoxon_p_holm"] = p_adjusted
        difference = row["mean_mae_difference_a_minus_b"]
        low = row["difference_ci95_low"]
        high = row["difference_ci95_high"]
        if (
            difference is not None
            and difference < 0
            and high is not None
            and high < 0
            and p_adjusted is not None
            and p_adjusted < 0.05
        ):
            row["conclusion"] = f"{row['mode_a']} superior"
        elif (
            difference is not None
            and difference > 0
            and low is not None
            and low > 0
            and p_adjusted is not None
            and p_adjusted < 0.05
        ):
            row["conclusion"] = f"{row['mode_b']} superior"
        else:
            row["conclusion"] = "inconclusive/mode-dependent"
    return rows


def _write_plots(output_dir, summary, trial_rows):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    scopes = ["contact", "separation", "combined"]
    x = np.arange(len(scopes))
    width = 0.24
    colors = ["#4C78A8", "#F58518", "#54A24B"]

    fig, ax = plt.subplots(figsize=(9, 5))
    for index, mode in enumerate(MODES):
        means = [
            summary["metrics"][scope][mode]["mean_frame_mae"]
            for scope in scopes
        ]
        lows = [
            summary["metrics"][scope][mode]["frame_mae_ci95"][0]
            for scope in scopes
        ]
        highs = [
            summary["metrics"][scope][mode]["frame_mae_ci95"][1]
            for scope in scopes
        ]
        clean = np.array([value if value is not None else np.nan for value in means])
        err = np.array(
            [
                [
                    value - low if value is not None and low is not None else 0
                    for value, low in zip(means, lows)
                ],
                [
                    high - value if value is not None and high is not None else 0
                    for value, high in zip(means, highs)
                ],
            ]
        )
        ax.bar(
            x + (index - 1) * width,
            clean,
            width,
            yerr=err,
            label=mode,
            color=colors[index],
            capsize=3,
        )
    ax.set_xticks(x, scopes)
    ax.set_ylabel("Frame MAE (95% paired-event bootstrap CI)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "mae_with_ci.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5))
    for index, mode in enumerate(MODES):
        values = [
            summary["metrics"][scope][mode]["mean_normalized_mae"]
            for scope in scopes
        ]
        ax.bar(
            x + (index - 1) * width,
            [value if value is not None else np.nan for value in values],
            width,
            label=mode,
            color=colors[index],
        )
    ax.set_xticks(x, scopes)
    ax.set_ylabel("Normalized MAE")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "normalized_mae.png", dpi=180)
    plt.close(fig)

    fallback = {}
    for mode in MODES:
        values = [
            row["neutral_fallback"]
            for row in trial_rows
            if row["mode"] == mode and row["neutral_fallback"] is not None
        ]
        fallback[mode] = float(np.mean(values)) if values else np.nan
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    combined = summary["metrics"]["combined"]
    axes[0].bar(
        MODES,
        [combined[mode]["failure_rate"] for mode in MODES],
        color=colors,
    )
    axes[0].set_ylabel("Event failure rate")
    axes[0].tick_params(axis="x", rotation=20)
    axes[1].bar(
        MODES, [fallback[mode] for mode in MODES], color=colors
    )
    axes[1].set_ylabel("Mean neutral fallback frames / trial")
    axes[1].tick_params(axis="x", rotation=20)
    fig.tight_layout()
    fig.savefig(output_dir / "failure_and_cue_fallback.png", dpi=180)
    plt.close(fig)


def generate_statistics(
    manifest_path=DEFAULT_MANIFEST,
    results_path=DEFAULT_RESULTS,
    output_dir=DEFAULT_OUTPUT,
    bootstrap_seed=20260908,
    bootstrap_samples=10000,
    allow_partial=False,
):
    try:
        import numpy as np  # noqa: F401
        import pandas as pd
        import scipy  # noqa: F401
        import matplotlib  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(
            "stats.py requires numpy, pandas, scipy, and matplotlib in egolocxyc"
        ) from exc

    manifest = load_manifest(manifest_path, validate=True, check_sources=False)
    records, torn = read_jsonl_tolerant(results_path)
    keys = [record.get("key") for record in records]
    duplicates = len(keys) != len(set(keys))
    expected = _expected_keys(manifest)
    missing = expected - set(keys)
    if (torn or duplicates or missing) and not allow_partial:
        raise RuntimeError(
            "results are incomplete/invalid; run audit.py first or pass "
            "--allow-partial for explicitly preliminary outputs "
            f"(missing={len(missing)}, duplicates={duplicates}, torn={torn})"
        )

    trial_rows, event_rows = _build_rows(manifest, records, allow_partial)
    metrics = {}
    for scope in (*TASKS, "combined"):
        metrics[scope] = {}
        for mode in MODES:
            scoped = [
                row
                for row in event_rows
                if row["mode"] == mode
                and (scope == "combined" or row["task"] == scope)
            ]
            metrics[scope][mode] = _metric_summary(
                scoped,
                _seed_for(bootstrap_seed, "metric", scope, mode),
                bootstrap_samples,
            )
    pairwise = _pairwise(event_rows, bootstrap_seed, bootstrap_samples)
    summary = {
        "dataset": "OccluBench",
        "preliminary": bool(allow_partial and (missing or torn or duplicates)),
        "complete": not (missing or torn or duplicates),
        "record_count": len(records),
        "missing_trial_keys": len(missing),
        "available_events": manifest["expected"]["events"],
        "unavailable_labels_ignored": manifest["expected"][
            "unavailable_labels"
        ],
        "ignored_unavailable": manifest["missing_labels"],
        "rounding_policy": (
            "int(round(np.mean(valid_predictions))); Python round uses "
            "banker's ties-to-even"
        ),
        "aggregation": (
            "N=3 trial predictions are averaged per event/mode using valid "
            "trials only, then rounded with the locked policy."
        ),
        "weighting": (
            "OccluBench only: sample-weighted and dataset-macro are identical "
            "because there is one dataset."
        ),
        "bootstrap": {
            "unit": "event",
            "paired_for_differences": True,
            "samples": int(bootstrap_samples),
            "seed": int(bootstrap_seed),
            "ci": 0.95,
        },
        "wilcoxon_family": (
            "9 comparisons: 3 mode pairs x contact/separation/combined; "
            "one Holm correction family"
        ),
        "metrics": metrics,
        "pairwise_tests": pairwise,
    }

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(trial_rows).to_csv(
        output_dir / "trial_results.csv", index=False
    )
    pd.DataFrame(event_rows).to_csv(
        output_dir / "event_results.csv", index=False
    )
    pd.DataFrame(pairwise).to_csv(
        output_dir / "pairwise_tests.csv", index=False
    )
    with open(output_dir / "summary.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")

    table = [
        "# OccluBench EgoLoc sampling ablation",
        "",
        "| Task | Mode | Success/total | Grid recall | Frame MAE (95% CI) | Median MAE | Normalized MAE | Failure |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for scope in (*TASKS, "combined"):
        for mode in MODES:
            metric = metrics[scope][mode]
            mean = metric["mean_frame_mae"]
            low, high = metric["frame_mae_ci95"]
            mae_text = (
                f"{mean:.3f} [{low:.3f}, {high:.3f}]"
                if None not in (mean, low, high)
                else "NA"
            )
            median = metric["median_frame_mae"]
            nmae = metric["mean_normalized_mae"]
            failure = metric["failure_rate"]
            table.append(
                f"| {scope} | {mode} | "
                f"{metric['successful_events']}/{metric['total_events']} | "
                f"{metric['candidate_grid_recall']:.3%} | "
                f"{mae_text} | "
                f"{median:.3f} | " if median is not None else
                f"| {scope} | {mode} | "
                f"{metric['successful_events']}/{metric['total_events']} | "
                f"{metric['candidate_grid_recall']:.3%} | "
                f"{mae_text} | NA | "
            )
            table[-1] += (
                f"{nmae:.5f} | {failure:.3%} |"
                if nmae is not None and failure is not None
                else "NA | NA |"
            )
    (output_dir / "table.md").write_text(
        "\n".join(table) + "\n", encoding="utf-8"
    )

    conclusions = [
        f"- {row['scope']}: {row['mode_a']} vs {row['mode_b']} — "
        f"{row['conclusion']} (Holm p={row['wilcoxon_p_holm']})"
        for row in pairwise
    ]
    report_lines = [
        "# OccluBench EgoLoc sampling-ablation report",
        "",
        (
            "**PRELIMINARY / INCOMPLETE.**"
            if summary["preliminary"]
            else "**Complete locked-protocol analysis.**"
        ),
        "",
        summary["aggregation"],
        "",
        summary["weighting"],
        "",
        (
            f"{summary['available_events']} available events were scored; "
            f"{summary['unavailable_labels_ignored']} explicitly unavailable "
            "labels were ignored without imputation."
        ),
        "",
        "Superiority is stated only when the MAE direction, paired bootstrap "
        "difference CI excluding zero, and Holm-corrected Wilcoxon p<0.05 agree.",
        "",
        "## Pairwise conclusions",
        "",
        *conclusions,
    ]
    (output_dir / "REPORT.md").write_text(
        "\n".join(report_lines) + "\n", encoding="utf-8"
    )
    _write_plots(output_dir, summary, trial_rows)
    return summary


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--results", default=str(DEFAULT_RESULTS))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--bootstrap-seed", type=int, default=20260908)
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--allow-partial", action="store_true")
    return parser.parse_args(argv)


def main(argv=None):
    args = _parse_args(argv)
    summary = generate_statistics(
        manifest_path=args.manifest,
        results_path=args.results,
        output_dir=args.output_dir,
        bootstrap_seed=args.bootstrap_seed,
        bootstrap_samples=args.bootstrap_samples,
        allow_partial=args.allow_partial,
    )
    print(
        json.dumps(
            {
                "complete": summary["complete"],
                "preliminary": summary["preliminary"],
                "record_count": summary["record_count"],
                "output_dir": args.output_dir,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
