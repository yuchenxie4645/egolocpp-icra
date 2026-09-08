import json

import pandas as pd

import stats
from tests.test_audit import write_complete_results
from tests.test_manifest_convert import build_fixture_manifest


def test_locked_rounding_policy_uses_python_ties_to_even():
    assert stats.rounded_mean_prediction([1, 2]) == 2
    assert stats.rounded_mean_prediction([2, 3]) == 2
    assert stats.rounded_mean_prediction([1, None, -1, 3]) == 2
    assert stats.rounded_mean_prediction([None, -1]) is None


def test_bootstrap_is_deterministic_and_paired():
    first = stats.bootstrap_mean_ci([1, 2, 3, 4], seed=7, n_boot=500)
    second = stats.bootstrap_mean_ci([1, 2, 3, 4], seed=7, n_boot=500)
    paired = stats.paired_bootstrap_difference(
        [2, 4, 6], [1, 3, 5], seed=3, n_boot=100
    )

    assert first == second
    assert first[0] <= 2.5 <= first[1]
    assert paired == (1.0, 1.0)


def test_wilcoxon_rank_biserial_and_holm_edge_cases():
    statistic, p_value = stats.robust_wilcoxon([0, 0, 0])

    assert statistic == 0.0
    assert p_value == 1.0
    assert stats.matched_rank_biserial([1, 2, 3]) == 1.0
    assert stats.matched_rank_biserial([-1, -2, -3]) == -1.0
    assert stats.holm_adjust([0.01, 0.04, 0.03]) == [0.03, 0.06, 0.06]


def test_complete_statistics_outputs_and_retains_trial_failure(tmp_path):
    _, manifest_path, _ = build_fixture_manifest(
        tmp_path, episodes=1, frames=12
    )
    results_path = tmp_path / "results.jsonl"
    write_complete_results(manifest_path, results_path)
    records = [
        json.loads(line) for line in results_path.read_text().splitlines()
    ]
    records[0]["status"] = "terminal_failure"
    records[0]["prediction_frame"] = None
    records[0]["error"] = {"class": "Synthetic", "message": "kept"}
    results_path.write_text(
        "\n".join(json.dumps(record) for record in records) + "\n"
    )
    output = tmp_path / "stats"

    summary = stats.generate_statistics(
        manifest_path=manifest_path,
        results_path=results_path,
        output_dir=output,
        bootstrap_samples=200,
    )

    assert summary["complete"] is True
    assert summary["metrics"]["combined"]["speed"]["total_events"] == 2
    assert all(
        row["conclusion"] == "inconclusive/mode-dependent"
        for row in summary["pairwise_tests"]
    )
    trials = pd.read_csv(output / "trial_results.csv")
    assert (trials["status"] == "terminal_failure").sum() == 1
    assert (output / "event_results.csv").exists()
    assert (output / "pairwise_tests.csv").exists()
    assert (output / "table.md").exists()
    assert (output / "REPORT.md").exists()
    assert (output / "mae_with_ci.png").exists()
    assert (output / "normalized_mae.png").exists()
    assert (output / "failure_and_cue_fallback.png").exists()
