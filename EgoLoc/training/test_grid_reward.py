#!/usr/bin/env python3
"""Unit tests for grid_reward.py (standard-library unittest only)."""

import math
import unittest

from grid_reward import (
    build_cell_rewards,
    exact_cell_metric,
    gaussian_cell_reward,
    gaussian_reward_for_cell,
    parse_pred_cell,
    parse_success_metric,
)


class ParsePredCellTests(unittest.TestCase):
    def test_parses_trailing_json_after_analysis(self):
        text = 'Contact begins here.\n{"points": [4]}'
        self.assertEqual(parse_pred_cell(text), 4)

    def test_uses_last_matching_json_block(self):
        text = '{"points": [2]} correction {"points": [7]}'
        self.assertEqual(parse_pred_cell(text), 7)

    def test_handles_conversational_completion(self):
        completion = [
            {"role": "assistant", "content": '{"points": [3]}'}
        ]
        self.assertEqual(parse_pred_cell(completion), 3)

    def test_rejects_multiple_points(self):
        self.assertIsNone(parse_pred_cell('{"points": [2, 3]}'))

    def test_rejects_missing_json(self):
        self.assertIsNone(parse_pred_cell("Cell 2"))


class GaussianRewardTests(unittest.TestCase):
    def test_gt_cell_one_vector(self):
        rewards = build_cell_rewards(1, sigma=2.0)
        expected = [
            1.0,
            math.exp(-1 / 8),
            math.exp(-4 / 8),
            math.exp(-9 / 8),
            math.exp(-16 / 8),
            math.exp(-25 / 8),
            math.exp(-36 / 8),
            math.exp(-49 / 8),
            math.exp(-64 / 8),
        ]
        self.assertEqual(len(rewards), 9)
        for actual, wanted in zip(rewards, expected):
            self.assertAlmostEqual(actual, wanted)

    def test_center_vector_is_symmetric(self):
        rewards = build_cell_rewards(5, sigma=2.0)
        self.assertEqual(rewards[4], 1.0)
        self.assertAlmostEqual(rewards[0], rewards[8])
        self.assertAlmostEqual(rewards[1], rewards[7])
        self.assertAlmostEqual(rewards[2], rewards[6])
        self.assertAlmostEqual(rewards[3], rewards[5])

    def test_exact_is_one_for_every_gt_cell(self):
        for gt_cell in range(1, 10):
            with self.subTest(gt_cell=gt_cell):
                self.assertEqual(
                    gaussian_reward_for_cell(gt_cell, gt_cell), 1.0
                )

    def test_invalid_predictions_receive_zero(self):
        for prediction in (None, -1, 0, 10):
            with self.subTest(prediction=prediction):
                self.assertEqual(
                    gaussian_reward_for_cell(prediction, 5), 0.0
                )

    def test_dataset_reward_lookup_callback(self):
        table_one = build_cell_rewards(1)
        table_five = build_cell_rewards(5)
        rewards = gaussian_cell_reward(
            completions=[
                '{"points": [2]}',
                "not json",
                '{"points": [-1]}',
            ],
            cell_rewards=[table_one, table_five, table_five],
            invalid_reward=[0.0, 0.0, 0.0],
        )
        self.assertAlmostEqual(rewards[0], math.exp(-1 / 2))
        self.assertEqual(rewards[1:], [0.0, 0.0])

    def test_diagnostic_metrics_do_not_change_reward(self):
        completions = [
            '{"points": [3]}',
            '{"points": [-1]}',
            "bad",
        ]
        self.assertEqual(
            parse_success_metric(completions), [1.0, 1.0, 0.0]
        )
        self.assertEqual(
            exact_cell_metric(completions, [3, 2, 1]),
            [1.0, 0.0, 0.0],
        )


if __name__ == "__main__":
    unittest.main()
