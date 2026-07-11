import unittest
import sys
from pathlib import Path

import torch

# Support both `python tests/test_node_specific_evaluation.py` and module-based
# test discovery without requiring the repository to be installed as a package.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hyper import evaluation


class NodeSpecificTypedEvaluationTest(unittest.TestCase):
    def test_rule_schema_is_hardcoded(self):
        expected = ("disease", "gene", "drug", "pathway")
        self.assertEqual(evaluation.node_types_for_dataset("HYPERRULE1"), expected)
        self.assertEqual(evaluation.node_types_for_dataset("HYPERRULE2"), expected)

    def test_metrics_are_grouped_by_position(self):
        rankings = evaluation.new_position_rankings()
        evaluation.add_position_rankings(rankings, 0, torch.tensor([1, 2, 4]))
        evaluation.add_position_rankings(rankings, 1, torch.tensor([1, 3]))

        metrics = evaluation.node_specific_typed_metrics(
            rankings, ("disease", "gene"), torch.device("cpu"), distributed=False
        )

        self.assertEqual(metrics["typed/disease/count"], 3)
        self.assertAlmostEqual(metrics["typed/disease/mrr"].item(), (1 + 0.5 + 0.25) / 3)
        self.assertAlmostEqual(metrics["typed/disease/hits@3"].item(), 2 / 3)
        self.assertEqual(metrics["typed/gene/hits@1"].item(), 0.5)
        self.assertEqual(metrics["typed/gene/hits@10"].item(), 1.0)

    def test_empty_positions_are_omitted(self):
        metrics = evaluation.node_specific_typed_metrics(
            evaluation.new_position_rankings(), ("disease",), torch.device("cpu"), distributed=False
        )
        self.assertEqual(metrics, {})


if __name__ == "__main__":
    unittest.main()
