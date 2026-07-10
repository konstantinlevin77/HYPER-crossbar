import unittest
import sys
import types

import torch
from torch_geometric.data import Data

# The sampling tests do not exercise weighted-graph construction. Permit them to
# run in lightweight environments where this optional compiled package is absent.
try:
    import torch_scatter  # noqa: F401
except ImportError:
    torch_scatter = types.ModuleType("torch_scatter")
    torch_scatter.scatter_add = None
    sys.modules["torch_scatter"] = torch_scatter

from hyper import tasks


class StrictTypedNegativeSamplingTest(unittest.TestCase):
    def setUp(self):
        # Relation 0 has position types {1, 2, 3} x {4, 5}.
        # Relation 1 has disjoint position types {6, 7} x {8, 9}.
        self.data = Data(
            edge_index=torch.tensor([
                [1, 2, 3, 6, 7],
                [4, 4, 5, 8, 9],
            ]),
            edge_type=torch.tensor([0, 0, 0, 1, 1]),
            num_nodes=10,
        )

    def test_mask_keeps_same_relation_and_position_type(self):
        batch = torch.tensor([
            [1, 4, 0],
            [6, 8, 1],
        ])

        masks = tasks.strict_typed_negative_mask(self.data, batch)

        # Entity 2 has the correct type but is another known true completion for
        # (?, 4, relation 0), so strict filtering also removes it.
        self.assertEqual(masks[0][0].nonzero().flatten().tolist(), [3])
        self.assertEqual(masks[0][1].nonzero().flatten().tolist(), [7])
        self.assertEqual(masks[1][0].nonzero().flatten().tolist(), [5])
        self.assertEqual(masks[1][1].nonzero().flatten().tolist(), [9])

    def test_sampler_only_draws_typed_strict_candidates(self):
        torch.manual_seed(0)
        batch = torch.tensor([[1, 4, 0]])

        sampled = tasks.negative_sampling(
            self.data,
            batch,
            num_negative=20,
            max_positions_per_edge=1,
            sampling_mode="strict_typed",
        )

        corrupted_position = torch.nonzero(
            torch.any(sampled[:-1, 0, 1:] != sampled[:-1, 0, :1], dim=1),
            as_tuple=False,
        ).item()
        allowed = {3} if corrupted_position == 0 else {5}
        self.assertTrue(set(sampled[corrupted_position, 0, 1:].tolist()) <= allowed)
        self.assertNotIn(0, sampled[corrupted_position, 0, 1:].tolist())

    def test_legacy_strict_flag_still_works(self):
        batch = torch.tensor([[1, 4, 0]])
        sampled = tasks.negative_sampling(self.data, batch, 2, strict=True)
        self.assertEqual(sampled.shape[-1], 3)

    def test_unknown_mode_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Unknown negative sampling mode"):
            tasks.negative_sampling(
                self.data, torch.tensor([[1, 4, 0]]), 2,
                sampling_mode="not_a_mode",
            )


if __name__ == "__main__":
    unittest.main()
