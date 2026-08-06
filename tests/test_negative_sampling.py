import unittest
from unittest import mock
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


class PreparedEdgeMatchTest(unittest.TestCase):
    def setUp(self):
        self.data = Data(
            edge_index=torch.tensor([
                [1, 1, 2, 3],
                [2, 2, 3, 1],
            ]),
            edge_type=torch.tensor([0, 0, 1, 1]),
            num_nodes=4,
        )

    def test_prepared_lookup_preserves_duplicates_and_missing_queries(self):
        query_index = torch.tensor([
            [1, 2, 3],
            [2, 3, 3],
            [0, 1, 0],
        ])

        edge_ids, counts = tasks.edge_match_prepared(
            tasks.get_graph_edge_match_index(self.data), query_index
        )

        self.assertEqual(counts.tolist(), [2, 1, 0])
        self.assertEqual(edge_ids.tolist(), [0, 1, 2])

    def test_graph_cache_reuses_each_logical_index(self):
        full = tasks.get_graph_edge_match_index(self.data)
        without_first = tasks.get_graph_edge_match_index(
            self.data, excluded_position=0
        )

        self.assertIs(full, tasks.get_graph_edge_match_index(self.data))
        self.assertIs(
            without_first,
            tasks.get_graph_edge_match_index(self.data, excluded_position=0),
        )
        self.assertIsNot(
            without_first,
            tasks.get_graph_edge_match_index(self.data, excluded_position=1),
        )
        self.assertNotIn("_edge_match_cache", self.data.keys())

    def test_graph_cache_is_invalidated_after_in_place_mutation(self):
        original = tasks.get_graph_edge_match_index(
            self.data, excluded_position=0
        )
        self.data.edge_index[0, 0] = 3

        rebuilt = tasks.get_graph_edge_match_index(
            self.data, excluded_position=0
        )

        self.assertIsNot(original, rebuilt)

    def test_strict_mask_prepares_a_position_only_once(self):
        batch = torch.tensor([[1, 2, 0]])
        positions = torch.tensor([0])

        with mock.patch.object(
            tasks, "prepare_edge_match", wraps=tasks.prepare_edge_match
        ) as prepare:
            tasks.strict_negative_mask(self.data, batch, positions=positions)
            tasks.strict_negative_mask(self.data, batch, positions=positions)

        self.assertEqual(prepare.call_count, 1)


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

    def test_position_allowlist_keeps_position_zero_fixed(self):
        batch = torch.tensor([[1, 4, 0]])
        sampled = tasks.negative_sampling(
            self.data,
            batch,
            num_negative=20,
            max_positions_per_edge=1,
            sampling_mode="strict_typed",
            corrupt_positions=[1],
        )

        self.assertTrue(torch.all(sampled[0] == 1))
        self.assertTrue(torch.all(sampled[1, 0, 1:] == 5))

    def test_position_allowlist_rejects_out_of_range_positions(self):
        with self.assertRaisesRegex(ValueError, "allowed positions must be between"):
            tasks.get_active_positions(
                torch.tensor([[1, 4]]), allowed_positions=[2]
            )

    def test_active_positions_are_intersected_with_allowlist(self):
        positions = tasks.get_active_positions(
            torch.tensor([[1, 2, 3, 4]]), allowed_positions=[1, 2, 3]
        )
        self.assertEqual(positions.tolist(), [1, 2, 3])

    def test_unknown_mode_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Unknown negative sampling mode"):
            tasks.negative_sampling(
                self.data, torch.tensor([[1, 4, 0]]), 2,
                sampling_mode="not_a_mode",
            )


if __name__ == "__main__":
    unittest.main()
