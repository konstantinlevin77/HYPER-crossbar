import os
import random
import numpy as np
from collections import defaultdict
from pathlib import Path

SEED = 42
TRAIN_RATIO = 0.80
VAL_RATIO = 0.10
TEST_RATIO = 0.10

BASE_DIR = Path("hypergraph_dataset")
RULES = ["HYPER-RULE1", "HYPER-RULE2"]

random.seed(SEED)
np.random.seed(SEED)


def read_all_lines(rule_dir):
    """Read train, valid, test files and return all lines + disease-to-line mapping."""
    disease_to_lines = defaultdict(list)
    total = 0
    for split_file in ["train.txt", "valid.txt", "test.txt"]:
        path = rule_dir / split_file
        print(f"  Reading {path.name} ...")
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split("\t")
                disease_field = parts[1]  # e.g. "disease:286"
                disease_id = disease_field.split(":")[1]
                disease_to_lines[disease_id].append(line)
                total += 1
    return disease_to_lines, total


def stratified_split_lines(disease_to_lines):
    """Split diseases 80/10/10 and collect the corresponding lines."""
    disease_ids = sorted(disease_to_lines.keys())
    random.shuffle(disease_ids)

    n_total = len(disease_ids)
    n_train = int(np.ceil(n_total * TRAIN_RATIO))
    n_val = int(np.ceil(n_total * VAL_RATIO))

    train_diseases = set(disease_ids[:n_train])
    val_diseases = set(disease_ids[n_train : n_train + n_val])
    test_diseases = set(disease_ids[n_train + n_val :])

    train_lines = []
    val_lines = []
    test_lines = []

    for disease in train_diseases:
        train_lines.extend(disease_to_lines[disease])
    for disease in val_diseases:
        val_lines.extend(disease_to_lines[disease])
    for disease in test_diseases:
        test_lines.extend(disease_to_lines[disease])

    random.shuffle(train_lines)
    random.shuffle(val_lines)
    random.shuffle(test_lines)

    return (
        train_lines,
        val_lines,
        test_lines,
        train_diseases,
        val_diseases,
        test_diseases,
    )


def write_split(output_dir, train_lines, val_lines, test_lines):
    output_dir.mkdir(parents=True, exist_ok=True)
    for fname, lines in [
        ("train.txt", train_lines),
        ("valid.txt", val_lines),
        ("test.txt", test_lines),
    ]:
        path = output_dir / fname
        print(f"  Writing {path.name} ({len(lines):,} lines) ...")
        with open(path, "w") as f:
            for line in lines:
                f.write(line + "\n")


def verify_no_leakage(
    train_diseases, val_diseases, test_diseases, total_edges, train_edges, val_edges, test_edges
):
    overlaps = (
        (train_diseases & val_diseases)
        | (train_diseases & test_diseases)
        | (val_diseases & test_diseases)
    )
    if overlaps:
        print(f"  *** LEAKAGE DETECTED: {len(overlaps)} diseases in multiple splits!")
        return False
    print(f"  Disease overlap check: PASSED (0 overlapping diseases)")
    print(f"  Edge distribution: train={train_edges/total_edges*100:.2f}%  "
          f"val={val_edges/total_edges*100:.2f}%  test={test_edges/total_edges*100:.2f}%")
    return True


for rule in RULES:
    print(f"\n{'='*60}")
    print(f"Processing {rule}")
    print(f"{'='*60}")

    rule_dir = BASE_DIR / rule
    output_dir = BASE_DIR / f"{rule}_STF"

    disease_to_lines, total_lines = read_all_lines(rule_dir)
    n_diseases = len(disease_to_lines)
    print(f"  Total edges: {total_lines:,}")
    print(f"  Unique diseases: {n_diseases:,}")

    edges_per_disease = {d: len(lines) for d, lines in disease_to_lines.items()}
    print(f"  Edges per disease: min={min(edges_per_disease.values()):,}, "
          f"max={max(edges_per_disease.values()):,}, "
          f"mean={np.mean(list(edges_per_disease.values())):,.0f}, "
          f"median={np.median(list(edges_per_disease.values())):,.0f}")

    train_lines, val_lines, test_lines, train_diseases, val_diseases, test_diseases = \
        stratified_split_lines(disease_to_lines)

    train_edges = len(train_lines)
    val_edges = len(val_lines)
    test_edges = len(test_lines)

    print(f"\n  --- Split stats ---")
    for name, n, dset in [("train", train_edges, train_diseases),
                           ("val", val_edges, val_diseases),
                           ("test", test_edges, test_diseases)]:
        print(f"  {name}: {n:,} edges from {len(dset):,} diseases")

    print(f"\n  --- Writing {output_dir} ---")
    write_split(output_dir, train_lines, val_lines, test_lines)

    print(f"\n  --- Verification ---")
    verify_no_leakage(
        train_diseases, val_diseases, test_diseases,
        total_lines, train_edges, val_edges, test_edges,
    )

print(f"\n{'='*60}")
print("Done.")
