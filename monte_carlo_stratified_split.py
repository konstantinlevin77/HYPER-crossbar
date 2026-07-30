import csv
import random
import numpy as np
from collections import defaultdict
from pathlib import Path

SEED = 42
SIMULATIONS = 10_000
TRAIN_RATIO = 0.80
VAL_RATIO = 0.10
TEST_RATIO = 0.10

DATA_DIR = Path("DATA_GENERATION_V3/final_hyperedges")
RULE1_PATH = DATA_DIR / "rule1_hyperedges.csv"
RULE2_PATH = DATA_DIR / "rule2_hyperedges.csv"

def build_disease_edge_map(csv_path):
    disease_counts = defaultdict(int)
    total = 0
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            disease_counts[row["Disease"]] += 1
            total += 1
    return dict(disease_counts), total

print("Reading rule1_hyperedges.csv ...")
rule1_disease_map, rule1_total = build_disease_edge_map(RULE1_PATH)
print(f"  {rule1_total:,} edges, {len(rule1_disease_map):,} unique diseases")

print("Reading rule2_hyperedges.csv ...")
rule2_disease_map, rule2_total = build_disease_edge_map(RULE2_PATH)
print(f"  {rule2_total:,} edges, {len(rule2_disease_map):,} unique diseases")

all_diseases = set(rule1_disease_map) | set(rule2_disease_map)
print(f"Total unique diseases across both: {len(all_diseases):,}")

def count_edges_for_disease(disease):
    return rule1_disease_map.get(disease, 0) + rule2_disease_map.get(disease, 0)

disease_edges = {d: count_edges_for_disease(d) for d in all_diseases}
total_edges = sum(disease_edges.values())
print(f"Total combined edges: {total_edges:,}")

rng = random.Random(SEED)
disease_list = sorted(all_diseases)

train_pcts = []
val_pcts = []
test_pcts = []

print(f"\nRunning {SIMULATIONS:,} Monte Carlo simulations...")
for sim in range(SIMULATIONS):
    rng.shuffle(disease_list)
    n_total = len(disease_list)
    n_train = int(np.ceil(n_total * TRAIN_RATIO))
    n_val = int(np.ceil(n_total * VAL_RATIO))

    train_diseases = set(disease_list[:n_train])
    val_diseases = set(disease_list[n_train:n_train + n_val])
    test_diseases = set(disease_list[n_train + n_val:])

    train_edges = sum(disease_edges[d] for d in train_diseases)
    val_edges = sum(disease_edges[d] for d in val_diseases)
    test_edges = sum(disease_edges[d] for d in test_diseases)

    train_pcts.append(train_edges / total_edges * 100)
    val_pcts.append(val_edges / total_edges * 100)
    test_pcts.append(test_edges / total_edges * 100)

print("\n===== Monte Carlo Results (stratified by disease, 80/10/10) =====\n")
print(f"{'Split':>6} | {'Mean %':>8} | {'Std %':>7} | {'Min %':>7} | {'Max %':>7} | {'Median %':>8}")
print("-" * 60)
for name, data in [("Train", train_pcts), ("Val", val_pcts), ("Test", test_pcts)]:
    arr = np.array(data)
    print(f"{name:>6} | {arr.mean():7.2f}% | {arr.std():6.2f}% | {arr.min():6.2f}% | {arr.max():6.2f}% | {np.median(arr):7.2f}%")
