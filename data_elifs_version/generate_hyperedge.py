import os
import torch
import random
import numpy as np
import argparse
from collections import defaultdict

def build_offsets_and_node_types_full(
    data,
    type_order=("disease", "gene", "drug", "pathway", "phenotype"),
    type_to_int=None
):
    if type_to_int is None:
        type_to_int = {"disease": 0, "gene": 1, "drug": 2, "pathway": 3, "phenotype": 4}

    offsets = {}
    cur = 0
    for t in type_order:
        offsets[t] = cur
        cur += int(data[t].num_nodes)

    node_types_array = np.full(cur, -1, dtype=np.int32)
    for t in type_order:
        s = offsets[t]
        e = s + int(data[t].num_nodes)
        node_types_array[s:e] = int(type_to_int[t])

    return node_types_array, offsets, type_order, type_to_int

def gid(offsets, node_type, local_id):
    return offsets[node_type] + int(local_id)

def build_disease_hyperedges_full(data, offsets):
    """
    Builds hyperedges for each disease node: [anchor, neighbor1, neighbor2, ...]
    Only keeps hyperedges with >=3 nodes.
    """
    disease_hyper = defaultdict(set)

    def cpu_edge_index(key):
        ei = data[key].edge_index
        if ei.is_cuda:
            ei = ei.cpu()
        return ei[0], ei[1]

    # Add all edges that connect to disease anchor
    # gene -> disease
    for g, d in zip(*cpu_edge_index(("gene", "gene_to_disease", "disease"))):
        neighbor = gid(offsets, "gene", g)
        disease_hyper[int(d)].add(neighbor)

    # pubtator gene -> disease
    for g, d in zip(*cpu_edge_index(("gene", "pubtator_gene", "disease"))):
        neighbor = gid(offsets, "gene", g)
        disease_hyper[int(d)].add(neighbor)

    # disease -> drug
    for d, r in zip(*cpu_edge_index(("disease", "disease_to_drug", "drug"))):
        neighbor = gid(offsets, "drug", r)
        disease_hyper[int(d)].add(neighbor)

    # disease -> pubtator drug
    for d, r in zip(*cpu_edge_index(("disease", "pubtator_drug", "drug"))):
        neighbor = gid(offsets, "drug", r)
        disease_hyper[int(d)].add(neighbor)

    # disease -> pathway
    for d, p in zip(*cpu_edge_index(("disease", "disease_to_pathway", "pathway"))):
        neighbor = gid(offsets, "pathway", p)
        disease_hyper[int(d)].add(neighbor)

    # phenotype -> disease
    for p, d in zip(*cpu_edge_index(("phenotype", "phenotype_to_disease", "disease"))):
        neighbor = gid(offsets, "phenotype", p)
        disease_hyper[int(d)].add(neighbor)

    # disease <-> disease (various types, some may not exist!)
    dd_edges = [
        ("disease", "associated_with", "disease"),
        ("disease", "associated_with_rev", "disease"),
        ("disease", "comorbid_with", "disease"),
        ("disease", "comorbid_with_rev", "disease"),
        ("disease", "is_a", "disease"),
        ("disease", "has_child", "disease"),
    ]
    for et in dd_edges:
        if et not in data.edge_types:
            continue
        src, dst = cpu_edge_index(et)
        for d1, d2 in zip(src.tolist(), dst.tolist()):
            neighbor = gid(offsets, "disease", d2)
            disease_hyper[int(d1)].add(neighbor)

    # Compose to hyperedges: anchor, sorted neighbors
    hyperedges = []
    for d_local, nbrs in disease_hyper.items():
        he = [gid(offsets, "disease", d_local)] + sorted(nbrs)
        if len(he) > 2:
            hyperedges.append(he)
    return hyperedges

def write_hyperedge_list_txt(out_dir, dataset_name, split_name, he_list):
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{dataset_name}-POS_{split_name}.txt")
    with open(path, "w") as f:
        for he in he_list:
            f.write(" ".join(map(str, he)) + "\n")
    return path

# -------------- Main entrypoint -----------------
def generate_hyperedges_from_hetero(
    data_path,
    out_dir,
    dataset_name="full_data",
    split_ratios=(0.8, 0.1, 0.1),
    random_seed=42
):
    # 1. Load data
    data = torch.load(data_path, weights_only=False)

    # 2. Get node types and offsets
    node_types_array, offsets, _, _ = build_offsets_and_node_types_full(data)

    # 3. Build all hyperedges
    hyperedges = build_disease_hyperedges_full(data, offsets)
    print(f"Total hyperedges: {len(hyperedges)}")

    # 4. Shuffle and split
    random.seed(random_seed)
    random.shuffle(hyperedges)
    n = len(hyperedges)
    n_train = int(n * split_ratios[0])
    n_val = int(n * split_ratios[1])
    n_test = n - n_train - n_val

    high_order_train = hyperedges[:n_train]
    high_order_val = hyperedges[n_train:n_train+n_val]
    high_order_test = hyperedges[n_train+n_val:]

    # 5. Write outputs
    out_dir = os.path.join(out_dir, dataset_name)
    write_hyperedge_list_txt(out_dir, dataset_name, "train", high_order_train)
    write_hyperedge_list_txt(out_dir, dataset_name, "val", high_order_val)
    write_hyperedge_list_txt(out_dir, dataset_name, "test", high_order_test)
    print(f"Wrote POS hyperedges: train({len(high_order_train)}), val({len(high_order_val)}), test({len(high_order_test)}) to {out_dir}")
    return out_dir

# --- USAGE EXAMPLE ---
# generate_hyperedges_from_hetero(
#     data_path="./data/hetero_data.pt",
#     out_dir="./generated_hypergraphs/",
#     dataset_name="mydataset"
# )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate disease-centered hyperedges from a PyG HeteroData file.")
    parser.add_argument("--data-path", default="data_elifs_version/hetero_data.pt")
    parser.add_argument("--out-dir", default="generated_hypergraphs")
    parser.add_argument("--dataset-name", default="full_data")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    generate_hyperedges_from_hetero(
        data_path=args.data_path,
        out_dir=args.out_dir,
        dataset_name=args.dataset_name,
        random_seed=args.seed,
    )
