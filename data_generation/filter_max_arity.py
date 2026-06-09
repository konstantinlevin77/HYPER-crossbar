#!/usr/bin/env python3
"""
Create a filtered copy of a hypergraph dataset, dropping hyperedges above a
maximum arity without modifying the source data.
"""

import argparse
import json
import shutil
from pathlib import Path


DEFAULT_EDGE_FILES = {
    "train.txt",
    "valid.txt",
    "test.txt",
    "aux.txt",
    "hypergraph_inference.txt",
}


def split_tokens(line):
    stripped = line.strip()
    if not stripped:
        return []
    if "\t" in stripped:
        return stripped.split("\t")
    return stripped.split()


def infer_format(tokens):
    if not tokens:
        return "entities_only"
    try:
        float(tokens[0])
        return "entities_only"
    except ValueError:
        return "relation_first"


def edge_arity(tokens, file_format):
    if file_format == "relation_first":
        return max(len(tokens) - 1, 0)
    if file_format == "entities_only":
        return len(tokens)
    inferred = infer_format(tokens)
    return edge_arity(tokens, inferred)


def should_process(path, process_all_txt):
    if path.suffix != ".txt":
        return False
    return process_all_txt or path.name in DEFAULT_EDGE_FILES or path.name.startswith("test_")


def filter_file(input_path, output_path, max_arity, file_format):
    stats = {
        "input": str(input_path),
        "output": str(output_path),
        "kept": 0,
        "removed": 0,
        "total": 0,
        "max_arity_before": 0,
        "max_arity_after": 0,
    }

    with input_path.open("r") as src, output_path.open("w") as dst:
        for line in src:
            tokens = split_tokens(line)
            if not tokens:
                dst.write(line)
                continue

            arity = edge_arity(tokens, file_format)
            stats["total"] += 1
            stats["max_arity_before"] = max(stats["max_arity_before"], arity)

            if arity > max_arity:
                stats["removed"] += 1
                continue

            dst.write(line)
            stats["kept"] += 1
            stats["max_arity_after"] = max(stats["max_arity_after"], arity)

    return stats


def copy_or_filter(input_dir, output_dir, max_arity, file_format, process_all_txt):
    all_stats = []

    for input_path in sorted(input_dir.rglob("*")):
        relative_path = input_path.relative_to(input_dir)
        output_path = output_dir / relative_path

        if input_path.is_dir():
            output_path.mkdir(parents=True, exist_ok=True)
            continue

        output_path.parent.mkdir(parents=True, exist_ok=True)
        if should_process(input_path, process_all_txt):
            stats = filter_file(input_path, output_path, max_arity, file_format)
            all_stats.append(stats)
            print(
                f"{relative_path}: kept {stats['kept']}/{stats['total']} "
                f"removed {stats['removed']} "
                f"max_arity {stats['max_arity_before']} -> {stats['max_arity_after']}"
            )
        else:
            shutil.copy2(input_path, output_path)

    return all_stats


def parse_args():
    parser = argparse.ArgumentParser(
        description="Copy a hypergraph dataset and remove edges above a max arity."
    )
    parser.add_argument(
        "--input_dir",
        type=Path,
        default=Path("generated_hypergraphs/hyper_compliant"),
        help="Source dataset directory. This directory is never modified.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("generated_hypergraphs/hyper_compliant_max_arity_150"),
        help="Destination directory for the filtered copy.",
    )
    parser.add_argument(
        "--max_arity",
        type=int,
        default=150,
        help="Remove hyperedges with arity strictly greater than this value.",
    )
    parser.add_argument(
        "--format",
        choices=("auto", "entities_only", "relation_first"),
        default="auto",
        help=(
            "Input edge format. Use entities_only for files where every token is "
            "a node, relation_first for relation + entities, or auto to infer per line."
        ),
    )
    parser.add_argument(
        "--process_all_txt",
        action="store_true",
        help="Filter every .txt file. By default only split files are filtered.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow writing into an existing output directory.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if not args.input_dir.exists():
        raise FileNotFoundError(f"Input directory does not exist: {args.input_dir}")
    if args.output_dir.exists() and not args.overwrite:
        raise FileExistsError(
            f"Output directory already exists: {args.output_dir}. "
            "Choose a new --output_dir or pass --overwrite."
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    stats = copy_or_filter(
        args.input_dir,
        args.output_dir,
        args.max_arity,
        args.format,
        args.process_all_txt,
    )

    summary = {
        "input_dir": str(args.input_dir),
        "output_dir": str(args.output_dir),
        "max_arity": args.max_arity,
        "format": args.format,
        "processed_files": len(stats),
        "total_edges": sum(item["total"] for item in stats),
        "kept_edges": sum(item["kept"] for item in stats),
        "removed_edges": sum(item["removed"] for item in stats),
        "files": stats,
    }

    summary_path = args.output_dir / "filter_max_arity_summary.json"
    with summary_path.open("w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nWrote filtered dataset to: {args.output_dir}")
    print(f"Wrote summary to: {summary_path}")
    print(
        f"Removed {summary['removed_edges']} of {summary['total_edges']} "
        f"processed hyperedges."
    )


if __name__ == "__main__":
    main()
