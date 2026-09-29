"""Measure the local merge model's time to its first phrase.

The stored value is the 90th percentile. Run from the repo root with Ollama
already serving the configured edge model:

    python Testing/benchmark_merge_latency.py
    python Testing/benchmark_merge_latency.py --write
"""

import argparse
import math
import os
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
os.chdir(ROOT)

from modules.edge_mllm_module import EdgeMLLMModule
from modules.fusion_module import build_merge_prompt, split_ready_phrases

CONFIG_PATH = ROOT / "src" / "config" / "config.yaml"
LOCAL = "The door is red. It is open. The room looks safe."
CLOUD = "The door is blue. It is closed."
SPOKEN = "The door is red."


def percentile_90(samples):
    ordered = sorted(samples)
    index = max(0, math.ceil(0.90 * len(ordered)) - 1)
    return ordered[index]


def first_phrase_seconds(edge, prompt):
    started = time.perf_counter()
    buffer = ""
    for chunk in edge.run_inference(prompt):
        buffer += str(chunk)
        phrases, buffer = split_ready_phrases(buffer)
        if phrases:
            return time.perf_counter() - started
    if buffer.strip():
        return time.perf_counter() - started
    raise RuntimeError("Merge model returned no text.")


def write_p90(value):
    text = CONFIG_PATH.read_text(encoding="utf-8")
    pattern = re.compile(r"^(\s*merge_first_phrase_p90_s:\s*)([0-9.]+)\s*$", re.M)
    updated, count = pattern.subn(rf"\g<1>{value:.3f}", text, count=1)
    if count != 1:
        raise SystemExit(f"Could not find merge_first_phrase_p90_s in {CONFIG_PATH}")
    CONFIG_PATH.write_text(updated, encoding="utf-8")


def main(argv=None):
    parser = argparse.ArgumentParser(description="Benchmark local merge first-phrase latency.")
    parser.add_argument("--samples", type=int, default=20)
    parser.add_argument("--write", action="store_true", help="Store the 90th percentile in config.yaml")
    args = parser.parse_args(argv)

    edge = EdgeMLLMModule()
    edge.load_model()
    prompt = build_merge_prompt(LOCAL, CLOUD, SPOKEN)
    samples = []
    for index in range(args.samples):
        elapsed = first_phrase_seconds(edge, prompt)
        samples.append(elapsed)
        print(f"sample {index + 1}: {elapsed:.3f}s")
    p90 = percentile_90(samples)
    print(f"90th percentile first phrase: {p90:.3f}s")
    print("Fusion reads fusion.merge_first_phrase_p90_s.")
    if args.write:
        write_p90(p90)
        print(f"Wrote merge_first_phrase_p90_s: {p90:.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
