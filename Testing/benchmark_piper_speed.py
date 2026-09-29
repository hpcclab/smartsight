"""Measure Piper characters per second and optionally store it in config.

Run from the repo root:

    python Testing/benchmark_piper_speed.py
    python Testing/benchmark_piper_speed.py --write
"""

import argparse
import os
import re
import sys
import wave
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
os.chdir(ROOT)

from modules.TTS_module import TTSModule

PASSAGE = (
    "The quick brown fox jumps over the lazy dog. "
    "Pack my box with five dozen liquor jugs. "
    "How vexingly quick daft zebras jump. "
    "The five boxing wizards jump quickly."
)
CONFIG_PATH = ROOT / "src" / "config" / "config.yaml"


def chars_per_second(wav_path, text):
    with wave.open(wav_path, "rb") as wav_file:
        frame_rate = wav_file.getframerate() or 1
        duration = wav_file.getnframes() / float(frame_rate)
    if duration <= 0:
        raise RuntimeError("Piper wrote a WAV with no duration.")
    return len(text) / duration


def write_chars_per_second(value):
    text = CONFIG_PATH.read_text(encoding="utf-8")
    pattern = re.compile(r"^(\s*chars_per_second:\s*)([0-9.]+)\s*$", re.M)
    replacement = rf"\g<1>{value:.2f}"
    updated, count = pattern.subn(replacement, text, count=1)
    if count != 1:
        raise SystemExit(f"Could not find chars_per_second in {CONFIG_PATH}")
    CONFIG_PATH.write_text(updated, encoding="utf-8")


def main(argv=None):
    parser = argparse.ArgumentParser(description="Benchmark Piper speaking rate.")
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument("--write", action="store_true", help="Store the mean in config.yaml")
    args = parser.parse_args(argv)

    tts = TTSModule()
    tts.load_model()
    if tts.model is None:
        raise SystemExit("Piper model did not load. Check piper_tts.model_path.")

    rates = []
    for index in range(args.samples):
        wav_path = tts.speak(PASSAGE)
        rate = chars_per_second(wav_path, PASSAGE)
        rates.append(rate)
        print(f"sample {index + 1}: {rate:.2f} chars/s")
    mean = sum(rates) / len(rates)
    print(f"mean: {mean:.2f} chars/s")
    print("Fusion reads piper_tts.chars_per_second.")
    if args.write:
        write_chars_per_second(mean)
        print(f"Wrote chars_per_second: {mean:.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
