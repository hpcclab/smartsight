"""Try merge prompts against spoofed edge and cloud answers.

The live merge sends build_merge_prompt(edge, cloud, spoken) to the local
Ollama model as a plain user message. This script does that call with your
texts and your template. It does not run speech, the camera, or FusionModule.handle.

Ollama must already be serving the model, same as benchmark_merge_latency.py.
--prompt-only prints the prompt and does not load the model.

    python Testing/fusion_merge_prompt.py --prompt-only
    python Testing/fusion_merge_prompt.py --edge "The door is red. It is open." --cloud "The door is blue." --spoken "The door is red."
    python Testing/fusion_merge_prompt.py --case door.json --template draft.txt --template production
    python Testing/fusion_merge_prompt.py --spoken-chars 18

    python Testing/fusion_merge_prompt.py --case Testing/door.json --template Testing/mergeprompt.txt

A template is plain text. These tokens are replaced literally:
{local_response}  {cloud_response}  {spoken_text}
The name production uses build_merge_prompt.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
os.chdir(ROOT)

from modules.fusion_module import build_merge_prompt, snap_cut, split_ready_phrases

PLACEHOLDERS = ("{local_response}", "{cloud_response}", "{spoken_text}")
BUILTIN = {
    "name": "built-in",
    "edge": "The door is red. It is open. The room looks safe.",
    "cloud": "The door is blue. It is closed.",
    "spoken": "The door is red.",
}


def spoken_prefix(edge, spoken_chars):
    """Snap a character count the way fusion cuts, then take that prefix."""
    return edge[:snap_cut(edge, spoken_chars)].strip()


def render_template(template, edge, cloud, spoken):
    """Replace the three tokens in one pass so values are not expanded again."""
    values = {
        "{local_response}": edge,
        "{cloud_response}": cloud,
        "{spoken_text}": spoken,
    }
    missing = [token for token in PLACEHOLDERS if token not in template]
    pieces = []
    index = 0
    while index < len(template):
        matched = None
        for token in PLACEHOLDERS:
            if template.startswith(token, index):
                matched = token
                break
        if matched is None:
            pieces.append(template[index])
            index += 1
            continue
        pieces.append(values[matched])
        index += len(matched)
    return "".join(pieces), missing


def _text_field(data, key, path):
    if key not in data:
        raise SystemExit(f"{path} is missing {key}")
    value = data[key]
    if not isinstance(value, str):
        raise SystemExit(f"{path} field {key} must be a string")
    return value


def load_case_file(path):
    file_path = Path(path)
    if not file_path.is_file():
        raise SystemExit(f"No case file at {file_path}")
    try:
        data = json.loads(file_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise SystemExit(f"Could not parse {file_path}: {exc}") from exc
    if not isinstance(data, dict):
        raise SystemExit(f"{file_path} must be a JSON object")
    edge = _text_field(data, "edge", file_path)
    cloud = _text_field(data, "cloud", file_path)
    spoken = _text_field(data, "spoken", file_path) if "spoken" in data else ""
    if "spoken_chars" in data:
        count = data["spoken_chars"]
        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            raise SystemExit(f"{file_path} field spoken_chars must be an integer >= 0")
        spoken = spoken_prefix(edge, count)
    return {"name": file_path.stem, "edge": edge, "cloud": cloud, "spoken": spoken}


def load_cases(parser, args):
    using_text = args.edge is not None or args.cloud is not None or args.spoken is not None
    if args.case and using_text:
        parser.error("pass either --case or --edge/--cloud")
    if args.spoken is not None and args.spoken_chars is not None:
        parser.error("pass either --spoken or --spoken-chars")
    if args.spoken_chars is not None and args.spoken_chars < 0:
        parser.error("--spoken-chars must be >= 0")
    if using_text:
        if args.edge is None or args.cloud is None:
            parser.error("--edge and --cloud are required together")
        cases = [{
            "name": "cli",
            "edge": args.edge,
            "cloud": args.cloud,
            "spoken": "" if args.spoken is None else args.spoken,
        }]
    elif args.case:
        cases = [load_case_file(path) for path in args.case]
    else:
        cases = [dict(BUILTIN)]
    if args.spoken_chars is not None:
        for case in cases:
            case["spoken"] = spoken_prefix(case["edge"], args.spoken_chars)
    return cases


def load_templates(args):
    if not args.template:
        return [("production", None)]
    loaded = []
    for item in args.template:
        if item == "production":
            loaded.append(("production", None))
            continue
        path = Path(item)
        if not path.is_file():
            raise SystemExit(f"No template file at {path}")
        loaded.append((item, path.read_text(encoding="utf-8")))
    return loaded


def prompt_for(template, edge, cloud, spoken):
    if template is None:
        return build_merge_prompt(edge, cloud, spoken), []
    return render_template(template, edge, cloud, spoken)


def stream_merge(edge, prompt):
    """Print the model stream and return (first phrase seconds, full seconds)."""
    started = time.perf_counter()
    first = None
    buffer = ""
    pieces = []
    for chunk in edge.run_inference(prompt):
        text = "" if chunk is None else str(chunk)
        if not text:
            continue
        pieces.append(text)
        sys.stdout.write(text)
        sys.stdout.flush()
        buffer += text
        if first is None:
            phrases, buffer = split_ready_phrases(buffer)
            if phrases:
                first = time.perf_counter() - started
    sys.stdout.write("\n")
    full = time.perf_counter() - started
    if not "".join(pieces).strip():
        raise SystemExit("Merge model returned no text.")
    if first is None:
        first = full
    return first, full


def load_edge(model_name):
    from modules.edge_mllm_module import EdgeMLLMModule

    edge = EdgeMLLMModule()
    if model_name:
        edge.model_name = model_name
    edge.load_model()
    return edge


def build_parser():
    parser = argparse.ArgumentParser(
        description="Run the fusion merge prompt on spoofed edge and cloud answers."
    )
    parser.add_argument("--edge", help="Spoofed edge (local) response.")
    parser.add_argument("--cloud", help="Spoofed cloud response.")
    parser.add_argument("--spoken", help="Text already spoken. Defaults to empty with --edge.")
    parser.add_argument(
        "--case",
        action="append",
        default=[],
        help="JSON file with edge, cloud, and optional spoken or spoken_chars. Repeatable.",
    )
    parser.add_argument(
        "--template",
        action="append",
        default=[],
        help="Prompt file, or 'production' for build_merge_prompt. Repeatable.",
    )
    parser.add_argument(
        "--spoken-chars",
        type=int,
        help="Replace spoken text with a snapped prefix of the edge response.",
    )
    parser.add_argument(
        "--prompt-only",
        action="store_true",
        help="Print the prompt and do not call the model.",
    )
    parser.add_argument("--model", help="Ollama model name. Defaults to ollama_llm.model.")
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    cases = load_cases(parser, args)
    templates = load_templates(args)
    edge_module = None if args.prompt_only else load_edge(args.model)
    first_block = True
    for case in cases:
        for label, template in templates:
            if not first_block:
                print()
            first_block = False
            prompt, missing = prompt_for(template, case["edge"], case["cloud"], case["spoken"])
            print(f"case: {case['name']}")
            print(f"template: {label}")
            print("--- prompt ---")
            print(prompt)
            if missing:
                print("missing placeholders: " + ", ".join(missing))
            if edge_module is not None:
                print("--- merge ---")
                first, full = stream_merge(edge_module, prompt)
                print(f"first phrase: {first:.3f}s")
                print(f"full: {full:.3f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
