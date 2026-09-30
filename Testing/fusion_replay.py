"""Replay a fusion request with scripted local, cloud, and merge text.

Arriving tokens, phrases sent to speech, finished speech, the skipped local
tail, and the merge are shown in different colors. No model is loaded.

    python Testing/fusion_replay.py
    python Testing/fusion_replay.py --scenario path.json
    python Testing/fusion_replay.py --log logs/<run>/smartsight.log
    python Testing/fusion_replay.py --log logs/<run> --index 0
"""

import argparse
import json
import os
import re
import shutil
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from modules.fusion_module import REPLAY_PREFIX, FusionModule

CYAN = "\033[36m"
YELLOW = "\033[33m"
GREEN = "\033[32m"
RED = "\033[31m"
MAGENTA = "\033[35m"
BLUE = "\033[34m"
RESET = "\033[0m"

ANSI = {
    "buf": CYAN,
    "queue": YELLOW,
    "spoken": GREEN,
    "skip": RED,
    "merge": MAGENTA,
}

_ANSI_RE = re.compile(r"\033\[[0-9;]*m")


def _term_size():
    try:
        size = shutil.get_terminal_size(fallback=(120, 24))
        columns, lines = size.columns, size.lines
    except OSError:
        columns, lines = 120, 24
    return max(40, columns), max(2, lines)


def _section_rows(label, text, width):
    """Wrap one labeled transcript into terminal rows.

    Newlines start a new row. Other text wraps so the visible width of each
    row is at most ``width``. Color codes take no columns. Continuation rows
    indent under the label, a row that starts inside an open color reopens
    it, and a row that opened a color ends in reset.
    """
    width = max(len(label) + 1, width)
    content_width = width - len(label)
    indent = " " * len(label)
    rows = []
    pieces = []
    visible = 0
    active = ""
    first = True

    def close_row():
        nonlocal pieces, visible, first
        body = "".join(pieces)
        if body and active and not body.endswith(RESET):
            body += RESET
        rows.append((label if first else indent) + body)
        pieces = []
        visible = 0
        first = False

    def open_color():
        if active and not pieces:
            pieces.append(active)

    index = 0
    text = text or ""
    while index < len(text):
        if text.startswith("\r\n", index):
            close_row()
            index += 2
            continue
        if text[index] in "\n\r":
            close_row()
            index += 1
            continue
        match = _ANSI_RE.match(text, index)
        if match:
            seq = match.group()
            open_color()
            pieces.append(seq)
            params = seq[2:-1]
            active = "" if params in ("", "0") else seq
            index = match.end()
            continue
        if visible >= content_width:
            close_row()
            continue
        open_color()
        pieces.append(text[index])
        visible += 1
        index += 1
    if pieces or not rows:
        close_row()
    return rows


def _hud_rows(edge, merge, cloud, cut, width):
    rows = []
    rows.extend(_section_rows("EDGE  ", edge, width))
    rows.extend(_section_rows("MERGE ", merge, width))
    rows.extend(_section_rows("CLOUD ", cloud, width))
    rows.extend(_section_rows("", cut, width))
    return rows

BUILTIN = {
    "assumed_tts_chars_per_second": 40,
    "actual_tts_chars_per_second": 12,
    "merge_first_phrase_p90_s": 0.35,
    "edge_tokens": [
        "The door is red. ",
        "It is open. ",
        "The room looks safe. ",
        "Everyone seems fine.",
    ],
    "edge_first_delay_s": 0.2,
    "edge_token_delay_s": 0.25,
    "cloud_tokens": ["The door is blue. ", "It is closed."],
    "cloud_first_delay_s": 1.6,
    "cloud_token_delay_s": 0.15,
    "merge_tokens": [" It is blue and closed."],
    "merge_first_delay_s": 0.3,
    "merge_token_delay_s": 0.08,
}


def enable_vt():
    if os.name != "nt":
        return
    import ctypes
    kernel = ctypes.windll.kernel32
    handle = kernel.GetStdHandle(-11)
    mode = ctypes.c_uint()
    if kernel.GetConsoleMode(handle, ctypes.byref(mode)):
        kernel.SetConsoleMode(handle, mode.value | 0x0004)


class Display:
    """Color state for the local transcript and the merge transcript."""

    def __init__(self):
        self.local = ""
        self.local_state = []
        self.merge = ""
        self.merge_state = []
        self.spans = {}
        self.cloud = ""
        self.cut = None
        self._lock = threading.Lock()

    def apply(self, event):
        with self._lock:
            kind = event.get("type")
            source = event.get("source")
            if kind == "token" and source == "local":
                self.local += event["text"]
                self.local_state.extend(["buf"] * len(event["text"]))
            elif kind == "token" and source == "merge":
                self.merge += event["text"]
                self.merge_state.extend(["merge"] * len(event["text"]))
            elif kind == "phrase" and source == "local":
                _paint(self.local_state, event.get("start"), event.get("end"), "queue")
                self.spans[event["order"]] = event
            elif kind == "phrase" and source == "merge":
                _paint_text(self.merge, self.merge_state, event.get("text", ""), "merge")
                self.spans[event["order"]] = event
            elif kind == "cancel" and source == "local":
                _paint(self.local_state, event.get("start"), event.get("end"), "skip")
            elif kind == "playback" and event.get("event") == "done":
                span = self.spans.get(event.get("order"))
                if span and span.get("source") == "local":
                    _paint(self.local_state, span.get("start"), span.get("end"), "spoken")
                elif span and span.get("source") == "merge":
                    _paint_text(self.merge, self.merge_state, span.get("text", ""), "spoken")
            elif kind == "cloud_done":
                self.cloud = event.get("text", "")
            elif kind == "cut":
                self.cut = event
                snapped = event.get("snapped")
                if snapped is not None:
                    for index in range(snapped, len(self.local_state)):
                        if self.local_state[index] == "buf":
                            self.local_state[index] = "skip"

    def markers(self):
        with self._lock:
            return _group(self.local, self.local_state), _group(self.merge, self.merge_state)

    def ansi_lines(self):
        with self._lock:
            edge = _ansi(self.local, self.local_state)
            merge = _ansi(self.merge, self.merge_state)
            cloud = f"{BLUE}{self.cloud}{RESET}" if self.cloud else ""
            cut = ""
            if self.cut:
                cut = (
                    f"cut callback={self.cut.get('callback_index')} "
                    f"time_estimate={self.cut.get('time_index')} "
                    f"snapped={self.cut.get('snapped')}"
                )
            return edge, merge, cloud, cut


def _paint(states, start, end, name):
    if start is None or end is None:
        return
    for index in range(max(0, start), min(len(states), end)):
        states[index] = name


def _paint_text(text, states, phrase, name):
    if not phrase:
        return
    found = text.find(phrase)
    if found < 0:
        return
    start = found
    while start > 0 and text[start - 1].isspace() and states[start - 1] == "merge":
        start -= 1
    _paint(states, start, found + len(phrase), name)


def _group(text, states):
    if not text:
        return ""
    parts = []
    index = 0
    while index < len(text):
        name = states[index] if index < len(states) else "buf"
        end = index + 1
        while end < len(text) and (states[end] if end < len(states) else "buf") == name:
            end += 1
        parts.append("{" + name + "}" + text[index:end])
        index = end
    return "".join(parts)


def _ansi(text, states):
    if not text:
        return ""
    parts = []
    index = 0
    while index < len(text):
        name = states[index] if index < len(states) else "buf"
        end = index + 1
        while end < len(text) and (states[end] if end < len(states) else "buf") == name:
            end += 1
        parts.append(ANSI.get(name, "") + text[index:end] + RESET)
        index = end
    return "".join(parts)


class PlaybackSimulator:
    """Speak queued phrases at a fixed character rate and report progress."""

    def __init__(self, chars_per_second, sleep=time.sleep):
        self.chars_per_second = chars_per_second if chars_per_second > 0 else 1
        self.sleep = sleep
        self._queue = []
        self._callbacks = {}
        self._spoken = {}
        self._doomed = {}
        self._lock = threading.Lock()
        self._seq = 0
        self._pending = 0
        self._running = True
        self._wake = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def add_message(self, message, message_time=None, message_expiration=1.0, *, priority=None, response_id=None, order=None):
        with self._lock:
            self._seq += 1
            self._pending += 1
            self._queue.append((priority if priority is not None else 10, self._seq, message, response_id, order))
            self._queue.sort()
        self._wake.set()

    def bind_playback_callback(self, response_id, callback):
        with self._lock:
            self._callbacks[response_id] = callback
            self._spoken.setdefault(response_id, "")

    def cancel_unspoken(self, response_id, keep_through_order):
        with self._lock:
            doomed = self._doomed.setdefault(response_id, set())
            kept = []
            for item in self._queue:
                _priority, _seq, _message, item_id, order = item
                if item_id == response_id and order is not None and order > keep_through_order:
                    doomed.add(order)
                    self._pending -= 1
                    continue
                kept.append(item)
            self._queue = kept
        return []

    def cancel_orders(self, response_id, orders):
        wanted = {order for order in orders if order is not None}
        with self._lock:
            doomed = self._doomed.setdefault(response_id, set())
            doomed.update(wanted)
            kept = []
            for item in self._queue:
                _priority, _seq, _message, item_id, order = item
                if item_id == response_id and order in wanted:
                    self._pending -= 1
                    continue
                kept.append(item)
            self._queue = kept
        return []

    def wait_until_idle(self, timeout):
        deadline = time.time() + timeout
        while time.time() < deadline:
            with self._lock:
                if self._pending <= 0 and not self._queue:
                    return True
            time.sleep(0.02)
        return False

    def stop(self):
        self._running = False
        self._wake.set()
        self._thread.join(timeout=2)

    def _loop(self):
        while self._running:
            item = None
            with self._lock:
                if self._queue:
                    item = self._queue.pop(0)
            if item is None:
                self._wake.wait(timeout=0.05)
                self._wake.clear()
                continue
            _priority, _seq, message, response_id, order = item
            with self._lock:
                if order in self._doomed.get(response_id, ()):
                    self._pending -= 1
                    continue
                callback = self._callbacks.get(response_id)
                spoken = self._spoken.get(response_id, "")
            duration = len(message) / self.chars_per_second
            started = time.monotonic()
            if callback is not None and order is not None:
                callback(response_id, {
                    "event": "start",
                    "order": order,
                    "phrase": message,
                    "spoken_text": spoken,
                    "phrase_duration_s": duration,
                    "phrase_started_at": started,
                })
            self.sleep(duration)
            with self._lock:
                prev = self._spoken.get(response_id, "")
                spoken = f"{prev} {message}".strip() if prev else message
                self._spoken[response_id] = spoken
                self._pending -= 1
            if callback is not None and order is not None:
                callback(response_id, {
                    "event": "done",
                    "order": order,
                    "phrase": message,
                    "spoken_text": spoken,
                    "phrase_duration_s": duration,
                    "phrase_started_at": started,
                })


def _scripted(tokens, first_delay, token_delay, sleep):
    def _stream(*_args, **_kwargs):
        if first_delay:
            sleep(first_delay)
        for index, token in enumerate(tokens):
            yield token
            if token_delay and index != len(tokens) - 1:
                sleep(token_delay)
    return _stream


def _scripted_events(events, sleep, clock=None):
    """Yield saved chunks, waiting until each chunk's own-stream timestamp."""
    clock = clock or time.monotonic

    def _stream(*_args, **_kwargs):
        started = clock()
        for event in events or []:
            wait = float(event.get("t") or 0) - (clock() - started)
            if wait > 0:
                sleep(wait)
            yield event.get("text") or ""
    return _stream


def _stream_for(scenario, events_key, tokens_key, first_key, delay_key, sleep):
    if events_key in scenario:
        return _scripted_events(scenario.get(events_key) or [], sleep)
    return _scripted(scenario[tokens_key], scenario[first_key], scenario[delay_key], sleep)


def load_replay_records(text):
    """Return every fusion record embedded in a SmartSight log."""
    records = []
    for line in text.splitlines():
        index = line.find(REPLAY_PREFIX)
        if index < 0:
            continue
        records.append(json.loads(line[index + len(REPLAY_PREFIX):].strip()))
    return records


def scenario_from_record(record):
    """Turn one logged fusion response into the dict the replay already plays."""
    assumed = float(record["assumed_tts_chars_per_second"])
    actual = float(record.get("actual_tts_chars_per_second") or 0)
    if actual <= 0:
        actual = assumed if assumed > 0 else 15.0
    if assumed <= 0:
        assumed = actual
    return {
        "assumed_tts_chars_per_second": assumed,
        "actual_tts_chars_per_second": actual,
        "merge_first_phrase_p90_s": record["merge_first_phrase_p90_s"],
        "prompt": record.get("prompt") or "",
        "edge_enabled": record.get("edge_enabled", True),
        "edge_events": record.get("edge_events") or [],
        "cloud_events": record.get("cloud_events") or [],
        "merge_events": record.get("merge_events") or [],
        "edge_mllm": record.get("edge_mllm"),
        "llm": record.get("llm"),
        "cloud_model": record.get("cloud_model"),
    }


def _fusion_for(scenario, playback, sleep, listener=None):
    edge_enabled = scenario.get("edge_enabled", True)
    ollama = {"ollama_llm_enabled": edge_enabled}
    model = scenario.get("edge_mllm") or scenario.get("llm")
    if model:
        ollama["model"] = model
    api = {}
    if scenario.get("cloud_model"):
        api["model_priority"] = [scenario["cloud_model"]]
    config = {
        "fusion": {"merge_first_phrase_p90_s": scenario["merge_first_phrase_p90_s"]},
        "piper_tts": {"chars_per_second": scenario["assumed_tts_chars_per_second"]},
        "ollama_llm": ollama,
        "openrouter_api": api,
    }
    return FusionModule(
        playback,
        edge_stream=_stream_for(
            scenario, "edge_events", "edge_tokens", "edge_first_delay_s", "edge_token_delay_s", sleep
        ),
        cloud_stream=_stream_for(
            scenario, "cloud_events", "cloud_tokens", "cloud_first_delay_s", "cloud_token_delay_s", sleep
        ),
        merge_stream=_stream_for(
            scenario, "merge_events", "merge_tokens", "merge_first_delay_s", "merge_token_delay_s", sleep
        ),
        config=config,
        frame_source=lambda: None,
        listener=listener,
        edge_enabled=edge_enabled,
    )


def run_scenario(scenario, sleep=time.sleep):
    display = Display()
    playback = PlaybackSimulator(scenario["actual_tts_chars_per_second"], sleep=sleep)
    fusion = _fusion_for(scenario, playback, sleep, listener=display.apply)
    try:
        spoken = fusion.handle(scenario.get("prompt", "What is in front of me?"))
        playback.wait_until_idle(timeout=30)
    finally:
        playback.stop()
    return display, spoken


def _print_live(display, lines_drawn):
    edge, merge, cloud, cut = display.ansi_lines()
    columns, height = _term_size()
    rows = _hud_rows(edge, merge, cloud, cut, columns)
    limit = max(1, height - 1)
    if len(rows) > limit:
        rows = rows[-limit:]
    if lines_drawn:
        sys.stdout.write(f"\033[{lines_drawn}A")
    sys.stdout.write("\033[J")
    for line in rows:
        sys.stdout.write("\033[2K" + line + RESET + "\n")
    sys.stdout.flush()
    return len(rows)


def _scenario_from_args(parser, args):
    if args.log and args.scenario:
        parser.error("--log and --scenario cannot be used together")
    if args.log:
        path = Path(args.log)
        if path.is_dir():
            path = path / "smartsight.log"
        if not path.is_file():
            parser.error(f"No log file at {path}")
        records = load_replay_records(path.read_text(encoding="utf-8"))
        if not records:
            parser.error(f"No {REPLAY_PREFIX.strip()} records in {path}")
        try:
            record = records[args.index]
        except IndexError:
            parser.error(f"Replay index {args.index} is outside {len(records)} record(s)")
        return scenario_from_record(record)
    if args.scenario:
        return json.loads(Path(args.scenario).read_text(encoding="utf-8"))
    return BUILTIN


def main(argv=None):
    parser = argparse.ArgumentParser(description="Replay the fusion system with scripted text.")
    parser.add_argument("--scenario", help="JSON file with token timings. Defaults to a built-in scene.")
    parser.add_argument("--log", help="SmartSight log file or run folder containing FUSION_REPLAY lines.")
    parser.add_argument("--index", type=int, default=-1, help="Which logged response to play. Default: the last one.")
    args = parser.parse_args(argv)
    scenario = _scenario_from_args(parser, args)
    enable_vt()
    print("cyan buffer | yellow sent to speech | green spoken | red skipped | magenta merge")
    display = Display()
    playback = PlaybackSimulator(scenario["actual_tts_chars_per_second"])
    drawn = {"n": 0}
    lock = threading.Lock()

    def listener(event):
        display.apply(event)
        with lock:
            drawn["n"] = _print_live(display, drawn["n"])

    fusion = _fusion_for(scenario, playback, time.sleep, listener=listener)
    sys.stdout.write("\033[?25l")
    try:
        spoken = fusion.handle(scenario.get("prompt", "What is in front of me?"))
        playback.wait_until_idle(timeout=30)
        with lock:
            _print_live(display, drawn["n"])
    finally:
        playback.stop()
        sys.stdout.write("\033[?25h")
        sys.stdout.flush()
    edge, merge = display.markers()
    print("FINAL EDGE: " + edge)
    print("FINAL MERGE: " + merge)
    print("SPOKEN: " + spoken)
    return 0


if __name__ == "__main__":
    sys.exit(main())
