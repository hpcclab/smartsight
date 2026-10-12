import sys
import os
import logging
import threading
import wave

import numpy as np

_script_dir = os.path.dirname(__file__)
_project_root = os.path.abspath(os.path.join(_script_dir, '..'))
_src_dir = os.path.join(_project_root, 'src')
for _p in (_project_root, _src_dir):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from modules.fusion_module import split_ready_phrases, snap_cut, project_index, build_merge_prompt
from modules.passive_detector_module import PassiveDetectorModule, paused_for_active_inference
from modules.active_module import _rule_matches
from utilities.logging_setup import _redact
from utilities.frame_buffer import PingPongBuffer


# ---------------------------------------------------------
# Fusion: phrase splitting and speech-cut math
# ---------------------------------------------------------
def test_split_ready_phrases():
    assert split_ready_phrases("Hi there. How are you? I'm") == (["Hi there.", "How are you?"], " I'm")
    assert split_ready_phrases("no punctuation") == ([], "no punctuation")
    assert split_ready_phrases("") == ([], "")


def test_snap_cut():
    assert snap_cut("", 5) == 0
    assert snap_cut("Hello.", 0) == 0
    assert snap_cut("Hello.", 10) == 6
    # Snaps forward to the end of the sentence being spoken
    assert snap_cut("The door is red. It is open.", 5) == 16
    # No punctuation: snaps to the end of the current word
    assert snap_cut("The door is red", 6) == 8
    # Inside an unfinished last word: back off to the previous word
    assert snap_cut("The door is red", 14) == 11


def test_project_index():
    assert project_index(0, 0, 0, 10) == 0
    assert project_index(5, 10, 0.5, 100) == 10
    assert project_index(5, 10, 10, 20) == 20
    assert project_index(5, 10, 1, 0) == 0


def test_build_merge_prompt():
    prompt = build_merge_prompt("LOCAL", "CLOUD", "SPOKEN")
    assert "'LOCAL'" in prompt and "'CLOUD'" in prompt and "'SPOKEN'" in prompt


# ---------------------------------------------------------
# Passive detection: result parsing and pausing
# ---------------------------------------------------------
def test_parse_detections():
    parse = PassiveDetectorModule._parse_detections
    assert parse(None, "") == {}
    assert parse(None, "2 persons, 1 dog") == {"person": 2, "dog": 1}
    assert parse(None, "bus") == {"bus": 1}
    assert parse(None, "2 persons, person") == {"person": 3}


def test_detections_round_trip():
    detections = {"person": 2, "dog": 1}
    text = PassiveDetectorModule._format_detections(None, detections)
    assert text == "2 persons, 1 dog"
    assert PassiveDetectorModule._parse_detections(None, text) == detections


def test_paused_for_active_inference(monkeypatch):
    # Skip __init__: it loads detection modules we don't need here
    detector = PassiveDetectorModule.__new__(PassiveDetectorModule)
    detector.pause_during_active_stt = True
    detector.pause_during_active_llm = False
    detector._pause_depth = 0
    detector._pause_lock = threading.Lock()
    detector.logger = logging.getLogger("test")
    monkeypatch.setattr(PassiveDetectorModule, "_current", detector)

    with paused_for_active_inference("stt"):
        assert detector.is_paused()
    assert not detector.is_paused()

    with paused_for_active_inference("llm"):
        assert not detector.is_paused()


# ---------------------------------------------------------
# Active mode command routing
# ---------------------------------------------------------
def test_rule_matches():
    assert _rule_matches("weather", "What's the Weather?")
    assert _rule_matches(r"read (this|that)", "please read that sign")
    assert not _rule_matches("weather", "what time is it")
    assert not _rule_matches("(", "invalid regex is a miss, not a crash")


# ---------------------------------------------------------
# Logging: secrets never reach the per-run config copy
# ---------------------------------------------------------
def test_redact():
    config = {
        "openrouter_api": {"api_key": "sk-123", "model": "m"},
        "dollar_detection": {"roboflow_api_key": "rf-123"},
        "hosts": [{"password": "p", "name": "pi"}],
    }
    assert _redact(config) == {
        "openrouter_api": {"api_key": "***", "model": "m"},
        "dollar_detection": {"roboflow_api_key": "***"},
        "hosts": [{"password": "***", "name": "pi"}],
    }
    assert config["openrouter_api"]["api_key"] == "sk-123"  # original untouched


# ---------------------------------------------------------
# Frame buffer
# ---------------------------------------------------------
def test_ping_pong_buffer():
    buf = PingPongBuffer()
    assert buf.retrieve_frame() is None
    assert buf.retrieve_both() == (None, None)

    first, second = np.zeros((2, 2)), np.ones((2, 2))
    buf.update(first)
    buf.update(second)
    current, previous = buf.retrieve_both()
    assert np.array_equal(current, second) and np.array_equal(previous, first)

    frame = buf.retrieve_frame()
    frame[:] = 7  # callers get a copy, not the stored frame
    assert np.array_equal(buf.retrieve_frame(), second)


# ---------------------------------------------------------
# TTS: the bundled Piper model synthesises a wav
# ---------------------------------------------------------
def test_tts_writes_wav(tmp_path, monkeypatch):
    from modules.TTS_module import TTSModule

    tts = TTSModule()
    # Config paths are repo-relative; speak() writes into cwd, so run from tmp
    tts.config = {
        **tts.config,
        "model_path": os.path.join(_project_root, tts.config["model_path"]),
        "config_path": os.path.join(_project_root, tts.config["config_path"]),
    }
    monkeypatch.chdir(tmp_path)

    tts.load_model()
    assert tts.model is not None, "Piper model failed to load"
    # speak() only; speak_aloud() needs winsound (Windows only)
    path = tts.speak("SmartSight test.")
    with wave.open(str(tmp_path / path)) as wav:
        assert wav.getnframes() > 0
