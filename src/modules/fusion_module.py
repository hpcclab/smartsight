"""Merge a fast local answer with a slower cloud answer while speech is already playing."""

import json
import math
import threading
import time
import uuid

from utilities.logging_setup import get_logger

HARD_PUNCTUATION = ".?!"
PHRASE_PRIORITY = 10
PHRASE_LIFETIME = 120.0
FAILURE_TEXT = "Failed to get response from the model."
REPLAY_PREFIX = "FUSION_REPLAY "


def split_ready_phrases(buffer):
    """Split completed hard-punctuation phrases out of buffer.

    Returns (phrases, remainder). Each phrase keeps the punctuation and is
    stripped of surrounding whitespace. The remainder is unsplit text.
    """
    phrases = []
    start = 0
    for index, char in enumerate(buffer):
        if char not in HARD_PUNCTUATION:
            continue
        raw = buffer[start:index + 1]
        if raw.strip():
            phrases.append(raw.strip())
        start = index + 1
    return phrases, buffer[start:]


def snap_cut(text, spoken_chars):
    """Snap a character count to a speakable boundary.

    Follows Algorithm 1, lines 19-27, of the Audo-Sight paper. spoken_chars
    is how many characters will already have been said. A count of 0 stays 0
    so an unstarted answer is not pulled forward to the first sentence.
    """
    if not text or spoken_chars <= 0:
        return 0
    if spoken_chars >= len(text):
        return len(text)

    # `spoken_chars` is a count, so the last included character is at this index.
    start_at = spoken_chars - 1
    for index in range(start_at, len(text)):
        if text[index] in HARD_PUNCTUATION:
            return index + 1

    spans = _word_spans(text)
    if not spans:
        return 0
    position = spoken_chars - 1
    containing = None
    for start, end in spans:
        if start <= position < end:
            containing = (start, end)
            break
    if containing is None:
        previous = 0
        for start, end in spans:
            if end <= spoken_chars:
                previous = end
            else:
                break
        return previous

    start, end = containing
    last_start, _last_end = spans[-1]
    in_last_word = start == last_start
    if in_last_word and text[-1] not in HARD_PUNCTUATION:
        previous = 0
        for word_start, word_end in spans:
            if word_end <= start:
                previous = word_end
            else:
                break
        return previous
    return end


def project_index(spoken_chars, chars_per_second, merge_wait_s, length):
    """Character index speech is projected to reach, clamped to text length."""
    if length <= 0:
        return 0
    if spoken_chars <= 0 and chars_per_second * merge_wait_s <= 0:
        return 0
    projected = spoken_chars + (chars_per_second * merge_wait_s)
    index = math.ceil(projected - 1e-12)
    if index < 0:
        return 0
    if index > length:
        return length
    return index


def build_merge_prompt(local_response, cloud_response, spoken_text):
    """Prompt the local model to continue from what has already been said."""
    return (
        f"You are a helpful assistant. "
        f"You have already said this part of a response: '{spoken_text}'. Don't say it again. Continue from here "
        f"The truth is: '{cloud_response}'. "
        f"Pick up seamlessly from where you left off, using the new information from the truth. "
        f"Do not repeat what you already said. Just continue the response."
        f"If you have stated any information that is conflicting with the truth, correct it in favor of the truth."
    )


def _word_spans(text):
    spans = []
    index = 0
    length = len(text)
    while index < length:
        while index < length and text[index].isspace():
            index += 1
        if index >= length:
            break
        end = index
        while end < length and not text[end].isspace():
            end += 1
        spans.append((index, end))
        index = end
    return spans


class FusionModule:
    """Speak the local stream immediately, then continue with a cloud merge.

    edge_stream(prompt, frame), cloud_stream(prompt, frame), and
    merge_stream(prompt) are iterators of text chunks. Tests pass these in.
    Production builds them from AIManager on first use.
    """

    def __init__(
        self,
        global_response_module,
        *,
        edge_stream=None,
        cloud_stream=None,
        merge_stream=None,
        config=None,
        frame_source=None,
        clock=None,
        listener=None,
        edge_enabled=None,
        wait_timeout=120,
    ):
        self.logger = get_logger(self.__class__.__name__)
        self.grm = global_response_module
        self._edge_stream = edge_stream
        self._cloud_stream = cloud_stream
        self._merge_stream = merge_stream
        self._frame_source = frame_source
        self.clock = clock or time.monotonic
        self.listener = listener
        self.wait_timeout = wait_timeout
        self.chars_per_second, self.merge_wait_s, self.edge_enabled, self._config = _settings(
            config, edge_enabled
        )
        self._lock = threading.Lock()
        self._reset_state()

    def handle(self, text_input):
        """Run one active request. Returns the text handed to speech."""
        self._reset_state()
        self._t0 = self.clock()
        self._prompt = text_input
        frame = self._snapshot_frame()
        self.grm.bind_playback_callback(self.response_id, self._on_playback)

        local_thread = None
        cloud_thread = None
        try:
            if self.edge_enabled:
                self.edge_mllm_used = _configured_model(self._config, "ollama_llm", "model")
                local_thread = threading.Thread(
                    target=self._run_local, args=(text_input, frame), daemon=True
                )
                local_thread.start()
            else:
                with self._lock:
                    self._set_mode_locked("cloud")

            cloud_thread = threading.Thread(
                target=self._run_cloud, args=(text_input, frame), daemon=True
            )
            cloud_thread.start()

            if not self._mode_event.wait(self.wait_timeout):
                self.logger.warning("Timed out waiting for the first local or cloud phrase.")

            mode = self._read_mode()
            if mode == "cloud":
                self._route = "cloud-only" if not self.edge_enabled else "cloud-first"
                self._finish_cloud_only(local_thread, cloud_thread)
            elif mode == "local":
                self._finish_local_first(local_thread, cloud_thread)
            else:
                self._stop_local.set()
                if local_thread is not None:
                    local_thread.join(timeout=self.wait_timeout)
                cloud_thread.join(timeout=self.wait_timeout)
                if self.cloud_raw.strip():
                    self._route = "cloud-only"
                    self._speak_text_as_phrases(self.cloud_raw, "cloud")
                elif self.local_raw.strip():
                    self._route = "local-only"
                    self._flush_local_remainder()
                else:
                    self._route = "failure"
        finally:
            self._stop_local.set()
            if local_thread is not None and local_thread.is_alive():
                local_thread.join(timeout=2)
            if cloud_thread is not None and cloud_thread.is_alive():
                cloud_thread.join(timeout=2)

        spoken = " ".join(phrase["text"] for phrase in self._handed if not phrase["cancelled"])
        if not spoken.strip():
            self._route = "failure"
            spoken = FAILURE_TEXT
        self._log_result(spoken)
        return spoken

    def _finish_cloud_only(self, local_thread, cloud_thread):
        self._stop_local.set()
        if local_thread is not None:
            local_thread.join(timeout=self.wait_timeout)
        local_orders = [phrase["order"] for phrase in self.local_phrases]
        if local_orders:
            self.grm.cancel_orders(self.response_id, local_orders)
            for phrase in self.local_phrases:
                phrase["cancelled"] = True
                self._trace({
                    "type": "cancel",
                    "source": "local",
                    "order": phrase["order"],
                    "text": phrase["text"],
                    "start": phrase["start"],
                    "end": phrase["end"],
                })
        cloud_thread.join(timeout=self.wait_timeout)

    def _finish_local_first(self, local_thread, cloud_thread):
        if not self._cloud_done.wait(self.wait_timeout):
            self.logger.warning("Timed out waiting for the cloud response.")
        cloud_thread.join(timeout=2)
        self._stop_local.set()
        if local_thread is not None:
            local_thread.join(timeout=self.wait_timeout)
        if self.cloud_raw.strip():
            self._route = "local-first"
            self._merge()
        else:
            self._route = "local-only"
            self._flush_local_remainder()

    def _merge(self):
        now = self.clock()
        with self._lock:
            local_raw = self.local_raw
            playback = self._playback
        measured = _measured_spoken_chars(local_raw, self.local_phrases, playback, now)
        time_spoken = self._time_spoken_chars(now, len(local_raw))
        time_index = snap_cut(
            local_raw,
            project_index(time_spoken, self.chars_per_second, self.merge_wait_s, len(local_raw)),
        )
        if measured is None:
            chosen = time_spoken
            callback_index = time_index
        else:
            chosen = measured
            callback_index = snap_cut(
                local_raw,
                project_index(measured, self.chars_per_second, self.merge_wait_s, len(local_raw)),
            )
        snapped = callback_index if measured is not None else time_index
        snapped = _extend_inflight(snapped, self.local_phrases, playback)
        self._cut = {
            "callback_index": callback_index,
            "time_index": time_index,
            "snapped": snapped,
        }
        self._emit_fragment_up_to(snapped)

        keep = -1
        for phrase in self.local_phrases:
            if phrase["end"] <= snapped and not phrase["cancelled"]:
                keep = max(keep, phrase["order"])
        self.grm.cancel_unspoken(self.response_id, keep)
        for phrase in self.local_phrases:
            if phrase["order"] > keep and not phrase["cancelled"]:
                phrase["cancelled"] = True
                self._trace({
                    "type": "cancel",
                    "source": "local",
                    "order": phrase["order"],
                    "text": phrase["text"],
                    "start": phrase["start"],
                    "end": phrase["end"],
                })

        spoken_text = " ".join(
            phrase["text"] for phrase in self.local_phrases if not phrase["cancelled"]
        )
        self._trace({
            "type": "cut",
            "time_index": time_index,
            "callback_index": callback_index,
            "snapped": snapped,
            "spoken_text": spoken_text,
        })
        prompt = build_merge_prompt(local_raw.strip(), self.cloud_raw.strip(), spoken_text)
        self._last_merge_prompt = prompt
        try:
            produced = self._consume_merge(prompt)
        except Exception as exc:
            self.logger.error(f"Merge model failed: {exc}")
            produced = False
        if not produced:
            self._route = "merge-fallback"
            self.logger.warning("Merge produced no text. Speaking the cloud response.")
            self._speak_text_as_phrases(self.cloud_raw, "cloud")

    def _run_local(self, prompt, frame):
        stream = None
        try:
            stream = self._edge_call()(prompt, frame)
            if stream is None:
                return
            leg_t0 = self.clock()
            for chunk in stream:
                stopped = self._stop_local.is_set()
                text = _chunk_text(chunk)
                if text:
                    self._edge_events.append(_timed_chunk(text, leg_t0, self.clock()))
                    self._trace({"type": "token", "source": "local", "text": text})
                    with self._lock:
                        self.local_raw += text
                    self._emit_completed("local")
                if stopped or self._stop_local.is_set():
                    break
        except Exception as exc:
            self.logger.error(f"Local stream failed: {exc}")
        finally:
            _close_stream(stream)
            self._mark_finished("local")

    def _run_cloud(self, prompt, frame):
        stream = None
        try:
            stream = self._cloud_call()(prompt, frame)
            if stream is None:
                return
            leg_t0 = self.clock()
            for chunk in stream:
                text = _chunk_text(chunk)
                if not text:
                    continue
                self._cloud_events.append(_timed_chunk(text, leg_t0, self.clock()))
                self._trace({"type": "token", "source": "cloud", "text": text})
                with self._lock:
                    self.cloud_raw += text
                self._emit_completed("cloud")
            self._emit_tail("cloud")
            self._trace({"type": "cloud_done", "text": self.cloud_raw})
        except Exception as exc:
            self.logger.error(f"Cloud stream failed: {exc}")
        finally:
            _close_stream(stream)
            if self._cloud_done_at is None:
                self._cloud_done_at = self.clock()
            self._cloud_done.set()
            self._mark_finished("cloud")

    def _emit_completed(self, source):
        ready = []
        with self._lock:
            raw = self.local_raw if source == "local" else self.cloud_raw
            scan = self.local_scan if source == "local" else self.cloud_scan
            while True:
                punct = None
                for index in range(scan, len(raw)):
                    if raw[index] in HARD_PUNCTUATION:
                        punct = index
                        break
                if punct is None:
                    break
                start = scan
                scan = punct + 1
                piece = raw[start:scan]
                if piece.strip():
                    ready.append((piece.strip(), start, scan))
            if source == "local":
                self.local_scan = scan
            else:
                self.cloud_scan = scan
        for text, start, end in ready:
            if source == "local":
                self._emit_local(text, start, end)
            else:
                self._emit_cloud(text, start, end)

    def _emit_tail(self, source):
        """Speak a final fragment that never reached punctuation."""
        with self._lock:
            if source == "local":
                raw = self.local_raw
                scan = self.local_scan
            else:
                raw = self.cloud_raw
                scan = self.cloud_scan
            piece = raw[scan:]
            start = scan
            end = len(raw)
            if source == "local":
                self.local_scan = end
            else:
                self.cloud_scan = end
        if not piece.strip():
            return
        if source == "local":
            if self._stop_local.is_set() or self._read_mode() == "cloud":
                return
            self._emit_local(piece.strip(), start, end)
        else:
            self._emit_cloud(piece.strip(), start, end)

    def _emit_local(self, text, start, end):
        phrase = None
        with self._lock:
            now = self.clock()
            if self.mode == "cloud":
                return
            if self.local_first is None:
                self.local_first = now
                if self.cloud_first is not None and self.cloud_first <= now:
                    self._set_mode_locked("cloud")
                    return
                self._set_mode_locked("local")
            if self.first_local_enqueued_at is None:
                self.first_local_enqueued_at = now
            phrase = self._allocate_phrase(text, "local", start, end)
            self.local_phrases.append(phrase)
        self._send(phrase)

    def _emit_cloud(self, text, start, end):
        phrase = None
        with self._lock:
            now = self.clock()
            if self.cloud_first is None:
                self.cloud_first = now
                if self.mode != "local" and (self.local_first is None or self.cloud_first <= self.local_first):
                    self._set_mode_locked("cloud")
            if self.mode != "cloud":
                return
            phrase = self._allocate_phrase(text, "cloud", start, end)
        self._send(phrase)

    def _emit_fragment_up_to(self, snapped):
        with self._lock:
            if snapped <= self.local_scan:
                return
            piece = self.local_raw[self.local_scan:snapped]
            start = self.local_scan
            self.local_scan = snapped
        if not piece.strip():
            return
        with self._lock:
            phrase = self._allocate_phrase(piece.strip(), "local", start, snapped)
            self.local_phrases.append(phrase)
        self._send(phrase)

    def _flush_local_remainder(self):
        with self._lock:
            piece = self.local_raw[self.local_scan:]
            start = self.local_scan
            end = len(self.local_raw)
            self.local_scan = end
        if not piece.strip():
            return
        with self._lock:
            if self.local_first is None:
                self.local_first = self.clock()
                self._set_mode_locked("local")
                self.first_local_enqueued_at = self.local_first
            phrase = self._allocate_phrase(piece.strip(), "local", start, end)
            self.local_phrases.append(phrase)
        self._send(phrase)

    def _consume_merge(self, prompt):
        self.llm_used = _configured_model(self._config, "ollama_llm", "model")
        stream = self._merge_call()(prompt)
        produced = False
        if stream is None:
            return False
        buffer = ""
        leg_t0 = self.clock()
        try:
            for chunk in stream:
                text = _chunk_text(chunk)
                if not text:
                    continue
                produced = True
                self._merge_events.append(_timed_chunk(text, leg_t0, self.clock()))
                self._trace({"type": "token", "source": "merge", "text": text})
                buffer += text
                phrases, buffer = split_ready_phrases(buffer)
                for phrase_text in phrases:
                    self._emit_other(phrase_text, "merge")
        finally:
            _close_stream(stream)
        if buffer.strip():
            produced = True
            self._emit_other(buffer.strip(), "merge")
        return produced

    def _speak_text_as_phrases(self, text, source):
        phrases, rest = split_ready_phrases(text)
        for phrase_text in phrases:
            self._emit_other(phrase_text, source)
        if rest.strip():
            self._emit_other(rest.strip(), source)

    def _emit_other(self, text, source):
        with self._lock:
            if source == "merge" and self._merge_first_at is None:
                self._merge_first_at = self.clock()
            phrase = self._allocate_phrase(text, source, None, None)
        self._send(phrase)

    def _allocate_phrase(self, text, source, start, end):
        phrase = {
            "order": self._next_order,
            "text": text,
            "source": source,
            "start": start,
            "end": end,
            "cancelled": False,
        }
        self._next_order += 1
        self._handed.append(phrase)
        return phrase

    def _send(self, phrase):
        self.grm.add_message(
            phrase["text"],
            message_expiration=PHRASE_LIFETIME,
            priority=PHRASE_PRIORITY,
            response_id=self.response_id,
            order=phrase["order"],
        )
        self._trace({
            "type": "phrase",
            "source": phrase["source"],
            "text": phrase["text"],
            "start": phrase["start"],
            "end": phrase["end"],
            "order": phrase["order"],
        })

    def _on_playback(self, response_id, payload):
        if response_id != self.response_id:
            return
        with self._lock:
            self._playback = dict(payload)
            if payload.get("event") == "done":
                phrase = payload.get("phrase") or ""
                duration = payload.get("phrase_duration_s") or 0
                if phrase and duration > 0:
                    self._playback_samples.append((len(phrase), float(duration)))
        self._trace({"type": "playback", **payload})

    def _time_spoken_chars(self, now, length):
        if self.first_local_enqueued_at is None or length <= 0:
            return 0
        elapsed = max(0.0, now - self.first_local_enqueued_at)
        return min(length, self.chars_per_second * elapsed)

    def _snapshot_frame(self):
        if self._frame_source is not None:
            return self._frame_source()
        try:
            from modules.shared_buffer import video_buffer
            return video_buffer.retrieve_frame()
        except Exception as exc:
            self.logger.warning(f"Could not snapshot a frame: {exc}")
            return None

    def _edge_call(self):
        if self._edge_stream is not None:
            return self._edge_stream

        def _stream(prompt, frame):
            yield from _production_edge_stream(prompt, frame, on_model=self._note_edge_model)
        return _stream

    def _cloud_call(self):
        if self._cloud_stream is not None:
            return self._cloud_stream

        def _stream(prompt, frame):
            yield from _production_cloud_stream(prompt, frame, on_model=self._note_cloud_model)
        return _stream

    def _merge_call(self):
        if self._merge_stream is not None:
            return self._merge_stream

        def _stream(prompt):
            yield from _production_merge_stream(prompt, on_model=self._note_llm)
        return _stream

    def _note_edge_model(self, name):
        if name:
            self.edge_mllm_used = name

    def _note_llm(self, name):
        if name:
            self.llm_used = name

    def _note_cloud_model(self, name):
        if name and not self.cloud_model_used:
            self.cloud_model_used = name

    def _elapsed(self, moment):
        if moment is None or self._t0 is None:
            return None
        return round(max(0.0, moment - self._t0), 4)

    def _log_result(self, spoken):
        """Log the route, model text, and one line the fusion replay can play."""
        try:
            record = self._replay_record(spoken)
            cut = record["cut"] or {}
            timings = record["timings"]
            self.logger.info(
                f"Fusion result route={record['route']} response_id={record['response_id']} "
                f"edge_mllm={record['edge_mllm']} llm={record['llm']} cloud_model={record['cloud_model']} "
                f"local_first={timings['local_first_s']} cloud_done={timings['cloud_done_s']} "
                f"cut_callback={cut.get('callback_index')} cut_time={cut.get('time_index')} "
                f"snapped={cut.get('snapped')}"
            )
            self.logger.info(f"Fusion local: {record['local_response']}")
            self.logger.info(f"Fusion cloud: {record['cloud_response']}")
            self.logger.info(f"Fusion merge: {record['merge_response']}")
            self.logger.info(f"Fusion spoken: {record['spoken']}")
            payload = json.dumps(record, ensure_ascii=False, separators=(",", ":"))
            self.logger.info(f"{REPLAY_PREFIX}{payload}")
        except Exception as exc:
            self.logger.error(f"Could not write the fusion record: {exc}")

    def _replay_record(self, spoken):
        with self._lock:
            samples = list(self._playback_samples)
        sample_chars = sum(count for count, _duration in samples)
        sample_seconds = sum(duration for _count, duration in samples)
        if sample_seconds > 0:
            actual = sample_chars / sample_seconds
        else:
            actual = self.chars_per_second
        cloud_model = self.cloud_model_used
        if cloud_model is None and self.cloud_raw.strip():
            cloud_model = _configured_cloud_model(self._config)
        return {
            "prompt": self._prompt,
            "route": self._route or "failure",
            "response_id": self.response_id,
            "edge_enabled": bool(self.edge_enabled),
            "edge_mllm": self.edge_mllm_used if self.edge_enabled else None,
            "llm": self.llm_used,
            "cloud_model": cloud_model,
            "assumed_tts_chars_per_second": self.chars_per_second,
            "actual_tts_chars_per_second": round(actual, 4),
            "merge_first_phrase_p90_s": self.merge_wait_s,
            "edge_events": list(self._edge_events),
            "cloud_events": list(self._cloud_events),
            "merge_events": list(self._merge_events),
            "local_response": self.local_raw,
            "cloud_response": self.cloud_raw,
            "merge_response": "".join(event["text"] for event in self._merge_events),
            "spoken": spoken,
            "cut": dict(self._cut) if self._cut else None,
            "timings": {
                "local_first_s": self._elapsed(self.local_first),
                "cloud_done_s": self._elapsed(self._cloud_done_at),
                "merge_first_phrase_s": self._elapsed(self._merge_first_at),
            },
            "phrases": [
                {
                    "order": phrase["order"],
                    "source": phrase["source"],
                    "text": phrase["text"],
                    "cancelled": phrase["cancelled"],
                }
                for phrase in self._handed
            ],
        }

    def _set_mode_locked(self, mode):
        if self.mode is None:
            self.mode = mode
            self.logger.info(f"Fusion mode: {mode}")
            self._trace({"type": "mode", "mode": mode})
            self._mode_event.set()

    def _read_mode(self):
        with self._lock:
            return self.mode

    def _mark_finished(self, which):
        with self._lock:
            self._finished.add(which)
            needed = {"cloud"} if not self.edge_enabled else {"local", "cloud"}
            if self.mode is None and needed <= self._finished:
                self._mode_event.set()

    def _trace(self, event):
        if self.listener is None:
            return
        try:
            self.listener(dict(event))
        except Exception as exc:
            self.logger.error(f"Fusion listener failed: {exc}")

    def _reset_state(self):
        self.response_id = uuid.uuid4().hex
        self.mode = None
        self.local_raw = ""
        self.cloud_raw = ""
        self.local_scan = 0
        self.cloud_scan = 0
        self.local_phrases = []
        self._handed = []
        self._next_order = 0
        self.local_first = None
        self.cloud_first = None
        self.first_local_enqueued_at = None
        self._playback = None
        self._finished = set()
        self._last_merge_prompt = None
        self._stop_local = threading.Event()
        self._mode_event = threading.Event()
        self._cloud_done = threading.Event()
        self._t0 = None
        self._prompt = ""
        self._route = None
        self._cut = None
        self._edge_events = []
        self._cloud_events = []
        self._merge_events = []
        self._cloud_done_at = None
        self._merge_first_at = None
        self._playback_samples = []
        self.edge_mllm_used = None
        self.llm_used = None
        self.cloud_model_used = None


def _settings(config, edge_enabled):
    if config is None:
        from config.config import get_config
        config = get_config()
    fusion = config.get("fusion", {})
    piper = config.get("piper_tts", {})
    ollama = config.get("ollama_llm", {})
    chars_per_second = float(piper.get("chars_per_second", 15))
    merge_wait_s = float(fusion.get("merge_first_phrase_p90_s", 0.5))
    if edge_enabled is None:
        edge_enabled = bool(ollama.get("ollama_llm_enabled", False))
    return chars_per_second, merge_wait_s, edge_enabled, config


def _configured_model(config, section, key):
    if not config:
        return None
    return (config.get(section) or {}).get(key)


def _configured_cloud_model(config):
    if not config:
        return None
    api = config.get("openrouter_api") or {}
    models = list(api.get("model_priority") or [])
    if models:
        return models[0]
    return api.get("model")


def _timed_chunk(text, leg_t0, now):
    return {"text": text, "t": round(max(0.0, now - leg_t0), 4)}


def _chunk_text(chunk):
    if chunk is None:
        return ""
    return str(chunk)


def _close_stream(stream):
    closer = getattr(stream, "close", None)
    if not callable(closer):
        return
    try:
        closer()
    except Exception:
        pass


def _measured_spoken_chars(local_raw, phrases, playback, now):
    if not playback:
        return None
    phrase = next((item for item in phrases if item["order"] == playback.get("order")), None)
    if phrase is None or phrase["start"] is None:
        return None
    if playback.get("event") == "done":
        return phrase["end"]
    start = phrase["start"]
    end = phrase["end"]
    raw = local_raw[start:end]
    lead = len(raw) - len(raw.lstrip())
    body = len(raw.strip())
    duration = playback.get("phrase_duration_s") or 0
    started = playback.get("phrase_started_at")
    if duration <= 0 or started is None:
        return start + lead
    fraction = min(1.0, max(0.0, (now - started) / duration))
    return start + lead + int(fraction * body)


def _extend_inflight(snapped, phrases, playback):
    if not playback or playback.get("event") != "start":
        return snapped
    phrase = next((item for item in phrases if item["order"] == playback.get("order")), None)
    if phrase is None or phrase["start"] is None:
        return snapped
    if phrase["start"] < snapped < phrase["end"]:
        return phrase["end"]
    return snapped


def _production_edge_stream(prompt, frame, on_model=None):
    from modules.ai_manager import AI_manager
    _report_edge_model(AI_manager, on_model)
    result = AI_manager.execute_module(
        lambda module: module.__class__.__name__ == "EdgeMLLMModule",
        "fusion_edge",
        prompt,
        use_image=frame is not None,
        frame=frame,
    )
    if result is None:
        return
        yield ""
    yield from result


def _production_merge_stream(prompt, on_model=None):
    from modules.ai_manager import AI_manager
    _report_edge_model(AI_manager, on_model)
    result = AI_manager.execute_module(
        lambda module: module.__class__.__name__ == "EdgeMLLMModule",
        "fusion_merge",
        prompt,
    )
    if result is None:
        return
        yield ""
    yield from result


def _report_edge_model(ai_manager, on_model):
    if on_model is None:
        return
    module = ai_manager.get_module(lambda item: item.__class__.__name__ == "EdgeMLLMModule")
    if module is None:
        return
    name = getattr(module, "model_name", None)
    if name:
        on_model(name)


def _production_cloud_stream(prompt, frame, on_model=None):
    from config.config import get_config
    from modules.ai_manager import AI_manager

    cfg = get_config().get("openrouter_api", {})
    models = list(cfg.get("model_priority") or [])
    if not models:
        legacy = cfg.get("model")
        models = [legacy] if legacy else []
    logger = get_logger("FusionModule")
    for model_name in models:
        logger.info(f"Cloud model: {model_name}")
        try:
            result = AI_manager.execute_module(
                lambda module: module.__class__.__name__ == "APIMLLMModule",
                "fusion_cloud",
                prompt,
                use_image=frame is not None,
                model=model_name,
                stream=True,
                frame=frame,
            )
        except Exception as exc:
            logger.warning(f"Cloud model {model_name} failed to start: {exc}")
            continue
        if result is None:
            continue
        got = False
        try:
            for chunk in result:
                if chunk:
                    if not got and on_model is not None:
                        on_model(model_name)
                    got = True
                    yield chunk
        except Exception as exc:
            logger.warning(f"Cloud model {model_name} stream failed: {exc}")
        finally:
            _close_stream(result)
        if got:
            return
    logger.warning("Cloud models produced no text.")
