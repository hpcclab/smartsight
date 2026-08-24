import os
import time
import math
import queue
import threading
import logging
from config.config import get_config

class EdgeCloudMerger:
    """
    Response Fusion Engine implementing Algorithm 1:
    Hybrid edge-cloud MLLM output streaming with real-time fusion for seamless text-to-speech.
    """
    def __init__(self, ttsr: float = 15.0, t_est: float = 0.5):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.config = get_config()
        self.ttsr = ttsr    # TTS playback rate in characters per second
        self.t_est = t_est  # Estimated 95th Percentile fusion first phrase time in seconds

    def run_inference(self, text_input: str, use_image: bool = True, cloud_model: str = None, edge_model: str = None, frame=None):
        """
        Main entry point for active request inference using edge-cloud fusion (Algorithm 1).
        Yields tokens for streaming consumption.
        """
        from .shared_buffer import video_buffer
        if frame is not None:
            video_buffer.update(frame)

        from .ai_manager import AI_manager

        cloud_queue = queue.Queue()
        edge_queue = queue.Queue()

        cloud_stop_event = threading.Event()
        edge_stop_event = threading.Event()

        tau_c1 = [None]  # Timestamp of first phrase from Cloud MLLM
        tau_e1 = [None]  # Timestamp of first phrase from Edge MLLM

        full_cloud_tokens = []
        full_edge_tokens = []

        def run_cloud():
            try:
                self.logger.info("Starting Cloud MLLM worker thread...")
                generator = AI_manager.execute_module(
                    lambda m: m.__class__.__name__ == "APIMLLMModule",
                    "active_request_cloud",
                    text_input,
                    use_image=use_image,
                    model=cloud_model,
                    stream=True
                )

                if generator and not isinstance(generator, str) and (hasattr(generator, "__next__") or hasattr(generator, "__iter__")):
                    for token in generator:
                        if cloud_stop_event.is_set():
                            break
                        if tau_c1[0] is None:
                            tau_c1[0] = time.time()
                        full_cloud_tokens.append(token)
                        cloud_queue.put(token)
                elif isinstance(generator, str) and generator.strip():
                    if tau_c1[0] is None:
                        tau_c1[0] = time.time()
                    full_cloud_tokens.append(generator)
                    cloud_queue.put(generator)
            except Exception as e:
                self.logger.error(f"Error in Cloud MLLM worker: {e}")
            finally:
                cloud_queue.put(None)

        def run_edge():
            temp_img_path = None
            try:
                self.logger.info("Starting Edge MLLM worker thread...")
                kwargs = {}
                if use_image:
                    from .shared_buffer import video_buffer
                    import cv2
                    import tempfile

                    frame = video_buffer.retrieve_frame()
                    if frame is not None:
                        fd, temp_img_path = tempfile.mkstemp(suffix=".jpg")
                        os.close(fd)
                        cv2.imwrite(temp_img_path, frame)
                        kwargs["image_path"] = temp_img_path
                    else:
                        self.logger.warning("use_image is True, but no frame available for Edge MLLM.")

                generator = AI_manager.execute_module(
                    lambda m: m.__class__.__name__ == "EdgeMLLMModule",
                    "active_request_edge",
                    text_input,
                    **kwargs
                )

                if generator and not isinstance(generator, str) and (hasattr(generator, "__next__") or hasattr(generator, "__iter__")):
                    for token in generator:
                        if edge_stop_event.is_set():
                            break
                        if tau_e1[0] is None:
                            tau_e1[0] = time.time()
                        full_edge_tokens.append(token)
                        edge_queue.put(token)
                elif isinstance(generator, str) and generator.strip():
                    if tau_e1[0] is None:
                        tau_e1[0] = time.time()
                    full_edge_tokens.append(generator)
                    edge_queue.put(generator)
            except Exception as e:
                self.logger.error(f"Error in Edge MLLM worker: {e}")
            finally:
                edge_queue.put(None)
                if temp_img_path and os.path.exists(temp_img_path):
                    try:
                        os.remove(temp_img_path)
                    except Exception:
                        pass

        # Start parallel workers (Lines 2-4)
        t0 = time.time()
        t_cloud = threading.Thread(target=run_cloud, daemon=True)
        t_edge = threading.Thread(target=run_edge, daemon=True)

        t_cloud.start()
        t_edge.start()

        # Monitor which MLLM begins generating first (Lines 6-10)
        # Wait up to 3 seconds for at least one worker to yield a token
        while tau_c1[0] is None and tau_e1[0] is None and (time.time() - t0 < 3.0):
            time.sleep(0.01)

        # Check condition tau_c1 <= tau_e1
        # If Cloud generated first or Edge failed to start before Cloud
        cloud_first = False
        if tau_c1[0] is not None and (tau_e1[0] is None or tau_c1[0] <= tau_e1[0]):
            cloud_first = True

        if cloud_first:
            # Line 7: Stop Edge MLLM
            self.logger.info("Cloud MLLM generated first (or Edge unavailable). Stopping Edge MLLM.")
            edge_stop_event.set()

            # Line 8: Append all Cloud MLLM phrases to Q_TTS
            while True:
                token = cloud_queue.get()
                if token is None:
                    break
                yield token
        else:
            # Line 10: First phrase arrives from Edge MLLM
            self.logger.info("Edge MLLM generated first. Starting Edge response streaming...")
            t1 = tau_e1[0] or time.time()

            # Lines 12-14: While Cloud MLLM Last Token hasn't arrived
            # Stream Edge tokens out
            while t_cloud.is_alive():
                try:
                    token = edge_queue.get(timeout=0.02)
                    if token is not None:
                        yield token
                except queue.Empty:
                    pass

            # Drain any remaining tokens in edge_queue
            while not edge_queue.empty():
                try:
                    token = edge_queue.get_nowait()
                    if token is not None:
                        yield token
                except queue.Empty:
                    break

            # Line 15: t2 = CurrentTime() when full Cloud response arrived
            t2 = time.time()

            # Line 17: Stop Edge MLLM
            edge_stop_event.set()

            r_cloud = "".join(full_cloud_tokens).strip()
            r_edge = "".join(full_edge_tokens).strip()

            if not r_cloud:
                # If Cloud failed completely, continue streaming remaining Edge if any
                self.logger.warning("Cloud response was empty. Continuing with Edge MLLM response.")
                while True:
                    try:
                        token = edge_queue.get(timeout=0.05)
                        if token is None:
                            break
                        yield token
                    except queue.Empty:
                        if not t_edge.is_alive():
                            break
                return

            # Line 18: Calculate p = ceil(TTSR * (t2 - t1 + T_est))
            p = math.ceil(self.ttsr * (t2 - t1 + self.t_est))

            # Lines 19-27: Boundary adjustments for p
            p = self._calculate_aligned_index(p, r_edge)
            self.logger.info(f"Calculated character truncation index p={p} (len(R_edge)={len(r_edge)})")

            r_edge_p = r_edge[:p]

            # Line 29: R_Fusion = StartFusionEngineLLM(R_Edge[0:p], R_Cloud)
            self.logger.info(f"Triggering Fusion Engine LLM with R_edge[:{p}] and full R_cloud.")
            fusion_prompt = (
                f"You are a real-time Response Fusion Engine for a vision AI assistant.\n"
                f"The user was already spoken the following partial response: '{r_edge_p}'\n"
                f"The full ground-truth cloud model response is: '{r_cloud}'\n\n"
                f"Task: Continue the response smoothly starting right after the partial response. "
                f"Treat the cloud response as the ground truth. Correct any mistakes or conflicts in the partial response seamlessly, "
                f"and do not repeat information that was already spoken."
            )

            try:
                # Execute Fusion Engine LLM using EdgeMLLMModule (or fallback to APIMLLMModule if needed)
                fusion_gen = AI_manager.execute_module(
                    lambda m: m.__class__.__name__ in ("EdgeMLLMModule", "APIMLLMModule"),
                    "fusion_engine_request",
                    fusion_prompt,
                    stream=True
                )

                if fusion_gen and not isinstance(fusion_gen, str) and (hasattr(fusion_gen, "__next__") or hasattr(fusion_gen, "__iter__")):
                    # Line 30-32: While Fusion Engine LLM is generating tokens, append all FE phrases to Q_TTS
                    for token in fusion_gen:
                        yield token
                elif isinstance(fusion_gen, str) and fusion_gen.strip():
                    yield fusion_gen
            except Exception as e:
                self.logger.error(f"Error during Fusion Engine LLM generation: {e}")

    def _calculate_aligned_index(self, p: int, r_edge: str) -> int:
        """
        Adjust index p based on Lines 19-27 of Algorithm 1:
        - If p > len(r_edge): p = len(r_edge)
        - Else if earliest punctuation in r_edge from p to end: p = index of punctuation
        - Else if p is in last word and end char is not punctuation: p = last character of previous word
        - Else: p = last character of current word
        """
        if not r_edge:
            return 0

        if p > len(r_edge):
            return len(r_edge)

        punctuation_set = set(".,;:!?\n")

        # Line 21: Check for earliest punctuation from p to end
        earliest_punc = -1
        for i in range(p, len(r_edge)):
            if r_edge[i] in punctuation_set:
                earliest_punc = i + 1  # Include the punctuation character
                break

        if earliest_punc != -1:
            return earliest_punc

        # Line 23: Check if p is in middle of a word and end char is not punctuation
        if p < len(r_edge) and r_edge[p - 1].isalnum() and r_edge[p].isalnum():
            # Find last character of previous word
            last_space_punc = -1
            for i in range(p - 1, -1, -1):
                if r_edge[i] in set(" \n\r\t.,;:!?"):
                    last_space_punc = i + 1
                    break
            if last_space_punc != -1:
                return last_space_punc
            return p
        else:
            # Line 25: Last character of current word
            last_char_idx = p
            while last_char_idx < len(r_edge) and r_edge[last_char_idx].isalnum():
                last_char_idx += 1
            return last_char_idx

    @staticmethod
    def truncate_token_list(token_list: list, p: int):
        """
        Line 28: Truncate Q_TTS after the first p characters of R_Edge.
        Mutates token_list in place so that its accumulated string length equals p.
        """
        accumulated = 0
        cutoff_idx = len(token_list)
        remainder = ""

        for idx, tok in enumerate(token_list):
            if tok is None:
                cutoff_idx = idx
                break
            tok_str = str(tok)
            if accumulated + len(tok_str) >= p:
                cutoff_idx = idx
                remainder = tok_str[:p - accumulated]
                break
            accumulated += len(tok_str)

        del token_list[cutoff_idx:]
        if remainder:
            token_list.append(remainder)
