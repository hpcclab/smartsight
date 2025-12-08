# operations/TTS/TestingThread.py

import threading
import json
import time

import keyboard  # used to detect 'T' key presses


class TestingThread(threading.Thread):
    """
    Thread that listens for the 'T' key and toggles a local
    "testing mode" on/off.

    While testing mode is ON, it streams messages from
    src\\operations\\testData.json to the callback.

    callback(text, priority) should match TTSThread.add_message.
    """

    def __init__(self, callback, daemon=True, key: str = "t"):
        super().__init__(daemon=daemon, name="TestingThread")
        self.callback = callback
        self.stop_running = threading.Event()
        self.key = key.lower()
        self._was_pressed = False      # for edge detection of key
        self._testing_active = False   # ON/OFF state of testing mode
        print(f"{self.name} initialized!")

    def run(self):
        data_file = r"src\operations\TTS\testData.json"
        print(f"{self.name} listening for '{self.key.upper()}' key to toggle testing mode...")

        while not self.stop_running.is_set():
            try:
                is_down = keyboard.is_pressed(self.key)

                # Rising edge: key just pressed -> toggle testing state
                if is_down and not self._was_pressed:
                    self._was_pressed = True
                    self._testing_active = not self._testing_active
                    state = "ON" if self._testing_active else "OFF"
                    print(f"{self.name}: testing mode toggled {state}.")

                    # Optional spoken feedback via callback
                    if self.callback is not None:
                        msg = f"Testing mode {state.lower()}."
                        try:
                            self.callback(msg, "testing")
                        except TypeError:
                            self.callback(msg)

                # Key released: arm for next press
                if not is_down:
                    self._was_pressed = False

                # If testing is active, play data (non-blocking style)
                if self._testing_active:
                    self._play_step(data_file)

                time.sleep(0.05)

            except Exception as e:
                print(f"{self.name} error while polling key '{self.key}': {e}")
                time.sleep(0.1)

        print(f"{self.name} exiting run loop.")

    # --- incremental JSON playback state ---
    _data = None
    _index = 0
    _data_loaded = False

    def _load_data_if_needed(self, data_file: str):
        if self._data_loaded:
            return
        try:
            with open(data_file, "r", encoding="utf-8") as file:
                self._data = json.load(file)
            self._index = 0
            self._data_loaded = True
            print(f"{self.name}: loaded {len(self._data)} test messages.")
        except FileNotFoundError:
            print(f"{self.name}: file not found: {data_file}")
            self._data = []
            self._index = 0
            self._data_loaded = True
        except Exception as e:
            print(f"{self.name} error loading JSON: {e}")
            self._data = []
            self._index = 0
            self._data_loaded = True

    def _play_step(self, data_file: str):
        """
        Send at most one JSON entry per call so that
        we can stop immediately when testing is toggled OFF.
        """
        self._load_data_if_needed(data_file)

        if not self._data or self._index >= len(self._data):
            return  # nothing left to play

        item = self._data[self._index]
        self._index += 1

        if self.callback is None:
            return

        text = item.get("text", "")
        priority = item.get("priority", "testing")
        try:
            self.callback(text, priority)
        except TypeError:
            self.callback(text)

    def stop(self):
        """Signal the thread to stop completely."""
        self.stop_running.set()
        print(f"{self.name} has been asked to stop.")


# Optional local test harness
class Receiver:
    def handle_message(self, text, priority):
        print(f"Got message [{priority}]: {text}")


def main():
    receiver = Receiver()
    thread = TestingThread(callback=receiver.handle_message)
    thread.start()
    print("Press 'T' to toggle testing mode ON/OFF. Ctrl+C to quit.")
    try:
        while True:
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("KeyboardInterrupt received, stopping TestingThread...")
        thread.stop()
        time.sleep(0.2)


if __name__ == "__main__":
    main()
