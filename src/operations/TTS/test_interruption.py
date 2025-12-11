"""
Interruption Feature Test Simulation for PiperTTSThread

Adapted from the original pyttsx3-based TextToSpeechThread tests.
This version targets the Piper-based TTS thread while preserving
the original logic and scenarios.
"""

import threading
import time

from TextToSpeechThread import PiperTTSThread as TTSThread, Console
# Default model path (update if your model is in a different location)
DEFAULT_MODEL_PATH = r"C:\Users\Crack\Downloads\TTSTest\piper\en_US-lessac-low.onnx"

# ---------- Common test text and helpers ----------

LONG_MESSAGES = {
    "passive": (
        "This is a very long passive message that will take several seconds "
        "to complete. It is designed so that interruption by higher priority "
        "messages can be clearly observed during playback."
    ),
    "active": (
        "This is a long active priority message designed to test interruption "
        "scenarios. It will take several seconds to complete so the urgent "
        "message can preempt it mid-sentence."
    ),
    "urgent": (
        "This is an urgent priority message that should interrupt lower "
        "priority messages immediately when it arrives in the queue."
    ),
}

SHORT_MESSAGES = {
    "passive_first": "First passive message. This should complete fully without interruption.",
    "passive_second": "Second passive message. This should wait for the first to finish, not interrupt.",
    "passive_wait": "This passive message should wait and not interrupt the urgent one.",
    "active_interrupt": "ACTIVE INTERRUPTION! This active message should interrupt the passive one!",
    "urgent_interrupt_1": "URGENT INTERRUPTION! This urgent message should stop the passive one immediately!",
    "urgent_interrupt_2": "URGENT! This urgent message should interrupt the active one!",
}

DEFAULT_WAIT_STEP = 0.05  # finer granularity => faster test completion


def wait_for_idle_thread(tts, timeout=10.0, step=DEFAULT_WAIT_STEP):
    """
    Wait for PiperTTSThread to finish speaking and empty its queue.

    Returns True if idle within timeout, False otherwise.
    """
    start = time.time()
    while time.time() - start < timeout:
        with tts.speaking_lock:
            speaking = tts.is_speaking
        if not speaking:
            try:
                if tts.queue.empty():
                    return True
            except Exception:
                pass
        time.sleep(step)
    return False


# ---------- Interruption Test Suite ----------

class InterruptionTestSuite:
    """Test suite for Piper TTS interruption feature."""

    def __init__(self, model_path=None):
        # Use provided model_path or fall back to DEFAULT_MODEL_PATH
        self.tts = TTSThread()
        self.tts.start()
        Console.log("TestSuite", "Initialized Piper interruption test suite", Console.GREEN)

    def test_passive_to_urgent_interruption(self):
        """Test 1: Passive message interrupted by urgent message."""
        Console.log("TestSuite", "=" * 60, Console.CYAN)
        Console.log("TestSuite", "TEST 1: Passive -> Urgent Interruption", Console.BOLD + Console.CYAN)
        Console.log("TestSuite", "=" * 60, Console.CYAN)

        long_passive = LONG_MESSAGES["passive"]
        Console.log("TestSuite", "Sending long passive message...", Console.CYAN)
        self.tts.add_message(long_passive, priority="passive")

        Console.log("TestSuite", "Waiting briefly for passive to start...", Console.YELLOW)
        time.sleep(2.0)

        urgent_msg = SHORT_MESSAGES["urgent_interrupt_1"]
        Console.log("TestSuite", f"Sending urgent message: '{urgent_msg}'", Console.MAGENTA)
        self.tts.add_message(urgent_msg, priority="urgent")

        Console.log("TestSuite", "Waiting for messages to complete...", Console.CYAN)
        wait_for_idle_thread(self.tts, timeout=12.0)
        Console.log("TestSuite", "Test 1 Complete\n", Console.GREEN)

    def test_passive_to_active_interruption(self):
        """Test 2: Passive message interrupted by active message."""
        Console.log("TestSuite", "=" * 60, Console.CYAN)
        Console.log("TestSuite", "TEST 2: Passive -> Active Interruption", Console.BOLD + Console.CYAN)
        Console.log("TestSuite", "=" * 60, Console.CYAN)

        long_passive = LONG_MESSAGES["passive"]
        Console.log("TestSuite", "Sending long passive message...", Console.CYAN)
        self.tts.add_message(long_passive, priority="passive")

        time.sleep(2.0)

        active_msg = SHORT_MESSAGES["active_interrupt"]
        Console.log("TestSuite", f"Sending active message: '{active_msg}'", Console.BLUE)
        self.tts.add_message(active_msg, priority="active")

        Console.log("TestSuite", "Waiting for messages to complete...", Console.CYAN)
        wait_for_idle_thread(self.tts, timeout=12.0)
        Console.log("TestSuite", "Test 2 Complete\n", Console.GREEN)

    def test_active_to_urgent_interruption(self):
        """Test 3: Active message interrupted by urgent message."""
        Console.log("TestSuite", "=" * 60, Console.CYAN)
        Console.log("TestSuite", "TEST 3: Active -> Urgent Interruption", Console.BOLD + Console.CYAN)
        Console.log("TestSuite", "=" * 60, Console.CYAN)

        long_active = LONG_MESSAGES["active"]
        Console.log("TestSuite", "Sending long active message...", Console.BLUE)
        self.tts.add_message(long_active, priority="active")

        time.sleep(2.0)

        urgent_msg = SHORT_MESSAGES["urgent_interrupt_2"]
        Console.log("TestSuite", f"Sending urgent message: '{urgent_msg}'", Console.MAGENTA)
        self.tts.add_message(urgent_msg, priority="urgent")

        Console.log("TestSuite", "Waiting for messages to complete...", Console.CYAN)
        wait_for_idle_thread(self.tts, timeout=12.0)
        Console.log("TestSuite", "Test 3 Complete\n", Console.GREEN)

    def test_no_interruption_same_priority(self):
        """Test 4: Same priority messages should not interrupt."""
        Console.log("TestSuite", "=" * 60, Console.CYAN)
        Console.log("TestSuite", "TEST 4: Same Priority (No Interruption)", Console.BOLD + Console.CYAN)
        Console.log("TestSuite", "=" * 60, Console.CYAN)

        first_passive = SHORT_MESSAGES["passive_first"]
        Console.log("TestSuite", "Sending first passive message...", Console.CYAN)
        self.tts.add_message(first_passive, priority="passive")

        time.sleep(0.8)

        second_passive = SHORT_MESSAGES["passive_second"]
        Console.log("TestSuite", "Sending second passive message (should queue, not interrupt)...", Console.CYAN)
        self.tts.add_message(second_passive, priority="passive")

        Console.log("TestSuite", "Waiting for messages to complete...", Console.CYAN)
        wait_for_idle_thread(self.tts, timeout=10.0)
        Console.log("TestSuite", "Test 4 Complete\n", Console.GREEN)

    def test_no_interruption_lower_priority(self):
        """Test 5: Lower priority should not interrupt higher priority."""
        Console.log("TestSuite", "=" * 60, Console.CYAN)
        Console.log("TestSuite", "TEST 5: Lower Priority (No Interruption)", Console.BOLD + Console.CYAN)
        Console.log("TestSuite", "=" * 60, Console.CYAN)

        urgent_msg = LONG_MESSAGES["urgent"]
        Console.log("TestSuite", "Sending urgent message...", Console.MAGENTA)
        self.tts.add_message(urgent_msg, priority="urgent")

        time.sleep(1.5)

        passive_msg = SHORT_MESSAGES["passive_wait"]
        Console.log("TestSuite", "Sending passive message (should queue, not interrupt)...", Console.CYAN)
        self.tts.add_message(passive_msg, priority="passive")

        Console.log("TestSuite", "Waiting for messages to complete...", Console.CYAN)
        wait_for_idle_thread(self.tts, timeout=12.0)
        Console.log("TestSuite", "Test 5 Complete\n", Console.GREEN)

    def test_priority_queueing_order(self):
        """Test 6: Messages should be processed in priority order."""
        Console.log("TestSuite", "=" * 60, Console.CYAN)
        Console.log("TestSuite", "TEST 6: Priority Queue Ordering", Console.BOLD + Console.CYAN)
        Console.log("TestSuite", "=" * 60, Console.CYAN)

        messages = [
            ("First passive message", "passive"),
            ("First urgent message", "urgent"),
            ("Second passive message", "passive"),
            ("First active message", "active"),
            ("Third passive message", "passive"),
            ("Second urgent message", "urgent"),
        ]

        Console.log("TestSuite", "Sending messages in this order:", Console.CYAN)
        for msg, prio in messages:
            Console.log("TestSuite", f"  - {prio.upper()}: {msg}", Console.CYAN)
            self.tts.add_message(msg, priority=prio)

        Console.log("TestSuite", "\nExpected order (by priority):", Console.YELLOW)
        Console.log("TestSuite", "  1. First urgent message (urgent)", Console.MAGENTA)
        Console.log("TestSuite", "  2. Second urgent message (urgent)", Console.MAGENTA)
        Console.log("TestSuite", "  3. First active message (active)", Console.BLUE)
        Console.log("TestSuite", "  4. First passive message (passive)", Console.CYAN)
        Console.log("TestSuite", "  5. Second passive message (passive)", Console.CYAN)
        Console.log("TestSuite", "  6. Third passive message (passive)", Console.CYAN)

        Console.log("TestSuite", "\nWaiting for all messages to complete...", Console.CYAN)
        wait_for_idle_thread(self.tts, timeout=18.0)
        Console.log("TestSuite", "Test 6 Complete\n", Console.GREEN)

    def run_all_tests(self):
        """Run all interruption tests sequentially."""
        Console.log("TestSuite", "\n" + "=" * 60, Console.BOLD)
        Console.log("TestSuite", "RUNNING ALL INTERRUPTION TESTS (Piper)", Console.BOLD)
        Console.log("TestSuite", "=" * 60 + "\n", Console.BOLD)

        try:
            self.test_passive_to_urgent_interruption()
            time.sleep(1.0)

            self.test_passive_to_active_interruption()
            time.sleep(1.0)

            self.test_active_to_urgent_interruption()
            time.sleep(1.0)

            self.test_no_interruption_same_priority()
            time.sleep(1.0)

            self.test_no_interruption_lower_priority()
            time.sleep(1.0)

            self.test_priority_queueing_order()

            Console.log("TestSuite", "\n" + "=" * 60, Console.BOLD + Console.GREEN)
            Console.log("TestSuite", "ALL TESTS COMPLETE!", Console.BOLD + Console.GREEN)
            Console.log("TestSuite", "=" * 60 + "\n", Console.BOLD + Console.GREEN)

        except KeyboardInterrupt:
            Console.log("TestSuite", "\nTests interrupted by user", Console.YELLOW)
        finally:
            Console.log("TestSuite", "Cleaning up...", Console.YELLOW)
            self.tts.stop()
            self.tts.join(timeout=2.0)


# ---------- Continuous Interruption Simulator ----------

class ContinuousInterruptionSimulator:
    """Continuous simulation of interruption scenarios (Piper)."""

    def __init__(self, model_path=None):
       
        self.tts = TTSThread()
        self.tts.start()
        self.running = threading.Event()
        self.sim_thread = None
        Console.log("Simulator", "Initialized Piper continuous interruption simulator", Console.GREEN)

    def _simulation_loop(self, duration=0, interval=3.0):
        start_time = time.time()
        message_count = 0

        scenarios = [
            ("passive", "This is a passive message for continuous testing."),
            ("active", "Active message in continuous mode."),
            ("urgent", "URGENT: Continuous test message!"),
        ]

        long_messages = LONG_MESSAGES

        try:
            while self.running.is_set():
                if duration > 0 and (time.time() - start_time) >= duration:
                    Console.log("Simulator", f"Duration ({duration}s) reached. Stopping...", Console.YELLOW)
                    break

                if message_count % 3 == 0:
                    priority = "passive"
                    message = long_messages["passive"]
                elif message_count % 3 == 1:
                    priority = "urgent"
                    message = long_messages["urgent"]
                else:
                    priority, message = scenarios[message_count % len(scenarios)]

                message_count += 1
                elapsed = time.time() - start_time

                color = Console.CYAN if priority == "passive" else Console.BLUE if priority == "active" else Console.MAGENTA
                Console.log(
                    "Simulator",
                    f"[{message_count}] [{elapsed:.1f}s] Sending {priority.upper()}: '{message[:50]}...'",
                    color,
                )

                self.tts.add_message(message, priority=priority)

                try:
                    queue_size = self.tts.queue.qsize()
                    with self.tts.speaking_lock:
                        is_speaking = self.tts.is_speaking
                        current_priority = self.tts.current_priority_val
                    p_name = self.tts.PRIORITY_NAMES.get(current_priority, "none") if current_priority is not None else "none"
                    Console.log("Simulator", f"  Status - Queue: {queue_size}, Speaking: {is_speaking} ({p_name})", Console.CYAN)
                except Exception:
                    pass

                wait_time = interval + (message_count % 3) * 0.5
                waited = 0.0
                while waited < wait_time and self.running.is_set():
                    time.sleep(0.1)
                    waited += 0.1

        except Exception as e:
            Console.log("Simulator", f"Error in simulation: {e}", Console.RED)
        finally:
            elapsed = time.time() - start_time
            Console.log("Simulator", f"\nSimulation stopped. Messages sent: {message_count}, Duration: {elapsed:.1f}s", Console.YELLOW)
            self.running.clear()

    def start_simulation(self, duration=60, interval=3.0):
        """Start continuous simulation in background thread."""
        if self.running.is_set():
            Console.log("Simulator", "Stopping existing simulation...", Console.YELLOW)
            self.stop_simulation()
            time.sleep(0.5)

        Console.log("Simulator", "=" * 60, Console.CYAN)
        Console.log("Simulator", "STARTING CONTINUOUS INTERRUPTION SIMULATION (Piper)", Console.BOLD + Console.CYAN)
        Console.log("Simulator", "=" * 60, Console.CYAN)
        Console.log("Simulator", f"Duration: {'Infinite' if duration == 0 else f'{duration}s'}", Console.CYAN)
        Console.log("Simulator", f"Interval: ~{interval}s between messages", Console.CYAN)
        Console.log("Simulator", "Simulation runs in background. Use stop_simulation() to stop.\n", Console.CYAN)

        self.running.set()
        self.sim_thread = threading.Thread(
            target=self._simulation_loop,
            args=(duration, interval),
            daemon=True,
        )
        self.sim_thread.start()

    def stop_simulation(self):
        """Stop the running simulation."""
        if self.running.is_set():
            Console.log("Simulator", "Stopping simulation...", Console.YELLOW)
            self.running.clear()
            if self.sim_thread and self.sim_thread.is_alive():
                self.sim_thread.join(timeout=2)
            Console.log("Simulator", "Simulation stopped.", Console.GREEN)
        else:
            Console.log("Simulator", "No simulation running.", Console.YELLOW)

    def cleanup(self):
        """Cleanup resources."""
        self.stop_simulation()
        self.tts.stop()
        self.tts.join(timeout=2)


# ---------- Interactive Menu / CLI ----------

def interactive_menu(model_path=None):
    """Interactive menu for testing Piper interruption feature."""
    suite = InterruptionTestSuite(model_path=model_path)
    simulator = ContinuousInterruptionSimulator(model_path=model_path)

    try:
        while True:
            print("\n" + "=" * 60)
            Console.log("Menu", "PIPER INTERRUPTION TEST MENU", Console.BOLD)
            print("=" * 60)
            print("\n1. Test: Passive -> Urgent Interruption")
            print("2. Test: Passive -> Active Interruption")
            print("3. Test: Active -> Urgent Interruption")
            print("4. Test: Same Priority (No Interruption)")
            print("5. Test: Lower Priority (No Interruption)")
            print("6. Test: Priority Queue Ordering")
            print("7. Run All Tests")
            print("8. Start Continuous Simulation (30s)")
            print("9. Start Continuous Simulation (2min)")
            print("10. Start Continuous Simulation (Infinite)")
            print("11. Stop Continuous Simulation")
            print("12. Exit")

            try:
                choice = input("\nEnter choice (1-12): ").strip()

                if choice == "1":
                    suite.test_passive_to_urgent_interruption()
                elif choice == "2":
                    suite.test_passive_to_active_interruption()
                elif choice == "3":
                    suite.test_active_to_urgent_interruption()
                elif choice == "4":
                    suite.test_no_interruption_same_priority()
                elif choice == "5":
                    suite.test_no_interruption_lower_priority()
                elif choice == "6":
                    suite.test_priority_queueing_order()
                elif choice == "7":
                    suite.run_all_tests()
                elif choice == "8":
                    simulator.start_simulation(duration=30, interval=2.0)
                elif choice == "9":
                    simulator.start_simulation(duration=120, interval=3.0)
                elif choice == "10":
                    interval = float(input("Enter interval in seconds (default 3.0): ") or "3.0")
                    simulator.start_simulation(duration=0, interval=interval)
                elif choice == "11":
                    simulator.stop_simulation()
                elif choice == "12":
                    break
                else:
                    Console.log("Menu", "Invalid choice!", Console.RED)

            except KeyboardInterrupt:
                Console.log("Menu", "\nInterrupted by user", Console.YELLOW)
                break
            except Exception as e:
                Console.log("Menu", f"Error: {e}", Console.RED)

    finally:
        Console.log("Menu", "Cleaning up...", Console.YELLOW)
        simulator.cleanup()
        suite.tts.stop()
        suite.tts.join(timeout=2)
        Console.log("Menu", "Exit complete.", Console.GREEN)


def main():
    """Main entry point."""
    import sys

    # Optional: pass model path via CLI, e.g. --model /path/to/model.onnx
    model_path = None
    if "--model" in sys.argv:
        idx = sys.argv.index("--model")
        if idx + 1 < len(sys.argv):
            model_path = sys.argv[idx + 1]

    if len(sys.argv) > 1 and sys.argv[1] not in ("--model",):
        arg = sys.argv[1]
        if arg == "--all":
            suite = InterruptionTestSuite(model_path=model_path)
            try:
                suite.run_all_tests()
            finally:
                suite.tts.stop()
                suite.tts.join(timeout=2)
        elif arg == "--sim":
            duration_arg = None
            for i, v in enumerate(sys.argv):
                if v == "--sim" and i + 1 < len(sys.argv) and sys.argv[i + 1].isdigit():
                    duration_arg = int(sys.argv[i + 1])
                    break
            duration = duration_arg if duration_arg is not None else 60
            simulator = ContinuousInterruptionSimulator(model_path=model_path)
            try:
                simulator.start_simulation(duration=duration, interval=3.0)
                print("Press Ctrl+C to stop simulation...")
                while simulator.running.is_set():
                    time.sleep(1)
            except KeyboardInterrupt:
                pass
            finally:
                simulator.cleanup()
        else:
            print("Usage: python test_interruption_piper.py [--model PATH] [--all|--sim [duration]]")
            print("  --model PATH    : Piper model path passed to PiperTTSThread (optional)")
            print("  --all           : Run all interruption tests")
            print("  --sim [sec]     : Run continuous simulation (default 60s)")
            print("  (no args)       : Interactive menu")
    else:
        interactive_menu(model_path=model_path)


if __name__ == "__main__":
    main()
