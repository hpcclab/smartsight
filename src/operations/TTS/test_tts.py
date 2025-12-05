import time
import threading
import sys
import random
import queue

# Try to import msvcrt for Windows non-blocking input
try:
    import msvcrt
except ImportError:
    msvcrt = None

# Import the provided class
from TextToSpeechThread import TTSThread

class TestColors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

class MessageFeeder:
    """
    A persistent thread that feeds messages at a specific interval.
    Improved to be pausable without killing the thread.
    """
    def __init__(self, tts_thread, priority, interval):
        self.tts = tts_thread
        self.priority = priority
        self.interval = interval
        self.active = False
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.counter = 1
        self.thread.start()

    def _run(self):
        while not self.stop_event.is_set():
            if self.active:
                msg = f"Continuous {self.priority} message #{self.counter}"
                # Add some length variance to passive messages to test overlap
                if self.priority == 'passive' and random.random() > 0.7:
                    msg += ". This is an extended sentence to occupy the engine."
                
                self.tts.add_message(msg, priority=self.priority)
                self.counter += 1
                time.sleep(self.interval)
            else:
                time.sleep(0.1)

    def toggle(self):
        self.active = not self.active
        return self.active

    def stop(self):
        self.active = False
        self.stop_event.set()
        self.thread.join(timeout=1)

class TTSTester:
    def __init__(self):
        self.tts = TTSThread()
        self.feeders = {}
        self.running = True

    def log(self, msg):
        print(f"{TestColors.OKCYAN}[TESTER] {msg}{TestColors.ENDC}")

    def setup(self):
        self.log("Initializing TTS Thread...")
        self.tts.start()
        
        # Initialize feeders (they start paused)
        self.feeders['passive'] = MessageFeeder(self.tts, 'passive', 3.0)
        self.feeders['active'] = MessageFeeder(self.tts, 'active', 1.5)
        self.feeders['urgent'] = MessageFeeder(self.tts, 'urgent', 5.0) # Urgent usually manual, but feeder exists for chaos
        
        time.sleep(1) # Let engine warm up

    def get_input(self):
        """Cross-platform non-blocking input handling."""
        if msvcrt:
            if msvcrt.kbhit():
                try:
                    return msvcrt.getch().decode('utf-8').lower()
                except:
                    return None
        return None

    def print_menu(self):
        print(f"\n{TestColors.HEADER}--- SmartSight TTS Control Panel ---{TestColors.ENDC}")
        print(f"{TestColors.BOLD}Continuous Feeders (Toggle):{TestColors.ENDC}")
        print(f"  [1] Passive (3.0s)  Status: {'ON' if self.feeders['passive'].active else 'OFF'}")
        print(f"  [2] Active  (1.5s)  Status: {'ON' if self.feeders['active'].active else 'OFF'}")
        print(f"  [3] Urgent  (5.0s)  Status: {'ON' if self.feeders['urgent'].active else 'OFF'}")
        print(f"\n{TestColors.BOLD}One-Shot Tests:{TestColors.ENDC}")
        print(f"  [i] Interruption Test (Active interrupting Passive)")
        print(f"  [s] Stress Test (50 mixed messages rapidly)")
        print(f"  [c] Queue Clear Test (Fill & Kill)")
        print(f"  [l] Long Message (Test specific reading)")
        print(f"\n{TestColors.BOLD}Controls:{TestColors.ENDC}")
        print(f"  [v] View Queue")
        print(f"  [x] Stop/Clear TTS Engine")
        print(f"  [q] Quit Application")
        print("-" * 40)

    def test_interruption(self):
        self.log("Starting Interruption Test...")
        self.log("1. Sending long PASSIVE message...")
        long_text = "This is a very long passive message intended to describe the scenery in great detail, allowing time for an interruption to occur properly."
        self.tts.add_message(long_text, "passive")
        
        time.sleep(1.5) # Wait for it to start speaking
        
        self.log("2. Sending URGENT message (Should cut off passive)...")
        self.tts.add_message("OBSTACLE DETECTED IMMEDIATE STOP.", "urgent")

    def test_stress(self):
        self.log("Starting Stress Test (50 messages)...")
        priorities = ['passive', 'active', 'urgent']
        for i in range(50):
            p = random.choice(priorities)
            self.tts.add_message(f"Rapid fire {p} {i}", p)
        self.log("50 messages injected. Check queue processing.")

    def test_clear_logic(self):
        self.log("Starting Queue Clear Test...")
        self.log("Injecting 10 messages...")
        for i in range(10):
            self.tts.add_message(f"Message {i} that should be deleted", "passive")
        
        self.log("Queue size is now: " + str(self.tts.queue.qsize()))
        time.sleep(0.5)
        self.log("Calling STOP (should clear queue)...")
        self.tts.stop()
        
        time.sleep(1)
        self.log("Restarting TTS thread for further tests...")
        # Since stop() sets the event, we need to create a new thread or reset the event
        # The provided TTS class isn't designed to be restarted easily after stop(), 
        # so we recreate it.
        self.tts = TTSThread()
        self.tts.start()
        # Update feeders to point to new tts instance
        for f in self.feeders.values():
            f.tts = self.tts

    def run(self):
        self.setup()
        self.print_menu()
        
        try:
            while self.running:
                key = self.get_input()
                
                if key:
                    if key == '1':
                        self.feeders['passive'].toggle()
                        self.print_menu()
                    elif key == '2':
                        self.feeders['active'].toggle()
                        self.print_menu()
                    elif key == '3':
                        self.feeders['urgent'].toggle()
                        self.print_menu()
                    
                    elif key == 'i':
                        self.test_interruption()
                        self.print_menu()
                    elif key == 's':
                        self.test_stress()
                        self.print_menu()
                    elif key == 'c':
                        self.test_clear_logic()
                        self.print_menu()
                    elif key == 'l':
                        self.tts.add_message("Reading a standard sentence to verify audio configuration.", "active")
                        
                    elif key == 'v':
                        q_list = list(self.tts.queue.queue)
                        self.log(f"Current Queue ({len(q_list)}):")
                        for idx, item in enumerate(q_list):
                            # item is (priority_int, message)
                            p_name = self.tts.PRIORITY_NAMES.get(item[0], '?')
                            print(f"   {idx+1}. [{p_name}] {item[1][:40]}...")

                    elif key == 'x':
                        self.log("Manual Stop Command Sent")
                        self.tts.stop()
                        # Simple restart logic for manual stop
                        time.sleep(1)
                        self.tts = TTSThread()
                        self.tts.start()
                        for f in self.feeders.values(): f.tts = self.tts
                        self.log("Engine Reset.")

                    elif key == 'q' or key == '\x1b': # ESC or q
                        self.running = False
                
                if not msvcrt:
                    # Fallback for non-windows environments (blocking input)
                    print("\nCommands: 1-3 (Toggles), i (Interrupt), s (Stress), c (Clear), v (View), q (Quit)")
                    key = input("Enter command: ").strip().lower()
                    # (Logic repeated here effectively, or just rely on msvcrt for best experience)
                    # For this specific script, msvcrt is highly recommended due to the realtime nature of TTS
                
                time.sleep(0.05)

        except KeyboardInterrupt:
            pass
        finally:
            self.cleanup()

    def cleanup(self):
        self.log("Shutting down...")
        for name, feeder in self.feeders.items():
            feeder.stop()
        
        if self.tts.is_alive():
            self.tts.stop()
            self.tts.join()
        print("Test Complete.")

if __name__ == "__main__":
    tester = TTSTester()
    tester.run()