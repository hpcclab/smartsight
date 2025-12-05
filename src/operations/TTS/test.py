import time
import threading
import sys

try:
    import msvcrt
except ImportError:
    msvcrt = None

from TextToSpeechThread import TTSThread


class FeederManager:
    """Manages feeder threads with start/stop toggle functionality."""
    
    def __init__(self, name, feeder_func, interval):
        self.name = name
        self.feeder_func = feeder_func
        self.interval = interval
        self.thread = None
        self.stop_event = None
    
    def is_running(self):
        return self.thread is not None and self.thread.is_alive()
    
    def start(self, tts):
        if self.is_running():
            return False
        print(f"Starting {self.name} feeder...")
        self.stop_event = threading.Event()
        self.thread = threading.Thread(
            target=self.feeder_func,
            args=(tts, self.stop_event, self.interval),
            daemon=True
        )
        self.thread.start()
        return True
    
    def stop(self):
        if not self.is_running():
            return False
        print(f"Stopping {self.name} feeder...")
        self.stop_event.set()
        self.thread.join(timeout=1)
        self.thread = None
        self.stop_event = None
        return True
    
    def toggle(self, tts):
        if self.is_running():
            self.stop()
        else:
            self.start(tts)


def create_feeder(priority):
    """Factory function to create feeder functions for different priorities."""
    def feeder(tts, stop_event, interval):
        i = 1
        prefix = priority.upper() if priority != "passive" else "Passive ambient"
        while not stop_event.is_set():
            tts.add_message(f"{prefix} message #{i}", priority=priority)
            i += 1
            time.sleep(interval)
    return feeder


def get_key_input():
    """Get keyboard input (non-blocking on Windows, blocking fallback otherwise)."""
    if msvcrt:
        if msvcrt.kbhit():
            ch = msvcrt.getch()
            try:
                return ch.decode('utf-8')
            except Exception:
                return None
        return None
    else:
        print("\n(No msvcrt) Enter command: SPACE=passive, T=active, Q=urgent, V=view, C=clear, A=simulation, 0=exit")
        return input().strip()


def view_queue(tts):
    """Display current queue contents."""
    q = list(tts.queue.queue)
    print(f"Queue size: {len(q)}")
    for idx, (priority, msg) in enumerate(q, 1):
        preview = (msg[:80] + '...') if len(msg) > 80 else msg
        print(f" {idx}. {priority} - {preview}")


def clear_queue(tts):
    """Clear all messages from the queue."""
    cleared = 0
    while not tts.queue.empty():
        try:
            tts.queue.get_nowait()
            cleared += 1
        except Exception:
            break
    print(f"Cleared {cleared} messages from queue")


def run_interruption_simulation(tts, passive_mgr, active_mgr):
    """Run the active-into-passive interruption simulation."""
    print("Interruption simulation: Active into Passive")
    print("Press 'A' to inject active messages, 'S' to stop simulation")
    
    passive_mgr.start(tts)
    
    while True:
        key = get_key_input()
        if key:
            k = key.lower()
            if k == 'a':
                if not active_mgr.is_running():
                    active_mgr.start(tts)
                else:
                    print("Active feeder already running")
            elif k == 's' or k == '':
                print("Stopping interruption simulation...")
                active_mgr.stop()
                passive_mgr.stop()
                break
        time.sleep(0.1)


def print_instructions():
    print("\n--- SmartSight TTS Simulation ---")
    print("SPACE  - Toggle passive ambient messages")
    print("T      - Toggle active feeder")
    print("Q      - Toggle urgent feeder")
    print("V      - View queue")
    print("C      - Clear queue")
    print("A      - Run interruption simulation (active into passive)")
    print("0/ESC  - Exit")


def main():
    tts = TTSThread()
    tts.start()
    
    # Create feeder managers
    feeders = {
        'passive': FeederManager('PASSIVE', create_feeder('passive'), interval=3.0),
        'active': FeederManager('ACTIVE', create_feeder('active'), interval=1.5),
        'urgent': FeederManager('URGENT', create_feeder('urgent'), interval=0.8),
    }
    
    print_instructions()
    
    try:
        while True:
            key = get_key_input()
            
            if key:
                k = key.lower()
                
                if k == ' ':
                    feeders['passive'].toggle(tts)
                elif k == 't':
                    feeders['active'].toggle(tts)
                elif k == 'q':
                    feeders['urgent'].toggle(tts)
                elif k == 'v':
                    view_queue(tts)
                elif k == 'c':
                    clear_queue(tts)
                elif k == 'a':
                    run_interruption_simulation(tts, feeders['passive'], feeders['active'])
                    print_instructions()
                elif k == '0' or (len(key) > 0 and ord(key[0]) == 27):
                    print("Exiting simulation...")
                    break
            
            time.sleep(0.1)
    
    except KeyboardInterrupt:
        print("\nKeyboardInterrupt, shutting down...")
    
    # Cleanup all feeders
    for feeder in feeders.values():
        feeder.stop()
    
    tts.stop()
    tts.join(timeout=3)
    print("Simulation stopped. Goodbye!")


if __name__ == '__main__':
    main()
