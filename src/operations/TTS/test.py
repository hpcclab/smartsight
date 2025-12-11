import time
from TextToSpeechThread import PiperTTSThread



def main():
    # 1. Start TTS thread
    tts = PiperTTSThread()
    tts.start()

    # 2. Queue a long passive message
    long_text = (
        "This is a long passive test message that should be interrupted "
        "by an urgent message coming shortly after it starts."
    )
    tts.add_message(long_text, priority="passive")
    print("[TEST] Queued long passive message")

    # 3. After a short delay, queue an urgent message
    time.sleep(2.0)
    urgent_text = "Urgent message. This should interrupt the previous speech."
    tts.add_message(urgent_text, priority="urgent")
    print("[TEST] Queued urgent interrupting message")

    # 4. Add another normal / active priority message
    time.sleep(1.0)
    active_text = "Active message after the urgent one."
    tts.add_message(active_text, priority="active")
    print("[TEST] Queued active message")

    # 5. Let everything play out
    time.sleep(15.0)

    # 6. Clean shutdown
    print("[TEST] Stopping TTS thread")
    tts.stop()
    tts.join(timeout=5.0)
    print("[TEST] Done")


if __name__ == "__main__":
    main()
