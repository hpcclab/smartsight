import time
from TextToSpeechThread import TTSThread  # Replace with your actual module name

def test_interruption():
    """Test the TTS interruption system with different priority levels."""
    
    # Create and start the TTS thread
    tts = TTSThread()
    tts.start()
    
    print("\n=== TEST 1: Passive -> Active Interruption ===")
    tts.add_message("This is a very long passive message that should be interrupted by an active message in about two seconds.", priority="passive")
    time.sleep(2)  # Let it start speaking
    tts.add_message("Active message interrupting!", priority="active")
    time.sleep(3)
    
    print("\n=== TEST 2: Active -> Urgent Interruption ===")
    tts.add_message("This is a long active message that will be interrupted by an urgent message.", priority="active")
    time.sleep(2)
    tts.add_message("URGENT! This is critical!", priority="urgent")
    time.sleep(3)
    
    print("\n=== TEST 3: Multiple Passive Messages (No Interruption) ===")
    tts.add_message("First passive message.", priority="passive")
    time.sleep(0.5)
    tts.add_message("Second passive message.", priority="passive")
    time.sleep(0.5)
    tts.add_message("Third passive message.", priority="passive")
    time.sleep(8)  # Wait for all to finish
    
    print("\n=== TEST 4: Rapid Urgent Messages ===")
    tts.add_message("Long passive message that will definitely be interrupted multiple times.", priority="passive")
    time.sleep(1)
    tts.add_message("First urgent interruption!", priority="urgent")
    time.sleep(0.5)
    tts.add_message("Second urgent interruption!", priority="urgent")
    time.sleep(5)
    
    print("\n=== TEST 5: Queue Buildup with Interruption ===")
    tts.add_message("Passive message one.", priority="passive")
    tts.add_message("Passive message two.", priority="passive")
    tts.add_message("Passive message three.", priority="passive")
    time.sleep(2)  # Let first one start
    tts.add_message("URGENT interruption cutting through queue!", priority="urgent")
    time.sleep(10)
    
    # Cleanup
    print("\n=== Stopping TTS Thread ===")
    tts.stop()
    tts.join(timeout=3)
    print("Test complete!")

if __name__ == "__main__":
    test_interruption()
