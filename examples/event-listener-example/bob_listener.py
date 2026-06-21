#!/usr/bin/env python3
"""
Bob's Event Listener Script

Bob (bob123) runs this script and waits for events from Alice.

Usage:
    python bob_listener.py
    
Then in another terminal, run:
    python alice_sender.py
"""

import sys
import os
import time
from datetime import datetime

# Add the altastata package to the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from altastata.altastata_functions import AltaStataFunctions


def bob_event_handler(event_name, data):
    """
    Bob's event handler - receives notifications when Alice performs actions.
    
    Args:
        event_name: Type of event (FILE_SHARED, FILE_DELETED, etc.)
        data: Event data
    """
    timestamp = datetime.now().strftime("%H:%M:%S")
    
    print("\n" + "=" * 80)
    print(f"🔔 [{timestamp}] BOB RECEIVED EVENT!")
    print("=" * 80)
    print(f"📋 Event Type: {event_name}")
    print(f"📄 Data: {data}")
    
    if event_name == "SHARE":
        print("✅ Alice shared a file with Bob!")
        print(f"   File path: {data}")
        print(f"   Bob can now access this file")
        
    elif event_name == "DELETE":
        print("🗑️  Alice deleted the file!")
        print(f"   File path: {data}")
        print(f"   File is no longer accessible")
        
    else:
        print(f"❓ Event type: {event_name}")
    
    print("=" * 80)


def main():
    print("\n" + "=" * 80)
    print("👤 BOB's EVENT LISTENER")
    print("=" * 80)
    print("\nBob (bob123) is waiting for events from Alice...")
    
    # Initialize Bob's AltaStata connection
    print("\n1️⃣  Connecting to AltaStata as bob123...")
    try:
        bob_altastata = AltaStataFunctions.from_account_dir(
            '/Users/sergevilvovsky/.altastata/accounts/amazon.rsa.bob123'
        )
        bob_altastata.set_password("123")
        print("✅ Bob connected successfully")
    except Exception as e:
        print(f"❌ Failed to connect as bob123: {e}")
        print("\nMake sure bob123 account exists at:")
        print("   /Users/sergevilvovsky/.altastata/accounts/amazon.rsa.bob123")
        return
    
    # Register Bob's event listener
    print("\n2️⃣  Registering Bob's event listener...")
    listener = bob_altastata.add_event_listener(bob_event_handler)
    print("✅ Event listener registered and active")
    print("   Listening for: SHARE, DELETE events")
    
    print("\n" + "=" * 80)
    print("🎧 BOB IS NOW LISTENING FOR EVENTS...")
    print("=" * 80)
    print("\n💡 Next steps:")
    print("   1. Keep this terminal running")
    print("   2. Open another terminal")
    print("   3. Run: python alice_sender.py")
    print("\n⏳ Waiting for Alice to perform actions...")
    print("   (Press Ctrl+C to stop)")
    print()
    
    try:
        # Keep the script running and listening for events
        while True:
            time.sleep(1)  # Sleep to reduce CPU usage
            
    except KeyboardInterrupt:
        print("\n\n⚠️  Bob stopped listening (interrupted by user)")
    finally:
        # Clean up
        print("\n🧹 Cleaning up...")
        bob_altastata.remove_event_listener(listener)
        print("✅ Event listener removed")
        print("\n👋 Bob disconnected\n")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

