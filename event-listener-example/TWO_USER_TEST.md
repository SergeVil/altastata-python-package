# Two-User Event Notification Test

This test demonstrates real-time event notifications between two AltaStata users:
- **Bob (bob123)** - Receives event notifications
- **Alice (alice222)** - Performs file operations

## Prerequisites

### 1. Create User Accounts

You need two AltaStata accounts set up:

```bash
# Bob's account should exist at:
/Users/sergevilvovsky/.altastata/accounts/amazon.rsa.bob123

# Alice's account should exist at:
/Users/sergevilvovsky/.altastata/accounts/amazon.rsa.alice222
```

Both accounts should have password `123` (or update the scripts accordingly).

### 2. Install Dependencies

```bash
pip install altastata
```

## Running the Test

### Terminal 1: Start Bob (Listener)

```bash
cd event-listener-example
python bob_listener.py
```

**Expected output:**
```
================================================================================
👤 BOB's EVENT LISTENER
================================================================================

Bob (bob123) is waiting for events from Alice...

1️⃣  Connecting to AltaStata as bob123...
✅ Bob connected successfully

2️⃣  Registering Bob's event listener...
✅ Event listener registered and active
   Listening for: SHARE, DELETE events

================================================================================
🎧 BOB IS NOW LISTENING FOR EVENTS...
================================================================================

💡 Next steps:
   1. Keep this terminal running
   2. Open another terminal
   3. Run: python alice_sender.py

⏳ Waiting for Alice to perform actions...
   (Press Ctrl+C to stop)
```

Bob is now listening and will display events as they arrive!

### Terminal 2: Start Alice (Sender)

Open a **second terminal** and run:

```bash
cd event-listener-example
python alice_sender.py
```

**Expected output:**
```
================================================================================
👤 ALICE's FILE OPERATIONS
================================================================================

Alice (alice222) will perform file operations...

1️⃣  Connecting to AltaStata as alice222...
✅ Alice connected successfully

================================================================================
📝 STEP 1: Creating File
================================================================================
File path: SharedEvents/alice_to_bob.txt
Creating file...
✅ File created: DONE

================================================================================
🤝 STEP 2: Sharing File with Bob
================================================================================
Sharing SharedEvents/alice_to_bob.txt with bob123...
✅ File shared successfully: 1 version(s)
   → Bob should receive a 'SHARE' event!

⏳ Waiting 15 seconds for Bob to receive SHARE event...
   (Bob's SecureCloudEventProcessor needs time to poll cloud storage)

================================================================================
🗑️  STEP 3: Deleting File
================================================================================
Deleting SharedEvents/alice_to_bob.txt...
✅ File deleted successfully: 1 version(s)
   → Bob should receive a 'DELETE' event!

================================================================================
✅ ALL OPERATIONS COMPLETED
================================================================================

📊 Summary:
   ✓ Created file
   ✓ Shared with bob123
   ✓ Deleted file

💡 Check Bob's terminal to see the events he received!
```

### Bob's Terminal: Events Received

Meanwhile, in Bob's terminal, you should see:

```
================================================================================
🔔 [08:17:05] BOB RECEIVED EVENT!
================================================================================
📋 Event Type: SHARE
📄 Data: SharedEvents/alice_to_bob.txt✹alice222_1761135341619
✅ Alice shared a file with Bob!
   File path: SharedEvents/alice_to_bob.txt✹alice222_1761135341619
   Bob can now access this file
================================================================================

================================================================================
🔔 [08:17:20] BOB RECEIVED EVENT!
================================================================================
📋 Event Type: DELETE
📄 Data: SharedEvents/alice_to_bob.txt✹alice222_1761135341619
🗑️  Alice deleted the file!
   File path: SharedEvents/alice_to_bob.txt✹alice222_1761135341619
   File is no longer accessible
================================================================================
```

## Test Flow Diagram

```
Terminal 1 (Bob)                    Terminal 2 (Alice)
================                    ==================

bob_listener.py starts
   ↓
Connects as bob123
   ↓
Registers event listener
   ↓
🎧 Waiting for events...
                                    alice_sender.py starts
                                       ↓
                                    Connects as alice222
                                       ↓
                                    Creates file
                                       ↓
                                    Shares file with bob123
                                       ↓
🔔 Event: SHARE ←────────────────────┘
   ↓
Display: "Alice shared
          file with Bob!"
                                    Deletes file
                                       ↓
🔔 Event: DELETE ←───────────────────┘
   ↓
Display: "Alice deleted
          the file!"
                                    Exits
   ↓
Still listening...
(Ctrl+C to stop)
```

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    AltaStata Cloud                           │
│                                                              │
│  ┌──────────────┐                    ┌──────────────┐       │
│  │  alice222    │                    │   bob123     │       │
│  │  (Sender)    │                    │  (Listener)  │       │
│  └──────┬───────┘                    └──────▲───────┘       │
│         │                                    │               │
│         │ 1. Create file                     │               │
│         │ 2. Share with bob123               │               │
│         │ 3. Delete file                     │               │
│         │                                    │               │
│         │                    Event System    │               │
│         └────────────────────────┬───────────┘               │
│                                  │                           │
└──────────────────────────────────┼───────────────────────────┘
                                   │
                    ┌──────────────┼──────────────┐
                    │              │              │
                    ▼              ▼              ▼
               CREATE         SHARE         DELETE
                    │              │              │
                    └──────────────┴──────────────┘
                                   │
                                   ▼
                    ┌──────────────────────────────┐
                    │   bob_listener.py            │
                    │   bob_event_handler()        │
                    │   → Displays events          │
                    └──────────────────────────────┘
```

## Troubleshooting

### Alice Can't Connect

**Error:** `Failed to connect as alice222`

**Solution:** Create Alice's account:
```bash
# Create alice222 account directory
mkdir -p /Users/sergevilvovsky/.altastata/accounts/amazon.rsa.alice222

# Copy account files or set up new account
# (Follow AltaStata account setup instructions)
```

### Bob Doesn't Receive Events

**Possible causes:**

1. **Bob not running** - Make sure `bob_listener.py` is running before Alice performs operations

2. **Different event names** - The Java side might use different event names. Check Bob's output for actual event names received.

3. **Callback server issue** - Verify py4j callback server is enabled (should be automatic)

4. **Network/timing** - Add longer delays in `alice_sender.py` between operations

### Events Arrive Out of Order

This is normal! Events are asynchronous. The order may vary slightly, but all events should arrive.

## Cleanup

To stop Bob's listener:
- Press `Ctrl+C` in Bob's terminal
- The script will clean up and remove the event listener

## Next Steps

### Modify the Test

You can customize the test by editing:

**`bob_listener.py`:**
- Change the event handler logic
- Add logging to file/database
- Integrate with other systems

**`alice_sender.py`:**
- Try different file operations
- Share with multiple users
- Test with larger files

### Production Use

For production, consider:
- Error handling and retries
- Persistent event logging
- Event queueing for reliability
- Multiple listeners for different purposes

## See Also

- [Event Listener README](README.md) - Complete documentation
- [Main AltaStata Documentation](../README.md)

