# AltaStata Event Listener

Real-time event notifications for file operations (share, delete, create, etc.)

## Overview

The Event Listener feature allows you to receive real-time notifications when file operations occur in AltaStata, such as:
- **SHARE** - When files are shared with other users
- **DELETE** - When files are deleted
- **CREATE** - When new files are created
- And other custom events

## Architecture

```
┌─────────────────────────────────────────────────────┐
│                  Your Python App                     │
│                                                      │
│  def my_handler(event_name, data):                  │
│      print(f"Event: {event_name}")                  │
│      # Process event...                             │
│                                                      │
│  listener = altastata.add_event_listener(my_handler)│
└──────────────────────┬──────────────────────────────┘
                       │
                       │ Python Callback
                       │
┌──────────────────────▼──────────────────────────────┐
│            AltaStataEventListener (py4j)            │
│                                                      │
│  notify(altastata_event):                           │
│      event_name = event.getEventName()              │
│      data = event.getData()                         │
│      callback(event_name, data)                     │
└──────────────────────┬──────────────────────────────┘
                       │
                       │ Java Interface Implementation
                       │
┌──────────────────────▼──────────────────────────────┐
│         Java AltaStataFileSystem (JVM)              │
│                                                      │
│  addAltaStataEventListener(listener)                │
│  → Registers listener for events                    │
│                                                      │
│  When file operation occurs:                        │
│  → Creates AltaStataEvent                          │
│  → Calls listener.notify(event)                    │
└─────────────────────────────────────────────────────┘
```

## Quick Start

### 1. Install Requirements

The event listener functionality requires py4j with callback support (already configured):

```python
from altastata.altastata_functions import AltaStataFunctions
```

### 2. Create Event Handler

```python
def my_event_handler(event_name, data):
    """
    Handle AltaStata events.
    
    Args:
        event_name (str): Type of event (e.g., "SHARE", "DELETE")
        data (str): Event-specific data (typically file path with version)
    """
    print(f"Event: {event_name}")
    print(f"Data: {data}")
    
    if event_name == "SHARE":
        # Handle file sharing
        print("File was shared!")
    elif event_name == "DELETE":
        # Handle file deletion
        print("File was deleted!")
```

### 3. Register Listener

```python
# Initialize AltaStata with callback server enabled
altastata = AltaStataFunctions.from_account_dir(
    'path/to/account',
    enable_callback_server=True,
    callback_server_port=25334  # Or let it auto-select with port=0
)
altastata.set_password("your_password")

# Register event listener
listener = altastata.add_event_listener(my_event_handler)

# Listener is now active and will receive events
```

### 4. Unregister Listener

```python
# Remove specific listener
altastata.remove_event_listener(listener)

# Or remove all listeners
altastata.remove_all_event_listeners()
```

## Complete Example

```python
from altastata.altastata_functions import AltaStataFunctions
import time

# Event handler
def event_handler(event_name, data):
    print(f"📢 Event: {event_name}")
    print(f"   Data: {data}")

# Initialize
altastata = AltaStataFunctions.from_account_dir('/path/to/account')
altastata.set_password("password")

# Register listener
listener = altastata.add_event_listener(event_handler)

try:
    # Perform operations that trigger events
    altastata.create_file("test.txt", b"content")
    altastata.share_files("test.txt", False, None, None, ["user1"])
    altastata.delete_files("test.txt", False, None, None)
    
    # Wait for events to process
    time.sleep(2)
    
finally:
    # Clean up
    altastata.remove_event_listener(listener)
```

## Advanced: Multiple Listeners

You can register multiple listeners for different purposes:

```python
# Logger
def logger(event_name, data):
    with open('audit.log', 'a') as f:
        f.write(f"{time.time()}: {event_name} - {data}\n")

# Security monitor
def security_monitor(event_name, data):
    if event_name in ["SHARE", "DELETE"]:
        send_alert(f"Sensitive operation: {event_name}")

# Metrics
event_counts = {}
def metrics(event_name, data):
    event_counts[event_name] = event_counts.get(event_name, 0) + 1

# Register all
listener1 = altastata.add_event_listener(logger)
listener2 = altastata.add_event_listener(security_monitor)
listener3 = altastata.add_event_listener(metrics)

# Later: remove all at once
altastata.remove_all_event_listeners()
```

## Event Types

Common event types you might receive:

| Event Name | Description | Typical Data |
|------------|-------------|--------------|
| `SHARE` | File shared with users | File path with version (e.g., `SharedEvents/file.txt✹alice222_1761135341619`) |
| `DELETE` | File was deleted | File path with version |
| `CREATE` | New file was created | File path with version |
| Custom events | Your application events | Application-specific |

**Note:** The data typically includes the full file path with version information in the format: `path/filename✹username_timestamp`

## Use Cases

### 1. Audit Logging

```python
def audit_logger(event_name, data):
    timestamp = datetime.now().isoformat()
    log_entry = f"{timestamp} | {event_name} | {data}"
    
    # Write to database
    db.insert('audit_log', {
        'timestamp': timestamp,
        'event': event_name,
        'data': data
    })
```

### 2. Real-time Sync

```python
def sync_handler(event_name, data):
    if event_name == "CREATE":
        # Trigger backup
        backup_service.backup(data)
    elif event_name == "DELETE":
        # Update index
        search_index.remove(data)
```

### 3. Security Monitoring

```python
def security_handler(event_name, data):
    sensitive_events = ["SHARE", "DELETE"]
    
    if event_name in sensitive_events:
        # Send alert
        slack.send(f"⚠️ Security Alert: {event_name} on {data}")
        
        # Log for compliance
        compliance_db.log_security_event(event_name, data)
```

### 4. RAG Index Updates

```python
def rag_updater(event_name, data):
    """Update RAG vector store when files change"""
    
    if event_name == "CREATE":
        # Re-index the file
        # Extract file path from version data (e.g., "path/file.txt✹user_timestamp")
        file_path = data.split('✹')[0] if '✹' in data else data
        content = altastata.get_buffer(file_path, None, 0, 1, -1)
        chunks = text_splitter.split_text(content)
        embeddings = create_embeddings(chunks)
        vector_store.upsert(data, embeddings)
    
    elif event_name == "DELETE":
        # Remove from index
        vector_store.delete(data)
```

## Technical Details

### How It Works

1. **Callback Server**: py4j starts a callback server on a specified port (or auto-selects)
2. **Java Interface**: Python class implements `com.altastata.api.AltaStataEventListener`
3. **Event Polling**: Java `SecureCloudEventProcessor` polls cloud storage for changes (~5 second interval)
4. **Event Flow**: Java detects changes, fires events, and calls Python's `notify()` method via Py4J callback
5. **Thread Safety**: Events are delivered on Java's event thread

### Performance

- **Low overhead**: Direct Java→Python callback (no polling)
- **Asynchronous**: Events delivered as they occur
- **Thread-safe**: Callback handling is thread-safe

### Error Handling

If your callback raises an exception, it's caught and logged:

```python
def buggy_handler(event_name, data):
    raise Exception("Oops!")  # Won't crash the system

listener = altastata.add_event_listener(buggy_handler)
# Exception is caught and printed to console
```

## Running the Two-User Test

The best way to see event notifications in action is the two-user test:

**Terminal 1 - Bob (Listener):**
```bash
cd event-listener-example
python bob_listener.py
```

**Terminal 2 - Alice (Sender):**
```bash
cd event-listener-example
python alice_sender.py
```

Bob will receive real-time notifications as Alice creates, shares, and deletes files!

See [TWO_USER_TEST.md](TWO_USER_TEST.md) for complete instructions.

## Troubleshooting

### Events Not Firing

1. **Check callback server**: Ensure py4j callback server is enabled (should be automatic)
2. **Check event names**: Verify the Java side is firing events with expected names
3. **Add debug logging**: Print messages in your handler to confirm it's being called

### Memory Leaks

Always remove listeners when done:

```python
try:
    listener = altastata.add_event_listener(handler)
    # ... use listener ...
finally:
    altastata.remove_event_listener(listener)
```

### Threading Issues

Event callbacks run on Java's event thread. If you need to update UI or do heavy work:

```python
from threading import Thread

def event_handler(event_name, data):
    # Quick processing on event thread
    print(f"Event: {event_name}")
    
    # Heavy work on separate thread
    def process():
        heavy_operation(data)
    
    Thread(target=process, daemon=True).start()
```

## See Also

- [Main AltaStata Documentation](../README.md)
- [fsspec Integration](../fsspec-example/)
- [RAG Examples](../rag-example/)

---

**Need help?** Check the [main documentation](../README.md) or contact support.

