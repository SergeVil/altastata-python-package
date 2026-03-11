# Smoke tests (core API)

Manual scripts to verify AltaStataFunctions: store, retrieve, share, list, buffer, stream, delete.

- **test_script.py** – macOS/Linux (edit account path and Desktop paths for your machine).
- **test_script_windows.py** – Windows (edit account path and Desktop paths).

Run from repo root after `pip install -e .`:

```bash
python examples/smoke-test/test_script.py
# or
python examples/smoke-test/test_script_windows.py
```
