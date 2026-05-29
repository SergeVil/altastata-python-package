# JS gRPC playground

This folder contains a standalone browser page for testing all gRPC-Web APIs from one place.

## What it covers

- UsersService: bootstrap, list users, get account/user, set password
- FileOpsService: upload/create, get buffer, append, copy, delete, list versions, read stream
- AttributesService: set/get/delete metadata
- SharingService: share/revoke (by list and by query)
- EventsService: subscribe/stop stream

The page uses hardcoded config in `app.js` (`CONFIG`) and logs every request/response in the in-page Logs panel.

## Files

- `index.html` - full playground UI
- `app.js` - gRPC-Web logic
- `protobuf.min.js` - local protobuf runtime (no CDN dependency)

## Run

1. Ensure local gRPC server is running on `127.0.0.1:9877`.
2. Serve this directory:

```bash
cd tests/js-grpc-ui
python -m http.server 8081
```

3. Open `http://127.0.0.1:8081/`.
4. Click **Bootstrap user**.
5. Use buttons section by section.

Notes:
- Upload uses the selected file bytes; destination path is taken from `File path` (fallback: selected file name).
- Token format is `local-<user_name>`.
