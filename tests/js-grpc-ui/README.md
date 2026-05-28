# JS gRPC one-page test

This folder contains a standalone browser page that:

- bootstraps a local gRPC user (`setUserProperties`, `setPrivateKey`, `setPassword`)
- calls `UsersService/ListUsers` over gRPC-Web and renders returned users
- uses hardcoded config in `app.js` (minimal UI)

## Files

- `index.html` - the page UI
- `app.js` - gRPC-Web request/response logic

## Run

1. Make sure `altastata-grpc` is running.
2. Use a simple local web server from this directory:

```bash
cd tests/js-grpc-ui
python -m http.server 8081
```

3. Open `http://127.0.0.1:8081/` in a browser.
4. Edit `app.js` and set:
   - `grpcBaseUrl`
   - `userName`
   - `accountPassword`
   - `userProperties`
   - `privateKey`
5. Click **Bootstrap user** in the page.
6. Click **List users**.

The page uses token format `local-<user_name>` automatically.
