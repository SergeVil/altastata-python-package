(function () {
  "use strict";

  if (!window.protobuf) {
    const status = document.getElementById("status");
    const output = document.getElementById("output");
    if (status) status.textContent = "Startup error";
    if (output) output.textContent = "protobuf runtime not loaded. Ensure tests/js-grpc-ui/protobuf.min.js exists and is served.";
    return;
  }

  const CONFIG = {
    grpcBaseUrl: "http://127.0.0.1:9877",
    userName: "bob123_rsa",
    accountPassword: "123",
    userProperties: `#My Properties
#Sun Jun 01 19:41:40 EDT 2025
AWSSecretKey=YKPTKMO3GnSr/aJqJpW9QPDtWVrazsnVMvsUNpyHG9usjuaLf4rVt4fmzDtb/8cEPslDxh2AJGYFQntEkr4mDeFXORUxw1XHxxNa/RctYyzxqLTsh7Gaxamm6Bxmy6zG1p+OM78Ykjr736jdx4F0yJJmy1HK19ZuZSmqnkLgsBgqU5l1nEFYsFm5vemQLjC431TfSCbGcVNU1uW/zkwL9U+9KM2rN6HWlmsxqA7t71jslI8Ahf5JWVp1dqvGqf8xgbr5Gw==
media-player=vlcj
myuser=bob123
accounttype=amazon-s3-secure
AWSAccessKeyId=oZP9Iam4mhj0N9uzwzoNT/7xzaj+cWWgIg1MIn5Sgr9zBBahnI04FfcDq2uiYIpVW28H5GKPUZPhnvxnfcvcYZvwAN3oeUnB96o5bg0ABAfizy5r4FbHwxQpFzX0sJTjub0Jv+tvgFR7H/+fO9F8XTfbn/e7WE3n5EKp5nTkyvzcfxU/pef3GN90ut1fMVRLVE47vVINpipto+b2dD0/DwO/SovRz14rvTHYbuIpUPI=
region=us-east-1
kms-region=us-east-2
metadata-encryption=RSA
acccontainer-prefix=altastata-myorgrsa444-
logging.level.root=WARN
logging.level.com.altastata=WARN
logging.level.org.apache.http=ERROR
logging.level.software.amazon=ERROR`,
    privateKey: `-----BEGIN RSA PRIVATE KEY-----
Proc-Type: 4,ENCRYPTED
DEK-Info: DES-EDE3,F26EBECE6DDAEC52

poe21ejZGZQ0GOe+EJjDdJpNvJcq/Yig9aYXY2rCGyxXLGVFeYJFg7z6gMCjIpSd
aprW/0R8L1a2TKbs7f4K5LkSAZ98cd7N45DtIR6B4JFrDGK3LI48/XH3GT3c4OfS
3LYldvy4XeIOAtOTTCoyhN0145ZLSoeEQ7MO3rGK0va3RGLtPWKgeZXH9j5O1Ch4
BvPGMaKapUcgc1slj1GI4Lr+MDSrJKnUNovnVTIClS2rXTEkTri3cPLwcgWjyQIi
BKVnobUD8Gm9irtUD6GeHrkz6Z7ELF3ctSBRSYCg+1FCvRBuljmS2C2aIiE1cu0/
6KcqBnjEPAs250832uhAkZWj5WedIwJv+sJoGJaAUWyOfgG7DHa2HuKeR9KPD2kS
6EygoQtQlXgSvdgZNALtIEfStmnrblTyP9Bh4JU9UzKnE6Tu5h7CjyuzkE0wgIXB
RxgfbURfdDWs22ujLBbWPGfdY+KdNrnmSqxYahKtq6B+99+xuI0GMzX3/rLpOdF0
AGwfa1xNe8/B/Nt+e2FXIhT2xOuH8K3sDn3/FKwy1qIsK+4g5iL6Q0xj07ujkiSI
wZ0X2gtg3L2DW8Y6B8gBdSmDGH+vNX5/CLNn9Ly1VUoMGgs4fUmd3FFZTxiIbpim
rQgQBHP4l1NsSqDrEyplKG83ejloLaVG+hUY1MGv5tF7B1Ta7j8bwoMTmyVCtCrC
P+a7ShdrBUsD2TDhilZhwZcWl0a+FfzR47+faJs/9pSTkyFFp3D4xgKAdME1lvcI
wV5BUmp5CEmbeB4r/+BlFttRZBLBXT1sq80YyQIVLumq0Livao9mOg==
-----END RSA PRIVATE KEY-----`,
  };

  const protoDef = `
    syntax = "proto3";
    package altastata.v1;

    message Empty {}
    message UserSummary { string user_name = 1; bool initialized = 2; }
    message User { string user_name = 1; bool initialized = 2; string access_key = 3; }
    message GetUserRequest { string user_name = 1; }
    message GetMyAccountRequest {}
    message SetPasswordRequest { string account_password = 1; }
    message SetPasswordResponse { bool success = 1; }
    message SetUserPropertiesRequest { string user_name = 1; string user_properties = 2; }
    message SetUserPropertiesResponse { bool success = 1; }
    message SetPrivateKeyRequest { string user_name = 1; string private_key_encrypted = 2; }
    message SetPrivateKeyResponse { bool success = 1; }
    message SetPasswordForUserRequest { string user_name = 1; string account_password = 2; }
    message SetPasswordForUserResponse { bool success = 1; string access_key = 2; string secret_key = 3; }

    message FileStatus { string file_path = 1; string operation_state = 2; string error = 3; }
    message CreateFileRequest { string file_path = 1; bytes content = 2; }
    message CreateFileResponse { FileStatus status = 1; }
    message GetBufferRequest { string file_path = 1; int64 snapshot_time = 2; int64 start_position = 3; int32 parallel_chunks = 4; int32 size = 5; }
    message GetBufferResponse { bytes data = 1; }
    message DeleteRequest { string cloud_path_prefix = 1; bool including_subdirectories = 2; string time_interval_start = 3; string time_interval_end = 4; }
    message DeleteResponse { repeated FileStatus statuses = 1; }
    message ListVersionsRequest { string cloud_path_prefix = 1; bool including_subdirectories = 2; string time_interval_start = 3; string time_interval_end = 4; }
    message VersionEntry { repeated string versions = 1; }
    message AppendBufferToFileRequest { string file_path = 1; int64 snapshot_time = 2; bytes content = 3; }
    message AppendBufferToFileResponse { bool success = 1; }
    message CopyFileRequest { string from_cloud_file_path = 1; string to_cloud_file_path = 2; }
    message CopyFileResponse { FileStatus status = 1; }
    message ReadStreamRequest { string file_path = 1; int64 snapshot_time = 2; int64 start_position = 3; int32 parallel_chunks = 4; int32 chunk_size = 5; }
    message ReadStreamChunk { bytes data = 1; }

    message Attribute { string name = 1; string value = 2; }
    message AttributeMap { map<string, string> attributes = 1; }
    message GetAttributeRequest { string file_path = 1; int64 snapshot_time = 2; string name = 3; }
    message GetAttributesRequest { string file_path = 1; int64 snapshot_time = 2; repeated string names = 3; }
    message SetAttributeRequest { string file_path = 1; int64 snapshot_time = 2; string name = 3; string value = 4; }
    message DeleteAttributeRequest { string file_path = 1; int64 snapshot_time = 2; string name = 3; }

    message ShareRequest { repeated string file_paths = 1; repeated string readers = 2; }
    message RevokeRequest { repeated string file_paths = 1; repeated string readers = 2; }
    message ShareByQueryRequest { string cloud_path_prefix = 1; bool including_subdirectories = 2; string time_interval_start = 3; string time_interval_end = 4; repeated string readers = 5; }
    message RevokeByQueryRequest { string cloud_path_prefix = 1; bool including_subdirectories = 2; string time_interval_start = 3; string time_interval_end = 4; repeated string readers = 5; }
    message ShareResult { repeated FileStatus statuses = 1; }
    message RevokeResult { repeated FileStatus statuses = 1; }

    message SubscribeRequest {}
    message EventMessage { string event_name = 1; string data = 2; }
  `;

  const root = protobuf.parse(protoDef).root;
  const T = (name) => root.lookupType(`altastata.v1.${name}`);

  const statusEl = document.getElementById("status");
  const outputEl = document.getElementById("output");
  const logsOutputEl = document.getElementById("logsOutput");
  const usersBody = document.getElementById("usersBody");
  const eventsOutput = document.getElementById("eventsOutput");

  const userNameInput = document.getElementById("userNameInput");
  const userPasswordInput = document.getElementById("userPasswordInput");
  const filePathInput = document.getElementById("filePathInput");
  const copyToPathInput = document.getElementById("copyToPathInput");
  const appendTextInput = document.getElementById("appendTextInput");
  const attrNameInput = document.getElementById("attrNameInput");
  const attrValueInput = document.getElementById("attrValueInput");
  const attrNamesInput = document.getElementById("attrNamesInput");
  const readersInput = document.getElementById("readersInput");
  const sharePathsInput = document.getElementById("sharePathsInput");
  const queryPrefixInput = document.getElementById("queryPrefixInput");
  const dropZone = document.getElementById("dropZone");
  const fileInput = document.getElementById("fileInput");
  const fileInfo = document.getElementById("fileInfo");

  userNameInput.value = CONFIG.userName;
  userPasswordInput.value = CONFIG.accountPassword;
  queryPrefixInput.value = "";
  attrNameInput.value = "size";
  attrValueInput.value = "";
  attrNamesInput.value = "size,readers";

  let selectedFile = null;
  let subscribeController = null;
  let authBootstrapDone = false;
  let authBootstrapInFlight = null;

  function setStatus(msg) { statusEl.textContent = msg; }

  function setOutput(value) {
    if (typeof value === "string") outputEl.textContent = value;
    else outputEl.textContent = JSON.stringify(value, null, 2);
  }

  function log(message, details) {
    const timestamp = new Date().toISOString();
    let line = `[${timestamp}] ${message}`;
    if (details !== undefined) {
      if (typeof details === "string") line += ` | ${details}`;
      else line += ` | ${JSON.stringify(details)}`;
    }
    if (logsOutputEl) logsOutputEl.textContent += `\n${line}`;
    console.log(line);
  }

  function appendEvent(text) {
    if (!eventsOutput) return;
    const current = eventsOutput.textContent || "";
    if (current === "No events yet." || current === "Subscribed. Waiting for events...") {
      eventsOutput.textContent = text;
    } else {
      eventsOutput.textContent += `\n${text}`;
    }
    eventsOutput.scrollTop = eventsOutput.scrollHeight;
    log("Event received", text);
    setStatus(`Event: ${text}`);
  }

  function parseMyUserFromProperties(text) {
    const lines = (text || "").split("\n");
    for (const raw of lines) {
      const line = raw.trim();
      if (!line || line.startsWith("#")) continue;
      const idx = line.indexOf("=");
      if (idx < 0) continue;
      const key = line.slice(0, idx).trim();
      const value = line.slice(idx + 1).trim();
      if (key === "myuser") return value;
    }
    return "";
  }

  function logConfigSanity() {
    const tokenUser = userNameInput.value.trim() || CONFIG.userName.trim();
    const propertiesUser = parseMyUserFromProperties(CONFIG.userProperties);
    log("Config summary", { grpcBaseUrl: baseUrl(), tokenUser, propertiesUser });
    if (tokenUser && propertiesUser && tokenUser !== propertiesUser) {
      log("Config warning", "userName/token user differs from userProperties myuser; this can point calls to unexpected account context.");
    }
  }

  function resolveRegistryUser() {
    const inputUser = userNameInput.value.trim();
    const configUser = CONFIG.userName.trim();
    // Token validation uses GrpcUserRegistry keys (user_name from bootstrap),
    // not "myuser" inside account properties.
    return inputUser || configUser;
  }

  async function ensureAuthBootstrap() {
    if (authBootstrapDone) return;
    if (authBootstrapInFlight) {
      await authBootstrapInFlight;
      return;
    }
    const user = resolveRegistryUser();
    const pwd = userPasswordInput.value;
    if (!user) throw new Error("userName is empty");
    if (!pwd) throw new Error("account password is empty");

    authBootstrapInFlight = (async () => {
      log("Auth bootstrap started", { userName: user });
      await grpcUnary("altastata.v1.UsersService/SetUserProperties", "SetUserPropertiesRequest", {
        userName: user,
        userProperties: CONFIG.userProperties,
      }, "SetUserPropertiesResponse", false);
      await grpcUnary("altastata.v1.UsersService/SetPrivateKey", "SetPrivateKeyRequest", {
        userName: user,
        privateKeyEncrypted: CONFIG.privateKey,
      }, "SetPrivateKeyResponse", false);
      await grpcUnary("altastata.v1.UsersService/SetPasswordForUser", "SetPasswordForUserRequest", {
        userName: user,
        accountPassword: pwd,
      }, "SetPasswordForUserResponse", false);
      authBootstrapDone = true;
      log("Auth bootstrap complete", { userName: user });
    })();
    try {
      await authBootstrapInFlight;
    } finally {
      authBootstrapInFlight = null;
    }
  }

  function baseUrl() { return CONFIG.grpcBaseUrl.trim().replace(/\/+$/, ""); }
  function token() {
    const user = resolveRegistryUser();
    if (!user) throw new Error("userName is empty");
    return `local-${user}`;
  }

  function frameMessage(bytes) {
    const out = new Uint8Array(5 + bytes.length);
    out[0] = 0x00;
    const len = bytes.length >>> 0;
    out[1] = (len >>> 24) & 0xff;
    out[2] = (len >>> 16) & 0xff;
    out[3] = (len >>> 8) & 0xff;
    out[4] = len & 0xff;
    out.set(bytes, 5);
    return out;
  }

  function concat(a, b) {
    if (!a || a.length === 0) return b;
    const out = new Uint8Array(a.length + b.length);
    out.set(a, 0);
    out.set(b, a.length);
    return out;
  }

  function keepTailBytes(currentTail, nextChunk, maxBytes) {
    const merged = concat(currentTail, nextChunk);
    if (merged.length <= maxBytes) return merged;
    return merged.slice(merged.length - maxBytes);
  }

  function extractFrames(buffer) {
    const frames = [];
    let offset = 0;
    while (offset + 5 <= buffer.length) {
      const flags = buffer[offset];
      const len = (buffer[offset + 1] << 24) | (buffer[offset + 2] << 16) | (buffer[offset + 3] << 8) | buffer[offset + 4];
      offset += 5;
      if (len < 0 || offset + len > buffer.length) {
        offset -= 5;
        break;
      }
      frames.push({ trailer: (flags & 0x80) !== 0, payload: buffer.slice(offset, offset + len) });
      offset += len;
    }
    return { frames, rest: buffer.slice(offset) };
  }

  function parseTrailers(payload) {
    const txt = new TextDecoder().decode(payload);
    const map = new Map();
    for (const line of txt.split("\r\n")) {
      if (!line) continue;
      const idx = line.indexOf(":");
      if (idx < 0) continue;
      map.set(line.slice(0, idx).trim().toLowerCase(), line.slice(idx + 1).trim());
    }
    return map;
  }

  async function grpcUnary(methodPath, reqTypeName, reqObj, respTypeName, withAuth, _retriedAfterBootstrap) {
    if (withAuth) await ensureAuthBootstrap();
    const reqType = T(reqTypeName);
    const respType = T(respTypeName);
    const body = frameMessage(reqType.encode(reqType.create(reqObj || {})).finish());
    const headers = {
      "content-type": "application/grpc-web+proto",
      "x-grpc-web": "1",
      "x-user-agent": "grpc-web-playground",
    };
    if (withAuth) headers.authorization = `Bearer ${token()}`;

    log("grpc unary request", { methodPath, requestType: reqTypeName, authenticated: !!withAuth });
    const response = await fetch(`${baseUrl()}/${methodPath}`, { method: "POST", headers, body });
    if (!response.ok) throw new Error(`HTTP ${response.status} ${response.statusText}`);

    const bytes = new Uint8Array(await response.arrayBuffer());
    const parsed = extractFrames(bytes);
    let msg = null;
    let trailers = new Map();
    for (const f of parsed.frames) {
      if (f.trailer) trailers = parseTrailers(f.payload);
      else msg = respType.decode(f.payload);
    }
    const code = trailers.get("grpc-status") || "0";
    if (code !== "0") {
      const message = trailers.get("grpc-message") || "";
      if (withAuth && !_retriedAfterBootstrap && code === "16" && /invalid token/i.test(message)) {
        log("grpc unary retry", { methodPath, reason: "Invalid token", action: "re-bootstrap and retry once" });
        authBootstrapDone = false;
        await ensureAuthBootstrap();
        return grpcUnary(methodPath, reqTypeName, reqObj, respTypeName, withAuth, true);
      }
      throw new Error(`gRPC status=${code} message=${message}`);
    }
    log("grpc unary response", { methodPath, grpcStatus: code });
    return msg || respType.create({});
  }

  async function grpcServerStream(methodPath, reqTypeName, reqObj, respTypeName, onMessage, withAuth, signal, _retriedAfterBootstrap) {
    if (withAuth) await ensureAuthBootstrap();
    const reqType = T(reqTypeName);
    const respType = T(respTypeName);
    const body = frameMessage(reqType.encode(reqType.create(reqObj || {})).finish());
    const headers = {
      "content-type": "application/grpc-web+proto",
      "x-grpc-web": "1",
      "x-user-agent": "grpc-web-playground",
    };
    if (withAuth) headers.authorization = `Bearer ${token()}`;

    log("grpc stream open", { methodPath, requestType: reqTypeName, authenticated: !!withAuth });
    const response = await fetch(`${baseUrl()}/${methodPath}`, { method: "POST", headers, body, signal });
    if (!response.ok) throw new Error(`HTTP ${response.status} ${response.statusText}`);

    const reader = response.body.getReader();
    let stash = new Uint8Array(0);
    let trailers = new Map();
    let messageCount = 0;
    let sawTrailerFrame = false;

    while (true) {
      const { done, value } = await reader.read();
      if (value && value.length) {
        stash = concat(stash, value);
        const { frames, rest } = extractFrames(stash);
        stash = rest;
        for (const frame of frames) {
          if (frame.trailer) {
            trailers = parseTrailers(frame.payload);
            sawTrailerFrame = true;
          } else {
            messageCount += 1;
            onMessage(respType.decode(frame.payload));
          }
        }
      }
      if (done) break;
    }

    if (stash.length) throw new Error(`Incomplete gRPC stream frame (${stash.length} bytes left in buffer)`);

    // Trailers-only responses (e.g. an empty result set, or the server completing/erroring before
    // emitting a body trailer frame) carry grpc-status in the HTTP response headers instead of a
    // body trailer frame. Fall back to the headers so we surface the real status rather than a
    // misleading "missing trailer frame".
    if (!sawTrailerFrame) {
      const headerStatus = response.headers.get("grpc-status");
      if (headerStatus !== null) {
        trailers = new Map([["grpc-status", headerStatus]]);
        const headerMsg = response.headers.get("grpc-message");
        if (headerMsg !== null) trailers.set("grpc-message", headerMsg);
      } else {
        throw new Error("Missing gRPC trailer frame in stream response");
      }
    }

    const code = trailers.get("grpc-status") || "0";
    if (code !== "0") {
      const message = trailers.get("grpc-message") || "";
      if (withAuth && !_retriedAfterBootstrap && code === "16" && /invalid token/i.test(message)) {
        log("grpc stream retry", { methodPath, reason: "Invalid token", action: "re-bootstrap and retry once" });
        authBootstrapDone = false;
        await ensureAuthBootstrap();
        return grpcServerStream(methodPath, reqTypeName, reqObj, respTypeName, onMessage, withAuth, signal, true);
      }
      throw new Error(`gRPC status=${code} message=${message}`);
    }
    log("grpc stream closed", { methodPath, grpcStatus: code, messages: messageCount, trailersOnly: !sawTrailerFrame });
  }

  function asList(csv) {
    return (csv || "").split(",").map((v) => v.trim()).filter(Boolean);
  }

  function normalizeCloudPath(rawPath) {
    let p = (rawPath || "").trim();
    p = p.replace(/^\/+/, "").replace(/\/+$/, "");
    return p;
  }

  function assertValidCloudPath(rawPath, label) {
    const p = normalizeCloudPath(rawPath);
    if (!p) throw new Error(`${label} is empty`);
    if (p === "." || p === "..") throw new Error(`${label} is invalid`);
    if (p.startsWith("*") || p.startsWith("✹")) throw new Error(`${label} cannot start with version marker (* or ✹)`);
    if (p.includes("✹")) throw new Error(`${label} cannot include version marker (✹)`);
    return p;
  }

  logConfigSanity();

  function setUsers(users) {
    usersBody.innerHTML = "";
    if (!users.length) {
      usersBody.innerHTML = "<tr><td>No users returned</td></tr>";
      return;
    }
    for (const u of users) {
      const tr = document.createElement("tr");
      const td = document.createElement("td");
      td.textContent = u.user_name || u.userName || "";
      tr.appendChild(td);
      usersBody.appendChild(tr);
    }
  }

  function setSelectedFile(file) {
    selectedFile = file;
    if (!file) {
      fileInfo.textContent = "No file selected.";
      return;
    }
    fileInfo.textContent = `Selected: ${file.name} (${file.size} bytes)`;
    const safeName = assertValidCloudPath(file.name, "Selected file name");
    filePathInput.value = safeName;
    if (!copyToPathInput.value.trim()) copyToPathInput.value = `${safeName}.copy`;
  }

  dropZone.addEventListener("click", () => fileInput.click());
  fileInput.addEventListener("change", () => setSelectedFile(fileInput.files[0] || null));
  ["dragenter", "dragover"].forEach((eventName) => dropZone.addEventListener(eventName, (e) => {
    e.preventDefault();
    dropZone.classList.add("dragover");
  }));
  ["dragleave", "drop"].forEach((eventName) => dropZone.addEventListener(eventName, (e) => {
    e.preventDefault();
    dropZone.classList.remove("dragover");
  }));
  dropZone.addEventListener("drop", (e) => setSelectedFile(e.dataTransfer.files[0] || null));

  async function run(label, fn) {
    try {
      setStatus(`${label}...`);
      log(`${label} started`);
      const out = await fn();
      if (out !== undefined) setOutput(out);
      setStatus(`${label}: success`);
      log(`${label} success`);
    } catch (err) {
      setStatus(`${label}: error`);
      setOutput(err && err.message ? err.message : String(err));
      log(`${label} error`, err && err.message ? err.message : String(err));
    }
  }

  document.getElementById("bootstrapBtn").addEventListener("click", () => run("Bootstrap", async () => {
    const user = resolveRegistryUser();
    userNameInput.value = user;
    const pwd = userPasswordInput.value;
    log("Bootstrap identity", { userName: user, myuser: parseMyUserFromProperties(CONFIG.userProperties) });
    await grpcUnary("altastata.v1.UsersService/SetUserProperties", "SetUserPropertiesRequest", { userName: user, userProperties: CONFIG.userProperties }, "SetUserPropertiesResponse", false);
    await grpcUnary("altastata.v1.UsersService/SetPrivateKey", "SetPrivateKeyRequest", { userName: user, privateKeyEncrypted: CONFIG.privateKey }, "SetPrivateKeyResponse", false);
    const resp = await grpcUnary("altastata.v1.UsersService/SetPasswordForUser", "SetPasswordForUserRequest", { userName: user, accountPassword: pwd }, "SetPasswordForUserResponse", false);
    authBootstrapDone = true;
    return resp;
  }));

  document.getElementById("listUsersBtn").addEventListener("click", () => run("ListUsers", async () => {
    const users = [];
    await grpcServerStream("altastata.v1.UsersService/ListUsers", "Empty", {}, "UserSummary", (msg) => users.push(msg), true);
    setUsers(users);
    return { count: users.length, users };
  }));

  document.getElementById("getMyAccountBtn").addEventListener("click", () => run("GetMyAccount", () =>
    grpcUnary("altastata.v1.UsersService/GetMyAccount", "GetMyAccountRequest", {}, "User", true)
  ));
  document.getElementById("getUserBtn").addEventListener("click", () => run("GetUser", () =>
    grpcUnary("altastata.v1.UsersService/GetUser", "GetUserRequest", { userName: userNameInput.value.trim() }, "User", true)
  ));
  document.getElementById("setPasswordBtn").addEventListener("click", () => run("SetPassword", () =>
    grpcUnary("altastata.v1.UsersService/SetPassword", "SetPasswordRequest", { accountPassword: userPasswordInput.value }, "SetPasswordResponse", true)
  ));

  document.getElementById("createFileBtn").addEventListener("click", () => run("CreateFile", async () => {
    if (!selectedFile) throw new Error("Drop/select a file first");
    const path = assertValidCloudPath(filePathInput.value || selectedFile.name || "upload.bin", "File path");
    filePathInput.value = path;
    const bytes = new Uint8Array(await selectedFile.arrayBuffer());
    log("CreateFile payload", { filePath: path, bytes: bytes.length });
    const resp = await grpcUnary("altastata.v1.FileOpsService/CreateFile", "CreateFileRequest", { filePath: path, content: bytes }, "CreateFileResponse", true);
    const status = resp && resp.status ? resp.status : {};
    const storedPath = status.file_path || status.filePath || path;
    filePathInput.value = path;
    log("CreateFile completed", {
      requestPath: path,
      bytes: bytes.length,
      storedPath,
      operationState: status.operation_state || status.operationState || "",
      error: status.error || "",
    });
    return resp;
  }));

  document.getElementById("getBufferBtn").addEventListener("click", () => run("GetBuffer", async () => {
    const filePath = assertValidCloudPath(filePathInput.value, "File path");
    filePathInput.value = filePath;
    const resp = await grpcUnary("altastata.v1.FileOpsService/GetBuffer", "GetBufferRequest", {
      filePath,
      snapshotTime: 0,
      startPosition: 0,
      parallelChunks: 4,
      size: 1024 * 1024,
    }, "GetBufferResponse", true);
    const data = resp.data || new Uint8Array(0);
    const bytes = data.length || 0;
    const txt = new TextDecoder().decode(data instanceof Uint8Array ? data : new Uint8Array(data));
    return { bytes, headPreview: txt.slice(0, 500), tailPreview: txt.slice(-500) };
  }));

  document.getElementById("appendBtn").addEventListener("click", () => run("AppendBufferToFile", async () => {
    const filePath = assertValidCloudPath(filePathInput.value, "File path");
    filePathInput.value = filePath;
    const appendText = appendTextInput.value || "";
    await grpcUnary("altastata.v1.FileOpsService/AppendBufferToFile", "AppendBufferToFileRequest", {
      filePath,
      snapshotTime: 0,
      content: new TextEncoder().encode(appendText),
    }, "AppendBufferToFileResponse", true);

    let tail = new Uint8Array(0);
    await grpcServerStream("altastata.v1.FileOpsService/ReadStream", "ReadStreamRequest", {
      filePath,
      snapshotTime: 0,
      startPosition: 0,
      parallelChunks: 4,
      chunkSize: 65536,
    }, "ReadStreamChunk", (msg) => {
      const chunk = msg.data instanceof Uint8Array ? msg.data : new Uint8Array(msg.data || []);
      tail = keepTailBytes(tail, chunk, 64 * 1024);
    }, true);

    const tailText = new TextDecoder().decode(tail);
    const contains = appendText ? tailText.includes(appendText) : true;
    log("Append verification", { filePath, containsAppendedText: contains, appendTextLength: appendText.length });
    if (!contains) throw new Error("Append RPC succeeded but appended text was not found in file tail. Check file version/path or backend append behavior.");

    return { success: true, containsAppendedText: contains, tailPreview: tailText.slice(-500) };
  }));

  document.getElementById("copyBtn").addEventListener("click", () => run("CopyFile", () => {
    const fromPath = assertValidCloudPath(filePathInput.value, "Source file path");
    const toPath = assertValidCloudPath(copyToPathInput.value, "Destination file path");
    filePathInput.value = fromPath;
    copyToPathInput.value = toPath;
    return grpcUnary("altastata.v1.FileOpsService/CopyFile", "CopyFileRequest", {
      fromCloudFilePath: fromPath,
      toCloudFilePath: toPath,
    }, "CopyFileResponse", true);
  }));

  document.getElementById("deleteBtn").addEventListener("click", () => run("Delete", () => {
    const filePath = assertValidCloudPath(filePathInput.value, "Cloud path prefix");
    filePathInput.value = filePath;
    return grpcUnary("altastata.v1.FileOpsService/Delete", "DeleteRequest", {
      cloudPathPrefix: filePath,
      // true so file paths match directly; false makes core treat the prefix as a directory
      // and append "/", so deleting "sample.txt.copy" becomes "sample.txt.copy/" (no match).
      includingSubdirectories: true,
      timeIntervalStart: "",
      timeIntervalEnd: "",
    }, "DeleteResponse", true);
  }));

  document.getElementById("listVersionsBtn").addEventListener("click", () => run("ListVersions", async () => {
    const filePath = assertValidCloudPath(filePathInput.value, "Cloud path prefix");
    filePathInput.value = filePath;
    const entries = [];
    await grpcServerStream("altastata.v1.FileOpsService/ListVersions", "ListVersionsRequest", {
      cloudPathPrefix: filePath,
      // true so the prefix matches a FILE: with false the core treats the prefix as a directory
      // and appends "/", so "sample.txt" becomes "sample.txt/" and matches nothing.
      includingSubdirectories: true,
      timeIntervalStart: "",
      timeIntervalEnd: "",
    }, "VersionEntry", (msg) => entries.push(msg), true);
    return entries;
  }));

  document.getElementById("readStreamBtn").addEventListener("click", () => run("ReadStream", async () => {
    const filePath = assertValidCloudPath(filePathInput.value, "File path");
    filePathInput.value = filePath;
    let total = 0;
    await grpcServerStream("altastata.v1.FileOpsService/ReadStream", "ReadStreamRequest", {
      filePath,
      snapshotTime: 0,
      startPosition: 0,
      parallelChunks: 4,
      chunkSize: 65536,
    }, "ReadStreamChunk", (msg) => {
      const len = (msg.data && msg.data.length) ? msg.data.length : 0;
      total += len;
    }, true);
    return { totalBytes: total };
  }));

  document.getElementById("setAttrBtn").addEventListener("click", () => run("SetAttribute", () =>
    grpcUnary("altastata.v1.AttributesService/SetAttribute", "SetAttributeRequest", {
      filePath: filePathInput.value.trim(),
      snapshotTime: 0,
      name: attrNameInput.value.trim(),
      value: attrValueInput.value,
    }, "Empty", true)
  ));

  document.getElementById("getAttrBtn").addEventListener("click", () => run("GetAttribute", () =>
    grpcUnary("altastata.v1.AttributesService/GetAttribute", "GetAttributeRequest", {
      filePath: filePathInput.value.trim(),
      snapshotTime: 0,
      name: attrNameInput.value.trim(),
    }, "Attribute", true)
  ));

  document.getElementById("getAttrsBtn").addEventListener("click", () => run("GetAttributes", () =>
    grpcUnary("altastata.v1.AttributesService/GetAttributes", "GetAttributesRequest", {
      filePath: filePathInput.value.trim(),
      snapshotTime: 0,
      names: asList(attrNamesInput.value),
    }, "AttributeMap", true)
  ));

  document.getElementById("delAttrBtn").addEventListener("click", () => run("DeleteAttribute", () =>
    grpcUnary("altastata.v1.AttributesService/DeleteAttribute", "DeleteAttributeRequest", {
      filePath: filePathInput.value.trim(),
      snapshotTime: 0,
      name: attrNameInput.value.trim(),
    }, "Empty", true)
  ));

  document.getElementById("shareBtn").addEventListener("click", () => run("Share", () =>
    grpcUnary("altastata.v1.SharingService/Share", "ShareRequest", {
      filePaths: asList(sharePathsInput.value || filePathInput.value),
      readers: asList(readersInput.value),
    }, "ShareResult", true)
  ));

  document.getElementById("revokeBtn").addEventListener("click", () => run("Revoke", () =>
    grpcUnary("altastata.v1.SharingService/Revoke", "RevokeRequest", {
      filePaths: asList(sharePathsInput.value || filePathInput.value),
      readers: asList(readersInput.value),
    }, "RevokeResult", true)
  ));

  document.getElementById("shareByQueryBtn").addEventListener("click", () => run("ShareByQuery", () =>
    grpcUnary("altastata.v1.SharingService/ShareByQuery", "ShareByQueryRequest", {
      cloudPathPrefix: queryPrefixInput.value.trim(),
      includingSubdirectories: true,
      timeIntervalStart: "",
      timeIntervalEnd: "",
      readers: asList(readersInput.value),
    }, "ShareResult", true)
  ));

  document.getElementById("revokeByQueryBtn").addEventListener("click", () => run("RevokeByQuery", () =>
    grpcUnary("altastata.v1.SharingService/RevokeByQuery", "RevokeByQueryRequest", {
      cloudPathPrefix: queryPrefixInput.value.trim(),
      includingSubdirectories: true,
      timeIntervalStart: "",
      timeIntervalEnd: "",
      readers: asList(readersInput.value),
    }, "RevokeResult", true)
  ));

  document.getElementById("subscribeBtn").addEventListener("click", () => run("Subscribe", async () => {
    if (subscribeController) throw new Error("Already subscribed");
    eventsOutput.textContent = "Subscribed. Waiting for events...";
    subscribeController = new AbortController();
    try {
      await grpcServerStream("altastata.v1.EventsService/Subscribe", "SubscribeRequest", {}, "EventMessage", (msg) => {
        const eventName = msg.event_name || msg.eventName || "event";
        const data = msg.data || "";
        appendEvent(`${eventName}: ${data}`);
      }, true, subscribeController.signal);
    } finally {
      subscribeController = null;
    }
    return "Subscription ended";
  }));

  document.getElementById("stopSubscribeBtn").addEventListener("click", () => {
    if (subscribeController) {
      subscribeController.abort();
      subscribeController = null;
      setStatus("Subscribe stopped");
      log("Subscribe stopped by user");
    }
  });

  const clearLogsBtn = document.getElementById("clearLogsBtn");
  if (clearLogsBtn) {
    clearLogsBtn.addEventListener("click", () => {
      if (logsOutputEl) logsOutputEl.textContent = "Logs cleared.";
      log("Logs cleared by user");
    });
  }

  userNameInput.addEventListener("change", () => { authBootstrapDone = false; });
  userPasswordInput.addEventListener("change", () => { authBootstrapDone = false; });
})();
