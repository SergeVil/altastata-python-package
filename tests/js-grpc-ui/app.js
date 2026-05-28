(function () {
  "use strict";

  // Hardcoded test configuration. Edit these values directly.
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

  const statusEl = document.getElementById("status");
  const usersBody = document.getElementById("usersBody");
  const bootstrapBtn = document.getElementById("bootstrapBtn");
  const listUsersBtn = document.getElementById("listUsersBtn");

  const protoDef = `
    syntax = "proto3";
    package altastata.v1;
    message Empty {}
    message UserSummary {
      string user_name = 1;
      bool initialized = 2;
    }
    message SetUserPropertiesRequest {
      string user_name = 1;
      string user_properties = 2;
    }
    message SetUserPropertiesResponse {
      bool success = 1;
    }
    message SetPrivateKeyRequest {
      string user_name = 1;
      string private_key_encrypted = 2;
    }
    message SetPrivateKeyResponse {
      bool success = 1;
    }
    message SetPasswordForUserRequest {
      string user_name = 1;
      string account_password = 2;
    }
    message SetPasswordForUserResponse {
      bool success = 1;
      string access_key = 2;
      string secret_key = 3;
    }
  `;

  const root = protobuf.parse(protoDef).root;
  const Empty = root.lookupType("altastata.v1.Empty");
  const UserSummary = root.lookupType("altastata.v1.UserSummary");
  const SetUserPropertiesRequest = root.lookupType("altastata.v1.SetUserPropertiesRequest");
  const SetUserPropertiesResponse = root.lookupType("altastata.v1.SetUserPropertiesResponse");
  const SetPrivateKeyRequest = root.lookupType("altastata.v1.SetPrivateKeyRequest");
  const SetPrivateKeyResponse = root.lookupType("altastata.v1.SetPrivateKeyResponse");
  const SetPasswordForUserRequest = root.lookupType("altastata.v1.SetPasswordForUserRequest");
  const SetPasswordForUserResponse = root.lookupType("altastata.v1.SetPasswordForUserResponse");

  function setStatus(msg) {
    statusEl.textContent = msg;
  }

  function setUsers(users) {
    usersBody.innerHTML = "";
    if (!users.length) {
      usersBody.innerHTML = "<tr><td>No users returned</td></tr>";
      return;
    }
    for (const u of users) {
      const tr = document.createElement("tr");
      const tdName = document.createElement("td");
      tdName.textContent = u.user_name || u.userName || "";
      tr.appendChild(tdName);
      usersBody.appendChild(tr);
    }
  }

  function frameGrpcWebMessage(messageBytes) {
    const out = new Uint8Array(5 + messageBytes.length);
    out[0] = 0x00; // data frame
    const len = messageBytes.length >>> 0;
    out[1] = (len >>> 24) & 0xff;
    out[2] = (len >>> 16) & 0xff;
    out[3] = (len >>> 8) & 0xff;
    out[4] = len & 0xff;
    out.set(messageBytes, 5);
    return out;
  }

  function parseGrpcWebFrames(bytes) {
    const frames = [];
    let offset = 0;
    while (offset + 5 <= bytes.length) {
      const flags = bytes[offset];
      const len =
        (bytes[offset + 1] << 24) |
        (bytes[offset + 2] << 16) |
        (bytes[offset + 3] << 8) |
        bytes[offset + 4];
      offset += 5;
      if (len < 0 || offset + len > bytes.length) {
        throw new Error("Malformed gRPC-Web frame");
      }
      frames.push({
        trailer: (flags & 0x80) !== 0,
        payload: bytes.slice(offset, offset + len),
      });
      offset += len;
    }
    return frames;
  }

  function parseTrailerHeaders(payload) {
    const txt = new TextDecoder().decode(payload);
    const headers = new Map();
    for (const line of txt.split("\r\n")) {
      if (!line) continue;
      const idx = line.indexOf(":");
      if (idx < 0) continue;
      const k = line.slice(0, idx).trim().toLowerCase();
      const v = line.slice(idx + 1).trim();
      headers.set(k, v);
    }
    return headers;
  }

  function buildBaseUrl() {
    const baseUrl = CONFIG.grpcBaseUrl.trim().replace(/\/+$/, "");
    if (!baseUrl) {
      throw new Error("Please provide gRPC server URL.");
    }
    return baseUrl;
  }

  function buildToken() {
    if (!CONFIG.userName.trim()) {
      throw new Error("CONFIG.userName must be set.");
    }
    return `local-${CONFIG.userName.trim()}`;
  }

  async function grpcUnary(baseUrl, methodPath, reqType, reqObj, respType, token) {
    const reqBytes = reqType.encode(reqType.create(reqObj)).finish();
    const body = frameGrpcWebMessage(reqBytes);
    const headers = {
      "content-type": "application/grpc-web+proto",
      "x-grpc-web": "1",
      "x-user-agent": "grpc-web-js-test",
    };
    if (token) {
      headers["authorization"] = `Bearer ${token}`;
    }

    const response = await fetch(`${baseUrl}/${methodPath}`, {
      method: "POST",
      headers,
      body,
    });
    if (!response.ok) {
      throw new Error(`HTTP ${response.status} ${response.statusText}`);
    }

    const bytes = new Uint8Array(await response.arrayBuffer());
    const frames = parseGrpcWebFrames(bytes);
    let msg = null;
    let trailerHeaders = new Map();
    for (const frame of frames) {
      if (frame.trailer) {
        trailerHeaders = parseTrailerHeaders(frame.payload);
      } else {
        msg = respType.decode(frame.payload);
      }
    }
    const grpcStatus = trailerHeaders.get("grpc-status") || "0";
    const grpcMessage = trailerHeaders.get("grpc-message") || "";
    if (grpcStatus !== "0") {
      throw new Error(`gRPC status=${grpcStatus} message=${grpcMessage}`);
    }
    return msg;
  }

  async function bootstrapUser() {
    const baseUrl = buildBaseUrl();
    const userName = CONFIG.userName.trim();
    const password = CONFIG.accountPassword;
    const userProperties = CONFIG.userProperties;
    const privateKey = CONFIG.privateKey;

    if (!userName) {
      throw new Error("Please provide user name.");
    }
    if (!password) {
      throw new Error("Please provide account password.");
    }
    if (!userProperties.trim()) {
      throw new Error("Please provide user.properties content.");
    }
    if (!privateKey.trim()) {
      throw new Error("Please provide private.key content.");
    }

    setStatus("Bootstrapping user...");
    await grpcUnary(
      baseUrl,
      "altastata.v1.UsersService/SetUserProperties",
      SetUserPropertiesRequest,
      { user_name: userName, user_properties: userProperties },
      SetUserPropertiesResponse
    );
    await grpcUnary(
      baseUrl,
      "altastata.v1.UsersService/SetPrivateKey",
      SetPrivateKeyRequest,
      { user_name: userName, private_key_encrypted: privateKey },
      SetPrivateKeyResponse
    );
    const passwordResult = await grpcUnary(
      baseUrl,
      "altastata.v1.UsersService/SetPasswordForUser",
      SetPasswordForUserRequest,
      { user_name: userName, account_password: password },
      SetPasswordForUserResponse
    );

    const accessKey = passwordResult && passwordResult.accessKey ? passwordResult.accessKey : "";
    setStatus(
      `Bootstrap success for ${userName}. Token: local-${userName}. ` +
      (accessKey ? `Access key: ${accessKey}` : "Access key not returned.")
    );
  }

  async function listUsers() {
    const baseUrl = buildBaseUrl();
    const token = buildToken();

    setStatus("Calling ListUsers...");
    setUsers([]);

    const emptyPayload = Empty.encode(Empty.create({})).finish();
    const body = frameGrpcWebMessage(emptyPayload);

    const endpoint = `${baseUrl}/altastata.v1.UsersService/ListUsers`;
    const response = await fetch(endpoint, {
      method: "POST",
      headers: {
        "content-type": "application/grpc-web+proto",
        "x-grpc-web": "1",
        "x-user-agent": "grpc-web-js-test",
        "authorization": `Bearer ${token}`,
      },
      body,
    });

    if (!response.ok) {
      throw new Error(`HTTP ${response.status} ${response.statusText}`);
    }

    const bytes = new Uint8Array(await response.arrayBuffer());
    const frames = parseGrpcWebFrames(bytes);

    const users = [];
    let trailerHeaders = new Map();
    for (const frame of frames) {
      if (frame.trailer) {
        trailerHeaders = parseTrailerHeaders(frame.payload);
      } else {
        users.push(UserSummary.decode(frame.payload));
      }
    }

    const grpcStatus = trailerHeaders.get("grpc-status") || "0";
    const grpcMessage = trailerHeaders.get("grpc-message") || "";
    if (grpcStatus !== "0") {
      throw new Error(`gRPC status=${grpcStatus} message=${grpcMessage}`);
    }

    setUsers(users);
    setStatus(`Success. Received ${users.length} user(s).`);
  }

  listUsersBtn.addEventListener("click", async () => {
    try {
      await listUsers();
    } catch (err) {
      setStatus(`Error: ${err && err.message ? err.message : String(err)}`);
    }
  });

  bootstrapBtn.addEventListener("click", async () => {
    try {
      await bootstrapUser();
    } catch (err) {
      setStatus(`Error: ${err && err.message ? err.message : String(err)}`);
    }
  });
})();
