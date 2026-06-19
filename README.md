# Altastata Python Package

Secure, encrypted cloud storage for Python — with **fsspec**, **PyTorch/TensorFlow**, **LangChain**, **Databricks**, **Snowflake**, **boto3/S3**, **gRPC**, and a bundled **Web UI** (AltaStata Console).

```bash
pip install altastata
```

## What you get

| Capability | How |
|------------|-----|
| Encrypted files in S3, Azure, IBM COS, etc. | `AltaStataFunctions` |
| Standard Python file APIs | `fsspec` (`create_filesystem`) |
| ML datasets | `AltaStataPyTorchDataset`, `AltaStataTensorFlowDataset` |
| **LangChain** / RAG document loading | fsspec + `DirectoryLoader` / `TextLoader` |
| **Databricks** / Apache Spark | AltaStata Hadoop FS (`altastata-hadoop` JAR) |
| **Snowflake** external stages | S3 Gateway (port **9876**) or Snowpark Python + fsspec |
| S3 tools (boto3, aws CLI, s3fs) | S3-compatible API on port **9876** |
| gRPC API (Python `transport="grpc"`, JS clients) | `altastata-services` JVM on port **9877** |
| Real-time share/delete events | gRPC `EventsService`, Py4J callback, or Web UI |
| **Web UI** — Finder-style file manager in the browser | http://127.0.0.1:9877 (same JVM as gRPC) |

---

## Configure your account

Two equivalent ways to connect from Python:

### 1. Account folder on disk (typical)

Each user keeps a directory under `~/.altastata/accounts/<display-name>/`:

```
amazon.rsa.bob123/
  altastata-myorg-bob123.user.properties   # from your admin
  private.key                              # RSA (password-encrypted PEM)
  public.key
```

| Account type | Key files | Password |
|--------------|-----------|----------|
| **RSA** | `private.key`, `public.key` | Yes |
| **PQC** | `kyber_private.key`, `dilithium_private.key`, … | Yes |
| **HPCS** | `hpcs-privkey.blob`, `public.key`, `hpcs.marker` | No |
| **HSM** | `*user.properties` only | No |

```python
from altastata import AltaStataFunctions

f = AltaStataFunctions.from_account_dir(
    "/path/to/.altastata/accounts/amazon.rsa.bob123",
    transport="grpc",
    password="your_password",
)
```

### 2. Inline credentials (`user_properties` + `private_key`)

Pass the same text you would have in files — useful for notebooks, secrets managers, or CI:

```python
from altastata import AltaStataFunctions

user_properties = """#My Properties
#Sun Jan 05 12:10:23 EST 2025
AWSSecretKey=*****
AWSAccessKeyId=*****
myuser=bob123
accounttype=amazon-s3-secure
acccontainer-prefix=altastata-myorg-
region=us-east-1
metadata-encryption=RSA"""

private_key = """-----BEGIN RSA PRIVATE KEY-----
Proc-Type: 4,ENCRYPTED
DEK-Info: DES-EDE3,F26EBECE6DDAEC52

... encrypted PEM body ...
-----END RSA PRIVATE KEY-----"""

altastata_functions = AltaStataFunctions.from_credentials(user_properties, private_key)
altastata_functions.set_password("my_password")

# Or with gRPC transport:
altastata_functions = AltaStataFunctions.from_credentials(
    user_properties,
    private_key,
    transport="grpc",
    password="my_password",
)
```

Your org admin creates `*user.properties` after you send them `public.key` (RSA/PQC/HPCS).

---

## Quick start (gRPC — recommended)

`transport="grpc"` auto-starts the bundled Java gateway (Web UI + gRPC + S3).

```python
from altastata import AltaStataFunctions

# RSA / PQC
f = AltaStataFunctions.from_account_dir(
    "/path/to/.altastata/accounts/amazon.rsa.bob123",
    transport="grpc",
    password="your_password",
)

# HPCS / HSM — empty password
f = AltaStataFunctions.from_account_dir(
    "/path/to/.altastata/accounts/amazon.rsa.hpcs.myuser",
    transport="grpc",
    password="",
)

print(f.list_cloud_versions("Public/", True))
```

### Ports

One bundled Java process (`altastata-grpc-server` / `altastata-services`) listens on:

| Port | Service |
|------|---------|
| **9877** | gRPC (file ops, auth, events) + Web UI static files |
| **9876** | S3-compatible REST API |
| **25333** | Py4J (legacy in-process bridge to Python) |

### HPCS in Docker / Jupyter

Mount a populated `grep11client.yaml` (e.g. `/etc/ep11client/grep11client.yaml`) and `hpcs-privkey.blob`. See [`containers/jupyter/README-Docker.md`](containers/jupyter/README-Docker.md).

---

## Legacy Py4J transport (default)

```python
from altastata import AltaStataFunctions

f = AltaStataFunctions.from_account_dir("/path/to/account")
f.set_password("your_password")
```

---

## fsspec

```python
from altastata import AltaStataFunctions
from altastata.fsspec import create_filesystem

f = AltaStataFunctions.from_account_dir("/path/to/account", transport="grpc", password="secret")
fs = create_filesystem(f, "my_account")

with fs.open("Public/readme.txt", "r") as fh:
    print(fh.read())
```

Works with pandas, dask, and other fsspec consumers.

---

## LangChain, Databricks, Snowflake

### LangChain / RAG

Load encrypted documents without copying them to local disk:

```python
from altastata import AltaStataFunctions
from altastata.fsspec import create_filesystem
from langchain_core.documents import Document

f = AltaStataFunctions.from_account_dir("/path/to/account", transport="grpc", password="secret")
fs = create_filesystem(f, "my_account")

with fs.open("Public/docs/policy.txt", "r") as fh:
    docs = [Document(page_content=fh.read(), metadata={"source": "Public/docs/policy.txt"})]
```

`TextLoader`, `DirectoryLoader`, and other LangChain loaders work via the `altastata://` fsspec protocol once the filesystem is registered — see [`examples/fsspec-example/`](examples/fsspec-example/) and full RAG pipelines in [`examples/rag-example/`](examples/rag-example/).

### Databricks / Apache Spark

Use the AltaStata Hadoop filesystem implementation so Spark jobs read encrypted paths on cluster storage (`altastata://…` or configured Hadoop URI). Deploy the `altastata-hadoop` shadow JAR on Databricks / Spark clusters.

### Snowflake

- **External stage via S3:** point Snowflake at the bundled S3 Gateway (`http://host:9876`) as an S3-compatible endpoint for encrypted objects in your backing bucket.
- **Snowpark Python:** use fsspec / `create_filesystem` in Snowpark notebooks to read AltaStata paths with the same account credentials.

---

## S3-compatible API (boto3, aws CLI, s3fs)

```python
f = AltaStataFunctions.from_account_dir("/path/to/account", transport="grpc", password="secret")

s3 = f.boto3_s3()   # pip install boto3
s3.put_object(Bucket="altastata-bucket", Key="hello.txt", Body=b"hi")

f.install_aws_env()   # AWS_* for !aws s3 ls in Jupyter
```

---

## PyTorch & TensorFlow

```python
from altastata import AltaStataFunctions, AltaStataPyTorchDataset
from altastata.altastata_pytorch_dataset import register_altastata_functions_for_pytorch

f = AltaStataFunctions.from_account_dir("/path/to/account", transport="grpc", password="secret")
register_altastata_functions_for_pytorch(f, "my_account")
dataset = AltaStataPyTorchDataset("my_account", root_dir="Public/", file_pattern="*.jpg")
```

See `examples/pytorch-example/` and `examples/tensorflow-example/`.

---

## Event notifications

```python
def on_event(name, data):
    print(name, data)

f = AltaStataFunctions.from_account_dir("/path/to/account", enable_callback_server=True)
f.set_password("secret")
f.add_event_listener(on_event)
```

With gRPC / Web UI, SHARE and DELETE events also appear in the browser and via `EventsService.Watch`.

See `examples/event-listener-example/`.

---

## Docker Jupyter (optional)

```bash
cd containers/jupyter
docker compose -f docker-compose.yml -f docker-compose-ghcr.yml up -d
```

- JupyterLab: http://127.0.0.1:8888  
- **Web UI** / gRPC: http://127.0.0.1:9877  

Images: `ghcr.io/sergevil/altastata/jupyter-datascience-{arm64,amd64}:latest`

---

## Web UI (AltaStata Console)

The wheel ships a browser file manager. Start the gateway:

```bash
altastata-grpc-server
# same as: python -m altastata.grpc_server
```

Open **http://127.0.0.1:9877** — Miller-column browser, upload/download, share, generate keys, and live refresh on SHARE/DELETE events.

**Sign in:** Settings → **Choose account folder** → **Sign in**

| Account type | Password in Settings |
|--------------|-------------------|
| RSA / PQC | Your account password |
| HPCS / HSM | Leave blank |

Set `ALTASTATA_WEB_UI_DIR=` (empty) to disable the UI and run gRPC-only.

---

## More documentation

- **Developers** (build wheel, bundle JAR + Console SPA, PyPI): [`README-developer.md`](README-developer.md)
- **Examples**: [`examples/`](examples/)

## Questions?

Email [contact@altastata.com](mailto:contact@altastata.com).

## License

MIT License — see [LICENSE](LICENSE).
