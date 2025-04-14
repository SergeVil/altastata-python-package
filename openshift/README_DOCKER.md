# Docker Instructions for Altastata Workbench

This document provides instructions for building and running the Altastata workbench Docker container.

## Table of Contents
- [Directory Structure](#directory-structure)
- [Prerequisites](#prerequisites)
- [Quick Start](#quick-start)
- [Detailed Instructions](#detailed-instructions)
  - [Building the Image](#building-the-image)
  - [Running the Container](#running-the-container)
  - [Container Management](#container-management)
  - [Shell Access and Logs](#shell-access-and-logs)
  - [Local Registry Setup](#local-registry-setup)
- [Verification and Troubleshooting](#verification-and-troubleshooting)
- [Notes and Considerations](#notes-and-considerations)

## Directory Structure

```
altastata-python-package/
├── altastata/              # Package source code
│   ├── altastata_functions.py
│   ├── base_gateway.py
│   ├── __init__.py
│   └── lib/                # Java libraries
├── openshift/              # OpenShift/Docker related files
│   ├── Dockerfile          # Container build instructions
│   ├── requirements.txt    # Python package dependencies
│   ├── .dockerignore       # Files to exclude from Docker build
│   └── README_DOCKER.md    # This file
├── setup.py                # Package installation configuration
├── test_script.py          # Test scripts
└── README.md               # Main project documentation
```

## Prerequisites

- Docker installed on your system
- Git repository cloned
- Python package dependencies in `openshift/requirements.txt`

## Quick Start

```bash

# Build the image
docker build -t altastata:jupyter-datascience-py311_latest -f openshift/Dockerfile .

# Kill the existing one
docker rm -f altastata-jupyter

# Run the container with .altastata volume mount
docker run --platform linux/amd64 \
  --name altastata-jupyter \
  -p 8888:8888 \
  -v /Users/sergevilvovsky/.altastata:/opt/app-root/src/.altastata:rw \
  -v /Users/sergevilvovsky/Desktop:/opt/app-root/src/Desktop:rw \
  altastata:jupyter-datascience-py311_latest

# Access Jupyter Lab at http://127.0.0.1:8888/lab

# In your Python code, use the container path:
from altastata import AltaStataFunctions

# Initialize with account directory
altastata_functions = AltaStataFunctions.from_account_dir('/opt/app-root/src/.altastata/accounts/amazon.pqc.alice786')

or 

user_properties = """#My Properties
#Sun Jan 05 12:10:23 EST 2025
AWSSecretKey=vcJXbtg/YGApAUpY9sjsj1xvmpz9MUPTYMxY+hDn5zZ3Fmc1BuVS34zoTRDQJ7XAvu2Z0+piCEN3TA5OArj77FlL4doYDZx7YWXUopwUhMVyBvP+gT4buHc3hkf1FvHYElbUe3yX/57fnaYP1Nwg1zN9fupzEOGtCMjy39e9Xj4vvVgXo/+YW6ogG8uXi5JA9Fm2aG7hEWQstjwu5shcMT+Q6BR2SOtkAB8B9gYlCIt7ciJ4ikkAKqtfQ8TWkOsN
media-player=vlcj
myuser=bob123
accounttype=amazon-s3-secure
AWSAccessKeyId=ZWnrkxX43me3l1YBCGX42RhdzXmhP4q4rEOcquLZJIFWCEA9+sVA+hnRYTFcJoJ5nn0luDmQJJkYaayvtAP1IG6/0h4d4sWb+1NQ/hVozOdQMezUSp+z2Wruv4WX6TQpmz12N7zqQALMDD6qi5hTiv2QLJY084ufcoMZzmK1E0uw3jTG6Pci03Zy8TFbhhbuag88Stc9thyoN44ou/d5/8Id0AruvE0EK2Q7Jg0AZZI\\=
region=us-east-1
kms-region=us-east-2
metadata-encryption=RSA
password-timeout-interval=9000000000
acccontainer-prefix=altastata-myorgrsa444-"""

private_key = """-----BEGIN RSA PRIVATE KEY-----
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
-----END RSA PRIVATE KEY-----"""

# Initialize with credentials
altastata_functions = AltaStataFunctions.from_credentials(user_properties, private_key)

altastata_functions.set_password("123")

# Store
result = altastata_functions.store(['/opt/app-root/src/Desktop/serge.png',
                                    '/opt/app-root/src/Desktop/meeting_saved_chat.txt'],
                                   '/opt/app-root/src/Desktop', 'StoreTest', True)