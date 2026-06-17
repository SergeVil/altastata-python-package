#!/usr/bin/env python3
"""Integration test: virtual S3 object tagging via AltaStataFunctions.boto3_s3().

The bundled altastata-services JVM hosts the S3 gateway on :9876 in the same
process as py4j — no standalone ``gradle altastata-s3-gateway:run`` needed.

Environment:
  ALTASTATA_ACCOUNT_DIR  path to account dir (required)
  ALTASTATA_PASSWORD     account password (default: 123)
  ALTASTATA_BUCKET       S3 bucket name (default: altastata-bucket)
  ALTASTATA_TAGGING_SHARE_PRINCIPAL  optional override for add/revoke test;
      default: first user from listUsers() other than the logged-in myuser

Prerequisites:
  pip install -e . boto3
  bash scripts/build-bundled-artifacts.sh   # stages altastata-services uber jar

Share/revoke steps use AltaStata listUsers() (py4j/gRPC) to pick a peer principal;
boto3 has no org user list — only the S3 tagging PUT/GET calls.
"""

from __future__ import annotations

import os
import sys
import time

from botocore.exceptions import ClientError


def _tag_map(tagging_response: dict) -> dict[str, str]:
    return {t["Key"]: t["Value"] for t in tagging_response.get("TagSet", [])}


def _current_user_name(alt) -> str:
    """Logged-in AltaStata user via AltaStataFunctions (same path as s3_credentials)."""
    if alt.transport == "grpc" and alt.grpc_client is not None:
        return str(alt.grpc_client.get_my_account()["user_name"])
    user_name, _, _ = alt._read_bootstrap_material()
    return user_name


def _list_account_user_names(alt) -> list[str]:
    """Org users via AltaStata API (py4j listUsers or gRPC ListUsers)."""
    if alt.transport == "grpc":
        if alt.grpc_client is None:
            raise RuntimeError("gRPC client not initialized")
        return [u["user_name"] for u in alt.grpc_client.list_users()]

    if alt.altastata_file_system is None:
        raise RuntimeError("AltaStata filesystem not initialized")
    java_list = alt.altastata_file_system.listUsers()
    return [str(u) for u in java_list]


def _pick_peer_principal(alt, self_name: str) -> str:
    """First org user other than self (for readers_to_add / readers_to_revoke)."""
    override = os.environ.get("ALTASTATA_TAGGING_SHARE_PRINCIPAL", "").strip()
    if override:
        if override == self_name:
            raise RuntimeError(
                f"ALTASTATA_TAGGING_SHARE_PRINCIPAL={override!r} is the logged-in user; "
                "pick someone else"
            )
        return override

    users = _list_account_user_names(alt)
    print(f"listUsers(): {users}")
    for name in users:
        if name and name != self_name:
            return name
    raise RuntimeError(
        f"no peer user in listUsers() to share with (logged in as {self_name!r}). "
        "Provision another user in the org or set ALTASTATA_TAGGING_SHARE_PRINCIPAL"
    )


def _readers_tag(s3, bucket: str, key: str) -> str:
    return _tag_map(s3.get_object_tagging(Bucket=bucket, Key=key))["readers"]


def _wait_for_readers(s3, bucket: str, key: str, predicate, timeout_s: float = 15.0) -> str:
    """Poll GET tagging until readers satisfy predicate (share is async in core)."""
    deadline = time.time() + timeout_s
    last = ""
    while time.time() < deadline:
        last = _readers_tag(s3, bucket, key)
        if predicate(last):
            return last
        time.sleep(0.5)
    raise AssertionError(f"timed out waiting for readers; last={last!r}")


def run_object_tagging_tests() -> None:
    account_dir = os.environ.get("ALTASTATA_ACCOUNT_DIR")
    if not account_dir:
        raise RuntimeError("ALTASTATA_ACCOUNT_DIR is required")
    if not os.path.isdir(account_dir):
        raise RuntimeError(f"ALTASTATA_ACCOUNT_DIR is not a directory: {account_dir}")

    password = os.environ.get("ALTASTATA_PASSWORD", "123")
    bucket = os.environ.get("ALTASTATA_BUCKET", "altastata-bucket")
    test_key = f"S3TaggingTest/tagging-{int(time.time())}.txt"
    prefix_key = test_key.rsplit("/", 1)[0] + "/"

    from altastata import AltaStataFunctions

    alt = AltaStataFunctions.from_account_dir(account_dir)
    alt.set_password(password)
    s3 = alt.boto3_s3()
    self_user = _current_user_name(alt)
    peer = _pick_peer_principal(alt, self_user)

    try:
        print(f"Using bucket={bucket} key={test_key} self={self_user!r} peer={peer!r}")
        s3.put_object(Bucket=bucket, Key=test_key, Body=b"tagging integration test")

        resp = s3.get_object_tagging(Bucket=bucket, Key=test_key)
        tags = _tag_map(resp)
        if "owner" not in tags:
            raise AssertionError(f"expected owner tag, got {tags}")
        if "readers" not in tags:
            raise AssertionError(f"expected readers tag, got {tags}")
        print(f"GET tagging: owner={tags['owner']!r} readers={tags['readers']!r}")

        s3.put_object_tagging(
            Bucket=bucket,
            Key=test_key,
            Tagging={"TagSet": [{"Key": "readers_to_add", "Value": peer}]},
        )
        readers = _wait_for_readers(
            s3, bucket, test_key, lambda r: peer in r.split()
        )
        print(f"After readers_to_add: readers={readers!r}")

        s3.put_object_tagging(
            Bucket=bucket,
            Key=test_key,
            Tagging={"TagSet": [{"Key": "readers_to_revoke", "Value": peer}]},
        )
        readers = _wait_for_readers(
            s3, bucket, test_key, lambda r: peer not in r.split()
        )
        print(f"After readers_to_revoke: readers={readers!r}")

        try:
            s3.put_object_tagging(
                Bucket=bucket,
                Key=test_key,
                Tagging={"TagSet": [{"Key": "readers", "Value": self_user}]},
            )
            raise AssertionError("expected InvalidTag for PUT readers")
        except ClientError as exc:
            code = exc.response.get("Error", {}).get("Code")
            if code != "InvalidTag":
                raise AssertionError(f"expected InvalidTag, got {code}") from exc
        print("PUT readers correctly rejected with InvalidTag")

        try:
            s3.get_object_tagging(Bucket=bucket, Key=prefix_key)
            raise AssertionError("expected NoSuchKey for prefix GET")
        except ClientError as exc:
            code = exc.response.get("Error", {}).get("Code")
            if code != "NoSuchKey":
                raise AssertionError(f"expected NoSuchKey, got {code}") from exc
        print("GET prefix correctly returned NoSuchKey")

        s3.delete_object(Bucket=bucket, Key=test_key)
        print("All object tagging tests passed")
    finally:
        alt.shutdown()


def main() -> int:
    try:
        run_object_tagging_tests()
        return 0
    except Exception as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
