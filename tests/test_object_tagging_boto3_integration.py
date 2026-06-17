"""Optional boto3 integration test for S3 virtual object tagging.

Skipped unless ALTASTATA_ACCOUNT_DIR is set. Exercises the real bundled
altastata-services JVM + S3 gateway — not the mocked surface in
test_s3_helper.py.

Run manually:

    ALTASTATA_ACCOUNT_DIR=$HOME/.altastata/accounts/amazon.rsa.bob123 \\
        python3 -m unittest tests.test_object_tagging_boto3_integration
"""

import os
import subprocess
import sys
import unittest
from pathlib import Path


_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "test-object-tagging.py"


@unittest.skipUnless(
    os.environ.get("ALTASTATA_ACCOUNT_DIR"),
    "set ALTASTATA_ACCOUNT_DIR to run boto3 object tagging integration test",
)
class ObjectTaggingBoto3Integration(unittest.TestCase):
    def test_object_tagging_via_boto3_s3(self):
        result = subprocess.run(
            [sys.executable, str(_SCRIPT)],
            env=os.environ.copy(),
            capture_output=True,
            text=True,
        )
        self.assertEqual(
            0,
            result.returncode,
            (result.stdout or "") + (result.stderr or ""),
        )


if __name__ == "__main__":
    unittest.main()
