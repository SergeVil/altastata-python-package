#!/usr/bin/env bash
# Stop and remove RAG s390x containers and the 2026b_latest image on the LinuxONE server.
# Run from repo root. Uses same SSH_HOST/SSH_KEY as build-rag-s390x-on-server.sh.
#
# Usage: ./openshift/rag/stop-and-remove-rag-s390x-2026b-on-server.sh

set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
SSH_KEY="${SSH_KEY:-/Users/sergevilvovsky/Downloads/torontolinuxonesshkey_rsa.prv}"
SSH_HOST="${SSH_HOST:-root@163.66.89.80}"
SSH_OPTS="-i $SSH_KEY -o StrictHostKeyChecking=accept-new -o GSSAPIAuthentication=no -o PreferredAuthentications=publickey"
CONTAINER_NAME="rag-s390x-test"
OLD_TAG="2026b_latest"

echo "SSH: $SSH_HOST — stop/remove $CONTAINER_NAME and $OLD_TAG image(s)"

# Stop and remove the usual RAG container (may be 2026b or 2026c)
ssh $SSH_OPTS "$SSH_HOST" "docker stop $CONTAINER_NAME 2>/dev/null || true; docker rm $CONTAINER_NAME 2>/dev/null || true; echo '  Container stop/rm done'"

# Stop and remove any other containers using the 2026b image
ssh $SSH_OPTS "$SSH_HOST" "
  for img in altastata/rag-open-llm-s390x:$OLD_TAG icr.io/altastata/rag-open-llm-s390x:$OLD_TAG; do
    ids=\$(docker ps -a -q --filter ancestor=\$img 2>/dev/null)
    if [ -n \"\$ids\" ]; then
      echo \"  Stopping containers using \$img...\"
      for cid in \$ids; do docker stop \$cid 2>/dev/null || true; docker rm \$cid 2>/dev/null || true; done
    fi
  done
  echo '  Other containers done'
"

# Remove the 2026b_latest images (ignore errors if not present)
ssh $SSH_OPTS "$SSH_HOST" "
  for img in altastata/rag-open-llm-s390x:$OLD_TAG icr.io/altastata/rag-open-llm-s390x:$OLD_TAG; do
    if docker image inspect \$img >/dev/null 2>&1; then
      docker rmi \$img 2>/dev/null || docker rmi -f \$img 2>/dev/null || true
      echo \"  Removed image: \$img\"
    else
      echo \"  Image not present: \$img\"
    fi
  done
"

echo "Done. 2026b_latest containers and images cleared on server."
