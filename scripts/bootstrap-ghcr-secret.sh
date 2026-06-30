#!/usr/bin/env bash
# Create ghcr-secret for pulling private images from GitHub Container Registry.
# Usage:
#   export GHCR_USER=nottmv
#   export GHCR_TOKEN=<PAT with read:packages>
#   ./scripts/bootstrap-ghcr-secret.sh
set -euo pipefail

NS="${NAMESPACE:-default}"
SECRET_NAME="${SECRET_NAME:-ghcr-secret}"

: "${GHCR_USER:?Set GHCR_USER (GitHub username)}"
: "${GHCR_TOKEN:?Set GHCR_TOKEN (PAT with read:packages)}"

kubectl create namespace "$NS" 2>/dev/null || true
kubectl -n "$NS" delete secret "$SECRET_NAME" 2>/dev/null || true
kubectl -n "$NS" create secret docker-registry "$SECRET_NAME" \
  --docker-server=ghcr.io \
  --docker-username="$GHCR_USER" \
  --docker-password="$GHCR_TOKEN"
echo "OK: secret $SECRET_NAME in namespace $NS"
