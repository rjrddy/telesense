#!/bin/sh
set -eu
# Fly mounts a root-owned volume at runtime, hiding image-time ownership.
mkdir -p /data/hf-cache /data/storage
chown telesense:telesense /data /data/hf-cache /data/storage
exec gosu telesense "$@"
