#!/bin/bash
# Docker Compose 配置检查脚本
set -euo pipefail

cd "$(dirname "$0")/.."
echo "=== 开始 Docker Compose 检查 ==="
docker compose --env-file infra/.env.example -f infra/docker-compose.yml config
echo "=== Docker Compose 配置有效 ==="
