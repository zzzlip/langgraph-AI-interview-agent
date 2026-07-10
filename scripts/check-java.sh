#!/bin/bash
# Java 平台检查脚本
# 执行 Maven 编译、测试和架构检查
set -euo pipefail

cd "$(dirname "$0")/.."
echo "=== 开始 Java 平台检查 ==="
mvn -f java-platform/pom.xml -B clean verify
echo "=== Java 平台检查通过 ==="
