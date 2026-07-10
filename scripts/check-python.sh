#!/bin/bash
# Python Worker 检查脚本
# 执行 ruff、mypy、pytest、import-linter
set -euo pipefail

cd "$(dirname "$0")/.."
echo "=== 开始 Python Worker 检查 ==="

# 默认使用当前环境中的 Python 3.10；如需指定解释器，可传入 PYTHON_BIN=/path/to/python。
PYTHON="${PYTHON_BIN:-${PYTHON:-}}"
if [ -z "$PYTHON" ]; then
    for cmd in python3.10 python3 python; do
        if command -v "$cmd" &>/dev/null; then
            PYTHON="$cmd"
            break
        fi
    done
fi
if [ -z "$PYTHON" ]; then
    echo "ERROR: 未找到 Python 3.10 命令"
    exit 1
fi
echo "使用 Python: $($PYTHON --version 2>&1)"

PYTHON_MINOR="$($PYTHON -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
if [ "$PYTHON_MINOR" != "3.10" ]; then
    echo "ERROR: Python Worker 必须使用 Python 3.10，当前为 $PYTHON_MINOR"
    exit 1
fi

PYTHON_BIN_DIR="$(cd "$(dirname "$PYTHON")" && pwd)"
export PATH="$PYTHON_BIN_DIR:$PATH"

echo "--- ruff 静态检查 ---"
$PYTHON -m ruff check worker

echo "--- mypy 类型检查 ---"
$PYTHON -m mypy --config-file worker/pyproject.toml worker/src

echo "--- pytest 单元测试 ---"
$PYTHON -m pytest worker/tests -v

echo "--- import-linter 导入边界检查 ---"
PYTHONPATH=worker/src "$PYTHON_BIN_DIR/lint-imports" --config worker/pyproject.toml

echo "=== Python Worker 检查通过 ==="
