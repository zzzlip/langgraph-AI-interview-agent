#!/bin/bash
# 文档同步检查脚本
# 验收项覆盖：specs/01 工程基线与模块边界 需求文档.md 第 9.1 节 — 文档体系验收
set -euo pipefail

cd "$(dirname "$0")/.."
errors=0

echo "=== 文档同步检查 ==="

# ── 9.1-1: specs/ 作为统一需求文档根目录存在 ──
if [ -d "specs" ]; then
    echo "OK: specs/ 目录存在"
else
    echo "ERROR: specs/ 目录不存在"
    errors=$((errors + 1))
fi

# ── 9.1-2: 11 个 V1 模块均有独立目录和需求文档.md ──
spec_modules=(
    "01-工程基线与模块边界"
    "02-认证会话权限与安全基线"
    "03-任务中心与状态机"
    "04-文件结果与报告资产管理"
    "05-异步消息OutboxInbox与SSE"
    "06-PythonWorker基座与AIGateway"
    "07-简历评估业务闭环"
    "08-简历优化业务闭环"
    "09-算法测试业务闭环"
    "10-模拟面试业务闭环"
    "11-静态前端管理员排查与V1验收"
)

for module in "${spec_modules[@]}"; do
    doc_path="specs/${module}/需求文档.md"
    if [ -f "$doc_path" ]; then
        echo "OK: ${doc_path} 存在"
    else
        echo "ERROR: ${doc_path} 不存在"
        errors=$((errors + 1))
    fi
done

# ── 9.1-4: specs/README.md 存在 ──
if [ -f "specs/README.md" ]; then
    echo "OK: specs/README.md 存在"
else
    echo "ERROR: specs/README.md 不存在"
    errors=$((errors + 1))
fi

# ── 9.1-5: 当前项目架构.md 已记录需求文档体系和目标重构架构概览 ──
required_refs=("需求文档" "技术架构方案" "目标工程目录基线")
for ref in "${required_refs[@]}"; do
    if grep -q "$ref" "当前项目架构.md"; then
        echo "OK: 当前项目架构.md 包含"${ref}"索引"
    else
        echo "ERROR: 当前项目架构.md 未包含"${ref}"索引"
        errors=$((errors + 1))
    fi
done

# ── 汇总 ──
if [ $errors -gt 0 ]; then
    echo "=== 文档检查失败: $errors 个错误 ==="
    exit 1
else
    echo "=== 文档同步检查通过 ==="
fi
