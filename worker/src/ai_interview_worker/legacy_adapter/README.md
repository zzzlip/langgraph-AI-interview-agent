# Legacy 原型适配层

## 职责

本目录用于集中适配 `code/` 目录下的现有 Python LangGraph 原型代码。

## 适配规则

1. 新 Worker 业务代码不得直接 import `code/*`。
2. 适配层负责将平台任务输入转换为 legacy 输入，将 legacy 输出转换为 Worker 结果。
3. 遗留问题（本地路径、控制台输入、硬编码密钥、本机摄像头/麦克风依赖等）
   不在 spec01 直接重写，需在后续 spec 中逐步治理。
