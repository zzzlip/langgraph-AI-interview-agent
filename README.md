# AI Interview Agent Platform

面向求职准备的 AI 面试辅助平台。V1 采用 Java 平台层 + Python AI Worker + 静态前端 + Docker Compose 基础设施的组合，支持多用户认证、文件管理、异步任务、SSE 进度、简历评估、简历优化、算法测试和模拟面试。

历史 LangGraph 原型代码不作为 V1 发布内容；`specs/` 是需求、技术方案和 TDD 文档，不参与服务启动。

## 主要功能

### 简历评估

- 解析用户上传的简历文件，结合目标岗位说明提取教育背景、项目经历、技术栈、工作经历和表达结构等信息。
- 从未来潜力、教育背景、技术栈与岗位匹配、经历匹配、简历结构五个维度生成评分和解释。
- 输出可下载的评估报告和雷达图，帮助用户定位简历优势、风险点和下一步优化方向。

### 简历优化

- 基于岗位目标对简历内容进行结构化分析，识别表述空泛、职责结果不清、关键词覆盖不足等问题。
- 生成优化后的简历文档，保留用户真实经历，不使用固定模板伪造业务结果。
- 支持头像、基础信息、教育经历、项目经历、工作经历和技能描述等内容的组合处理。

### 算法测试

- 根据目标岗位和职位描述生成算法训练任务，并通过 Codeforces 等正式端口获取候选题目。
- 支持按难度、标签和岗位方向选择题目，记录任务状态、进度和结果。
- 用户提交后可触发检查流程，系统异步确认提交状态并回写任务反馈。

### 模拟面试

- 根据简历和岗位信息生成技术基础、项目经历、业务理解、软技能等不同类型的问题。
- 支持回答提交、追问决策、阶段状态更新、任务内记忆压缩和最终报告生成。
- 面试状态通过任务中心和 SSE 反馈给前端，便于用户持续查看当前轮次和结果。

### 多模态分析

- Worker 可处理文本、音频、视频等输入，调用正式 ASR/多模态 provider 分析表达、语速、停顿和视频表现。
- 多模态信号作为模拟面试评估的补充信息，不替代业务主链路的结构化判断。

### 题库与报告

- 支持基于岗位、公司背景和简历内容生成面试问题与参考分析。
- 简历评估、简历优化、模拟面试等能力会产出可验证文件，统一存储到 MinIO 并由 Java 平台绑定到任务结果。

## 项目架构

```text
frontend/static
  -> 同源 REST / SSE
java-platform
  -> MySQL: 账号、任务、文件、结果、Outbox/Inbox、审计
  -> Redis: Session、CSRF、限流、短期缓存
  -> MinIO: 用户文件和 Worker 产物
  -> RabbitMQ: task.command / task.feedback
worker
  -> 消费 RabbitMQ 命令
  -> 读取 MinIO 输入文件
  -> 调用 DeepSeek / DashScope / Codeforces 等正式端口
  -> 写入 MinIO 产物并发布反馈事件
```

### 目录说明

| 目录 | 作用 |
| --- | --- |
| `java-platform/` | Spring Boot 3 模块化单体，承载认证、权限、任务、文件、结果、异步消息、SSE、管理员排查和静态资源托管 |
| `worker/` | Python AI Worker，执行简历评估、简历优化、算法测试、模拟面试等真实业务逻辑 |
| `frontend/static/` | 无构建步骤的静态 HTML/CSS/JS 前端，调用同源 `/api/v1/**` |
| `infra/` | Docker Compose 编排 MySQL、Redis、RabbitMQ、MinIO、Java 平台和 Worker |
| `scripts/` | 本地质量检查入口 |
| `docs/adr/` | 架构决策记录 |
| `specs/` | spec01-spec11 的需求、技术方案和 TDD 开发方案 |

### Java 平台分层

| 模块 | 职责 |
| --- | --- |
| `platform-domain` | 领域模型、状态机、值对象、仓储接口，禁止依赖框架和基础设施 |
| `platform-application` | 用例编排、事务边界、任务创建、状态流转、Port 定义 |
| `platform-interface` | HTTP Controller、请求解析、响应组装、接口错误映射 |
| `platform-infrastructure` | MySQL、Redis、RabbitMQ、MinIO、安全过滤器和外部适配 |
| `platform-bootstrap` | Spring Boot 启动、配置装配、静态资源托管 |
| `platform-architecture-test` | ArchUnit 架构门禁 |

### Worker 分层

| 模块 | 职责 |
| --- | --- |
| `domain/` | Worker 命令、结果、失败、产物等纯领域对象 |
| `application/` | AI 业务 Handler、任务级 LLM 配置、错误映射、反馈事件组装 |
| `infrastructure/` | RabbitMQ、MinIO、Codeforces、AI Gateway、SQLite 幂等存储 |
| `interface/` | CLI 和 MQ Consumer 入口 |
| `legacy_adapter/` | 兼容导出正式 Handler，不维护独立占位业务逻辑 |

## 技术选型

| 层面 | 选型 | 说明 |
| --- | --- | --- |
| 前端 | 静态 HTML + 原生 JavaScript | V1 降低构建复杂度，同源访问 Java 平台 |
| Java 平台 | Spring Boot 3、Spring Security、MyBatis-Plus | 统一身份、权限、任务、文件、结果和异步协作 |
| Python Worker | Python 3.10、LangGraph、httpx、DashScope SDK | 复用 AI 工作流生态，执行真实模型、ASR、多模态和外部查询 |
| 数据库 | MySQL 8 | 核心事实源 |
| 缓存 | Redis 7 | Session、CSRF、限流和短期缓存 |
| 消息 | RabbitMQ 3 | 长任务命令、进度、结果、失败和死信 |
| 文件 | MinIO | 用户上传和 Worker 产物对象存储 |
| 部署 | Docker Compose | V1 本地与单机部署一致编排 |

## 配置

先复制环境变量样例：

```bash
cd infra
cp .env.example .env
```

上线或联调前必须修改 `.env` 中所有 `CHANGE_ME_*` 值。核心变量如下：

| 变量 | 说明 |
| --- | --- |
| `MYSQL_*` | MySQL 数据库、业务用户和 root 密码 |
| `REDIS_PASSWORD` | Redis 密码 |
| `RABBITMQ_DEFAULT_*` | RabbitMQ 用户、密码和 vhost |
| `MINIO_ROOT_*` | MinIO 管理账号 |
| `AI_INTERVIEW_MINIO_BUCKET` | Java 平台与 Worker 共享的业务文件 bucket |
| `AI_INTERVIEW_*` | Java 平台数据库、Redis、RabbitMQ、MinIO 和安全配置 |
| `AI_WORKER_ENABLED_PROVIDERS` | Worker 启用的 provider，默认 `deepseek,dashscope` |
| `AI_WORKER_DEEPSEEK_API_KEY` | 可选，全局 DeepSeek key；用户也可在前端保存任务级 key |
| `AI_WORKER_DASHSCOPE_API_KEY` | 启用 DashScope/ASR/多模态时必填 |
| `AI_WORKER_SERPER_API_KEY` | 启用 search provider 时必填 |

DeepSeek 支持用户级配置：前端保存 API Key 后，业务创建请求只携带公开配置，后端按账号复用已保存密钥；API Key 不进入用户可见响应、SSE、幂等 hash 或日志摘要。

## Docker 启动

```bash
cd infra
docker compose up --build
```

启动后访问：

| 服务 | 地址 |
| --- | --- |
| 前端入口 | `http://localhost:8080/` |
| Java health | `http://localhost:8080/actuator/health` |
| RabbitMQ 管理台 | `http://localhost:15672/` |
| MinIO 控制台 | `http://localhost:9001/` |

Java 镜像会把 `frontend/static/` 打入 Spring Boot classpath；Worker 镜像会安装 PDF、DOCX、图表、文件解析、`httpx` 和 DashScope 运行依赖，并使用 `/app/.data/processed_commands.sqlite` 保存消费幂等记录。

## 本地开发

本地开发需要先安装：

- JDK 21
- Maven 3.9+
- Python 3.10
- Docker Compose v2

Java 验证命令：

```bash
mvn -f java-platform/pom.xml test
```

如果 Maven 没有加入 `PATH`，请先在本机安装 Maven，或把命令中的 `mvn` 替换为自己的 Maven 可执行文件路径。

Worker 建议使用 Python 3.10 虚拟环境：

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install -r worker/requirements.txt -r worker/requirements-dev.txt
python -m pytest worker/tests -q
python -m ruff check worker
cd worker
python -m mypy src
PYTHONPATH=src lint-imports --config pyproject.toml
```

前端静态契约测试：

```bash
python -m pytest frontend/tests -q
```

Compose 配置校验：

```bash
cd infra
docker compose --env-file .env.example -f docker-compose.yml config
```

## 上线检查

- 所有受保护接口必须以 `SessionStore` 为登录态事实源，不能从 `HttpSession` 属性回退放行。
- 所有状态变更请求必须携带 `X-CSRF-Token`，需要幂等的接口必须携带 `Idempotency-Key`。
- Worker 必须调用正式 AI Gateway、Codeforces、MinIO 等端口执行真实业务逻辑，不允许固定模板、伪造结果或旧原型逻辑代替。
- `api_key`、`object_key`、原始 prompt、模型原始响应、完整回答转写不得进入公开响应、SSE、日志或测试快照。
- `AI_WORKER_ENABLED_PROVIDERS` 启用的非 DeepSeek provider 必须配置对应环境凭证；DeepSeek 可由用户任务级配置提供。
- 历史原型代码、本地缓存、数据库文件、构建产物和历史文档不作为运行时依赖发布。
