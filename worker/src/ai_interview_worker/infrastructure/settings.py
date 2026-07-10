"""Worker 集中配置与启动门禁。"""

import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from ai_interview_worker.application.resume_evaluation.scoring_config import parse_weights

DEFAULT_ENABLED_PROVIDERS = ("deepseek", "dashscope")
ALLOWED_PROVIDERS = {"deepseek", "dashscope", "search", "asr", "multimodal"}
PROVIDER_API_KEY_ENV = {
    "deepseek": "AI_WORKER_DEEPSEEK_API_KEY",
    "dashscope": "AI_WORKER_DASHSCOPE_API_KEY",
    "search": "AI_WORKER_SERPER_API_KEY",
    "asr": "AI_WORKER_DASHSCOPE_API_KEY",
    "multimodal": "AI_WORKER_DASHSCOPE_API_KEY",
}


@dataclass(frozen=True)
class Settings:
    """Worker 全局配置，由环境变量驱动。"""

    env: str
    worker_id: str
    log_level: str
    rabbitmq_url: str
    rabbitmq_username: str
    rabbitmq_password: str
    minio_endpoint: str
    minio_access_key: str
    minio_secret_key: str
    minio_bucket: str
    sqlite_path: str
    enabled_providers: tuple[str, ...]
    processing_timeout_seconds_raw: str
    resume_evaluation_weights_raw: str
    resume_optimization_context_max_chars_raw: str
    resume_optimization_min_length_ratio_raw: str
    resume_optimization_avatar_max_px_raw: str
    codeforces_timeout_seconds_raw: str
    codeforces_max_submissions_raw: str
    algorithm_test_problem_count_raw: str
    algorithm_test_pass_threshold_raw: str
    deepseek_api_key: str
    dashscope_api_key: str
    serper_api_key: str

    @classmethod
    def from_env(cls, environ: Mapping[str, str] | None = None) -> "Settings":
        """从环境变量加载配置。"""
        source = os.environ if environ is None else environ
        enabled_providers = _parse_enabled_providers(source.get("AI_WORKER_ENABLED_PROVIDERS", ""))
        return cls(
            env=source.get("AI_WORKER_ENV", "development"),
            worker_id=source.get("AI_WORKER_ID", "python-worker-001"),
            log_level=source.get("AI_WORKER_LOG_LEVEL", "INFO"),
            rabbitmq_url=source.get("AI_WORKER_RABBITMQ_URL", ""),
            rabbitmq_username=source.get("AI_WORKER_RABBITMQ_USERNAME", ""),
            rabbitmq_password=source.get("AI_WORKER_RABBITMQ_PASSWORD", ""),
            minio_endpoint=source.get("AI_WORKER_MINIO_ENDPOINT", ""),
            minio_access_key=source.get("AI_WORKER_MINIO_ACCESS_KEY", ""),
            minio_secret_key=source.get("AI_WORKER_MINIO_SECRET_KEY", ""),
            minio_bucket=source.get("AI_WORKER_MINIO_BUCKET", ""),
            sqlite_path=source.get("AI_WORKER_SQLITE_PATH", "worker/.data/processed_commands.sqlite"),
            enabled_providers=enabled_providers,
            processing_timeout_seconds_raw=source.get("AI_WORKER_PROCESSING_TIMEOUT_SECONDS", "1800"),
            resume_evaluation_weights_raw=source.get("AI_WORKER_RESUME_EVALUATION_WEIGHTS", ""),
            resume_optimization_context_max_chars_raw=source.get(
                "AI_WORKER_RESUME_OPTIMIZATION_CONTEXT_MAX_CHARS",
                "12000",
            ),
            resume_optimization_min_length_ratio_raw=source.get(
                "AI_WORKER_RESUME_OPTIMIZATION_MIN_LENGTH_RATIO",
                "0.6",
            ),
            resume_optimization_avatar_max_px_raw=source.get("AI_WORKER_RESUME_OPTIMIZATION_AVATAR_MAX_PX", "150"),
            codeforces_timeout_seconds_raw=source.get("AI_WORKER_CODEFORCES_TIMEOUT_SECONDS", "10"),
            codeforces_max_submissions_raw=source.get("AI_WORKER_CODEFORCES_MAX_SUBMISSIONS", "200"),
            algorithm_test_problem_count_raw=source.get("AI_WORKER_ALGORITHM_TEST_PROBLEM_COUNT", "5"),
            algorithm_test_pass_threshold_raw=source.get("AI_WORKER_ALGORITHM_TEST_PASS_THRESHOLD", "2"),
            deepseek_api_key=source.get("AI_WORKER_DEEPSEEK_API_KEY", ""),
            dashscope_api_key=source.get("AI_WORKER_DASHSCOPE_API_KEY", ""),
            serper_api_key=source.get("AI_WORKER_SERPER_API_KEY", ""),
        )

    @property
    def processing_timeout_seconds(self) -> int:
        """返回合法的 processing 恢复超时时间。"""
        return int(self.processing_timeout_seconds_raw)

    @property
    def resume_optimization_context_max_chars(self) -> int:
        """返回简历优化上下文最大字符数。"""
        return int(self.resume_optimization_context_max_chars_raw)

    @property
    def resume_optimization_min_length_ratio(self) -> float:
        """返回简历优化输出最小长度比例。"""
        return float(self.resume_optimization_min_length_ratio_raw)

    @property
    def resume_optimization_avatar_max_px(self) -> int:
        """返回简历优化头像最大边长。"""
        return int(self.resume_optimization_avatar_max_px_raw)

    @property
    def codeforces_timeout_seconds(self) -> int:
        """返回 Codeforces 请求超时时间。"""
        return int(self.codeforces_timeout_seconds_raw)

    @property
    def codeforces_max_submissions(self) -> int:
        """返回提交检查最多拉取条数。"""
        return int(self.codeforces_max_submissions_raw)

    @property
    def algorithm_test_problem_count(self) -> int:
        """返回算法测试题目数量。"""
        return int(self.algorithm_test_problem_count_raw)

    @property
    def algorithm_test_pass_threshold(self) -> int:
        """返回算法测试通过阈值。"""
        return int(self.algorithm_test_pass_threshold_raw)

    def validate(self) -> list[str]:
        """返回不满足启动门禁的配置项名称，禁止输出配置值。"""
        invalid: list[str] = []
        required_values = {
            "AI_WORKER_RABBITMQ_URL": self.rabbitmq_url,
            "AI_WORKER_RABBITMQ_USERNAME": self.rabbitmq_username,
            "AI_WORKER_RABBITMQ_PASSWORD": self.rabbitmq_password,
            "AI_WORKER_MINIO_ENDPOINT": self.minio_endpoint,
            "AI_WORKER_MINIO_ACCESS_KEY": self.minio_access_key,
            "AI_WORKER_MINIO_SECRET_KEY": self.minio_secret_key,
            "AI_WORKER_MINIO_BUCKET": self.minio_bucket,
        }
        invalid.extend(key for key, value in required_values.items() if not value.strip())

        invalid.extend(self._validate_enabled_providers())
        invalid.extend(self._validate_processing_timeout())
        invalid.extend(self._validate_resume_evaluation_weights())
        invalid.extend(self._validate_resume_optimization_config())
        invalid.extend(self._validate_algorithm_test_config())
        invalid.extend(self._validate_sqlite_path())
        return _deduplicate(invalid)

    def _validate_enabled_providers(self) -> list[str]:
        invalid = []
        unknown_providers = [provider for provider in self.enabled_providers if provider not in ALLOWED_PROVIDERS]
        if unknown_providers:
            invalid.append("AI_WORKER_ENABLED_PROVIDERS")

        # DeepSeek 可由任务级 llm_config 注入；其他 provider 必须靠环境变量提供真实凭证。
        for provider in self.enabled_providers:
            if provider == "deepseek" or provider not in PROVIDER_API_KEY_ENV:
                continue
            env_name = PROVIDER_API_KEY_ENV[provider]
            if not self._api_key_for_env(env_name).strip():
                invalid.append(env_name)
        return invalid

    def _api_key_for_env(self, env_name: str) -> str:
        return {
            "AI_WORKER_DEEPSEEK_API_KEY": self.deepseek_api_key,
            "AI_WORKER_DASHSCOPE_API_KEY": self.dashscope_api_key,
            "AI_WORKER_SERPER_API_KEY": self.serper_api_key,
        }.get(env_name, "")

    def _validate_algorithm_test_config(self) -> list[str]:
        invalid: list[str] = []
        problem_count: int | None = None
        pass_threshold: int | None = None

        try:
            timeout = int(self.codeforces_timeout_seconds_raw)
            if timeout <= 0:
                invalid.append("AI_WORKER_CODEFORCES_TIMEOUT_SECONDS")
        except ValueError:
            invalid.append("AI_WORKER_CODEFORCES_TIMEOUT_SECONDS")

        try:
            max_submissions = int(self.codeforces_max_submissions_raw)
            if max_submissions <= 0:
                invalid.append("AI_WORKER_CODEFORCES_MAX_SUBMISSIONS")
        except ValueError:
            invalid.append("AI_WORKER_CODEFORCES_MAX_SUBMISSIONS")

        try:
            problem_count = int(self.algorithm_test_problem_count_raw)
            if problem_count != 5:
                invalid.append("AI_WORKER_ALGORITHM_TEST_PROBLEM_COUNT")
        except ValueError:
            invalid.append("AI_WORKER_ALGORITHM_TEST_PROBLEM_COUNT")

        try:
            pass_threshold = int(self.algorithm_test_pass_threshold_raw)
            if pass_threshold <= 0:
                invalid.append("AI_WORKER_ALGORITHM_TEST_PASS_THRESHOLD")
        except ValueError:
            invalid.append("AI_WORKER_ALGORITHM_TEST_PASS_THRESHOLD")

        if problem_count is not None and pass_threshold is not None and pass_threshold > problem_count:
            invalid.append("AI_WORKER_ALGORITHM_TEST_PASS_THRESHOLD")

        return invalid

    def _validate_processing_timeout(self) -> list[str]:
        try:
            timeout = int(self.processing_timeout_seconds_raw)
        except ValueError:
            return ["AI_WORKER_PROCESSING_TIMEOUT_SECONDS"]
        if timeout < 60:
            return ["AI_WORKER_PROCESSING_TIMEOUT_SECONDS"]
        return []

    def _validate_resume_evaluation_weights(self) -> list[str]:
        try:
            parse_weights(self.resume_evaluation_weights_raw)
        except ValueError:
            return ["AI_WORKER_RESUME_EVALUATION_WEIGHTS"]
        return []

    def _validate_resume_optimization_config(self) -> list[str]:
        invalid: list[str] = []
        try:
            context_max_chars = int(self.resume_optimization_context_max_chars_raw)
            if context_max_chars < 1000:
                invalid.append("AI_WORKER_RESUME_OPTIMIZATION_CONTEXT_MAX_CHARS")
        except ValueError:
            invalid.append("AI_WORKER_RESUME_OPTIMIZATION_CONTEXT_MAX_CHARS")

        try:
            min_length_ratio = float(self.resume_optimization_min_length_ratio_raw)
            if min_length_ratio <= 0 or min_length_ratio > 1:
                invalid.append("AI_WORKER_RESUME_OPTIMIZATION_MIN_LENGTH_RATIO")
        except ValueError:
            invalid.append("AI_WORKER_RESUME_OPTIMIZATION_MIN_LENGTH_RATIO")

        try:
            avatar_max_px = int(self.resume_optimization_avatar_max_px_raw)
            if avatar_max_px <= 0:
                invalid.append("AI_WORKER_RESUME_OPTIMIZATION_AVATAR_MAX_PX")
        except ValueError:
            invalid.append("AI_WORKER_RESUME_OPTIMIZATION_AVATAR_MAX_PX")

        return invalid

    def _validate_sqlite_path(self) -> list[str]:
        path = Path(self.sqlite_path)
        parent = path.parent
        if not parent.exists() or not os.access(parent, os.W_OK):
            return ["AI_WORKER_SQLITE_PATH"]
        return []


def _parse_enabled_providers(raw_value: str) -> tuple[str, ...]:
    if not raw_value.strip():
        return DEFAULT_ENABLED_PROVIDERS
    providers = tuple(provider.strip() for provider in raw_value.split(",") if provider.strip())
    return providers or DEFAULT_ENABLED_PROVIDERS


def _deduplicate(values: list[str]) -> list[str]:
    seen = set()
    result = []
    for value in values:
        if value not in seen:
            seen.add(value)
            result.append(value)
    return result
