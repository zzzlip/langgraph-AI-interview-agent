"""简历 Handler 兼容导出。

旧测试和少量兼容入口曾从本模块导入 Handler。真实业务实现已经迁移到
application 层，本模块只保留导出，不再维护独立占位逻辑。
"""

from ai_interview_worker.application.resume_evaluation.handler import ResumeEvaluationHandler
from ai_interview_worker.application.resume_optimization.handler import ResumeOptimizationHandler

__all__ = ["ResumeEvaluationHandler", "ResumeOptimizationHandler"]
