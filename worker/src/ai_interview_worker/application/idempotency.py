"""命令幂等与 ACK/NACK 决策类型。"""

from dataclasses import dataclass
from enum import Enum


class IdempotencyDecisionType(str, Enum):
    """幂等检查结果。"""

    PROCESS = "process"
    PROCESS_RECOVERED = "process_recovered"
    ACK_DUPLICATE = "ack_duplicate"
    ACK_CONFLICT = "ack_conflict"
    NACK_REQUEUE = "nack_requeue"


@dataclass(frozen=True)
class IdempotencyDecision:
    """命令消费后的 ACK/NACK 决策。"""

    decision_type: IdempotencyDecisionType
    reason: str

    @property
    def should_ack(self) -> bool:
        """是否可以 ACK 原命令。"""
        return self.decision_type in {
            IdempotencyDecisionType.ACK_DUPLICATE,
            IdempotencyDecisionType.ACK_CONFLICT,
        }

    @property
    def should_nack_requeue(self) -> bool:
        """是否应 NACK 并 requeue。"""
        return self.decision_type is IdempotencyDecisionType.NACK_REQUEUE

    @property
    def should_process(self) -> bool:
        """是否应进入业务处理。"""
        return self.decision_type in {
            IdempotencyDecisionType.PROCESS,
            IdempotencyDecisionType.PROCESS_RECOVERED,
        }
