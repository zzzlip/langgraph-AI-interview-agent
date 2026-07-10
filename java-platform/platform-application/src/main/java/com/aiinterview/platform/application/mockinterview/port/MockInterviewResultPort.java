package com.aiinterview.platform.application.mockinterview.port;

import com.aiinterview.platform.application.mockinterview.dto.MockInterviewStateSnapshot;
import com.aiinterview.platform.application.mockinterview.dto.MockInterviewCurrentState;
import com.aiinterview.platform.application.mockinterview.dto.MockInterviewAnswerFile;
import java.util.Map;
import java.util.Optional;

/**
 * 模拟面试中间状态保存端口。
 */
public interface MockInterviewResultPort {

    void saveInitialPlan(MockInterviewStateSnapshot snapshot);

    Optional<MockInterviewCurrentState> findCurrentState(String taskId, int attemptNo);

    boolean markAnswerSubmitted(MockInterviewCurrentState state, MockInterviewAnswerFile answerFile);

    boolean updateCurrentState(
            MockInterviewCurrentState state,
            String currentQuestionId,
            String currentRoundId,
            String waitingInputType,
            Map<String, Object> statePayload
    );
}
