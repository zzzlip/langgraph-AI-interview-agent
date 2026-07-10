package com.aiinterview.platform.application.mockinterview.port;

import com.aiinterview.platform.application.mockinterview.dto.MockInterviewCreatedResult;

/**
 * 模拟面试任务端口。
 */
public interface MockInterviewTaskPort {

    MockInterviewCreatedResult createMockInterviewTask(MockInterviewTaskCreation creation);
}
