package com.aiinterview.platform.application.mockinterview.port;

import com.aiinterview.platform.application.mockinterview.dto.MockInterviewAnswerFile;
import com.aiinterview.platform.application.mockinterview.dto.MockInterviewResumeFile;

/**
 * 模拟面试文件校验端口。
 */
public interface MockInterviewFileBindingPort {

    MockInterviewResumeFile validateResumeFile(String userId, String resumeFileId);

    MockInterviewAnswerFile validateAnswerFile(String userId, String answerFileId);
}
