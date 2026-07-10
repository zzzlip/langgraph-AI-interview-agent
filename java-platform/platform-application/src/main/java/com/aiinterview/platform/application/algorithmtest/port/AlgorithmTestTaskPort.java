package com.aiinterview.platform.application.algorithmtest.port;

import com.aiinterview.platform.application.algorithmtest.dto.AlgorithmTestCreatedResult;
import com.aiinterview.platform.application.algorithmtest.dto.AlgorithmCheckRequestedResult;

/**
 * 算法测试任务端口。
 */
public interface AlgorithmTestTaskPort {

    AlgorithmTestCreatedResult createAlgorithmTestTask(AlgorithmTestTaskCreation creation);

    AlgorithmCheckRequestedResult requestAlgorithmCheck(AlgorithmTestCheckRequest request);
}
