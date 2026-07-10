package com.aiinterview.platform.infrastructure.persistence.mockinterview;

import com.aiinterview.platform.application.mockinterview.dto.MockInterviewAnswerFile;
import com.aiinterview.platform.application.mockinterview.dto.MockInterviewCurrentState;
import com.aiinterview.platform.application.mockinterview.dto.MockInterviewStateSnapshot;
import com.aiinterview.platform.application.mockinterview.port.MockInterviewResultPort;
import com.aiinterview.platform.infrastructure.persistence.mockinterview.entity.MockInterviewStateEntity;
import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import com.baomidou.mybatisplus.core.conditions.update.LambdaUpdateWrapper;
import com.fasterxml.jackson.core.type.TypeReference;
import com.fasterxml.jackson.databind.ObjectMapper;
import java.time.Instant;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.function.Supplier;
import org.springframework.stereotype.Component;

/**
 * 模拟面试中间状态持久化适配器。
 */
@Component
public class MockInterviewResultPortImpl implements MockInterviewResultPort {

    private static final TypeReference<Map<String, Object>> PAYLOAD_TYPE = new TypeReference<>() {
    };

    private final MockInterviewStateMapper mapper;
    private final ObjectMapper objectMapper;
    private final Supplier<String> idSupplier;

    public MockInterviewResultPortImpl(
            MockInterviewStateMapper mapper,
            ObjectMapper objectMapper,
            Supplier<String> idSupplier
    ) {
        this.mapper = mapper;
        this.objectMapper = objectMapper;
        this.idSupplier = idSupplier;
    }

    @Override
    public void saveInitialPlan(MockInterviewStateSnapshot snapshot) {
        MockInterviewStateEntity entity = new MockInterviewStateEntity();
        entity.setId(idSupplier.get());
        entity.setTaskId(snapshot.taskId());
        entity.setUserId(snapshot.userId());
        entity.setAttemptNo(snapshot.attemptNo());
        entity.setCurrentQuestionId(snapshot.currentQuestionId());
        entity.setCurrentRoundId(snapshot.currentRoundId());
        entity.setWaitingInputType(snapshot.waitingInputType());
        entity.setInputExpired(false);
        entity.setWaitingSince(snapshot.waitingSince());
        entity.setExpiresAt(snapshot.expiresAt());
        entity.setStatePayload(writePayload(snapshot));
        entity.setVersion(0L);
        entity.setCreatedAt(snapshot.waitingSince());
        entity.setUpdatedAt(snapshot.waitingSince());
        mapper.insert(entity);
    }

    @Override
    public Optional<MockInterviewCurrentState> findCurrentState(String taskId, int attemptNo) {
        return mapper.selectList(new LambdaQueryWrapper<MockInterviewStateEntity>()
                        .eq(MockInterviewStateEntity::getTaskId, taskId)
                        .eq(MockInterviewStateEntity::getAttemptNo, attemptNo))
                .stream()
                .findFirst()
                .map(this::toCurrentState);
    }

    @Override
    public boolean markAnswerSubmitted(MockInterviewCurrentState state, MockInterviewAnswerFile answerFile) {
        Map<String, Object> payload = new LinkedHashMap<>(state.statePayload());
        List<Object> rounds = new ArrayList<>();
        Object rawRounds = payload.get("rounds");
        if (rawRounds instanceof List<?> list) {
            for (Object item : list) {
                if (item instanceof Map<?, ?> map) {
                    Map<String, Object> round = new LinkedHashMap<>();
                    map.forEach((key, value) -> round.put(String.valueOf(key), value));
                    if (state.currentRoundId().equals(round.get("round_id"))) {
                        round.put("status", "submitted");
                        round.put("answer_file", answerFilePayload(answerFile));
                    }
                    rounds.add(round);
                }
            }
        }
        payload.put("rounds", rounds);
        payload.put("waiting_input_type", "processing_answer");
        Instant now = Instant.now();
        int updated = mapper.update(null, new LambdaUpdateWrapper<MockInterviewStateEntity>()
                .eq(MockInterviewStateEntity::getTaskId, state.taskId())
                .eq(MockInterviewStateEntity::getAttemptNo, state.attemptNo())
                .eq(MockInterviewStateEntity::getVersion, state.version())
                .set(MockInterviewStateEntity::getWaitingInputType, "processing_answer")
                .set(MockInterviewStateEntity::getStatePayload, writePayload(payload))
                .set(MockInterviewStateEntity::getVersion, state.version() + 1)
                .set(MockInterviewStateEntity::getUpdatedAt, now));
        return updated == 1;
    }

    @Override
    public boolean updateCurrentState(
            MockInterviewCurrentState state,
            String currentQuestionId,
            String currentRoundId,
            String waitingInputType,
            Map<String, Object> statePayload
    ) {
        Instant now = Instant.now();
        int updated = mapper.update(null, new LambdaUpdateWrapper<MockInterviewStateEntity>()
                .eq(MockInterviewStateEntity::getTaskId, state.taskId())
                .eq(MockInterviewStateEntity::getAttemptNo, state.attemptNo())
                .eq(MockInterviewStateEntity::getVersion, state.version())
                .set(MockInterviewStateEntity::getCurrentQuestionId, currentQuestionId)
                .set(MockInterviewStateEntity::getCurrentRoundId, currentRoundId)
                .set(MockInterviewStateEntity::getWaitingInputType, waitingInputType)
                .set(MockInterviewStateEntity::getInputExpired, false)
                .set(MockInterviewStateEntity::getWaitingSince, now)
                .set(MockInterviewStateEntity::getExpiresAt, now.plusSeconds(72 * 3600L))
                .set(MockInterviewStateEntity::getStatePayload, writePayload(statePayload))
                .set(MockInterviewStateEntity::getVersion, state.version() + 1)
                .set(MockInterviewStateEntity::getUpdatedAt, now));
        return updated == 1;
    }

    private String writePayload(MockInterviewStateSnapshot snapshot) {
        try {
            return objectMapper.writeValueAsString(snapshot.statePayload());
        } catch (Exception ex) {
            throw new IllegalStateException("模拟面试中间状态序列化失败", ex);
        }
    }

    private String writePayload(Map<String, Object> payload) {
        try {
            return objectMapper.writeValueAsString(payload);
        } catch (Exception ex) {
            throw new IllegalStateException("模拟面试中间状态序列化失败", ex);
        }
    }

    private MockInterviewCurrentState toCurrentState(MockInterviewStateEntity entity) {
        return new MockInterviewCurrentState(
                entity.getTaskId(),
                entity.getAttemptNo(),
                entity.getCurrentQuestionId(),
                entity.getCurrentRoundId(),
                entity.getWaitingInputType(),
                entity.getVersion(),
                parsePayload(entity.getStatePayload())
        );
    }

    private Map<String, Object> parsePayload(String payload) {
        try {
            return objectMapper.readValue(payload, PAYLOAD_TYPE);
        } catch (Exception ex) {
            throw new IllegalStateException("模拟面试中间状态反序列化失败", ex);
        }
    }

    private static Map<String, Object> answerFilePayload(MockInterviewAnswerFile answerFile) {
        Map<String, Object> payload = new LinkedHashMap<>();
        payload.put("file_id", answerFile.fileId());
        payload.put("object_key", answerFile.objectKey());
        payload.put("file_type", answerFile.fileType());
        payload.put("file_purpose", answerFile.filePurpose());
        payload.put("media_type", answerFile.mediaType());
        payload.put("mime_type", answerFile.mimeType());
        return payload;
    }
}
