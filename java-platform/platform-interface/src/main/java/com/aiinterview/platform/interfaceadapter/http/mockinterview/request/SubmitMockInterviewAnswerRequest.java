package com.aiinterview.platform.interfaceadapter.http.mockinterview.request;

import com.fasterxml.jackson.annotation.JsonProperty;

/**
 * 提交模拟面试回答请求。
 */
public class SubmitMockInterviewAnswerRequest {

    @JsonProperty("question_id")
    private String questionId;

    @JsonProperty("round_id")
    private String roundId;

    @JsonProperty("answer_file_id")
    private String answerFileId;

    public String getQuestionId() {
        return questionId;
    }

    public void setQuestionId(String questionId) {
        this.questionId = questionId;
    }

    public String getRoundId() {
        return roundId;
    }

    public void setRoundId(String roundId) {
        this.roundId = roundId;
    }

    public String getAnswerFileId() {
        return answerFileId;
    }

    public void setAnswerFileId(String answerFileId) {
        this.answerFileId = answerFileId;
    }
}
