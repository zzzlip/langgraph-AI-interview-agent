package com.aiinterview.platform.interfaceadapter.error;

import com.aiinterview.platform.application.asyncmessage.AsyncMessageException;
import com.aiinterview.platform.application.algorithmtest.AlgorithmTestException;
import com.aiinterview.platform.application.auth.AuthException;
import com.aiinterview.platform.application.file.FileException;
import com.aiinterview.platform.application.mockinterview.MockInterviewException;
import com.aiinterview.platform.application.resumeevaluation.ResumeEvaluationException;
import com.aiinterview.platform.application.resumeoptimization.ResumeOptimizationException;
import com.aiinterview.platform.application.task.TaskException;
import com.aiinterview.platform.domain.common.BaseResult;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.MethodArgumentNotValidException;
import org.springframework.web.bind.annotation.ExceptionHandler;
import org.springframework.web.bind.annotation.RestControllerAdvice;

/**
 * 全局异常处理器。
 *
 * 统一拦截所有 Controller 层异常，确保：
 * - 响应格式统一为 {@link BaseResult}
 * - 不暴露内部堆栈、SQL 错误或密钥信息给客户端
 */
@RestControllerAdvice
public class GlobalExceptionHandler {

    private static final Logger log = LoggerFactory.getLogger(GlobalExceptionHandler.class);

    /** 参数校验异常 -> 400 */
    @ExceptionHandler(ValidationException.class)
    public ResponseEntity<BaseResult<Void>> handleValidation(ValidationException ex) {
        log.warn("参数校验失败: code={}, message={}", ex.getCode(), ex.getMessage());
        return ResponseEntity
                .status(HttpStatus.BAD_REQUEST)
                .body(BaseResult.error(ex.getCode(), ex.getMessage()));
    }

    /** 资源不存在 -> 404 */
    @ExceptionHandler(ResourceNotFoundException.class)
    public ResponseEntity<BaseResult<Void>> handleNotFound(ResourceNotFoundException ex) {
        log.warn("资源不存在: code={}, message={}", ex.getCode(), ex.getMessage());
        return ResponseEntity
                .status(HttpStatus.NOT_FOUND)
                .body(BaseResult.error(ex.getCode(), ex.getMessage()));
    }

    /** 通用业务异常 */
    @ExceptionHandler(BusinessException.class)
    public ResponseEntity<BaseResult<Void>> handleBusiness(BusinessException ex) {
        log.warn("业务异常: code={}, message={}", ex.getCode(), ex.getMessage());
        return ResponseEntity
                .status(HttpStatus.BAD_REQUEST)
                .body(BaseResult.error(ex.getCode(), ex.getMessage()));
    }

    /** 请求参数校验失败 */
    @ExceptionHandler(MethodArgumentNotValidException.class)
    public ResponseEntity<BaseResult<Void>> handleMethodArgumentNotValid(MethodArgumentNotValidException ex) {
        String message = ex.getBindingResult().getFieldErrors().stream()
                .findFirst()
                .map(fieldError -> fieldError.getField() + " " + fieldError.getDefaultMessage())
                .orElse("请求参数校验失败");
        log.warn("请求参数校验失败: {}", message);
        return ResponseEntity
                .status(HttpStatus.BAD_REQUEST)
                .body(BaseResult.error(10001, message));
    }

    /** 认证业务异常 */
    @ExceptionHandler(AuthException.class)
    public ResponseEntity<BaseResult<Void>> handleAuthException(AuthException ex) {
        log.warn("认证异常: code={}, message={}", ex.getCode(), ex.getMessage());
        HttpStatus status = switch (ex.getCode()) {
            case 70200, 70201, 70202 -> HttpStatus.UNAUTHORIZED;
            case 70102 -> HttpStatus.CONFLICT;
            default -> HttpStatus.BAD_REQUEST;
        };
        return ResponseEntity
                .status(status)
                .body(BaseResult.error(ex.getCode(), ex.getMessage()));
    }

    /** 任务中心业务异常 */
    @ExceptionHandler(TaskException.class)
    public ResponseEntity<BaseResult<Void>> handleTaskException(TaskException ex) {
        log.warn("任务中心异常: code={}, message={}", ex.getCode(), ex.getMessage());
        HttpStatus status = switch (ex.getCode()) {
            case 20003 -> HttpStatus.FORBIDDEN;
            case 71003 -> HttpStatus.NOT_FOUND;
            case 71004, 71006, 71007, 71008, 71009, 71015 -> HttpStatus.CONFLICT;
            case 71013, 71014 -> HttpStatus.INTERNAL_SERVER_ERROR;
            default -> HttpStatus.BAD_REQUEST;
        };
        return ResponseEntity
                .status(status)
                .body(BaseResult.error(ex.getCode(), ex.getMessage()));
    }

    /** 文件管理业务异常 */
    @ExceptionHandler(FileException.class)
    public ResponseEntity<BaseResult<Void>> handleFileException(FileException ex) {
        log.warn("文件管理异常: code={}, message={}", ex.getCode(), ex.getMessage());
        HttpStatus status = switch (ex.getCode()) {
            case 72001, 71003 -> HttpStatus.NOT_FOUND;
            case 72007, 72008 -> HttpStatus.INTERNAL_SERVER_ERROR;
            default -> HttpStatus.BAD_REQUEST;
        };
        return ResponseEntity
                .status(status)
                .body(BaseResult.error(ex.getCode(), ex.getMessage()));
    }

    /** 异步消息业务异常 */
    @ExceptionHandler(AsyncMessageException.class)
    public ResponseEntity<BaseResult<Void>> handleAsyncMessageException(AsyncMessageException ex) {
        log.warn("异步消息异常: code={}, message={}", ex.getCode(), ex.getMessage());
        HttpStatus status = switch (ex.getCode()) {
            case 74011 -> HttpStatus.NOT_FOUND;
            case 74007, 74010, 74012 -> HttpStatus.CONFLICT;
            case 74013, 74003 -> HttpStatus.SERVICE_UNAVAILABLE;
            case 74014 -> HttpStatus.TOO_MANY_REQUESTS;
            case 74005 -> HttpStatus.BAD_REQUEST;
            default -> HttpStatus.INTERNAL_SERVER_ERROR;
        };
        return ResponseEntity
                .status(status)
                .body(BaseResult.error(ex.getCode(), ex.getMessage()));
    }

    /** 简历评估业务异常 */
    @ExceptionHandler(ResumeEvaluationException.class)
    public ResponseEntity<BaseResult<Void>> handleResumeEvaluationException(ResumeEvaluationException ex) {
        log.warn("简历评估异常: code={}, message={}", ex.getCode(), ex.getMessage());
        HttpStatus status = switch (ex.getCode()) {
            case 72001 -> HttpStatus.NOT_FOUND;
            case 72006 -> HttpStatus.UNPROCESSABLE_ENTITY;
            case 71008 -> HttpStatus.CONFLICT;
            case 71014 -> HttpStatus.INTERNAL_SERVER_ERROR;
            default -> HttpStatus.BAD_REQUEST;
        };
        return ResponseEntity
                .status(status)
                .body(BaseResult.error(ex.getCode(), ex.getMessage()));
    }

    /** 简历优化业务异常 */
    @ExceptionHandler(ResumeOptimizationException.class)
    public ResponseEntity<BaseResult<Void>> handleResumeOptimizationException(ResumeOptimizationException ex) {
        log.warn("简历优化异常: code={}, message={}", ex.getCode(), ex.getMessage());
        HttpStatus status = switch (ex.getCode()) {
            case 72001 -> HttpStatus.NOT_FOUND;
            case 72006 -> HttpStatus.UNPROCESSABLE_ENTITY;
            case 71008 -> HttpStatus.CONFLICT;
            case 71014 -> HttpStatus.INTERNAL_SERVER_ERROR;
            default -> HttpStatus.BAD_REQUEST;
        };
        return ResponseEntity
                .status(status)
                .body(BaseResult.error(ex.getCode(), ex.getMessage()));
    }

    /** 算法测试业务异常 */
    @ExceptionHandler(AlgorithmTestException.class)
    public ResponseEntity<BaseResult<Void>> handleAlgorithmTestException(AlgorithmTestException ex) {
        log.warn("算法测试异常: code={}, message={}", ex.getCode(), ex.getMessage());
        HttpStatus status = switch (ex.getCode()) {
            case 71003 -> HttpStatus.NOT_FOUND;
            case 71008, 71009, 76006, 76007, 76008 -> HttpStatus.CONFLICT;
            case 71014 -> HttpStatus.INTERNAL_SERVER_ERROR;
            default -> HttpStatus.BAD_REQUEST;
        };
        return ResponseEntity
                .status(status)
                .body(BaseResult.error(ex.getCode(), ex.getMessage()));
    }

    /** 模拟面试业务异常 */
    @ExceptionHandler(MockInterviewException.class)
    public ResponseEntity<BaseResult<Void>> handleMockInterviewException(MockInterviewException ex) {
        log.warn("模拟面试异常: code={}, message={}", ex.getCode(), ex.getMessage());
        HttpStatus status = switch (ex.getCode()) {
            case 71003 -> HttpStatus.NOT_FOUND;
            case 71008, 71009 -> HttpStatus.CONFLICT;
            case 71014 -> HttpStatus.INTERNAL_SERVER_ERROR;
            default -> HttpStatus.BAD_REQUEST;
        };
        return ResponseEntity
                .status(status)
                .body(BaseResult.error(ex.getCode(), ex.getMessage()));
    }

    /** 安全异常 */
    @ExceptionHandler(com.aiinterview.platform.application.security.SecurityException.class)
    public ResponseEntity<BaseResult<Void>> handleSecurityException(
            com.aiinterview.platform.application.security.SecurityException ex
    ) {
        log.warn("安全异常: code={}, message={}", ex.getCode(), ex.getMessage());
        HttpStatus status = switch (ex.getCode()) {
            case 30001 -> HttpStatus.NOT_FOUND;
            case 20001, 70201, 70202 -> HttpStatus.UNAUTHORIZED;
            default -> HttpStatus.FORBIDDEN;
        };
        return ResponseEntity
                .status(status)
                .body(BaseResult.error(ex.getCode(), ex.getMessage()));
    }

    /** 兜底异常 -> 500，不暴露内部细节 */
    @ExceptionHandler(Exception.class)
    public ResponseEntity<BaseResult<Void>> handleUnknown(Exception ex) {
        log.error("未预期异常: ", ex);
        return ResponseEntity
                .status(HttpStatus.INTERNAL_SERVER_ERROR)
                .body(BaseResult.error(60002, "系统繁忙，请稍后重试"));
    }
}
