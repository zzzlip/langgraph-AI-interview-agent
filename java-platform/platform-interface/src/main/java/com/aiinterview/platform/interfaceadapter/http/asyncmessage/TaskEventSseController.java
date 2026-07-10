package com.aiinterview.platform.interfaceadapter.http.asyncmessage;

import static com.aiinterview.platform.interfaceadapter.http.support.HttpSessionSupport.requireUserSession;

import com.aiinterview.platform.application.asyncmessage.SseTaskEventService;
import com.aiinterview.platform.application.asyncmessage.dto.SseTaskEventResult;
import com.aiinterview.platform.application.asyncmessage.port.SseConnectionRegistry;
import com.aiinterview.platform.application.session.CurrentSession;
import com.aiinterview.platform.application.session.SessionManager;
import com.fasterxml.jackson.databind.ObjectMapper;
import jakarta.servlet.http.HttpServletRequest;
import java.io.IOException;
import org.springframework.http.CacheControl;
import org.springframework.http.MediaType;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PathVariable;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;
import org.springframework.web.servlet.mvc.method.annotation.SseEmitter;

/**
 * 普通用户任务 SSE 订阅接口。
 */
@RestController
@RequestMapping("/api/v1/tasks")
public class TaskEventSseController {

    private final SseTaskEventService sseTaskEventService;
    private final SessionManager sessionManager;
    private final ObjectMapper objectMapper;

    public TaskEventSseController(SseTaskEventService sseTaskEventService, SessionManager sessionManager,
            ObjectMapper objectMapper) {
        this.sseTaskEventService = sseTaskEventService;
        this.sessionManager = sessionManager;
        this.objectMapper = objectMapper;
    }

    @GetMapping(path = "/{taskId}/events", produces = MediaType.TEXT_EVENT_STREAM_VALUE)
    public ResponseEntity<SseEmitter> subscribe(@PathVariable String taskId, HttpServletRequest request) {
        CurrentSession currentSession = requireUserSession(request, sessionManager);
        SseEmitter emitter = new SseEmitter(0L);
        EmitterSink sink = new EmitterSink(emitter, objectMapper);
        sseTaskEventService.subscribe(currentSession.getAccountId(), taskId, sink);
        Runnable cleanup = () -> sseTaskEventService.disconnect(currentSession.getAccountId(), taskId, sink);
        emitter.onCompletion(cleanup);
        emitter.onTimeout(cleanup);
        emitter.onError(ignored -> cleanup.run());
        return ResponseEntity.ok()
                .cacheControl(CacheControl.noStore().mustRevalidate())
                .header("Pragma", "no-cache")
                .header("X-Accel-Buffering", "no")
                .body(emitter);
    }

    private static final class EmitterSink implements SseConnectionRegistry.TaskEventSink {
        private final SseEmitter emitter;
        private final ObjectMapper objectMapper;

        private EmitterSink(SseEmitter emitter, ObjectMapper objectMapper) {
            this.emitter = emitter;
            this.objectMapper = objectMapper;
        }

        @Override
        public void send(SseTaskEventResult event) {
            try {
                emitter.send(SseEmitter.event()
                        .name(event.eventName())
                        .id(event.eventId())
                        .data(objectMapper.writeValueAsString(event)));
            } catch (IOException ex) {
                throw new IllegalStateException("SSE 推送失败", ex);
            }
        }
    }
}
