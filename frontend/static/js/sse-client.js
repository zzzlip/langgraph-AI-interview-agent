(() => {
    const POLLING_INTERVAL_MS = 5000;
    const allowedEvents = new Set(["task_snapshot", "task_progress", "task_completed", "task_failed", "heartbeat"]);

    class TaskSseClient {
        constructor({ taskId, onEvent, onPoll, onRecovering }) {
            this.taskId = taskId;
            this.onEvent = onEvent;
            this.onPoll = onPoll;
            this.onRecovering = onRecovering;
            this.eventSource = null;
            this.pollTimer = null;
        }

        connect() {
            if (typeof window.EventSource === "undefined") {
                this.startPolling();
                return;
            }
            this.eventSource = new window.EventSource(`/api/v1/tasks/${encodeURIComponent(this.taskId)}/events`, {
                withCredentials: true,
            });
            allowedEvents.forEach(eventName => {
                this.eventSource.addEventListener(eventName, event => this.handleEvent(eventName, event));
            });
            this.eventSource.onerror = () => {
                this.closeEventSource();
                this.startPolling();
            };
        }

        handleEvent(eventName, event) {
            if (!allowedEvents.has(eventName)) {
                return;
            }
            if (eventName === "heartbeat") {
                this.onEvent(eventName, { received_at: new Date().toISOString() });
                return;
            }
            const data = parseEventData(event);
            this.onEvent(eventName, data);
            if (eventName === "task_completed" || eventName === "task_failed") {
                this.stopPolling();
                this.close();
            }
        }

        startPolling() {
            this.stopPolling();
            if (this.onRecovering) {
                this.onRecovering();
            }
            this.pollTimer = window.setInterval(() => {
                this.onPoll();
            }, POLLING_INTERVAL_MS);
            this.onPoll();
        }

        stopPolling() {
            if (this.pollTimer) {
                window.clearInterval(this.pollTimer);
                this.pollTimer = null;
            }
        }

        closeEventSource() {
            const eventSource = this.eventSource;
            if (eventSource) {
                eventSource.close();
                this.eventSource = null;
            }
        }

        close() {
            this.stopPolling();
            this.closeEventSource();
        }
    }

    function parseEventData(event) {
        try {
            return event.data ? JSON.parse(event.data) : {};
        } catch {
            return {};
        }
    }

    window.AiInterviewSse = Object.freeze({
        POLLING_INTERVAL_MS,
        TaskSseClient,
    });
})();
