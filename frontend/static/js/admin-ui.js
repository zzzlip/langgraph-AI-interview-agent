(() => {
    // 管理员视图：ADMIN 角色可见完整详情，USER 角色被拒绝
    // 所有状态变更由 api.requestJson 统一注入 X-CSRF-Token 和 Idempotency-Key
    const api = window.AiInterviewApi;
    const auth = window.AiInterviewAuth;

    // ============================================================
    // 失败任务列表
    // ============================================================

    async function renderFailedTaskList(container) {
        if (!isAdmin()) {
            renderForbidden(container);
            return;
        }

        const panel = document.createElement("section");
        panel.className = "panel";
        panel.append(createEyebrow("管理员"));

        const h1 = document.createElement("h1");
        setText(h1, "失败排查");

        const form = createFilterForm();
        const list = document.createElement("div");
        list.className = "task-list";
        list.dataset.failedTaskList = "true";

        panel.append(h1, form, list);
        container.replaceChildren(panel);

        form.addEventListener("submit", event => {
            event.preventDefault();
            loadFailedTasks(container, {
                taskType: form.elements.task_type.value,
                retryable: form.elements.retryable.value,
            });
        });

        await loadFailedTasks(container, {});
    }

    function createFilterForm() {
        const form = document.createElement("form");
        form.className = "filter-form";
        form.append(createSelect("task_type", "任务类型", [
            ["", "全部类型"],
            ["resume_evaluation", "简历评估"],
            ["resume_optimization", "简历优化"],
            ["algorithm_test", "算法测试"],
            ["mock_interview", "模拟面试"],
        ]));
        form.append(createSelect("retryable", "可重试", [
            ["", "全部"],
            ["true", "可重试"],
            ["false", "不可重试"],
        ]));
        const button = document.createElement("button");
        button.type = "submit";
        setText(button, "筛选");
        form.append(button);
        return form;
    }

    async function loadFailedTasks(container, filters) {
        const list = container.querySelector("[data-failed-task-list]");
        setText(list, "加载中");
        const params = new URLSearchParams({ page: "1", page_size: "20" });
        if (filters.taskType) params.set("task_type", filters.taskType);
        if (filters.retryable) params.set("retryable", filters.retryable);

        try {
            const data = await api.requestJson(`/api/v1/admin/tasks/failed?${params.toString()}`);
            const items = data.items || [];
            if (items.length === 0) {
                setText(list, "暂无失败任务");
                return;
            }
            renderFailedTaskItems(list, items);
        } catch (error) {
            setText(list, api.sanitizeErrorMessage(error));
        }
    }

    function renderFailedTaskItems(list, items) {
        list.replaceChildren();
        items.forEach(task => {
            const row = document.createElement("a");
            row.className = "task-row";
            row.href = `#/admin/failures/${encodeURIComponent(task.task_id)}`;
            const type = mapTaskType(task.task_type);
            const retryable = task.retryable ? "可重试" : "不可重试";
            setText(row, `${type} · ${task.task_id} · ${retryable}`);
            list.append(row);
        });
    }

    // ============================================================
    // 失败详情
    // ============================================================

    async function renderFailureDetail(container, taskId) {
        if (!isAdmin()) {
            renderForbidden(container);
            return;
        }

        const panel = document.createElement("section");
        panel.className = "panel";
        panel.append(createEyebrow("管理员"));
        const h1 = document.createElement("h1");
        setText(h1, "失败详情");
        const loading = document.createElement("p");
        setText(loading, "加载中");
        panel.append(h1, loading);
        container.replaceChildren(panel);

        try {
            const data = await api.requestJson(`/api/v1/admin/tasks/${encodeURIComponent(taskId)}/failure`);
            renderFailureContent(panel, data);
        } catch (error) {
            setText(loading, api.sanitizeErrorMessage(error));
        }
    }

    function renderFailureContent(panel, data) {
        const loading = panel.querySelector("p");
        if (loading) loading.remove();

        const dl = document.createElement("dl");
        dl.className = "detail-grid";
        appendDetail(dl, "任务 ID", data.task_id);
        appendDetail(dl, "任务类型", mapTaskType(data.task_type));
        appendDetail(dl, "失败阶段", data.failure_stage || data.stage || "未知");
        appendDetail(dl, "失败码", String(data.failure_code || data.error_code || ""));
        appendDetail(dl, "失败原因", data.failure_reason || data.error_message || "未知");
        appendDetail(dl, "当前尝试", String(data.current_attempt || data.attempt || ""));
        appendDetail(dl, "重试次数", String(data.retry_count || data.attempt_count || "0"));
        panel.append(dl);

        // 事件历史
        const events = data.events || [];
        if (events.length > 0) {
            const eventSection = document.createElement("div");
            const h3 = document.createElement("h3");
            setText(h3, "事件历史");
            const timeline = document.createElement("ol");
            timeline.className = "timeline";
            events.forEach(ev => {
                const item = document.createElement("li");
                setText(item, `${ev.event_type || ev.type} — ${ev.message || ""}`);
                timeline.append(item);
            });
            eventSection.append(h3, timeline);
            panel.append(eventSection);
        }

        // 重试记录
        const attempts = data.attempts || [];
        if (attempts.length > 0) {
            const attemptSection = document.createElement("div");
            const h3 = document.createElement("h3");
            setText(h3, "重试记录");
            const attemptList = document.createElement("ul");
            attempts.forEach(att => {
                const item = document.createElement("li");
                setText(item, `${att.attempt || att.number} — ${att.status || ""} — ${att.created_at || ""}`);
                attemptList.append(item);
            });
            attemptSection.append(h3, attemptList);
            panel.append(attemptSection);
        }

        // 异步追踪
        const traceBtn = document.createElement("button");
        traceBtn.type = "button";
        traceBtn.className = "secondary-action";
        setText(traceBtn, "查看异步追踪");
        traceBtn.addEventListener("click", () => renderAsyncTrace(panel, data.task_id));

        // 死信入口
        const deadLetterBtn = document.createElement("button");
        deadLetterBtn.type = "button";
        deadLetterBtn.className = "secondary-action";
        setText(deadLetterBtn, "查看死信");
        deadLetterBtn.addEventListener("click", () => renderDeadLetters(panel));

        const actionRow = document.createElement("div");
        actionRow.className = "action-row";
        actionRow.append(traceBtn, deadLetterBtn);

        // 可重试任务显示重试按钮（需二次确认）
        if (data.retryable !== false) {
            const retryBtn = document.createElement("button");
            retryBtn.type = "button";
            retryBtn.className = "primary-action";
            setText(retryBtn, "重试任务");
            retryBtn.addEventListener("click", () => confirmRetry(panel, data.task_id, retryBtn));
            actionRow.append(retryBtn);
        }

        panel.append(actionRow);
    }

    function confirmRetry(panel, taskId, retryBtn) {
        const confirmed = window.confirm("确认重试该任务？重试将创建新的执行尝试，保留历史失败记录。");
        if (!confirmed) return;

        retryBtn.disabled = true;
        setText(retryBtn, "重试中...");
        api.requestJson(`/api/v1/admin/tasks/${encodeURIComponent(taskId)}/retry`, {
            method: "POST",
            idempotent: true,
            operationId: "admin_retry:" + taskId,
        }).then(() => {
            const success = document.createElement("p");
            success.className = "form-feedback";
            setText(success, "重试请求已提交");
            panel.append(success);
        }).catch(error => {
            retryBtn.disabled = false;
            setText(retryBtn, "重试任务");
            const err = document.createElement("p");
            err.className = "form-feedback";
            setText(err, api.sanitizeErrorMessage(error));
            panel.append(err);
        });
    }

    async function renderAsyncTrace(panel, taskId) {
        let tracePanel = panel.querySelector("[data-async-trace]");
        if (!tracePanel) {
            tracePanel = document.createElement("div");
            tracePanel.dataset.asyncTrace = "true";
            panel.append(tracePanel);
        }
        setText(tracePanel, "加载中...");
        try {
            const data = await api.requestJson(`/api/v1/admin/tasks/${encodeURIComponent(taskId)}/async-trace`);
            tracePanel.replaceChildren();
            const h3 = document.createElement("h3");
            setText(h3, "异步追踪");
            const dl = document.createElement("dl");
            dl.className = "detail-grid";
            appendDetail(dl, "Outbox", JSON.stringify(data.outbox || []));
            appendDetail(dl, "Inbox", JSON.stringify(data.inbox || []));
            tracePanel.append(h3, dl);
        } catch (error) {
            setText(tracePanel, api.sanitizeErrorMessage(error));
        }
    }

    async function renderDeadLetters(panel) {
        let deadPanel = panel.querySelector("[data-dead-letters]");
        if (!deadPanel) {
            deadPanel = document.createElement("div");
            deadPanel.dataset.deadLetters = "true";
            panel.append(deadPanel);
        }
        setText(deadPanel, "加载中...");
        try {
            const data = await api.requestJson("/api/v1/admin/async-dead-letters?page=1&page_size=10");
            deadPanel.replaceChildren();
            const h3 = document.createElement("h3");
            setText(h3, "死信列表");
            const items = data.items || [];
            if (items.length === 0) {
                const empty = document.createElement("p");
                setText(empty, "暂无死信");
                deadPanel.append(h3, empty);
                return;
            }
            const list = document.createElement("ul");
            items.forEach(dl => {
                const item = document.createElement("li");
                setText(item, `${dl.id || dl.dead_letter_id} — ${dl.source_type || dl.type}`);
                list.append(item);
            });
            deadPanel.append(h3, list);
        } catch (error) {
            setText(deadPanel, api.sanitizeErrorMessage(error));
        }
    }

    // ============================================================
    // 辅助函数
    // ============================================================

    function isAdmin() {
        const session = auth.session();
        return session && session.role === "ADMIN";
    }

    function renderForbidden(container) {
        const panel = document.createElement("section");
        panel.className = "panel";
        const eyebrow = document.createElement("p");
        eyebrow.className = "eyebrow";
        setText(eyebrow, "权限");
        const h1 = document.createElement("h1");
        setText(h1, "无权限");
        const desc = document.createElement("p");
        setText(desc, "当前账号无权访问该页面。");
        const backLink = document.createElement("a");
        backLink.href = "#/tasks";
        backLink.className = "secondary-action";
        setText(backLink, "返回任务中心");
        panel.append(eyebrow, h1, desc, backLink);
        container.replaceChildren(panel);
    }

    function mapTaskType(type) {
        const map = {
            resume_evaluation: "简历评估",
            resume_optimization: "简历优化",
            algorithm_test: "算法测试",
            mock_interview: "模拟面试",
        };
        return map[type] || type || "未知";
    }

    function createSelect(name, label, options) {
        const field = document.createElement("label");
        const text = document.createElement("span");
        const select = document.createElement("select");
        select.name = name;
        setText(text, label);
        options.forEach(([value, optionLabel]) => {
            const option = document.createElement("option");
            option.value = value;
            setText(option, optionLabel);
            select.append(option);
        });
        field.append(text, select);
        return field;
    }

    function appendDetail(summary, label, value) {
        const term = document.createElement("dt");
        const description = document.createElement("dd");
        setText(term, label);
        setText(description, value || "无");
        summary.append(term, description);
    }

    function createEyebrow(value) {
        const eyebrow = document.createElement("p");
        eyebrow.className = "eyebrow";
        setText(eyebrow, value);
        return eyebrow;
    }

    function setText(element, value) {
        element.textContent = String(value);
    }

    window.AiInterviewAdminUi = {
        renderFailedTaskList,
        renderFailureDetail,
    };
})();
