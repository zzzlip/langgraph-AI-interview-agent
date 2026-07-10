(() => {
    const api = window.AiInterviewApi;
    const mapping = window.AiInterviewStateMapping;
    const { TaskSseClient } = window.AiInterviewSse;
    const businessRenderers = window.AiInterviewBusinessRenderers;
    const mockUi = window.AiInterviewMockUi;
    let activeSseClient = null;

    async function renderTaskList(container) {
        closeActiveSseClient();
        container.replaceChildren(createTaskListShell());
        const form = container.querySelector("[data-task-filter-form]");
        form.addEventListener("submit", event => {
            event.preventDefault();
            loadTaskList(container, {
                taskType: form.elements.task_type.value,
                status: form.elements.status.value,
            });
        });
        await loadTaskList(container, {});
    }

    async function loadTaskList(container, filters) {
        const list = container.querySelector("[data-task-list]");
        setText(list, "加载中");
        const params = new URLSearchParams({ page: "1", page_size: "20" });
        if (filters.taskType) {
            params.set("task_type", filters.taskType);
        }
        if (filters.status) {
            params.set("status", filters.status);
        }
        try {
            const page = await api.requestJson(`/api/v1/tasks?${params.toString()}`);
            renderTaskItems(list, page.items || []);
        } catch (error) {
            setText(list, api.sanitizeErrorMessage(error));
        }
    }

    async function renderTaskDetail(container, taskId) {
        closeActiveSseClient();
        container.replaceChildren(createLoadingPanel("任务详情"));
        try {
            const task = await api.requestJson(`/api/v1/tasks/${encodeURIComponent(taskId)}`);
            renderTaskSnapshot(container, task);
            if (!mapping.isTerminalStatus(task.status)) {
                activeSseClient = new TaskSseClient({
                    taskId,
                    onEvent: (eventName, data) => handleTaskEvent(container, taskId, eventName, data),
                    onPoll: () => refreshTaskDetail(container, taskId),
                    onRecovering: () => showRecovering(container),
                });
                activeSseClient.connect();
            }
        } catch (error) {
            container.replaceChildren(createErrorPanel("任务详情", api.sanitizeErrorMessage(error)));
        }
    }

    async function refreshTaskDetail(container, taskId) {
        try {
            const task = await api.requestJson(`/api/v1/tasks/${encodeURIComponent(taskId)}`);
            renderTaskSnapshot(container, task);
            if (mapping.isTerminalStatus(task.status)) {
                closeActiveSseClient();
            }
        } catch (error) {
            showRecovering(container, api.sanitizeErrorMessage(error));
        }
    }

    function handleTaskEvent(container, taskId, eventName, data) {
        appendTimelineEvent(container, eventName, data);
        if (eventName === "task_snapshot") {
            renderTaskSnapshot(container, data);
        }
        if (eventName === "task_progress") {
            refreshTaskDetail(container, taskId);
        }
        if (eventName === "task_completed" || eventName === "task_failed") {
            refreshTaskDetail(container, taskId);
        }
    }

    function renderTaskSnapshot(container, task) {
        const status = mapping.mapTaskStatus(task.status);
        const panel = document.createElement("section");
        panel.className = "panel task-detail";
        panel.dataset.taskStatus = status.category;

        const title = document.createElement("h1");
        setText(title, mapping.mapTaskType(task.task_type));

        const summary = document.createElement("dl");
        summary.className = "detail-grid";
        appendDetail(summary, "任务 ID", task.task_id);
        appendDetail(summary, "状态", status.label);
        appendDetail(summary, "阶段", mapping.mapStage(task.stage));
        appendDetail(summary, "进度", formatProgress(task.progress));
        appendDetail(summary, "输入摘要", task.input_summary || "无");

        const timeline = document.createElement("ol");
        timeline.dataset.taskTimeline = "true";
        timeline.className = "timeline";
        (task.events || []).forEach(event => appendTimelineItem(timeline, event.event_type || event.type, event.message));

        panel.append(createEyebrow("任务详情"), title, summary, timeline);
        container.replaceChildren(panel);

        // 渲染业务结果
        if (businessRenderers && task.task_type) {
            businessRenderers.renderBusinessResult(panel, task);
        }
        if (mockUi && task.task_type === "mock_interview") {
            mockUi.renderMockInterviewDetail(panel, task);
            if (task.status === "success") {
                mockUi.renderMockInterviewResult(panel, task);
            }
        }
    }

    function createTaskListShell() {
        const panel = document.createElement("section");
        panel.className = "panel";
        panel.append(createEyebrow("任务"));
        const title = document.createElement("h1");
        setText(title, "任务中心");

        const form = document.createElement("form");
        form.className = "filter-form";
        form.dataset.taskFilterForm = "true";
        form.append(createSelect("task_type", "任务类型", [
            ["", "全部类型"],
            ["resume_evaluation", "简历评估"],
            ["resume_optimization", "简历优化"],
            ["algorithm_test", "算法测试"],
            ["mock_interview", "模拟面试"],
        ]));
        form.append(createSelect("status", "任务状态", [
            ["", "全部状态"],
            ["pending", "等待处理"],
            ["queued", "排队中"],
            ["running", "处理中"],
            ["waiting_user_input", "等待您的输入"],
            ["success", "已完成"],
            ["failed", "失败"],
            ["cancelled", "已取消"],
        ]));
        const button = document.createElement("button");
        button.type = "submit";
        setText(button, "筛选");
        form.append(button);

        const list = document.createElement("div");
        list.dataset.taskList = "true";
        list.className = "task-list";
        panel.append(title, form, list);
        return panel;
    }

    function renderTaskItems(container, items) {
        container.replaceChildren();
        if (items.length === 0) {
            setText(container, "暂无任务");
            return;
        }
        items.forEach(task => {
            const link = document.createElement("a");
            link.className = "task-row";
            link.href = `#/tasks/${encodeURIComponent(task.task_id)}`;
            const status = mapping.mapTaskStatus(task.status);
            setText(link, `${mapping.mapTaskType(task.task_type)} · ${status.label} · ${task.task_id}`);
            container.append(link);
        });
    }

    function appendDetail(summary, label, value) {
        const term = document.createElement("dt");
        const description = document.createElement("dd");
        setText(term, label);
        setText(description, value || "无");
        summary.append(term, description);
    }

    function appendTimelineEvent(container, eventName, data) {
        const timeline = container.querySelector("[data-task-timeline]");
        if (timeline) {
            appendTimelineItem(timeline, eventName, data.message || data.stage || "");
        }
    }

    function appendTimelineItem(timeline, eventName, message) {
        const item = document.createElement("li");
        setText(item, `${mapping.mapStage(eventName)} ${message || ""}`.trim());
        timeline.append(item);
    }

    function showRecovering(container, message) {
        const panel = container.querySelector(".task-detail");
        if (!panel) {
            return;
        }
        let banner = panel.querySelector("[data-recovering-banner]");
        if (!banner) {
            banner = document.createElement("p");
            banner.className = "recovering-banner";
            banner.dataset.recoveringBanner = "true";
            panel.prepend(banner);
        }
        setText(banner, message || "实时连接已断开，正在轮询恢复");
    }

    function closeActiveSseClient() {
        if (activeSseClient) {
            activeSseClient.close();
            activeSseClient = null;
        }
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

    function createLoadingPanel(titleText) {
        const panel = document.createElement("section");
        panel.className = "panel";
        panel.append(createEyebrow("任务"));
        const title = document.createElement("h1");
        setText(title, titleText);
        const text = document.createElement("p");
        setText(text, "加载中");
        panel.append(title, text);
        return panel;
    }

    function createErrorPanel(titleText, message) {
        const panel = document.createElement("section");
        panel.className = "panel";
        panel.append(createEyebrow("任务"));
        const title = document.createElement("h1");
        const text = document.createElement("p");
        setText(title, titleText);
        setText(text, message);
        panel.append(title, text);
        return panel;
    }

    function createEyebrow(value) {
        const eyebrow = document.createElement("p");
        eyebrow.className = "eyebrow";
        setText(eyebrow, value);
        return eyebrow;
    }

    function formatProgress(progress) {
        if (progress === undefined || progress === null || progress === "") {
            return "未提供";
        }
        return `${progress}%`;
    }

    function setText(element, value) {
        element.textContent = String(value);
    }

    window.AiInterviewTaskUi = Object.freeze({
        closeActiveSseClient,
        renderTaskDetail,
        renderTaskList,
    });
})();
