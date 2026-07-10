(() => {
    const api = window.AiInterviewApi;
    const validators = window.AiInterviewValidators;
    const fileUi = window.AiInterviewFileUi;
    const llmConfigUi = window.AiInterviewLlmConfigUi;

    // ============================================================
    // 创建面试任务
    // ============================================================

    function renderMockInterviewPage(container) {
        const panel = document.createElement("section");
        panel.className = "panel";
        const eyebrow = document.createElement("p");
        eyebrow.className = "eyebrow";
        setText(eyebrow, "业务");
        const h1 = document.createElement("h1");
        setText(h1, "模拟面试");
        const desc = document.createElement("p");
        setText(desc, "选择简历文件和目标岗位，创建 Human-in-the-loop 模拟面试任务。");
        panel.append(eyebrow, h1, desc);
        panel.append(createMockInterviewForm());
        container.replaceChildren(panel);
    }

    function createMockInterviewForm() {
        const form = document.createElement("form");
        form.className = "business-form";
        form.setAttribute("aria-label", "创建模拟面试任务");

        const resumeSection = createFileSelectorSection("resume", "选择简历文件");

        const titleField = createTextField("job_title", "目标岗位名称");
        titleField.querySelector("input").required = true;

        const descField = createTextAreaField("job_description", "岗位描述");
        descField.querySelector("textarea").required = true;

        const questionCountField = createTextField("question_count", "题目数量（默认4）");
        questionCountField.querySelector("input").required = false;

        const followUpField = createTextField("max_follow_up_rounds", "最大追问轮数（默认3）");
        followUpField.querySelector("input").required = false;

        const submitButton = document.createElement("button");
        submitButton.type = "submit";
        submitButton.className = "primary-action";
        setText(submitButton, "创建模拟面试任务");

        const feedback = createFeedback();

        form.append(resumeSection, titleField, descField, questionCountField, followUpField, submitButton, feedback);
        form.addEventListener("submit", async event => {
            event.preventDefault();
            submitButton.disabled = true;
            setText(feedback, "");

            const resumeFileId = form.querySelector("[data-file-selector]").dataset.selectedFileId;
            const jobTitle = form.elements.job_title.value.trim();
            const jobDescription = form.elements.job_description.value.trim();
            const questionCount = form.elements.question_count.value.trim();
            const maxFollowUp = form.elements.max_follow_up_rounds.value.trim();

            const errors = validators.validateTargetJob({ title: jobTitle, description: jobDescription });
            if (!resumeFileId) {
                errors.unshift("请选择简历文件");
            }
            const configErrors = validators.validateInterviewConfig({
                question_count: questionCount || undefined,
                max_follow_up_rounds: maxFollowUp || undefined,
            });
            errors.push(...configErrors);
            if (errors.length > 0) {
                setText(feedback, errors.join("；"));
                submitButton.disabled = false;
                return;
            }

            const body = {
                resume_file_id: resumeFileId,
                target_job: { title: jobTitle, description: jobDescription },
            };
            if (questionCount) body.question_count = parseInt(questionCount, 10);
            if (maxFollowUp) body.max_follow_up_rounds = parseInt(maxFollowUp, 10);

            try {
                const llmConfig = await llmConfigUi.loadUserLlmConfig();
                if (llmConfig) {
                    body.llm_config = llmConfigUi.buildLlmConfigPayload(llmConfig);
                }
                const result = await api.requestJson("/api/v1/mock-interviews", {
                    method: "POST",
                    idempotent: true,
                    operationId: "mock_interview",
                    body,
                });
                window.location.hash = `#/tasks/${encodeURIComponent(result.task_id)}`;
            } catch (error) {
                setText(feedback, api.sanitizeErrorMessage(error));
            } finally {
                submitButton.disabled = false;
            }
        });

        return form;
    }

    // ============================================================
    // 面试详情与交互
    // ============================================================

    async function renderMockInterviewDetail(container, task) {
        const panel = findOrCreateInterviewPanel(container);
        const interviewData = await loadInterviewData(task);

        // 题目列表
        const questions = interviewData.questions || [];
        if (questions.length > 0) {
            renderQuestionList(panel, questions, interviewData.current_question_id);
        } else if (!isTerminal(task.status)) {
            renderWaitingState(panel);
        }

        // 当前题和追问
        if (interviewData.current_question) {
            renderCurrentQuestion(panel, interviewData);
        }

        // 回答提交区域
        if (!isTerminal(task.status)) {
            renderAnswerSubmission(panel, task, interviewData);
        }
    }

    function renderQuestionList(panel, questions, currentQuestionId) {
        let list = panel.querySelector("[data-question-list]");
        if (!list) {
            list = document.createElement("ul");
            list.className = "question-list";
            list.dataset.questionList = "true";
            panel.append(list);
        }
        list.replaceChildren();
        questions.forEach((q, index) => {
            const item = document.createElement("li");
            const isCurrent = q.question_id === currentQuestionId;
            const isCompleted = q.status === "completed";
            let statusIcon = isCurrent ? "▶" : isCompleted ? "✅" : "○";
            setText(item, `${statusIcon} 第${index + 1}题：${q.title || q.question || ""}`);
            if (isCurrent) item.className = "current-question";
            list.append(item);
        });
    }

    function renderCurrentQuestion(panel, interviewData) {
        let currentPanel = panel.querySelector("[data-current-question-panel]");
        if (!currentPanel) {
            currentPanel = document.createElement("div");
            currentPanel.className = "current-question-panel";
            currentPanel.dataset.currentQuestionPanel = "true";
            panel.append(currentPanel);
        }
        currentPanel.replaceChildren();

        const q = interviewData.current_question;
        const qHeading = document.createElement("h3");
        setText(qHeading, q.title || q.question || "当前题目");
        currentPanel.append(qHeading);

        if (q.content || q.description) {
            const content = document.createElement("p");
            setText(content, q.content || q.description);
            currentPanel.append(content);
        }

        // 回答历史
        const history = interviewData.answer_history || interviewData.rounds || [];
        if (history.length > 0) {
            const historyList = document.createElement("ol");
            historyList.className = "answer-history";
            history.forEach(entry => {
                const item = document.createElement("li");
                const status = entry.status || "";
                setText(item, `轮次 ${entry.round_id || ""} ${status}`);
                historyList.append(item);
            });
            currentPanel.append(historyList);
        }

        // 当前轮次提示
        const currentRound = interviewData.current_round_id || interviewData.round_id || "";
        if (currentRound) {
            const roundHint = document.createElement("p");
            roundHint.className = "round-hint";
            setText(roundHint, `当前轮次：${currentRound}`);
            currentPanel.append(roundHint);
        }

        // 追问内容
        if (interviewData.follow_up || interviewData.current_follow_up) {
            const followUp = document.createElement("div");
            followUp.className = "follow-up-content";
            const fuHeading = document.createElement("strong");
            setText(fuHeading, "追问：");
            const fuText = document.createElement("span");
            setText(fuText, interviewData.follow_up || interviewData.current_follow_up);
            followUp.append(fuHeading, fuText);
            currentPanel.append(followUp);
        }

        // 重传提示
        if (interviewData.needs_resubmit) {
            const resubmitBanner = document.createElement("p");
            resubmitBanner.className = "resubmit-banner";
            setText(resubmitBanner, "当前回答无法识别，请重新上传当前轮回答");
            currentPanel.append(resubmitBanner);
        }
    }

    function renderAnswerSubmission(panel, task, interviewData) {
        let submitSection = panel.querySelector("[data-answer-submission]");
        if (!submitSection) {
            submitSection = document.createElement("div");
            submitSection.className = "answer-submission";
            submitSection.dataset.answerSubmission = "true";
            panel.append(submitSection);
        }
        submitSection.replaceChildren();

        const heading = document.createElement("h3");
        setText(heading, "提交回答");
        submitSection.append(heading);

        // 使用 MediaRecorder 录制或手动上传
        const canRecord = typeof window.MediaRecorder !== "undefined";
        const questionId = interviewData.current_question_id || interviewData.question_id || "";
        const roundId = interviewData.current_round_id || interviewData.round_id || "";

        // 手动上传文件选择
        const fileInput = document.createElement("input");
        fileInput.type = "file";
        fileInput.accept = canRecord ? "audio/*,video/*" : "audio/*,video/*,.mp3,.mp4,.webm,.wav";
        fileInput.className = "upload-input";

        const uploadLabel = document.createElement("label");
        const uploadText = document.createElement("span");
        setText(uploadText, "选择音频/视频回答文件");
        uploadLabel.append(uploadText, fileInput);

        // 录制按钮
        let recorderButton = null;
        if (canRecord) {
            recorderButton = document.createElement("button");
            recorderButton.type = "button";
            recorderButton.className = "secondary-action";
            setText(recorderButton, "开始录音/录像");
        }
        submitSection.append(uploadLabel);
        if (recorderButton) {
            submitSection.append(recorderButton);
        }

        // 如果没有录制能力，添加手动上传提示
        if (!canRecord) {
            const manualHint = document.createElement("p");
            manualHint.className = "manual-upload-hint";
            setText(manualHint, "您的浏览器不支持录音录像，请手动选择已录制的文件上传");
            submitSection.append(manualHint);
        }

        // 提交按钮
        const submitAnswerBtn = document.createElement("button");
        submitAnswerBtn.type = "button";
        submitAnswerBtn.className = "primary-action";
        setText(submitAnswerBtn, "提交回答");
        submitAnswerBtn.disabled = true;

        const submitFeedback = createFeedback();
        submitSection.append(submitAnswerBtn, submitFeedback);
        if (!questionId || !roundId) {
            setText(submitFeedback, "等待面试题目生成后再提交回答");
        }

        // 文件选择联动
        fileInput.addEventListener("change", () => {
            submitAnswerBtn.disabled = !questionId || !roundId || !fileInput.files || fileInput.files.length === 0;
        });

        // 提交回答
        submitAnswerBtn.addEventListener("click", async () => {
            const file = fileInput.files[0];
            if (!file) {
                setText(submitFeedback, "请选择回答文件");
                return;
            }

            submitAnswerBtn.disabled = true;
            setText(submitAnswerBtn, "提交中...");
            setText(submitFeedback, "");

            try {
                // 先上传回答文件
                const formData = new FormData();
                formData.append("file", file);
                formData.append("file_purpose", file.type && file.type.startsWith("video/") ? "interview_video" : "interview_audio");
                const uploadResult = await api.requestMultipart("/api/v1/files", formData);
                const answerFileId = uploadResult.file_id || uploadResult.fileId;

                // 提交回答
                await api.requestJson(`/api/v1/mock-interviews/${encodeURIComponent(task.task_id)}/answers`, {
                    method: "POST",
                    idempotent: true,
                    operationId: "mock_answer:" + questionId + ":" + roundId,
                    body: {
                        question_id: questionId,
                        round_id: roundId,
                        answer_file_id: answerFileId,
                    },
                });

                setText(submitFeedback, "回答已提交，等待分析");
                fileInput.value = "";
                // 触发任务刷新
                window.dispatchEvent(new CustomEvent("ai-task-refresh", { detail: { taskId: task.task_id } }));
            } catch (error) {
                setText(submitFeedback, api.sanitizeErrorMessage(error));
                submitAnswerBtn.disabled = false;
                setText(submitAnswerBtn, "提交回答");
            }
        });

        // 简单录制实现
        if (canRecord) {
            setupRecorder(recorderButton, fileInput, submitAnswerBtn);
        }
    }

    function setupRecorder(button, fileInput, submitButton) {
        let mediaRecorder = null;
        let chunks = [];
        let recording = false;

        button.addEventListener("click", async () => {
            if (recording) {
                mediaRecorder.stop();
                setText(button, "开始录音/录像");
                recording = false;
                return;
            }

            try {
                const stream = await navigator.mediaDevices.getUserMedia({ audio: true, video: true });
                const mimeType = MediaRecorder.isTypeSupported("video/webm") ? "video/webm" : "video/mp4";
                mediaRecorder = new MediaRecorder(stream, { mimeType });
                chunks = [];

                mediaRecorder.ondataavailable = event => {
                    if (event.data.size > 0) chunks.push(event.data);
                };

                mediaRecorder.onstop = () => {
                    const blob = new Blob(chunks, { type: mimeType });
                    const file = new File([blob], `interview_answer_${Date.now()}.webm`, { type: mimeType });
                    const dt = new DataTransfer();
                    dt.items.add(file);
                    fileInput.files = dt.files;
                    submitButton.disabled = false;
                    stream.getTracks().forEach(track => track.stop());
                };

                mediaRecorder.start();
                recording = true;
                setText(button, "停止录制");
            } catch {
                setText(button, "录制不可用，请手动上传");
                button.disabled = true;
            }
        });
    }

    // ============================================================
    // SSE 事件处理
    // ============================================================

    function handleMockInterviewEvent(container, task, eventName, data) {
        const businessData = data || {};
        switch (eventName) {
            case "mock_interview.questions.generated":
                refreshInterviewPanel(container, task);
                break;
            case "mock_interview.follow_up.generated":
                refreshInterviewPanel(container, task);
                break;
            case "mock_interview.question.completed":
                refreshInterviewPanel(container, task);
                break;
            case "mock_interview.answer.needs_resubmit":
                refreshInterviewPanel(container, task);
                break;
            case "task_completed":
                renderMockInterviewResult(container, task);
                break;
        }
    }

    function refreshInterviewPanel(container, task) {
        renderMockInterviewDetail(container, task);
    }

    // ============================================================
    // 最终结果渲染
    // ============================================================

    function renderMockInterviewResult(container, task) {
        let resultPanel = container.querySelector(".interview-result-panel");
        const h2 = document.createElement("h2");
        setText(h2, "模拟面试结果");
        if (!resultPanel) {
            resultPanel = document.createElement("section");
            resultPanel.className = "panel interview-result-panel";
            container.append(resultPanel);
        }
        resultPanel.replaceChildren(h2);

        const result = task.result || {};
        const interviewResult = resultSummary(task);

        if (interviewResult.overall_score !== undefined) {
            const overall = document.createElement("p");
            overall.className = "result-score";
            setText(overall, `综合评分：${interviewResult.overall_score}分`);
            resultPanel.append(overall);
        }

        // 维度评分
        const dimensions = interviewResult.dimensions || interviewResult.dimension_scores || [];
        if (dimensions.length > 0) {
            const dimHeader = document.createElement("h3");
            setText(dimHeader, "维度评分");
            const dimList = document.createElement("ul");
            dimList.className = "dimension-list";
            dimensions.forEach(dim => {
                const item = document.createElement("li");
                setText(item, `${dim.name || dim.dimension}：${dim.score || ""}分`);
                dimList.append(item);
            });
            resultPanel.append(dimHeader, dimList);
        }

        // 逐题分析
        const questionAnalysis = interviewResult.question_analysis || interviewResult.per_question || [];
        if (questionAnalysis.length > 0) {
            const qaHeader = document.createElement("h3");
            setText(qaHeader, "逐题分析");
            const qaList = document.createElement("ul");
            qaList.className = "question-analysis-list";
            questionAnalysis.forEach(qa => {
                const item = document.createElement("li");
                setText(item, `题目：${qa.question_title || ""} — ${qa.analysis || qa.summary || ""}`);
                qaList.append(item);
            });
            resultPanel.append(qaHeader, qaList);
        }

        // 改进建议
        const improvements = interviewResult.improvements || interviewResult.suggestions || [];
        if (improvements.length > 0) {
            const impHeader = document.createElement("h3");
            setText(impHeader, "改进建议");
            const impList = document.createElement("ul");
            impList.className = "improvement-list";
            improvements.forEach(imp => {
                const item = document.createElement("li");
                setText(item, imp);
                impList.append(item);
            });
            resultPanel.append(impHeader, impList);
        }

        // 报告下载
        const reportFile = firstReportFile(task) || firstOutputFile(task);
        if (reportFile) {
            const downloadBtn = document.createElement("button");
            downloadBtn.type = "button";
            downloadBtn.className = "primary-action";
            setText(downloadBtn, "下载面试报告");
            downloadBtn.addEventListener("click", async () => {
                await downloadBackendFile(reportFile);
            });
            resultPanel.append(downloadBtn);
        }
    }

    // ============================================================
    // 辅助组件
    // ============================================================

    function findOrCreateInterviewPanel(container) {
        let panel = container.querySelector(".mock-interview-detail");
        if (!panel) {
            panel = document.createElement("section");
            panel.className = "panel mock-interview-detail";
            const h2 = document.createElement("h2");
            setText(h2, "模拟面试详情");
            panel.append(h2);
            container.append(panel);
        }
        return panel;
    }

    async function loadInterviewData(task) {
        if (isTerminal(task.status)) {
            return resultSummary(task);
        }
        try {
            const state = await api.requestJson(`/api/v1/mock-interviews/${encodeURIComponent(task.task_id)}/state`);
            return normalizeStatePayload(state);
        } catch {
            return resultSummary(task);
        }
    }

    function normalizeStatePayload(state) {
        const payload = state.state_payload || state.statePayload || {};
        const plan = payload.interview_plan || {};
        const questions = plan.questions || payload.questions || [];
        const currentQuestionId = state.current_question_id || payload.current_question_id || "";
        const currentQuestion = findCurrentQuestion(questions, currentQuestionId);
        return {
            ...payload,
            questions,
            current_question_id: currentQuestionId,
            current_round_id: state.current_round_id || payload.current_round_id || "",
            waiting_input_type: state.waiting_input_type || payload.waiting_input_type || "",
            current_question: payload.current_question || currentQuestion,
            rounds: payload.rounds || [],
            needs_resubmit: hasOpenResubmit(payload),
            follow_up: currentFollowUp(payload),
        };
    }

    function findCurrentQuestion(questions, currentQuestionId) {
        return questions.find(question => question.question_id === currentQuestionId) || null;
    }

    function hasOpenResubmit(payload) {
        const records = payload.resubmit_records || [];
        return records.some(record => record.status !== "resolved");
    }

    function currentFollowUp(payload) {
        const rounds = payload.rounds || [];
        const latest = [...rounds].reverse().find(round => round.follow_up_question);
        return latest ? latest.follow_up_question : "";
    }

    function renderWaitingState(panel) {
        let waiting = panel.querySelector("[data-mock-waiting]");
        if (!waiting) {
            waiting = document.createElement("p");
            waiting.dataset.mockWaiting = "true";
            waiting.className = "user-action-hint";
            panel.append(waiting);
        }
        setText(waiting, "面试题目生成中，请等待任务进度更新");
    }

    function resultSummary(task) {
        const result = task && task.result ? task.result : {};
        return result.summary || result;
    }

    function firstReportFile(task) {
        const files = task && task.result && task.result.report_files;
        return files && files.length > 0 ? files[0] : null;
    }

    function firstOutputFile(task) {
        const files = task && task.result && task.result.output_files;
        return files && files.length > 0 ? files[0] : null;
    }

    async function downloadBackendFile(file) {
        const fileId = file.file_id || file.fileId;
        if (!fileId) {
            return;
        }
        const response = await api.requestFile(`/api/v1/files/${encodeURIComponent(fileId)}/download`);
        const url = URL.createObjectURL(response.blob);
        const link = document.createElement("a");
        link.href = url;
        link.download = file.file_name || file.fileName || safeDownloadName(response.filename) || "download";
        document.body.append(link);
        link.click();
        link.remove();
        URL.revokeObjectURL(url);
    }

    function safeDownloadName(contentDisposition) {
        try {
            const match = /filename\*?=(?:UTF-8'')?("?)([^";]+)\1/i.exec(contentDisposition || "");
            return match ? decodeURIComponent(match[2]) : "";
        } catch {
            return "";
        }
    }

    function isTerminal(status) {
        return status === "success" || status === "failed" || status === "cancelled";
    }

    function createFileSelectorSection(purpose, label) {
        const section = document.createElement("div");
        section.className = "file-selector-section";
        const heading = document.createElement("label");
        setText(heading, label);
        const selector = document.createElement("div");
        selector.className = "file-selector";
        selector.dataset.fileSelector = "true";
        selector.dataset.filePurpose = purpose;
        setText(selector, "加载中...");
        fileUi.loadFilesByPurpose(selector, purpose);
        section.append(heading, selector);
        return section;
    }

    function createTextField(name, label) {
        const field = document.createElement("label");
        field.className = "form-field";
        const text = document.createElement("span");
        const input = document.createElement("input");
        input.type = "text";
        input.name = name;
        input.required = true;
        setText(text, label);
        field.append(text, input);
        return field;
    }

    function createTextAreaField(name, label) {
        const field = document.createElement("label");
        field.className = "form-field";
        const text = document.createElement("span");
        const textarea = document.createElement("textarea");
        textarea.name = name;
        textarea.required = true;
        textarea.rows = 4;
        setText(text, label);
        field.append(text, textarea);
        return field;
    }

    function createFeedback() {
        const feedback = document.createElement("p");
        feedback.className = "form-feedback";
        feedback.setAttribute("role", "status");
        return feedback;
    }

    function setText(element, value) {
        element.textContent = String(value);
    }

    window.AiInterviewMockUi = {
        handleMockInterviewEvent,
        renderMockInterviewDetail,
        renderMockInterviewPage,
        renderMockInterviewResult,
    };
})();
