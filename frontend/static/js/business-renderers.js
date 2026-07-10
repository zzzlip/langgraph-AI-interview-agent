(() => {
    // 所有状态变更由 api.requestJson 统一注入 X-CSRF-Token、Idempotency-Key 和 X-Trace-Id
    const api = window.AiInterviewApi;
    const validators = window.AiInterviewValidators;
    const fileUi = window.AiInterviewFileUi;
    const mapping = window.AiInterviewStateMapping;
    const llmConfigUi = window.AiInterviewLlmConfigUi;

    // ============================================================
    // 简历评估
    // ============================================================

    async function renderResumeEvaluationPage(container) {
        const panel = createCreationPanel("简历评估", "选择简历文件和目标岗位，创建简历评估任务。");
        panel.append(createResumeEvaluationForm());
        container.replaceChildren(panel);
    }

    function createResumeEvaluationForm() {
        const form = document.createElement("form");
        form.className = "business-form";
        form.setAttribute("aria-label", "创建简历评估任务");

        // 简历文件选择
        const fileSection = createFileSelectorSection("resume", "选择简历文件");

        // 目标岗位
        const titleField = createTextField("job_title", "目标岗位名称");
        const descField = createTextAreaField("job_description", "岗位描述");

        const submitButton = document.createElement("button");
        submitButton.type = "submit";
        submitButton.className = "primary-action";
        setText(submitButton, "创建简历评估任务");

        const feedback = createFeedback();

        form.append(fileSection, titleField, descField, submitButton, feedback);
        form.addEventListener("submit", async event => {
            event.preventDefault();
            submitButton.disabled = true;
            setText(feedback, "");

            const resumeFileId = form.querySelector("[data-file-selector]").dataset.selectedFileId;
            const jobTitle = form.elements.job_title.value.trim();
            const jobDescription = form.elements.job_description.value.trim();

            const errors = validators.validateTargetJob({ title: jobTitle, description: jobDescription });
            if (!resumeFileId) {
                errors.unshift("请选择简历文件");
            }
            if (errors.length > 0) {
                setText(feedback, errors.join("；"));
                submitButton.disabled = false;
                return;
            }

            try {
                const llmConfig = await llmConfigUi.loadUserLlmConfig();
                const body = {
                    resume_file_id: resumeFileId,
                    target_job: { title: jobTitle, description: jobDescription },
                };
                if (llmConfig) {
                    body.llm_config = llmConfigUi.buildLlmConfigPayload(llmConfig);
                }
                const result = await api.requestJson("/api/v1/resume-evaluations", {
                    method: "POST",
                    idempotent: true,
                    operationId: "resume_evaluation",
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

    // 简历评估结果渲染
    function renderResumeEvaluationResult(container, task) {
        const panel = findOrCreateResultPanel(container, "简历评估结果");
        const result = resultSummary(task);

        if (result.overall_score !== undefined) {
            const scoreLine = document.createElement("p");
            scoreLine.className = "result-score";
            setText(scoreLine, `综合评分：${result.overall_score}分`);
            panel.append(scoreLine);
        }

        if (result.dimensions && result.dimensions.length > 0) {
            const dimList = document.createElement("ul");
            dimList.className = "dimension-list";
            result.dimensions.forEach(dim => {
                const item = document.createElement("li");
                setText(item, `${dim.name || dim.dimension}：${dim.score || ""}分 — ${dim.comment || dim.analysis || ""}`);
                dimList.append(item);
            });
            panel.append(dimList);
        }

        if (result.radar_chart_url) {
            const img = document.createElement("img");
            img.src = result.radar_chart_url;
            img.alt = "五维评分雷达图";
            img.className = "result-chart";
            panel.append(img);
        }

        const reportFile = firstReportFile(task);
        if (reportFile) {
            panel.append(createDownloadButton(reportFile, "下载评估报告"));
        }
        const outputFile = firstOutputFile(task);
        if (outputFile) {
            panel.append(createDownloadButton(outputFile, "下载评估产物"));
        }
    }

    // ============================================================
    // 简历优化
    // ============================================================

    async function renderResumeOptimizationPage(container) {
        const panel = createCreationPanel("简历优化", "选择简历文件和可选头像，创建简历优化任务。");
        panel.append(createResumeOptimizationForm());
        container.replaceChildren(panel);
    }

    function createResumeOptimizationForm() {
        const form = document.createElement("form");
        form.className = "business-form";
        form.setAttribute("aria-label", "创建简历优化任务");

        const resumeSection = createFileSelectorSection("resume", "选择简历文件");
        const avatarSection = createFileSelectorSection("avatar", "选择头像（可选）");

        const titleField = createTextField("job_title", "目标岗位名称");
        const descField = createTextAreaField("job_description", "岗位描述");

        const submitButton = document.createElement("button");
        submitButton.type = "submit";
        submitButton.className = "primary-action";
        setText(submitButton, "创建简历优化任务");

        const feedback = createFeedback();

        form.append(resumeSection, avatarSection, titleField, descField, submitButton, feedback);
        form.addEventListener("submit", async event => {
            event.preventDefault();
            submitButton.disabled = true;
            setText(feedback, "");

            const resumeFileId = form.querySelectorAll("[data-file-selector]")[0].dataset.selectedFileId;
            const avatarFileId = form.querySelectorAll("[data-file-selector]")[1].dataset.selectedFileId || null;
            const jobTitle = form.elements.job_title.value.trim();
            const jobDescription = form.elements.job_description.value.trim();

            const errors = validators.validateTargetJob({ title: jobTitle, description: jobDescription });
            if (!resumeFileId) {
                errors.unshift("请选择简历文件");
            }
            if (errors.length > 0) {
                setText(feedback, errors.join("；"));
                submitButton.disabled = false;
                return;
            }

            const body = {
                resume_file_id: resumeFileId,
                target_job: { title: jobTitle, description: jobDescription },
            };
            if (avatarFileId) {
                body.avatar_file_id = avatarFileId;
            }

            try {
                const llmConfig = await llmConfigUi.loadUserLlmConfig();
                if (llmConfig) {
                    body.llm_config = llmConfigUi.buildLlmConfigPayload(llmConfig);
                }
                const result = await api.requestJson("/api/v1/resume-optimizations", {
                    method: "POST",
                    idempotent: true,
                    operationId: "resume_optimization",
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

    // 简历优化结果渲染
    function renderResumeOptimizationResult(container, task) {
        const panel = findOrCreateResultPanel(container, "简历优化结果");
        const result = resultSummary(task);

        if (result.summary || result.optimization_summary) {
            const summary = document.createElement("p");
            summary.className = "result-summary";
            setText(summary, `优化摘要：${result.summary || result.optimization_summary}`);
            panel.append(summary);
        }

        if (result.optimization_items && result.optimization_items.length > 0) {
            const list = document.createElement("ul");
            list.className = "optimization-list";
            result.optimization_items.forEach(item => {
                const row = document.createElement("li");
                setText(row, `${item.section || "段落"}：${item.direction || item.summary || ""}`);
                list.append(row);
            });
            panel.append(list);
        }

        const resumeFile = firstReportFile(task) || firstOutputFile(task);
        if (resumeFile) {
            panel.append(createDownloadButton(resumeFile, "下载优化后简历"));
        }
    }

    // ============================================================
    // 算法测试
    // ============================================================

    async function renderAlgorithmTestPage(container) {
        const panel = createCreationPanel("算法测试", "填写 Codeforces handle 和目标岗位，创建算法测试任务。");
        panel.append(createAlgorithmTestForm());
        container.replaceChildren(panel);
    }

    function createAlgorithmTestForm() {
        const form = document.createElement("form");
        form.className = "business-form";
        form.setAttribute("aria-label", "创建算法测试任务");

        const handleField = createTextField("codeforces_handle", "Codeforces Handle");

        const titleField = createTextField("job_title", "目标岗位名称");
        const descField = createTextAreaField("job_description", "岗位描述");

        const submitButton = document.createElement("button");
        submitButton.type = "submit";
        submitButton.className = "primary-action";
        setText(submitButton, "创建算法测试任务");

        const feedback = createFeedback();

        form.append(handleField, titleField, descField, submitButton, feedback);
        form.addEventListener("submit", async event => {
            event.preventDefault();
            submitButton.disabled = true;
            setText(feedback, "");

            const handle = form.elements.codeforces_handle.value.trim();
            const jobTitle = form.elements.job_title.value.trim();
            const jobDescription = form.elements.job_description.value.trim();

            const errors = validators.validateCodeforcesHandle(handle);
            const jobErrors = validators.validateTargetJob({ title: jobTitle, description: jobDescription });
            errors.push(...jobErrors);
            if (errors.length > 0) {
                setText(feedback, errors.join("；"));
                submitButton.disabled = false;
                return;
            }

            try {
                const llmConfig = await llmConfigUi.loadUserLlmConfig();
                const body = {
                    target_job: { title: jobTitle, description: jobDescription },
                    codeforces_handle: handle,
                };
                if (llmConfig) {
                    body.llm_config = llmConfigUi.buildLlmConfigPayload(llmConfig);
                }
                const result = await api.requestJson("/api/v1/algorithm-tests", {
                    method: "POST",
                    idempotent: true,
                    operationId: "algorithm_test",
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

    // 算法测试题单渲染
    function renderAlgorithmTestProblemSet(container, task) {
        const panel = findOrCreateResultPanel(container, "算法测试题单");
        const result = resultSummary(task);
        const problems = result.problems || result.problem_set || [];

        if (problems.length === 0) {
            const empty = document.createElement("p");
            setText(empty, "题单生成中...");
            panel.append(empty);
            return;
        }

        const problemList = document.createElement("ul");
        problemList.className = "problem-list";
        problems.forEach(problem => {
            const item = document.createElement("li");
            item.className = "problem-item";

            const info = document.createElement("span");
            const title = problem.title || problem.name || "未知题目";
            const difficulty = problem.difficulty || "";
            const tags = (problem.tags || []).join("、");
            setText(info, `${title} ${difficulty ? "· " + difficulty : ""} ${tags ? "· " + tags : ""}`);

            const link = document.createElement("a");
            link.href = problem.url || problem.link || "#";
            link.target = "_blank";
            link.rel = "noopener";
            setText(link, "打开题目");
            link.className = "problem-link";

            item.append(info, link);
            problemList.append(item);
        });
        panel.append(problemList);

        // 等待用户完成的提示和提交检查按钮
        if (task.status === "waiting_user_input") {
            const hint = document.createElement("p");
            hint.className = "user-action-hint";
            setText(hint, "请在外部平台完成所有题目后，点击下方按钮提交检查");
            panel.append(hint);

            const checkButton = document.createElement("button");
            checkButton.type = "button";
            checkButton.className = "primary-action";
            setText(checkButton, "我已完成，提交检查");
            checkButton.addEventListener("click", async () => {
                checkButton.disabled = true;
                setText(checkButton, "提交中...");
                try {
                    await api.requestJson(`/api/v1/algorithm-tests/${encodeURIComponent(task.task_id)}/checks`, {
                        method: "POST",
                        idempotent: true,
                        operationId: "algorithm_check:" + task.task_id,
                        body: {},
                    });
                    window.dispatchEvent(new CustomEvent("ai-task-refresh", { detail: { taskId: task.task_id } }));
                } catch (error) {
                    setText(checkButton, api.sanitizeErrorMessage(error));
                    checkButton.disabled = false;
                }
            });
            panel.append(checkButton);
        }
    }

    // 算法测试检查结果渲染
    function renderAlgorithmCheckResult(container, task) {
        const panel = findOrCreateResultPanel(container, "算法测试检查结果");
        const result = resultSummary(task);
        const checks = result.problem_results || result.submission_results || [];

        if (checks.length > 0) {
            const checkList = document.createElement("ul");
            checkList.className = "check-result-list";
            checks.forEach(check => {
                const item = document.createElement("li");
                const passed = check.passed || check.status === "passed";
                const statusIcon = passed ? "✅" : "❌";
                setText(item, `${statusIcon} ${check.problem_title || check.title || ""} — ${passed ? "通过" : "未通过"}`);
                checkList.append(item);
            });
            panel.append(checkList);
        }

        if (result.overall_conclusion || result.conclusion) {
            const conclusion = document.createElement("p");
            conclusion.className = "result-conclusion";
            setText(conclusion, `整体结论：${result.overall_conclusion || result.conclusion}`);
            panel.append(conclusion);
        }
    }

    // ============================================================
    // 任务详情的业务结果调度
    // ============================================================

    function renderBusinessResult(container, task) {
        if (!task || !task.task_type) return;

        switch (task.task_type) {
            case "resume_evaluation":
                renderResumeEvaluationResult(container, task);
                break;
            case "resume_optimization":
                renderResumeOptimizationResult(container, task);
                break;
            case "algorithm_test": {
                const result = resultSummary(task);
                if (result.problems || result.problem_set) {
                    renderAlgorithmTestProblemSet(container, task);
                }
                if (result.problem_results || result.submission_results) {
                    renderAlgorithmCheckResult(container, task);
                }
                break;
            }
            case "mock_interview":
                // 由 mock-interview-ui.js 处理
                break;
        }
    }

    // ============================================================
    // 辅助组件
    // ============================================================

    function createCreationPanel(title, description) {
        const panel = document.createElement("section");
        panel.className = "panel";
        const eyebrow = document.createElement("p");
        eyebrow.className = "eyebrow";
        setText(eyebrow, "业务");
        const h1 = document.createElement("h1");
        setText(h1, title);
        const desc = document.createElement("p");
        setText(desc, description);
        panel.append(eyebrow, h1, desc);
        return panel;
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

        // 异步加载符合用途的文件
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

    function findOrCreateResultPanel(container, title) {
        let panel = Array.from(container.querySelectorAll(".result-panel"))
                .find(item => item.dataset.resultTitle === title);
        const h2 = document.createElement("h2");
        setText(h2, title);
        if (!panel) {
            panel = document.createElement("section");
            panel.className = "panel result-panel";
            panel.dataset.resultTitle = title;
            container.append(panel);
        }
        panel.replaceChildren(h2);
        return panel;
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

    function createDownloadButton(file, label) {
        const button = document.createElement("button");
        button.type = "button";
        button.className = "primary-action";
        setText(button, label);
        button.addEventListener("click", async () => {
            await downloadBackendFile(file);
        });
        return button;
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

    function setText(element, value) {
        element.textContent = String(value);
    }

    window.AiInterviewBusinessRenderers = {
        renderAlgorithmCheckResult,
        renderAlgorithmTestPage,
        renderAlgorithmTestProblemSet,
        renderBusinessResult,
        renderResumeEvaluationPage,
        renderResumeEvaluationResult,
        renderResumeOptimizationPage,
        renderResumeOptimizationResult,
    };
})();
