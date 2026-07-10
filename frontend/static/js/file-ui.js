(() => {
    const api = window.AiInterviewApi;

    // ============================================================
    // 文件管理视图
    // ============================================================

    async function renderFileManager(container) {
        container.replaceChildren(createFileManagerShell());
        await loadFileList(container);
    }

    function createFileManagerShell() {
        const panel = document.createElement("section");
        panel.className = "panel";
        panel.append(createEyebrow("文件"));

        const title = document.createElement("h1");
        setText(title, "文件管理");

        // 上传区域
        const uploadSection = createUploadSection();

        // 文件列表
        const list = document.createElement("div");
        list.className = "file-list";
        list.dataset.fileList = "true";

        panel.append(title, uploadSection, list);
        return panel;
    }

    function createUploadSection() {
        const section = document.createElement("div");
        section.className = "upload-section";

        const heading = document.createElement("h2");
        setText(heading, "上传文件");

        const form = document.createElement("form");
        form.className = "upload-form";
        form.dataset.uploadForm = "true";

        const fileInput = document.createElement("input");
        fileInput.type = "file";
        fileInput.name = "file";
        fileInput.required = true;

        const purposeSelect = createSelect("file_purpose", "文件用途", [
            ["resume", "简历"],
            ["avatar", "头像"],
            ["interview_audio", "面试音频"],
            ["interview_video", "面试视频"],
        ]);

        const submitButton = document.createElement("button");
        submitButton.type = "submit";
        setText(submitButton, "上传");

        const feedback = document.createElement("p");
        feedback.className = "form-feedback";
        feedback.dataset.uploadFeedback = "true";

        form.append(fileInput, purposeSelect, submitButton);

        form.addEventListener("submit", async event => {
            event.preventDefault();
            submitButton.disabled = true;
            setText(feedback, "");

            const file = fileInput.files[0];
            const purpose = form.elements.file_purpose.value;

            if (!file || file.size === 0) {
                setText(feedback, "文件不能为空");
                submitButton.disabled = false;
                return;
            }

            try {
                const formData = new FormData();
                formData.append("file", file);
                formData.append("file_purpose", purpose);
                await api.requestMultipart("/api/v1/files", formData);
                setText(feedback, "上传成功");
                fileInput.value = "";
                await loadFileList(findPanelContainer(form));
            } catch (error) {
                setText(feedback, api.sanitizeErrorMessage(error));
            } finally {
                submitButton.disabled = false;
            }
        });

        section.append(heading, form, feedback);
        return section;
    }

    async function loadFileList(container) {
        const panel = findPanelContainer(container);
        const list = panel.querySelector("[data-file-list]");
        if (!list) return;

        setText(list, "加载中");
        try {
            const data = await api.requestJson("/api/v1/files?page=1&page_size=50");
            const files = data.items || [];
            if (files.length === 0) {
                setText(list, "暂无文件");
                return;
            }
            renderFileList(list, files);
        } catch (error) {
            setText(list, api.sanitizeErrorMessage(error));
        }
    }

    function renderFileList(list, files) {
        list.replaceChildren();
        files.forEach(file => {
            const fileId = file.file_id || file.fileId;
            const row = document.createElement("div");
            row.className = "file-row";

            const info = document.createElement("div");
            info.className = "file-info";

            const name = document.createElement("span");
            name.className = "file-name";
            setText(name, file.file_name || file.fileName || "未知文件");

            const meta = document.createElement("span");
            meta.className = "file-meta";
            setText(meta, formatFileMeta(file));

            info.append(name, meta);

            const actions = document.createElement("div");
            actions.className = "file-actions";

            const downloadBtn = createDownloadButton(fileId);
            actions.append(downloadBtn);

            const previewBtn = createPreviewButton(fileId);
            actions.append(previewBtn);

            row.append(info, actions);
            list.append(row);
        });
    }

    async function downloadFile(fileId) {
        await api.requestFile(`/api/v1/files/${encodeURIComponent(fileId)}/download`);
        const iframe = document.createElement("iframe");
        iframe.style.display = "none";
        iframe.src = `/api/v1/files/${encodeURIComponent(fileId)}/download`;
        document.body.append(iframe);
        setTimeout(() => iframe.remove(), 5000);
    }

    async function previewFile(fileId) {
        await api.requestFile(`/api/v1/files/${encodeURIComponent(fileId)}/preview`);
        const win = window.open(`/api/v1/files/${encodeURIComponent(fileId)}/preview`, "_blank");
        if (win) win.focus();
    }

    function createDownloadButton(fileId) {
        const button = document.createElement("button");
        button.type = "button";
        setText(button, "下载");
        button.className = "file-action-btn";
        button.addEventListener("click", () => downloadFile(fileId));
        return button;
    }

    function createPreviewButton(fileId) {
        const button = document.createElement("button");
        button.type = "button";
        setText(button, "预览");
        button.className = "file-action-btn";
        button.addEventListener("click", () => previewFile(fileId));
        return button;
    }

    function formatFileMeta(file) {
        const parts = [];
        const type = mapFilePurpose(file.file_purpose || file.filePurpose);
        if (type) parts.push(type);
        const size = formatFileSize(file.file_size || file.fileSize);
        if (size) parts.push(size);
        parts.push(file.created_at || file.createdAt || "");
        return parts.join(" · ");
    }

    function mapFilePurpose(purpose) {
        const map = {
            resume: "简历",
            avatar: "头像",
            report: "报告",
            interview_audio: "面试音频",
            interview_video: "面试视频",
        };
        return map[purpose] || purpose || "";
    }

    function formatFileSize(bytes) {
        if (bytes === undefined || bytes === null) return "";
        if (bytes < 1024) return `${bytes}B`;
        if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)}KB`;
        return `${(bytes / (1024 * 1024)).toFixed(1)}MB`;
    }

    // ============================================================
    // 业务创建页文件选择器
    // ============================================================

    async function loadFilesByPurpose(container, purpose) {
        try {
            const data = await api.requestJson("/api/v1/files?page=1&page_size=50");
            const files = (data.items || []).filter(file => file.file_purpose === purpose);
            renderPurposeFileList(container, files);
        } catch (error) {
            setText(container, api.sanitizeErrorMessage(error));
        }
    }

    function renderPurposeFileList(container, files) {
        container.replaceChildren();
        if (files.length === 0) {
            const empty = document.createElement("p");
            setText(empty, "暂无可用文件，请先上传");
            container.append(empty);
            return;
        }
        files.forEach(file => {
            const option = document.createElement("div");
            option.className = "file-option";
            const name = document.createElement("span");
            setText(name, file.file_name || file.fileName || "未知文件");
            option.append(name);
            option.dataset.fileId = file.file_id || file.fileId;
            option.addEventListener("click", () => {
                option.closest("[data-file-selector]").querySelectorAll(".file-option").forEach(o => o.classList.remove("selected"));
                option.classList.add("selected");
                option.closest("[data-file-selector]").dataset.selectedFileId = option.dataset.fileId;
            });
            container.append(option);
        });
    }

    // ============================================================
    // 任务文件展示
    // ============================================================

    async function loadTaskFiles(container, taskId) {
        try {
            const data = await api.requestJson(`/api/v1/tasks/${encodeURIComponent(taskId)}/files?page=1&page_size=20`);
            const files = data.items || [];
            if (files.length === 0) {
                const empty = document.createElement("p");
                setText(empty, "暂无任务文件");
                container.append(empty);
                return;
            }
            renderFileList(container, files);
        } catch (error) {
            setText(container, api.sanitizeErrorMessage(error));
        }
    }

    // ============================================================
    // 辅助函数
    // ============================================================

    function createEyebrow(value) {
        const eyebrow = document.createElement("p");
        eyebrow.className = "eyebrow";
        setText(eyebrow, value);
        return eyebrow;
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

    function findPanelContainer(element) {
        while (element) {
            if (element.classList.contains("panel")) return element;
            element = element.parentElement;
        }
        return document.querySelector("[data-route-view]");
    }

    function setText(element, value) {
        element.textContent = String(value);
    }

    window.AiInterviewFileUi = {
        loadFilesByPurpose,
        loadTaskFiles,
        renderFileManager,
        renderPurposeFileList,
    };
})();
