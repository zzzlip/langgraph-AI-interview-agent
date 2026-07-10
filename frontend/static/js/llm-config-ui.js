(() => {
    const api = window.AiInterviewApi;

    // ============================================================
    // LLM 配置页面
    // ============================================================

    async function renderLlmConfigPage(container) {
        const panel = document.createElement("section");
        panel.className = "panel";
        const eyebrow = document.createElement("p");
        eyebrow.className = "eyebrow";
        setText(eyebrow, "配置");
        const h1 = document.createElement("h1");
        setText(h1, "DeepSeek API 配置");
        const desc = document.createElement("p");
        setText(desc, "设置你的 DeepSeek API 密钥，用于所有 AI 任务的模型调用。所有任务共用此配置。API 密钥加密存储在服务器端。");
        panel.append(eyebrow, h1, desc);

        // 加载当前配置
        try {
            const current = await api.requestJson("/api/v1/user/llm-config");
            panel.append(createConfigForm(current));
        } catch {
            panel.append(createConfigForm(null));
        }

        container.replaceChildren(panel);
    }

    function createConfigForm(current) {
        const form = document.createElement("form");
        form.className = "business-form";
        form.setAttribute("aria-label", "LLM 配置");

        // provider 固定为 deepseek
        const providerField = createReadonlyField("provider", "Provider", "deepseek");

        // base_url
        const baseUrlField = createTextField("base_url", "API 地址",
            current && current.base_url || "https://api.deepseek.com/v1");

        // model
        const modelField = createTextField("model", "模型名称",
            current && current.model || "deepseek-chat");

        // api_key
        const apiKeyField = createTextField("api_key", "API Key", "");
        const apiKeyInput = apiKeyField.querySelector("input");
        apiKeyInput.required = !current || !current.has_api_key;
        const apiKeyHint = document.createElement("p");
        apiKeyHint.className = "field-hint";
        setText(apiKeyHint, current && current.has_api_key ? "不填写则保留已保存的 API Key" : "首次保存必须填写 API Key");
        apiKeyField.append(apiKeyHint);

        const submitButton = document.createElement("button");
        submitButton.type = "submit";
        submitButton.className = "primary-action";
        setText(submitButton, "保存配置");

        const feedback = createFeedback();

        form.append(providerField, baseUrlField, modelField, apiKeyField, submitButton, feedback);
        form.addEventListener("submit", async event => {
            event.preventDefault();
            submitButton.disabled = true;
            setText(feedback, "");
            const body = {
                provider: "deepseek",
                base_url: form.elements.base_url.value.trim(),
                model: form.elements.model.value.trim(),
                api_key: form.elements.api_key.value.trim()
            };
            if (!body.base_url || !body.model) {
                setText(feedback, "API 地址和模型名称不能为空");
                submitButton.disabled = false;
                return;
            }
            const apiKey = form.elements.api_key.value.trim();
            const keepExistingKey = current && current.has_api_key && !apiKey;
            if (!keepExistingKey && !apiKey) {
                setText(feedback, "API Key 不能为空");
                submitButton.disabled = false;
                return;
            }
            try {
                await api.requestJson("/api/v1/user/llm-config", {
                    method: "PUT",
                    body: {
                        provider: body.provider,
                        base_url: body.base_url,
                        model: body.model,
                        api_key: keepExistingKey ? "" : body.api_key,
                    },
                });
                setText(feedback, "配置已保存");
            } catch (error) {
                setText(feedback, api.sanitizeErrorMessage(error));
            } finally {
                submitButton.disabled = false;
            }
        });

        return form;
    }

    // ============================================================
    // 查询当前用户 LLM 配置，用于创建任务时自动附带
    // ============================================================

    async function loadUserLlmConfig() {
        try {
            const config = await api.requestJson("/api/v1/user/llm-config");
            return config && config.has_api_key ? config : null;
        } catch {
            return null;
        }
    }

    /**
     * 返回用于创建任务请求的 llm_config 字段。
     * 需要调用方在创建任务前先调用 loadUserLlmConfig 获取 config，
     * 然后传入页面表单中用户可能覆盖的值。
     */
    function buildLlmConfigPayload(savedConfig, formOverrides) {
        const provider = (formOverrides && formOverrides.provider) || (savedConfig && savedConfig.provider) || "deepseek";
        const baseUrl = (formOverrides && formOverrides.base_url) || (savedConfig && savedConfig.base_url) || "https://api.deepseek.com/v1";
        const model = (formOverrides && formOverrides.model) || (savedConfig && savedConfig.model) || "deepseek-chat";
        return { provider, base_url: baseUrl, model };
    }

    // ============================================================
    // 辅助
    // ============================================================

    function createTextField(name, label, defaultValue) {
        const field = document.createElement("label");
        field.className = "form-field";
        const text = document.createElement("span");
        const input = document.createElement("input");
        input.type = name === "api_key" ? "password" : "text";
        input.name = name;
        input.required = false;
        input.value = defaultValue || "";
        setText(text, label);
        field.append(text, input);
        return field;
    }

    function createReadonlyField(name, label, value) {
        const field = document.createElement("label");
        field.className = "form-field";
        const text = document.createElement("span");
        const input = document.createElement("input");
        input.type = "text";
        input.name = name;
        input.value = value;
        input.readOnly = true;
        setText(text, label);
        field.append(text, input);
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

    window.AiInterviewLlmConfigUi = {
        buildLlmConfigPayload,
        loadUserLlmConfig,
        renderLlmConfigPage,
    };
})();
