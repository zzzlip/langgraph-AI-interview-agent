(() => {
    const csrf = window.AiInterviewCsrf;
    const idempotency = window.AiInterviewIdempotency;
    const safeMethods = new Set(["GET", "HEAD", "OPTIONS"]);

    function createTraceId() {
        if (window.crypto && window.crypto.randomUUID) {
            return window.crypto.randomUUID();
        }
        return `trace-${Date.now()}-${Math.random().toString(16).slice(2)}`;
    }

    function sanitizeErrorMessage(error) {
        const raw = typeof error === "string" ? error : (error && error.message) || "";
        const forbidden = ["Exception", "stack", "Cookie", "Token", "SQL", "object key", "object_key", "/Users/", "/tmp/"];
        if (forbidden.some(item => raw.indexOf(item) >= 0)) {
            return "请求处理失败，请稍后重试";
        }
        return raw || "请求处理失败，请稍后重试";
    }

    function createApiError(payload, response) {
        const error = new Error(sanitizeErrorMessage(payload && payload.message));
        error.status = response.status;
        error.code = payload && payload.code;
        return error;
    }

    function shouldRefreshCsrf(response, method, options) {
        return response.status === 403
            && options.csrf !== false
            && !options._csrfRetried
            && !safeMethods.has(method);
    }

    function completeOperation(operationId) {
        if (operationId) {
            idempotency.clearOperationKey(operationId);
        }
    }

    async function requestJson(path, options = {}) {
        const method = (options.method || "GET").toUpperCase();
        const operationId = options.idempotent ? (options.operationId || `${method}:${path}`) : "";
        const headers = {
            Accept: "application/json",
            ...(options.headers || {}),
        };
        headers["X-Trace-Id"] = createTraceId();
        if (!safeMethods.has(method) && options.csrf !== false) {
            headers["X-CSRF-Token"] = await csrf.ensureCsrfToken();
        }
        if (options.idempotent) {
            headers["Idempotency-Key"] = idempotency.keyFor(operationId);
        }
        if (options.body !== undefined) {
            headers["Content-Type"] = "application/json";
        }

        const response = await window.fetch(path, {
            method,
            credentials: "same-origin",
            headers,
            body: options.body === undefined ? undefined : JSON.stringify(options.body),
        });
        // 403 可能是 CSRF Token 过期，刷新后重试一次
        if (shouldRefreshCsrf(response, method, options)) {
            try {
                await csrf.resetAndRefresh();
                headers["X-CSRF-Token"] = await csrf.ensureCsrfToken();
            } catch {
                // 刷新失败则继续走原有错误处理
            }
            const retryResponse = await window.fetch(path, {
                method,
                credentials: "same-origin",
                headers,
                body: options.body === undefined ? undefined : JSON.stringify(options.body),
            });
            const retryPayload = await readJson(retryResponse);
            if (retryResponse.status === 401) {
                window.dispatchEvent(new CustomEvent("ai-session-expired"));
            }
            if (!retryResponse.ok) {
                throw createApiError(retryPayload, retryResponse);
            }
            if (!retryPayload || retryPayload.code !== 0) {
                throw createApiError(retryPayload, retryResponse);
            }
            completeOperation(operationId);
            return retryPayload.data || {};
        }
        const payload = await readJson(response);
        if (response.status === 401) {
            window.dispatchEvent(new CustomEvent("ai-session-expired"));
        }
        if (!response.ok) {
            throw createApiError(payload, response);
        }
        if (!payload || payload.code !== 0) {
            throw createApiError(payload, response);
        }
        completeOperation(operationId);
        return payload.data || {};
    }

    async function requestFile(path, options = {}) {
        const method = (options.method || "GET").toUpperCase();
        const headers = {
            Accept: options.accept || "application/octet-stream",
            ...(options.headers || {}),
        };
        headers["X-Trace-Id"] = createTraceId();
        if (!safeMethods.has(method) && options.csrf !== false) {
            headers["X-CSRF-Token"] = await csrf.ensureCsrfToken();
        }
        const response = await window.fetch(path, {
            method,
            credentials: "same-origin",
            headers,
            body: options.body,
        });
        if (response.status === 401) {
            window.dispatchEvent(new CustomEvent("ai-session-expired"));
        }
        if (!response.ok) {
            throw createApiError(null, response);
        }
        return {
            blob: await response.blob(),
            contentType: response.headers.get("Content-Type") || "",
            filename: response.headers.get("Content-Disposition") || "",
        };
    }

    async function requestMultipart(path, formData, options = {}) {
        const headers = {
            Accept: "application/json",
            ...(options.headers || {}),
        };
        headers["X-Trace-Id"] = createTraceId();
        if (options.csrf !== false) {
            headers["X-CSRF-Token"] = await csrf.ensureCsrfToken();
        }
        const response = await window.fetch(path, {
            method: "POST",
            credentials: "same-origin",
            headers,
            body: formData,
        });
        const payload = await readJson(response);
        if (response.status === 401) {
            window.dispatchEvent(new CustomEvent("ai-session-expired"));
        }
        if (!response.ok) {
            throw createApiError(payload, response);
        }
        if (!payload || payload.code !== 0) {
            throw createApiError(payload, response);
        }
        return payload.data || {};
    }

    async function readJson(response) {
        try {
            return await response.json();
        } catch {
            return null;
        }
    }

    window.AiInterviewApi = Object.freeze({
        requestFile,
        requestJson,
        requestMultipart,
        sanitizeErrorMessage,
        shouldRefreshCsrf,
    });
})();
