(() => {
    let csrfToken = "";

    function setToken(value) {
        csrfToken = value || "";
    }

    async function refreshCsrfToken() {
        const response = await window.fetch("/api/v1/csrf-token", {
            method: "GET",
            credentials: "same-origin",
            headers: { Accept: "application/json" },
        });
        let payload = null;
        try {
            payload = await response.json();
        } catch {
            payload = null;
        }
        if (!response.ok || !payload || payload.code !== 0) {
            throw new Error("CSRF Token 获取失败，请刷新页面或重新登录");
        }
        setToken(payload.data && (payload.data.csrf_token || payload.data.csrfToken));
        if (!csrfToken) {
            throw new Error("CSRF Token 获取失败，请刷新页面或重新登录");
        }
        return csrfToken;
    }

    async function ensureCsrfToken() {
        if (csrfToken) {
            return csrfToken;
        }
        return refreshCsrfToken();
    }

    /** 清除当前 token 并重新获取，用于 403 重试场景 */
    async function resetAndRefresh() {
        csrfToken = "";
        return refreshCsrfToken();
    }

    window.AiInterviewCsrf = Object.freeze({
        ensureCsrfToken,
        refreshCsrfToken,
        resetAndRefresh,
        setToken,
    });
})();
