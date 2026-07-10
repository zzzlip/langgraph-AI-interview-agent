(() => {
    const api = window.AiInterviewApi;
    const csrf = window.AiInterviewCsrf;
    const SESSION_KEY = "ai_interview_session";
    const REDIRECT_KEY = "ai_interview_redirect_to";
    let currentSession = readSession();

    function safeSession(session) {
        return {
            account_id: session.account_id || session.accountId || "",
            username: session.username || "",
            role: session.role || "USER",
        };
    }

    function readSession() {
        try {
            const raw = window.sessionStorage.getItem(SESSION_KEY);
            if (!raw) {
                return null;
            }
            const parsed = JSON.parse(raw);
            const session = safeSession(parsed);
            return session.account_id && session.username ? session : null;
        } catch {
            window.sessionStorage.removeItem(SESSION_KEY);
            return null;
        }
    }

    function saveSession(session) {
        const snapshot = safeSession(session);
        currentSession = snapshot;
        window.sessionStorage.setItem(SESSION_KEY, JSON.stringify(safeSession(session)));
    }

    function clearSession() {
        currentSession = null;
        window.sessionStorage.removeItem(SESSION_KEY);
    }

    function session() {
        return currentSession;
    }

    function saveRedirectTarget(path) {
        window.sessionStorage.setItem(REDIRECT_KEY, path || "/tasks");
    }

    function takeRedirectTarget() {
        const target = window.sessionStorage.getItem(REDIRECT_KEY) || "/tasks";
        window.sessionStorage.removeItem(REDIRECT_KEY);
        return target;
    }

    function sanitizeErrorMessage(error) {
        const raw = typeof error === "string" ? error : (error && error.message) || "";
        const forbidden = ["Exception", "stack", "Cookie", "Token", "SQL", "object key", "object_key", "/Users/", "/tmp/"];
        if (forbidden.some(item => raw.indexOf(item) >= 0)) {
            return "请求处理失败，请稍后重试";
        }
        return api.sanitizeErrorMessage(raw);
    }

    async function login(username, password) {
        const data = await api.requestJson("/api/v1/auth/login", {
            method: "POST",
            csrf: false,
            body: { username, password },
        });
        csrf.setToken(data.csrf_token || data.csrfToken || "");
        saveSession(data);
        return session();
    }

    async function register(username, password) {
        return api.requestJson("/api/v1/auth/register", {
            method: "POST",
            csrf: false,
            body: { username, password },
        });
    }

    async function logout() {
        try {
            await api.requestJson("/api/v1/auth/logout", {
                method: "POST",
            });
        } finally {
            clearSession();
        }
    }

    window.AiInterviewAuth = Object.freeze({
        clearSession,
        login,
        logout,
        register,
        sanitizeErrorMessage,
        saveRedirectTarget,
        session,
        takeRedirectTarget,
    });
})();
