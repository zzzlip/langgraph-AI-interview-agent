(() => {
    const auth = window.AiInterviewAuth;
    const taskUi = window.AiInterviewTaskUi;
    const fileUi = window.AiInterviewFileUi;
    const businessRenderers = window.AiInterviewBusinessRenderers;
    const mockUi = window.AiInterviewMockUi;
    const adminUi = window.AiInterviewAdminUi;
    const llmConfigUi = window.AiInterviewLlmConfigUi;
    const routes = new Map([
        ["/", {
            eyebrow: "工作台",
            title: "AI 模拟面试平台",
            description: "登录后可创建任务、查看进度、下载报告，并在任务详情中接收实时事件。",
            actions: [
                { label: "登录", href: "#/login", kind: "primary-action" },
                { label: "注册账号", href: "#/register", kind: "secondary-action" },
            ],
        }],
        ["/login", {
            eyebrow: "账号",
            title: "登录",
            description: "使用平台账号进入任务工作台。",
            render: renderLogin,
        }],
        ["/register", {
            eyebrow: "账号",
            title: "注册账号",
            description: "创建账号后即可使用简历、算法和模拟面试任务。",
            render: renderRegister,
        }],
        ["/tasks", {
            eyebrow: "任务",
            title: "任务中心",
            description: "查看任务列表、筛选状态并进入任务详情。",
            requiresAuth: true,
            render: renderTaskListRoute,
        }],
        ["/files", {
            eyebrow: "文件",
            title: "文件管理",
            description: "上传简历、头像、音频或视频，并通过后端受控接口下载和预览。",
            requiresAuth: true,
            render: renderFileManagerRoute,
        }],
        ["/resume-evaluation", {
            eyebrow: "业务",
            title: "简历评估",
            description: "选择简历文件和目标岗位，创建简历评估任务。",
            requiresAuth: true,
            render: renderResumeEvaluationRoute,
        }],
        ["/resume-optimization", {
            eyebrow: "业务",
            title: "简历优化",
            description: "选择简历文件和可选头像，创建简历优化任务。",
            requiresAuth: true,
            render: renderResumeOptimizationRoute,
        }],
        ["/algorithm-test", {
            eyebrow: "业务",
            title: "算法测试",
            description: "填写 Codeforces handle 和目标岗位，创建算法测试任务。",
            requiresAuth: true,
            render: renderAlgorithmTestRoute,
        }],
        ["/mock-interview", {
            eyebrow: "业务",
            title: "模拟面试",
            description: "选择简历文件和目标岗位，创建 Human-in-the-loop 模拟面试任务。",
            requiresAuth: true,
            render: renderMockInterviewRoute,
        }],
        ["/llm-config", {
            eyebrow: "配置",
            title: "DeepSeek API 配置",
            description: "设置 API Key 后即可使用所有 AI 任务。所有业务共用此配置。",
            requiresAuth: true,
            render: renderLlmConfigRoute,
        }],
        ["/admin/failures", {
            eyebrow: "管理员",
            title: "失败排查",
            description: "管理员可查看失败任务、异步追踪和重试入口。",
            requiresAuth: true,
            adminOnly: true,
            render: renderFailuresRoute,
        }],
    ]);

    function currentPath() {
        const hash = window.location.hash.replace(/^#/, "");
        if (hash.startsWith("/tasks/")) {
            return hash;
        }
        if (hash.startsWith("/admin/failures/")) {
            return hash;
        }
        return routes.has(hash) ? hash : "/";
    }

    function resolveRoute(path) {
        if (path.startsWith("/tasks/")) {
            return {
                eyebrow: "任务",
                title: "任务详情",
                description: "查看任务快照和实时进度。",
                requiresAuth: true,
                render: () => renderTaskDetailRoute(path.slice("/tasks/".length)),
            };
        }
        if (path.startsWith("/admin/failures/")) {
            return {
                eyebrow: "管理员",
                title: "失败详情",
                description: "查看失败任务的阶段、失败码、事件历史和重试记录。",
                requiresAuth: true,
                adminOnly: true,
                render: () => renderAdminFailureRoute(path.slice("/admin/failures/".length)),
            };
        }
        return routes.get(path) || routes.get("/");
    }

    function setText(element, value) {
        element.textContent = value;
    }

    function createAction(action) {
        const link = document.createElement("a");
        link.className = action.kind;
        link.href = action.href;
        setText(link, action.label);
        return link;
    }

    function renderRoute() {
        const path = currentPath();
        const route = resolveRoute(path);
        const activeSession = auth.session();
        updateSessionNavigation(activeSession);
        if (route.requiresAuth && !activeSession) {
            const { saveRedirectTarget } = auth;
            saveRedirectTarget(path);
            window.location.hash = "#/login";
            return;
        }
        if (route.adminOnly && activeSession.role !== "ADMIN") {
            renderForbidden();
            updateActiveNavigation(path);
            return;
        }
        if (route.render) {
            route.render(route);
        } else {
            renderPanel(route);
        }
        updateActiveNavigation(path);
    }

    function renderPanel(route) {
        const view = document.querySelector("[data-route-view]");
        view.replaceChildren(createPanel(route));
    }

    function createPanel(route) {
        const section = document.createElement("section");
        section.className = "panel";

        const eyebrow = document.createElement("p");
        eyebrow.className = "eyebrow";
        setText(eyebrow, route.eyebrow);

        const title = document.createElement("h1");
        setText(title, route.title);

        const description = document.createElement("p");
        setText(description, route.description);

        section.append(eyebrow, title, description);
        if (route.actions) {
            const actionRow = document.createElement("div");
            actionRow.className = "action-row";
            route.actions.map(createAction).forEach(action => actionRow.append(action));
            section.append(actionRow);
        }
        return section;
    }

    function renderLogin(route) {
        const section = createPanel(route);
        const form = createAuthForm("登录", "登录");
        const feedback = createFeedback();
        form.addEventListener("submit", async event => {
            event.preventDefault();
            const submitButton = form.querySelector("button");
            submitButton.disabled = true;
            setText(feedback, "");
            const username = form.elements.username.value.trim();
            const password = form.elements.password.value;
            try {
                await auth.login(username, password);
                updateSessionNavigation(auth.session());
                const target = auth.takeRedirectTarget() || "/tasks";
                window.location.hash = `#${target}`;
            } catch (error) {
                setText(feedback, auth.sanitizeErrorMessage(error));
            } finally {
                submitButton.disabled = false;
            }
        });
        section.append(form, feedback);
        document.querySelector("[data-route-view]").replaceChildren(section);
    }

    function renderFileManagerRoute() {
        const view = document.querySelector("[data-route-view]");
        fileUi.renderFileManager(view);
    }

    function renderResumeEvaluationRoute() {
        const view = document.querySelector("[data-route-view]");
        businessRenderers.renderResumeEvaluationPage(view);
    }

    function renderResumeOptimizationRoute() {
        const view = document.querySelector("[data-route-view]");
        businessRenderers.renderResumeOptimizationPage(view);
    }

    function renderAlgorithmTestRoute() {
        const view = document.querySelector("[data-route-view]");
        businessRenderers.renderAlgorithmTestPage(view);
    }

    function renderMockInterviewRoute() {
        const view = document.querySelector("[data-route-view]");
        mockUi.renderMockInterviewPage(view);
    }

    function renderFailuresRoute() {
        const view = document.querySelector("[data-route-view]");
        adminUi.renderFailedTaskList(view);
    }

    function renderAdminFailureRoute(taskId) {
        const view = document.querySelector("[data-route-view]");
        adminUi.renderFailureDetail(view, decodeURIComponent(taskId));
    }

    function renderLlmConfigRoute() {
        const view = document.querySelector("[data-route-view]");
        llmConfigUi.renderLlmConfigPage(view);
    }

    function renderTaskListRoute() {
        const view = document.querySelector("[data-route-view]");
        taskUi.renderTaskList(view);
    }

    function renderTaskDetailRoute(taskId) {
        const view = document.querySelector("[data-route-view]");
        taskUi.renderTaskDetail(view, decodeURIComponent(taskId));
    }

    function renderRegister(route) {
        const section = createPanel(route);
        const form = createAuthForm("注册账号", "注册账号");
        const feedback = createFeedback();
        form.addEventListener("submit", async event => {
            event.preventDefault();
            const username = form.elements.username.value.trim();
            const password = form.elements.password.value;
            const confirmPassword = form.elements.confirm_password.value;
            if (password !== confirmPassword) {
                setText(feedback, "两次输入的密码不一致");
                return;
            }
            const submitButton = form.querySelector("button");
            submitButton.disabled = true;
            setText(feedback, "");
            try {
                await auth.register(username, password);
                setText(feedback, "注册成功，请登录");
                window.location.hash = "#/login";
            } catch (error) {
                setText(feedback, auth.sanitizeErrorMessage(error));
            } finally {
                submitButton.disabled = false;
            }
        });
        form.insertBefore(createField("confirm_password", "确认密码", "password"), form.querySelector("button"));
        section.append(form, feedback);
        document.querySelector("[data-route-view]").replaceChildren(section);
    }

    function createAuthForm(label, submitText) {
        const form = document.createElement("form");
        form.className = "auth-form";
        form.setAttribute("aria-label", label);
        form.append(createField("username", "用户名", "text"));
        form.append(createField("password", "密码", "password"));
        const button = document.createElement("button");
        button.type = "submit";
        setText(button, submitText);
        form.append(button);
        return form;
    }

    function createField(name, label, type) {
        const field = document.createElement("label");
        const text = document.createElement("span");
        const input = document.createElement("input");
        setText(text, label);
        input.name = name;
        input.type = type;
        input.required = true;
        input.autocomplete = name === "password" || name === "confirm_password" ? "current-password" : "username";
        field.append(text, input);
        return field;
    }

    function createFeedback() {
        const feedback = document.createElement("p");
        feedback.className = "form-feedback";
        feedback.setAttribute("role", "status");
        return feedback;
    }

    function renderForbidden() {
        renderPanel({
            eyebrow: "权限",
            title: "无权限",
            description: "当前账号无权访问该页面。",
            actions: [{ label: "返回任务中心", href: "#/tasks", kind: "secondary-action" }],
        });
    }

    function updateSessionNavigation(activeSession) {
        const adminLink = document.querySelector("[data-admin-link]");
        const role = activeSession ? activeSession.role : "";
        adminLink.hidden = role !== "ADMIN";

        document.querySelectorAll("[data-auth-anon]").forEach(link => {
            link.hidden = Boolean(activeSession);
        });

        const status = document.querySelector("#auth-status");
        status.replaceChildren();
        if (!activeSession) {
            const loginLink = document.createElement("a");
            loginLink.href = "#/login";
            setText(loginLink, "登录");
            status.append(loginLink);
            return;
        }

        const username = document.createElement("span");
        username.className = "session-user";
        setText(username, activeSession.username);

        const logoutButton = document.createElement("button");
        logoutButton.type = "button";
        setText(logoutButton, "登出");
        logoutButton.addEventListener("click", async () => {
            logoutButton.disabled = true;
            await auth.logout();
            updateSessionNavigation(null);
            window.location.hash = "#/login";
        });
        status.append(username, logoutButton);
    }

    function updateActiveNavigation(path) {
        document.querySelectorAll("[data-nav-link]").forEach(link => {
            const href = link.getAttribute("href") || "";
            link.toggleAttribute("aria-current", href === `#${path}`);
        });
    }

    function initRouter() {
        window.addEventListener("hashchange", renderRoute);
        window.addEventListener("ai-session-expired", () => {
            auth.clearSession();
            updateSessionNavigation(null);
            const path = currentPath();
            if (path !== "/login") {
                auth.saveRedirectTarget(path);
                window.location.hash = "#/login";
            }
        });
        renderRoute();
    }

    window.AiInterviewApp = Object.freeze({
        initRouter,
    });

    document.addEventListener("DOMContentLoaded", initRouter);
})();
