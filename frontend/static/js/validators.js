(() => {
    // 校验 "title" 和 "description" 为必填字段
    function validateTargetJob(job) {
        const errors = [];
        if (!job || !job.title || !job.title.trim()) {
            errors.push("目标岗位名称不能为空");
        }
        if (job && job.title && job.title.length > 100) {
            errors.push("目标岗位名称不能超过100个字符");
        }
        if (!job || !job.description || !job.description.trim()) {
            errors.push("目标岗位描述不能为空");
        }
        if (job && job.description && job.description.length > 5000) {
            errors.push("目标岗位描述不能超过5000个字符");
        }
        return errors;
    }

    // Codeforces handle 校验
    function validateCodeforcesHandle(handle) {
        const errors = [];
        if (!handle || !handle.trim()) {
            errors.push("Codeforces handle 不能为空");
            return errors;
        }
        const trimmed = handle.trim();
        if (trimmed.length < 3 || trimmed.length > 40) {
            errors.push("Codeforces handle 长度应在 3-40 个字符之间");
        }
        if (!/^[a-zA-Z0-9._-]+$/.test(trimmed)) {
            errors.push("Codeforces handle 只能包含字母、数字、点、下划线和连字符");
        }
        return errors;
    }

    // 模拟面试字段校验
    function validateInterviewConfig(config) {
        const errors = [];
        if (config.question_count !== undefined) {
            const count = parseInt(config.question_count, 10);
            if (isNaN(count) || count < 4 || count > 8) {
                errors.push("题目数量应在 4-8 之间");
            }
        }
        if (config.max_follow_up_rounds !== undefined) {
            const rounds = parseInt(config.max_follow_up_rounds, 10);
            if (isNaN(rounds) || rounds < 1 || rounds > 3) {
                errors.push("最大追问轮数应在 1-3 之间");
            }
        }
        return errors;
    }

    // 模拟面试回答文件校验
    function validateInterviewAnswer(answer) {
        const errors = [];
        if (!answer || !answer.question_id) {
            errors.push("题目 ID 不能为空");
        }
        if (!answer || !answer.answer_file_id) {
            errors.push("请选择或录制回答文件");
        }
        return errors;
    }

    window.AiInterviewValidators = {
        validateCodeforcesHandle,
        validateInterviewAnswer,
        validateInterviewConfig,
        validateTargetJob,
    };
})();
