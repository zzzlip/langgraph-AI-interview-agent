(() => {
    const statusMap = {
        pending: { label: "等待处理", category: "neutral" },
        queued: { label: "排队中", category: "neutral" },
        running: { label: "处理中", category: "active" },
        waiting_user_input: { label: "等待您的输入", category: "warning" },
        success: { label: "已完成", category: "success" },
        failed: { label: "失败", category: "danger" },
        cancelled: { label: "已取消", category: "muted" },
    };

    const typeMap = {
        resume_evaluation: "简历评估",
        resume_optimization: "简历优化",
        algorithm_test: "算法测试",
        mock_interview: "模拟面试",
    };

    const stageMap = {
        task_created: "任务已创建",
        task_queued: "任务排队中",
        worker_started: "开始处理",
        file_uploaded: "文件已上传",
        result_generated: "结果已生成",
        waiting_user_input: "等待您的输入",
        task_completed: "任务完成",
        task_failed: "任务失败",
    };

    function mapTaskStatus(status) {
        return statusMap[status] || { label: "未知状态", category: "neutral" };
    }

    function mapTaskType(type) {
        return typeMap[type] || type || "未知任务";
    }

    function mapStage(stage) {
        if (!stage) {
            return "处理中";
        }
        return stageMap[stage] || "处理中";
    }

    function isTerminalStatus(status) {
        return status === "success" || status === "failed" || status === "cancelled";
    }

    window.AiInterviewStateMapping = Object.freeze({
        isTerminalStatus,
        mapStage,
        mapTaskStatus,
        mapTaskType,
    });
})();
