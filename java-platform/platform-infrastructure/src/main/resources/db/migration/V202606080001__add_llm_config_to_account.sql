-- 在 account 表添加用户级 LLM 配置字段
ALTER TABLE `account`
    ADD COLUMN `llm_config` JSON DEFAULT NULL COMMENT '用户 DeepSeek API 配置：{provider, base_url, model, api_key}' AFTER `status`;
