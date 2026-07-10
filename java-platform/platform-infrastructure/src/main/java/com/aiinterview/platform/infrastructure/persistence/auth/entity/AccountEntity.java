package com.aiinterview.platform.infrastructure.persistence.auth.entity;

import com.baomidou.mybatisplus.annotation.TableField;
import com.baomidou.mybatisplus.annotation.TableId;
import com.baomidou.mybatisplus.annotation.TableName;
import com.baomidou.mybatisplus.extension.handlers.JacksonTypeHandler;
import java.time.Instant;

/**
 * 账号数据库实体。
 */
@TableName(value = "`account`", autoResultMap = true)
public class AccountEntity {

    @TableId
    private String id;

    private String username;

    @TableField("password_hash")
    private String passwordHash;

    private String role;

    private String status;

    @TableField("created_at")
    private Instant createdAt;

    @TableField("updated_at")
    private Instant updatedAt;

    @TableField("deleted_at")
    private Instant deletedAt;

    /** 用户级 LLM 配置 JSON 字段，通过 JacksonTypeHandler 序列化。 */
    @TableField(value = "llm_config", typeHandler = JacksonTypeHandler.class)
    private java.util.Map<String, Object> llmConfig;

    public String getId() {
        return id;
    }

    public void setId(String id) {
        this.id = id;
    }

    public String getUsername() {
        return username;
    }

    public void setUsername(String username) {
        this.username = username;
    }

    public String getPasswordHash() {
        return passwordHash;
    }

    public void setPasswordHash(String passwordHash) {
        this.passwordHash = passwordHash;
    }

    public String getRole() {
        return role;
    }

    public void setRole(String role) {
        this.role = role;
    }

    public String getStatus() {
        return status;
    }

    public void setStatus(String status) {
        this.status = status;
    }

    public Instant getCreatedAt() {
        return createdAt;
    }

    public void setCreatedAt(Instant createdAt) {
        this.createdAt = createdAt;
    }

    public Instant getUpdatedAt() {
        return updatedAt;
    }

    public void setUpdatedAt(Instant updatedAt) {
        this.updatedAt = updatedAt;
    }

    public Instant getDeletedAt() {
        return deletedAt;
    }

    public void setDeletedAt(Instant deletedAt) {
        this.deletedAt = deletedAt;
    }

    public java.util.Map<String, Object> getLlmConfig() {
        return llmConfig;
    }

    public void setLlmConfig(java.util.Map<String, Object> llmConfig) {
        this.llmConfig = llmConfig;
    }
}
