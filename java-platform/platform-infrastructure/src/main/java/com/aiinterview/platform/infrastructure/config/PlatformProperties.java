package com.aiinterview.platform.infrastructure.config;

import org.springframework.boot.context.properties.ConfigurationProperties;
import org.springframework.validation.annotation.Validated;
import jakarta.validation.constraints.Min;
import jakarta.validation.constraints.NotBlank;

/**
 * 平台全局配置绑定类。
 *
 * 配置来源优先级：启动参数 > 环境变量 > profile 配置 > application.yml
 * 关键配置缺失时，@Validated + @NotBlank 导致启动失败（fail-closed）。
 */
@ConfigurationProperties(prefix = "ai-interview")
@Validated
public class PlatformProperties {

    private Db db = new Db();
    private Redis redis = new Redis();
    private Rabbitmq rabbitmq = new Rabbitmq();
    private Minio minio = new Minio();
    private Security security = new Security();
    private AsyncMessage asyncMessage = new AsyncMessage();

    public Db getDb() { return db; }
    public void setDb(Db db) { this.db = db; }
    public Redis getRedis() { return redis; }
    public void setRedis(Redis redis) { this.redis = redis; }
    public Rabbitmq getRabbitmq() { return rabbitmq; }
    public void setRabbitmq(Rabbitmq rabbitmq) { this.rabbitmq = rabbitmq; }
    public Minio getMinio() { return minio; }
    public void setMinio(Minio minio) { this.minio = minio; }
    public Security getSecurity() { return security; }
    public void setSecurity(Security security) { this.security = security; }
    public AsyncMessage getAsyncMessage() { return asyncMessage; }
    public void setAsyncMessage(AsyncMessage asyncMessage) { this.asyncMessage = asyncMessage; }

    public static class Db {
        @NotBlank private String url = "";
        @NotBlank private String username = "";
        @NotBlank private String password = "";

        public String getUrl() { return url; }
        public void setUrl(String url) { this.url = url; }
        public String getUsername() { return username; }
        public void setUsername(String username) { this.username = username; }
        public String getPassword() { return password; }
        public void setPassword(String password) { this.password = password; }
    }

    public static class Redis {
        @NotBlank private String host = "";
        @Min(1) private int port = 6379;
        private String password = "";

        public String getHost() { return host; }
        public void setHost(String host) { this.host = host; }
        public int getPort() { return port; }
        public void setPort(int port) { this.port = port; }
        public String getPassword() { return password; }
        public void setPassword(String password) { this.password = password; }
    }

    public static class Rabbitmq {
        @NotBlank private String host = "";
        @Min(1) private int port = 5672;
        @NotBlank private String username = "";
        @NotBlank private String password = "";
        @NotBlank private String commandExchange = "ai.task.command";
        @NotBlank private String feedbackQueue = "ai.platform.task.feedback";
        private String virtualHost = "/";

        public String getHost() { return host; }
        public void setHost(String host) { this.host = host; }
        public int getPort() { return port; }
        public void setPort(int port) { this.port = port; }
        public String getUsername() { return username; }
        public void setUsername(String username) { this.username = username; }
        public String getPassword() { return password; }
        public void setPassword(String password) { this.password = password; }
        public String getCommandExchange() { return commandExchange; }
        public void setCommandExchange(String commandExchange) { this.commandExchange = commandExchange; }
        public String getFeedbackQueue() { return feedbackQueue; }
        public void setFeedbackQueue(String feedbackQueue) { this.feedbackQueue = feedbackQueue; }
        public String getVirtualHost() { return virtualHost; }
        public void setVirtualHost(String virtualHost) { this.virtualHost = virtualHost; }
    }

    public static class Minio {
        @NotBlank private String endpoint = "";
        @NotBlank private String accessKey = "";
        @NotBlank private String secretKey = "";

        public String getEndpoint() { return endpoint; }
        public void setEndpoint(String endpoint) { this.endpoint = endpoint; }
        public String getAccessKey() { return accessKey; }
        public void setAccessKey(String accessKey) { this.accessKey = accessKey; }
        public String getSecretKey() { return secretKey; }
        public void setSecretKey(String secretKey) { this.secretKey = secretKey; }
    }

    public static class Security {
        private String allowedOrigins = "";
        private String anonymousPaths = "/,/index.html,/css/**,/js/**,/assets/**,/favicon.ico,/api/v1/auth/register,/api/v1/auth/login,/api/v1/csrf-token,/actuator/health,/actuator/info";
        private int rateLimitWindowSeconds = 300;
        private int loginRateLimit = 20;
        private int registerRateLimit = 10;
        private int csrfTokenRateLimit = 30;

        public String getAllowedOrigins() { return allowedOrigins; }
        public void setAllowedOrigins(String allowedOrigins) { this.allowedOrigins = allowedOrigins; }
        public String getAnonymousPaths() { return anonymousPaths; }
        public void setAnonymousPaths(String anonymousPaths) { this.anonymousPaths = anonymousPaths; }
        public int getRateLimitWindowSeconds() { return rateLimitWindowSeconds; }
        public void setRateLimitWindowSeconds(int rateLimitWindowSeconds) { this.rateLimitWindowSeconds = rateLimitWindowSeconds; }
        public int getLoginRateLimit() { return loginRateLimit; }
        public void setLoginRateLimit(int loginRateLimit) { this.loginRateLimit = loginRateLimit; }
        public int getRegisterRateLimit() { return registerRateLimit; }
        public void setRegisterRateLimit(int registerRateLimit) { this.registerRateLimit = registerRateLimit; }
        public int getCsrfTokenRateLimit() { return csrfTokenRateLimit; }
        public void setCsrfTokenRateLimit(int csrfTokenRateLimit) { this.csrfTokenRateLimit = csrfTokenRateLimit; }
    }

    public static class AsyncMessage {
        @Min(1) private int publishBatchSize = 50;
        @Min(1) private int maxConnectionsPerUser = 5;
        @Min(1) private int maxConnectionsPerTask = 3;
        @Min(1) private int heartbeatSeconds = 15;

        public int getPublishBatchSize() { return publishBatchSize; }
        public void setPublishBatchSize(int publishBatchSize) { this.publishBatchSize = publishBatchSize; }
        public int getMaxConnectionsPerUser() { return maxConnectionsPerUser; }
        public void setMaxConnectionsPerUser(int maxConnectionsPerUser) { this.maxConnectionsPerUser = maxConnectionsPerUser; }
        public int getMaxConnectionsPerTask() { return maxConnectionsPerTask; }
        public void setMaxConnectionsPerTask(int maxConnectionsPerTask) { this.maxConnectionsPerTask = maxConnectionsPerTask; }
        public int getHeartbeatSeconds() { return heartbeatSeconds; }
        public void setHeartbeatSeconds(int heartbeatSeconds) { this.heartbeatSeconds = heartbeatSeconds; }
    }
}
