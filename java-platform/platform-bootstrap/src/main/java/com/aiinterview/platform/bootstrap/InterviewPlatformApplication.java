package com.aiinterview.platform.bootstrap;

import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;

/**
 * AI 模拟面试平台 - Spring Boot 启动入口。
 *
 * 扫描路径覆盖所有模块的 com.aiinterview.platform 包。
 */
@SpringBootApplication(scanBasePackages = "com.aiinterview.platform")
public class InterviewPlatformApplication {

    public static void main(String[] args) {
        SpringApplication.run(InterviewPlatformApplication.class, args);
    }
}
