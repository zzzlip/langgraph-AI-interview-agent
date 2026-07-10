package com.aiinterview.platform.infrastructure.config;

import org.springframework.boot.context.properties.EnableConfigurationProperties;
import org.springframework.context.annotation.Configuration;
import org.mybatis.spring.annotation.MapperScan;

/**
 * 基础设施层配置装配类。
 */
@Configuration
@EnableConfigurationProperties(PlatformProperties.class)
@MapperScan("com.aiinterview.platform.infrastructure.persistence")
public class InfrastructureConfig {
}
