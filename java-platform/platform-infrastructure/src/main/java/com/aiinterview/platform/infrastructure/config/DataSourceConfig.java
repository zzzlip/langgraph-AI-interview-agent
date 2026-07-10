package com.aiinterview.platform.infrastructure.config;

import javax.sql.DataSource;
import org.springframework.boot.context.properties.ConfigurationProperties;
import org.springframework.boot.jdbc.DataSourceBuilder;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;

/**
 * 数据源配置，将 ai-interview.db.* 映射为 spring.datasource.* 并创建 DataSource Bean。
 */
@Configuration
public class DataSourceConfig {

    @Bean
    @ConfigurationProperties(prefix = "ai-interview.db")
    public DataSource dataSource(PlatformProperties properties) {
        return DataSourceBuilder.create()
                .url(properties.getDb().getUrl())
                .username(properties.getDb().getUsername())
                .password(properties.getDb().getPassword())
                .driverClassName("com.mysql.cj.jdbc.Driver")
                .build();
    }
}
