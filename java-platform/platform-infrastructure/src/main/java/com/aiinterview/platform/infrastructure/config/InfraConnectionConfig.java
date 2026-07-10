package com.aiinterview.platform.infrastructure.config;

import org.springframework.amqp.rabbit.connection.CachingConnectionFactory;
import org.springframework.amqp.rabbit.connection.ConnectionFactory;
import org.springframework.amqp.rabbit.core.RabbitTemplate;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.data.redis.connection.RedisConnectionFactory;
import org.springframework.data.redis.connection.RedisStandaloneConfiguration;
import org.springframework.data.redis.connection.lettuce.LettuceConnectionFactory;

/**
 * 基础设施连接配置，将 ai-interview.rabbitmq.* / ai-interview.redis.* 映射为 Spring 自动配置所需 Bean。
 */
@Configuration
public class InfraConnectionConfig {

    @Bean
    public ConnectionFactory rabbitConnectionFactory(PlatformProperties properties) {
        PlatformProperties.Rabbitmq rmq = properties.getRabbitmq();
        CachingConnectionFactory factory = new CachingConnectionFactory(rmq.getHost(), rmq.getPort());
        factory.setUsername(rmq.getUsername());
        factory.setPassword(rmq.getPassword());
        String vhost = rmq.getVirtualHost();
        if (vhost != null && !vhost.isEmpty() && !"/".equals(vhost)) {
            factory.setVirtualHost(vhost);
        }
        return factory;
    }

    @Bean
    public RabbitTemplate rabbitTemplate(ConnectionFactory rabbitConnectionFactory) {
        return new RabbitTemplate(rabbitConnectionFactory);
    }

    @Bean
    public RedisConnectionFactory redisConnectionFactory(PlatformProperties properties) {
        PlatformProperties.Redis redis = properties.getRedis();
        RedisStandaloneConfiguration config = new RedisStandaloneConfiguration(redis.getHost(), redis.getPort());
        String password = redis.getPassword();
        if (password != null && !password.isEmpty()) {
            config.setPassword(password);
        }
        return new LettuceConnectionFactory(config);
    }
}
