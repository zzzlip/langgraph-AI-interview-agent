package com.aiinterview.platform.application.port;

/**
 * 密码编码器接口。
 */
public interface PasswordEncoder {

    String encode(String rawPassword);

    boolean matches(String rawPassword, String encodedPassword);
}
