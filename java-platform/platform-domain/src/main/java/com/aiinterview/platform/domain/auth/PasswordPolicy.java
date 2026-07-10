package com.aiinterview.platform.domain.auth;

/**
 * 密码策略。
 */
public final class PasswordPolicy {

    private static final int MIN_LENGTH = 8;
    private static final int MAX_LENGTH = 128;

    private PasswordPolicy() {
    }

    /**
     * 判断密码是否满足基础强度要求。
     */
    public static boolean isValid(String password) {
        if (password == null) {
            return false;
        }

        int length = password.length();
        if (length < MIN_LENGTH || length > MAX_LENGTH) {
            return false;
        }

        boolean hasUpperCase = false;
        boolean hasLowerCase = false;
        boolean hasDigit = false;

        for (int i = 0; i < length; i++) {
            char current = password.charAt(i);
            if (Character.isUpperCase(current)) {
                hasUpperCase = true;
            } else if (Character.isLowerCase(current)) {
                hasLowerCase = true;
            } else if (Character.isDigit(current)) {
                hasDigit = true;
            }
        }

        return hasUpperCase && hasLowerCase && hasDigit;
    }
}
