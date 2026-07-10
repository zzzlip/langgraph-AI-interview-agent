package com.aiinterview.platform.bootstrap;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.time.Duration;
import java.util.ArrayList;
import java.util.List;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.context.annotation.Configuration;
import org.springframework.http.CacheControl;
import org.springframework.util.StringUtils;
import org.springframework.web.servlet.config.annotation.ResourceHandlerRegistry;
import org.springframework.web.servlet.config.annotation.ViewControllerRegistry;
import org.springframework.web.servlet.config.annotation.WebMvcConfigurer;

/**
 * 静态前端同源资源挂载配置。
 */
@Configuration
public class FrontendStaticResourceConfig implements WebMvcConfigurer {

    private final List<String> resourceLocations;

    public FrontendStaticResourceConfig(
            @Value("${ai-interview.frontend.static-location:}") String configuredStaticLocation
    ) {
        this.resourceLocations = resolveResourceLocations(configuredStaticLocation);
    }

    @Override
    public void addViewControllers(ViewControllerRegistry registry) {
        registry.addViewController("/").setViewName("forward:/index.html");
    }

    @Override
    public void addResourceHandlers(ResourceHandlerRegistry registry) {
        registry.addResourceHandler("/index.html", "/favicon.ico")
                .addResourceLocations(resourceLocations.toArray(String[]::new))
                .setCacheControl(CacheControl.maxAge(Duration.ofMinutes(10)).cachePublic());
        registry.addResourceHandler("/css/**")
                .addResourceLocations(subdirectoryLocations("css").toArray(String[]::new))
                .setCacheControl(CacheControl.maxAge(Duration.ofMinutes(10)).cachePublic());
        registry.addResourceHandler("/js/**")
                .addResourceLocations(subdirectoryLocations("js").toArray(String[]::new))
                .setCacheControl(CacheControl.maxAge(Duration.ofMinutes(10)).cachePublic());
        registry.addResourceHandler("/assets/**")
                .addResourceLocations(subdirectoryLocations("assets").toArray(String[]::new))
                .setCacheControl(CacheControl.maxAge(Duration.ofMinutes(10)).cachePublic());
    }

    private static List<String> resolveResourceLocations(String configuredStaticLocation) {
        List<String> locations = new ArrayList<>();
        if (StringUtils.hasText(configuredStaticLocation)) {
            locations.add(toResourceLocation(Path.of(configuredStaticLocation)));
        }
        findLocalFrontendStaticDir().ifPresent(path -> {
            String location = toResourceLocation(path);
            if (!locations.contains(location)) {
                locations.add(location);
            }
        });
        locations.add("classpath:/static/");
        return locations;
    }

    private static java.util.Optional<Path> findLocalFrontendStaticDir() {
        Path userDir = Path.of(System.getProperty("user.dir")).toAbsolutePath().normalize();
        List<Path> candidates = List.of(
                userDir.resolve("frontend/static"),
                userDir.resolve("../frontend/static"),
                userDir.resolve("../../frontend/static")
        );
        return candidates.stream()
                .map(Path::normalize)
                .filter(Files::isDirectory)
                .findFirst();
    }

    private static String toResourceLocation(Path path) {
        Path normalized = path.toAbsolutePath().normalize();
        try {
            normalized = normalized.toRealPath();
        } catch (IOException ignored) {
            // 测试启动时目录可能由动态属性提前声明，资源处理器后续按规范路径解析。
        }
        String location = normalized.toUri().toString();
        return location.endsWith("/") ? location : location + "/";
    }

    private List<String> subdirectoryLocations(String subdirectory) {
        return resourceLocations.stream()
                .map(location -> location + subdirectory + "/")
                .toList();
    }
}
