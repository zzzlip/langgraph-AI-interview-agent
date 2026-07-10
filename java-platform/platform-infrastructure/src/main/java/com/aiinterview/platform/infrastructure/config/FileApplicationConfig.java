package com.aiinterview.platform.infrastructure.config;

import com.aiinterview.platform.application.file.FileApplicationService;
import com.aiinterview.platform.application.file.FileAdminQueryService;
import com.aiinterview.platform.application.file.FileQueryService;
import com.aiinterview.platform.application.file.port.AdminFileStatusQueryPort;
import com.aiinterview.platform.application.file.port.AdminResultStatusQueryPort;
import com.aiinterview.platform.application.file.port.FileStoragePort;
import com.aiinterview.platform.application.result.ResultApplicationService;
import com.aiinterview.platform.application.result.ResultQueryService;
import com.aiinterview.platform.application.result.Spec04TaskResultReferencePort;
import com.aiinterview.platform.application.security.FileContentInspector;
import com.aiinterview.platform.application.task.port.AuditWritePort;
import com.aiinterview.platform.application.task.port.ResourceOwnershipPort;
import com.aiinterview.platform.application.task.port.TaskResultReferencePort;
import com.aiinterview.platform.domain.file.FileBindingRepository;
import com.aiinterview.platform.domain.file.FileRepository;
import com.aiinterview.platform.domain.result.TaskResultRepository;
import com.aiinterview.platform.infrastructure.storage.minio.MinioFileStorageAdapter;
import com.fasterxml.jackson.databind.ObjectMapper;
import io.minio.MinioClient;
import java.time.Clock;
import java.util.function.Supplier;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.boot.autoconfigure.condition.ConditionalOnMissingBean;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.context.annotation.Primary;

/**
 * 文件与结果资产应用服务装配。
 */
@Configuration
public class FileApplicationConfig {

    @Bean
    public FileContentInspector fileContentInspector() {
        return new FileContentInspector();
    }

    @Bean
    public FileApplicationService fileApplicationService(
            FileRepository fileRepository,
            FileBindingRepository fileBindingRepository,
            FileStoragePort fileStoragePort,
            FileContentInspector fileContentInspector,
            ResourceOwnershipPort resourceOwnershipPort,
            AuditWritePort auditWritePort,
            Supplier<String> idSupplier,
            Clock clock
    ) {
        return new FileApplicationService(
                fileRepository,
                fileBindingRepository,
                fileStoragePort,
                fileContentInspector,
                resourceOwnershipPort,
                auditWritePort,
                idSupplier,
                clock
        );
    }

    @Bean
    public FileQueryService fileQueryService(
            FileRepository fileRepository,
            FileBindingRepository fileBindingRepository,
            FileStoragePort fileStoragePort,
            ResourceOwnershipPort resourceOwnershipPort,
            AuditWritePort auditWritePort
    ) {
        return new FileQueryService(
                fileRepository,
                fileBindingRepository,
                fileStoragePort,
                resourceOwnershipPort,
                auditWritePort
        );
    }

    @Bean
    public FileAdminQueryService fileAdminQueryService(
            AdminFileStatusQueryPort fileStatusQueryPort,
            AdminResultStatusQueryPort resultStatusQueryPort,
            AuditWritePort auditWritePort
    ) {
        return new FileAdminQueryService(fileStatusQueryPort, resultStatusQueryPort, auditWritePort);
    }

    @Bean
    public ResultApplicationService resultApplicationService(
            TaskResultRepository taskResultRepository,
            FileRepository fileRepository,
            FileBindingRepository fileBindingRepository,
            AuditWritePort auditWritePort,
            Supplier<String> idSupplier,
            Clock clock,
            ObjectMapper objectMapper
    ) {
        return new ResultApplicationService(
                taskResultRepository,
                fileRepository,
                fileBindingRepository,
                auditWritePort,
                idSupplier,
                clock,
                objectMapper
        );
    }

    @Bean
    public ResultQueryService resultQueryService(
            TaskResultRepository taskResultRepository,
            FileRepository fileRepository,
            FileBindingRepository fileBindingRepository
    ) {
        return new ResultQueryService(taskResultRepository, fileBindingRepository, fileRepository);
    }

    @Bean
    @Primary
    public TaskResultReferencePort spec04TaskResultReferencePort(ResultApplicationService resultApplicationService) {
        return new Spec04TaskResultReferencePort(resultApplicationService);
    }

    @Bean
    @ConditionalOnMissingBean
    public FileStoragePort fileStoragePort(
            @Value("${ai-interview.minio.endpoint:}") String endpoint,
            @Value("${ai-interview.minio.access-key:}") String accessKey,
            @Value("${ai-interview.minio.secret-key:}") String secretKey,
            @Value("${ai-interview.minio.bucket:ai-interview-files}") String bucket
    ) {
        MinioClient minioClient = MinioClient.builder()
                .endpoint(endpoint)
                .credentials(accessKey, secretKey)
                .build();
        return new MinioFileStorageAdapter(minioClient, bucket);
    }
}
