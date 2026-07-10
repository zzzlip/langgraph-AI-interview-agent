package com.aiinterview.platform.infrastructure.storage.minio;

import com.aiinterview.platform.application.file.port.FileStoragePort;
import io.minio.BucketExistsArgs;
import io.minio.GetObjectArgs;
import io.minio.MakeBucketArgs;
import io.minio.MinioClient;
import io.minio.PutObjectArgs;
import java.io.InputStream;
import org.springframework.util.StringUtils;

/**
 * MinIO 文件存储适配器。
 */
public class MinioFileStorageAdapter implements FileStoragePort {

    private final MinioClient minioClient;
    private final String bucket;
    private volatile boolean bucketReady;

    public MinioFileStorageAdapter(MinioClient minioClient, String bucket) {
        this.minioClient = minioClient;
        this.bucket = bucket;
    }

    @Override
    public void store(String objectKey, InputStream content, long size) {
        ensureConfigured();
        ensureBucketExists();
        try {
            minioClient.putObject(PutObjectArgs.builder()
                    .bucket(bucket)
                    .object(objectKey)
                    .stream(content, size, -1)
                    .build());
        } catch (Exception ex) {
            throw new IllegalStateException("文件对象存储写入失败", ex);
        }
    }

    @Override
    public InputStream retrieve(String objectKey) {
        ensureConfigured();
        try {
            return minioClient.getObject(GetObjectArgs.builder()
                    .bucket(bucket)
                    .object(objectKey)
                    .build());
        } catch (Exception ex) {
            throw new IllegalStateException("文件对象存储读取失败", ex);
        }
    }

    private void ensureConfigured() {
        if (!StringUtils.hasText(bucket)) {
            throw new IllegalStateException("MinIO bucket 未配置");
        }
    }

    private void ensureBucketExists() {
        if (bucketReady) {
            return;
        }
        synchronized (this) {
            if (bucketReady) {
                return;
            }
            try {
                boolean exists = minioClient.bucketExists(BucketExistsArgs.builder().bucket(bucket).build());
                if (!exists) {
                    minioClient.makeBucket(MakeBucketArgs.builder().bucket(bucket).build());
                }
                bucketReady = true;
            } catch (Exception ex) {
                throw new IllegalStateException("MinIO bucket 初始化失败", ex);
            }
        }
    }
}
