package com.aiinterview.platform.infrastructure.persistence.result;

import com.aiinterview.platform.application.file.dto.AdminResultStatusItem;
import com.aiinterview.platform.application.file.dto.AdminResultStatusPageResult;
import com.aiinterview.platform.application.file.port.AdminResultStatusQuery;
import com.aiinterview.platform.application.file.port.AdminResultStatusQueryPort;
import com.aiinterview.platform.domain.file.FileBindType;
import com.aiinterview.platform.infrastructure.persistence.file.FileBindingMapper;
import com.aiinterview.platform.infrastructure.persistence.file.entity.FileBindingEntity;
import com.aiinterview.platform.infrastructure.persistence.result.entity.TaskResultEntity;
import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import java.util.List;
import org.springframework.stereotype.Component;

/**
 * 管理员结果状态查询基础设施适配器。
 */
@Component
public class AdminResultStatusQueryAdapter implements AdminResultStatusQueryPort {

    private final TaskResultMapper taskResultMapper;
    private final FileBindingMapper bindingMapper;

    public AdminResultStatusQueryAdapter(TaskResultMapper taskResultMapper, FileBindingMapper bindingMapper) {
        this.taskResultMapper = taskResultMapper;
        this.bindingMapper = bindingMapper;
    }

    @Override
    public AdminResultStatusPageResult search(AdminResultStatusQuery query) {
        long total = taskResultMapper.selectCount(wrapper(query));
        List<AdminResultStatusItem> items = taskResultMapper.selectList(wrapper(query)
                        .orderByDesc(TaskResultEntity::getCreatedAt)
                        .orderByDesc(TaskResultEntity::getId)
                        .last("LIMIT " + query.pageSize() + " OFFSET " + offset(query)))
                .stream()
                .map(this::toItem)
                .toList();
        return new AdminResultStatusPageResult(items, total, query.page(), query.pageSize());
    }

    private LambdaQueryWrapper<TaskResultEntity> wrapper(AdminResultStatusQuery query) {
        LambdaQueryWrapper<TaskResultEntity> wrapper = new LambdaQueryWrapper<>();
        if (query.taskId() != null) {
            wrapper.eq(TaskResultEntity::getTaskId, query.taskId());
        }
        if (query.resultType() != null) {
            wrapper.eq(TaskResultEntity::getResultType, query.resultType());
        }
        if (query.attemptNo() != null) {
            wrapper.eq(TaskResultEntity::getAttemptNo, query.attemptNo());
        }
        return wrapper;
    }

    private AdminResultStatusItem toItem(TaskResultEntity result) {
        return new AdminResultStatusItem(
                result.getId(),
                result.getTaskId(),
                result.getAttemptNo(),
                result.getResultType(),
                result.getResultStatus(),
                countFiles(result, FileBindType.REPORT),
                countFiles(result, FileBindType.OUTPUT),
                result.getCreatedAt(),
                result.getCompletedAt()
        );
    }

    private long countFiles(TaskResultEntity result, FileBindType bindType) {
        return bindingMapper.selectCount(new LambdaQueryWrapper<FileBindingEntity>()
                .eq(FileBindingEntity::getTaskId, result.getTaskId())
                .eq(FileBindingEntity::getAttemptNo, result.getAttemptNo())
                .eq(FileBindingEntity::getBindType, bindType.getCode()));
    }

    private static int offset(AdminResultStatusQuery query) {
        return (query.page() - 1) * query.pageSize();
    }
}
