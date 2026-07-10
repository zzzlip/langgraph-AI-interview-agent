package com.aiinterview.platform.infrastructure.messaging.rabbitmq;

import java.util.ArrayList;
import java.util.List;
import org.springframework.amqp.core.Binding;
import org.springframework.amqp.core.BindingBuilder;
import org.springframework.amqp.core.Declarable;
import org.springframework.amqp.core.Declarables;
import org.springframework.amqp.core.DirectExchange;
import org.springframework.amqp.core.Queue;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;

/**
 * spec05 RabbitMQ 拓扑声明。
 */
@Configuration
public class RabbitMqTopologyConfig {

    public static final String TASK_COMMAND_EXCHANGE = "ai.task.command";
    public static final String TASK_FEEDBACK_EXCHANGE = "ai.task.feedback";
    public static final String TASK_DLX_EXCHANGE = "ai.task.dlx";
    public static final String WORKER_TASK_COMMAND_QUEUE = "ai.worker.task.command";
    public static final String PLATFORM_TASK_FEEDBACK_QUEUE = "ai.platform.task.feedback";
    public static final String PLATFORM_TASK_DEAD_LETTER_QUEUE = "ai.platform.task.dead-letter";

    private static final List<String> TASK_TYPES = List.of(
            "resume_evaluation",
            "resume_optimization",
            "algorithm_test",
            "mock_interview"
    );

    @Bean
    public Declarables rabbitMqDeclarables() {
        DirectExchange commandExchange = new DirectExchange(TASK_COMMAND_EXCHANGE, true, false);
        DirectExchange feedbackExchange = new DirectExchange(TASK_FEEDBACK_EXCHANGE, true, false);
        DirectExchange dlxExchange = new DirectExchange(TASK_DLX_EXCHANGE, true, false);

        Queue commandQueue = new Queue(WORKER_TASK_COMMAND_QUEUE, true);
        Queue feedbackQueue = new Queue(PLATFORM_TASK_FEEDBACK_QUEUE, true);
        Queue deadLetterQueue = new Queue(PLATFORM_TASK_DEAD_LETTER_QUEUE, true);

        List<Declarable> declarables = new ArrayList<>();
        declarables.add(commandExchange);
        declarables.add(feedbackExchange);
        declarables.add(dlxExchange);
        declarables.add(commandQueue);
        declarables.add(feedbackQueue);
        declarables.add(deadLetterQueue);
        TASK_TYPES.forEach(taskType -> {
            declarables.add(BindingBuilder.bind(commandQueue)
                    .to(commandExchange)
                    .with("task." + taskType + ".command"));
            declarables.add(BindingBuilder.bind(feedbackQueue)
                    .to(feedbackExchange)
                    .with("task." + taskType + ".feedback"));
            declarables.add(BindingBuilder.bind(deadLetterQueue)
                    .to(dlxExchange)
                    .with("task." + taskType + ".dead"));
        });
        return new Declarables(declarables);
    }
}
