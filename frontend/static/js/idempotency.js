(() => {
    const activeOperationKeys = new Map();

    function createKey() {
        if (window.crypto && window.crypto.randomUUID) {
            return window.crypto.randomUUID();
        }
        return `idem-${Date.now()}-${Math.random().toString(16).slice(2)}`;
    }

    function keyFor(operationId) {
        const key = operationId || "default";
        if (!activeOperationKeys.has(key)) {
            activeOperationKeys.set(key, createKey());
        }
        return activeOperationKeys.get(key);
    }

    function clearOperationKey(operationId) {
        activeOperationKeys.delete(operationId || "default");
    }

    window.AiInterviewIdempotency = Object.freeze({
        clearOperationKey,
        keyFor,
    });
})();
