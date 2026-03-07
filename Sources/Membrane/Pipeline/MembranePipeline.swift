import MembraneCore

public enum PipelineMode: Sendable {
    case full
    case budgetOnly
}

public actor MembranePipeline {
    private let baseBudget: ContextBudget
    private let intakeStage: (any IntakeStage)?
    private let allocatorStage: (any BudgetStage)?
    private let compressStage: (any CompressStage)?
    private let pageStage: (any PageStage)?
    private let emitStage: (any EmitStage)?
    private let mode: PipelineMode
    private let stageTimeout: Duration?

    public init(
        budget: ContextBudget,
        intake: (any IntakeStage)? = nil,
        allocator: (any BudgetStage)? = nil,
        compress: (any CompressStage)? = nil,
        page: (any PageStage)? = nil,
        emit: (any EmitStage)? = nil,
        mode: PipelineMode = .full,
        stageTimeout: Duration? = nil
    ) {
        self.baseBudget = budget
        self.intakeStage = intake
        self.allocatorStage = allocator
        self.compressStage = compress
        self.pageStage = page
        self.emitStage = emit
        self.mode = mode
        self.stageTimeout = stageTimeout
    }

    public static func foundationModel(
        budget: ContextBudget = ContextBudget(totalTokens: 4096, profile: .foundationModels4K),
        intake: (any IntakeStage)? = nil,
        allocator: (any BudgetStage)? = nil,
        compress: (any CompressStage)? = nil,
        page: (any PageStage)? = nil,
        emit: (any EmitStage)? = nil,
        stageTimeout: Duration? = nil
    ) -> MembranePipeline {
        MembranePipeline(
            budget: budget,
            intake: intake,
            allocator: allocator,
            compress: compress,
            page: page,
            emit: emit,
            mode: .budgetOnly,
            stageTimeout: stageTimeout
        )
    }

    public static func openModel(
        budget: ContextBudget,
        intake: (any IntakeStage)? = nil,
        allocator: (any BudgetStage)? = nil,
        compress: (any CompressStage)? = nil,
        page: (any PageStage)? = nil,
        emit: (any EmitStage)? = nil,
        stageTimeout: Duration? = nil
    ) -> MembranePipeline {
        MembranePipeline(
            budget: budget,
            intake: intake,
            allocator: allocator,
            compress: compress,
            page: page,
            emit: emit,
            mode: .full,
            stageTimeout: stageTimeout
        )
    }

    private func runWithTimeout<T: Sendable>(
        stage stageName: String,
        _ work: @Sendable () async throws -> T
    ) async throws -> T {
        guard let stageTimeout else {
            return try await work()
        }
        let start = ContinuousClock.now
        return try await withThrowingTaskGroup(of: T.self) { group in
            group.addTask { try await work() }
            group.addTask {
                try await Task.sleep(for: stageTimeout)
                throw MembraneError.stageTimeout(stage: stageName, elapsed: ContinuousClock.now - start)
            }
            guard let result = try await group.next() else {
                throw MembraneError.stageTimeout(stage: stageName, elapsed: ContinuousClock.now - start)
            }
            group.cancelAll()
            return result
        }
    }

    public func prepare(_ request: ContextRequest) async throws -> PlannedRequest {
        var budget = baseBudget

        var window = ContextWindow(
            systemPrompt: ContextSlice(
                content: "",
                tokenCount: 0,
                importance: 1.0,
                source: .system,
                tier: .full,
                timestamp: .now
            ),
            memory: request.memories,
            tools: request.tools,
            toolPlan: request.toolPlan,
            history: request.history,
            retrieval: request.retrieval,
            pointers: request.pointers,
            metadata: ContextMetadata(modelProfile: baseBudget.profile)
        )

        if let intakeStage {
            try Task.checkCancellation()
            let currentBudget = budget
            window = try await runWithTimeout(stage: "intake") {
                try await intakeStage.process(request, budget: currentBudget)
            }
        }

        var budgeted = BudgetedContext(window: window, budget: budget)
        if let allocatorStage {
            try Task.checkCancellation()
            let input = budgeted
            budgeted = try await runWithTimeout(stage: "budget") {
                try await allocatorStage.process(input.window, budget: input.budget)
            }
        }
        budget = budgeted.budget

        var compressed = CompressedContext(
            window: budgeted.window,
            budget: budgeted.budget,
            compressionReport: CompressionReport(
                originalTokens: budgeted.window.totalTokenCount,
                compressedTokens: budgeted.window.totalTokenCount,
                techniquesApplied: []
            )
        )
        if let compressStage {
            try Task.checkCancellation()
            let input = BudgetedContext(window: compressed.window, budget: compressed.budget)
            let currentBudget = compressed.budget
            compressed = try await runWithTimeout(stage: "compress") {
                try await compressStage.process(input, budget: currentBudget)
            }
        }
        budget = compressed.budget

        var paged = PagedContext(window: compressed.window, budget: compressed.budget, pagedOut: [])
        if mode == .full, let pageStage {
            try Task.checkCancellation()
            let input = CompressedContext(
                window: paged.window,
                budget: paged.budget,
                compressionReport: compressed.compressionReport
            )
            let currentBudget = paged.budget
            paged = try await runWithTimeout(stage: "page") {
                try await pageStage.process(input, budget: currentBudget)
            }
        }
        budget = paged.budget

        var plannedRequest = PlannedRequest(
            prompt: request.userInput,
            systemPrompt: paged.window.systemPrompt.content,
            toolPlan: paged.window.toolPlan,
            budget: budget,
            metadata: paged.window.metadata
        )
        if mode == .full, let emitStage {
            try Task.checkCancellation()
            let input = paged
            let currentBudget = budget
            plannedRequest = try await runWithTimeout(stage: "emit") {
                try await emitStage.process(input, budget: currentBudget)
            }
        }

        return plannedRequest
    }
}
