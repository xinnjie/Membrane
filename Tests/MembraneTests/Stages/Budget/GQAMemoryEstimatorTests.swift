import Testing
@testable import Membrane
@testable import MembraneCore

@Suite struct GQAMemoryEstimatorTests {
    @Test func setsKVSizingWhenArchitectureProvided() async throws {
        let architecture = ModelArchitectureInfo(
            numLayers: 32,
            numQueryHeads: 32,
            numKVHeads: 8,
            headDim: 128
        )
        let estimator = GQAMemoryEstimator(architecture: architecture, kvMemoryBudgetBytes: 512_000_000)

        let window = makeWindow()
        let budget = ContextBudget(totalTokens: 8_192, profile: .openModel8K)

        let result = try await estimator.process(window, budget: budget)

        #expect(result.budget.kvBytesPerToken == architecture.kvBytesPerToken)
        #expect(result.budget.kvMemoryBudgetBytes == 512_000_000)
        #expect(result.budget.maxSequenceLength == 3_906)
    }

    @Test func noopsWhenArchitectureMissing() async throws {
        let estimator = GQAMemoryEstimator(architecture: nil, kvMemoryBudgetBytes: 512_000_000)
        let window = makeWindow()
        let budget = ContextBudget(totalTokens: 4_096, profile: .foundationModels4K)

        let result = try await estimator.process(window, budget: budget)

        #expect(result.budget.kvBytesPerToken == nil)
        #expect(result.budget.kvMemoryBudgetBytes == nil)
        #expect(result.budget.maxSequenceLength == nil)
    }

    @Test func preservesBudgetAllocationsAndContextWindow() async throws {
        let architecture = ModelArchitectureInfo(
            numLayers: 40,
            numQueryHeads: 40,
            numKVHeads: 8,
            headDim: 128
        )
        let estimator = GQAMemoryEstimator(architecture: architecture, kvMemoryBudgetBytes: 600_000_000)

        var budget = ContextBudget(totalTokens: 8_192, profile: .openModel8K)
        try budget.allocate(250, to: .system)
        try budget.allocate(600, to: .history)

        let window = makeWindow(history: [slice("h0", tokens: 120, source: .history)])
        let result = try await estimator.process(window, budget: budget)

        #expect(result.budget.allocated(for: .system) == 250)
        #expect(result.budget.allocated(for: .history) == 600)
        #expect(result.window.history.map { $0.content } == ["h0"])
        #expect(result.window.systemPrompt.content == window.systemPrompt.content)
    }

    @Test func propagationIsDeterministicAcrossRuns() async throws {
        let architecture = ModelArchitectureInfo(
            numLayers: 24,
            numQueryHeads: 24,
            numKVHeads: 6,
            headDim: 128
        )

        func runOnce() async throws -> BudgetedContext {
            let estimator = GQAMemoryEstimator(architecture: architecture, kvMemoryBudgetBytes: 384_000_000)
            let window = makeWindow(history: [slice("h0", tokens: 50, source: .history)])
            let budget = ContextBudget(totalTokens: 8_192, profile: .openModel8K)
            return try await estimator.process(window, budget: budget)
        }

        let first = try await runOnce()
        let second = try await runOnce()

        #expect(first.budget.kvBytesPerToken == second.budget.kvBytesPerToken)
        #expect(first.budget.kvMemoryBudgetBytes == second.budget.kvMemoryBudgetBytes)
        #expect(first.budget.maxSequenceLength == second.budget.maxSequenceLength)
        #expect(first.window.history.map { $0.content } == second.window.history.map { $0.content })
    }

    private func makeWindow(history: [ContextSlice] = []) -> ContextWindow {
        ContextWindow(
            systemPrompt: slice("sys", tokens: 20, source: .system),
            memory: [],
            tools: [],
            toolPlan: .allowAll,
            history: history,
            retrieval: [],
            pointers: [],
            metadata: ContextMetadata()
        )
    }

    private func slice(_ content: String, tokens: Int, source: ContextSource) -> ContextSlice {
        ContextSlice(
            content: content,
            tokenCount: tokens,
            importance: 0.5,
            source: source,
            tier: .full,
            timestamp: .now
        )
    }
}
