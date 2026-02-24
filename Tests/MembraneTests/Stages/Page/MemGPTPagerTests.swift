import Testing
@testable import Membrane
@testable import MembraneCore

@Suite struct MemGPTPagerTests {
    @Test func pagesWhenTokenPressureExceedsThreshold() async throws {
        let pager = MemGPTPager(pressureThreshold: 0.5, keepRecentHistoryTurns: 1)

        let window = makeWindow(
            system: [slice("sys", tokens: 10, importance: 1.0, source: .system)],
            retrieval: [slice("ret0", tokens: 20, importance: 0.1, source: .retrieval)],
            memory: [
                slice("mem0", tokens: 20, importance: 0.2, source: .memory),
                slice("mem1", tokens: 20, importance: 0.4, source: .memory),
            ],
            history: [
                slice("hist0", tokens: 20, importance: 0.3, source: .history),
                slice("hist1", tokens: 20, importance: 0.9, source: .history),
            ]
        )

        let budget = ContextBudget(totalTokens: 100, profile: .custom(buckets: [
            .system: 20,
            .history: 40,
            .memory: 40,
            .retrieval: 40,
            .tools: 0,
            .toolIO: 0,
            .outputReserve: 0,
            .protocolOverhead: 0,
            .safetyMargin: 0,
        ]))

        let input = CompressedContext(
            window: window,
            budget: budget,
            compressionReport: CompressionReport(originalTokens: window.totalTokenCount, compressedTokens: window.totalTokenCount, techniquesApplied: [])
        )

        let result = try await pager.process(input, budget: budget)

        #expect(result.window.totalTokenCount <= 50)
        #expect(result.pagedOut.map(\.content) == ["ret0", "mem0", "hist0"])
        #expect(result.window.history.map(\.content) == ["hist1"])
    }

    @Test func preservesMostRecentHistoryTurns() async throws {
        let pager = MemGPTPager(pressureThreshold: 0.6, keepRecentHistoryTurns: 2)

        let window = makeWindow(
            system: [slice("sys", tokens: 10, importance: 1.0, source: .system)],
            history: [
                slice("h0", tokens: 20, importance: 0.1, source: .history),
                slice("h1", tokens: 20, importance: 0.2, source: .history),
                slice("h2", tokens: 20, importance: 0.3, source: .history),
                slice("h3", tokens: 20, importance: 0.4, source: .history),
            ]
        )

        let budget = ContextBudget(totalTokens: 100, profile: .custom(buckets: [
            .system: 20,
            .history: 80,
            .memory: 0,
            .retrieval: 0,
            .tools: 0,
            .toolIO: 0,
            .outputReserve: 0,
            .protocolOverhead: 0,
            .safetyMargin: 0,
        ]))

        let input = CompressedContext(
            window: window,
            budget: budget,
            compressionReport: CompressionReport(originalTokens: window.totalTokenCount, compressedTokens: window.totalTokenCount, techniquesApplied: [])
        )

        let result = try await pager.process(input, budget: budget)

        #expect(result.window.totalTokenCount <= 60)
        #expect(result.pagedOut.map(\.content) == ["h0", "h1"])
        #expect(result.window.history.map(\.content) == ["h2", "h3"])
    }

    @Test func evictionOrderIsDeterministicAcrossRuns() async throws {
        func runOnce() async throws -> PagedContext {
            let pager = MemGPTPager(pressureThreshold: 0.5, keepRecentHistoryTurns: 0)

            let window = makeWindow(
                system: [slice("sys", tokens: 10, importance: 1.0, source: .system)],
                retrieval: [
                    slice("r0", tokens: 20, importance: 0.2, source: .retrieval),
                    slice("r1", tokens: 20, importance: 0.2, source: .retrieval),
                ],
                memory: [
                    slice("m0", tokens: 20, importance: 0.2, source: .memory),
                    slice("m1", tokens: 20, importance: 0.2, source: .memory),
                ],
                history: [
                    slice("h0", tokens: 20, importance: 0.2, source: .history),
                    slice("h1", tokens: 20, importance: 0.2, source: .history),
                ]
            )

            let budget = ContextBudget(totalTokens: 100, profile: .custom(buckets: [
                .system: 20,
                .history: 40,
                .memory: 40,
                .retrieval: 40,
                .tools: 0,
                .toolIO: 0,
                .outputReserve: 0,
                .protocolOverhead: 0,
                .safetyMargin: 0,
            ]))

            let input = CompressedContext(
                window: window,
                budget: budget,
                compressionReport: CompressionReport(originalTokens: window.totalTokenCount, compressedTokens: window.totalTokenCount, techniquesApplied: [])
            )

            return try await pager.process(input, budget: budget)
        }

        let first = try await runOnce()
        let second = try await runOnce()

        #expect(first.pagedOut.map(\.content) == ["r0", "r1", "m0", "m1"])
        #expect(first.pagedOut.map(\.content) == second.pagedOut.map(\.content))
    }

    @Test func returnsUnchangedWindowWhenUnderThreshold() async throws {
        let pager = MemGPTPager(pressureThreshold: 0.9, keepRecentHistoryTurns: 1)
        let window = makeWindow(
            system: [slice("sys", tokens: 10, importance: 1.0, source: .system)],
            history: [slice("h0", tokens: 20, importance: 0.5, source: .history)]
        )

        let budget = ContextBudget(totalTokens: 100, profile: .foundationModels4K)
        let input = CompressedContext(
            window: window,
            budget: budget,
            compressionReport: CompressionReport(originalTokens: window.totalTokenCount, compressedTokens: window.totalTokenCount, techniquesApplied: [])
        )

        let result = try await pager.process(input, budget: budget)

        #expect(result.pagedOut.isEmpty)
        #expect(result.window.totalTokenCount == window.totalTokenCount)
        #expect(result.window.history.map(\.content) == ["h0"])
    }

    private func makeWindow(
        system: [ContextSlice] = [],
        retrieval: [ContextSlice] = [],
        memory: [ContextSlice] = [],
        history: [ContextSlice] = []
    ) -> ContextWindow {
        ContextWindow(
            systemPrompt: system.first ?? slice("", tokens: 0, importance: 1.0, source: .system),
            memory: memory,
            tools: [],
            toolPlan: .allowAll,
            history: history,
            retrieval: retrieval,
            pointers: [],
            metadata: ContextMetadata()
        )
    }

    private func slice(
        _ content: String,
        tokens: Int,
        importance: Double,
        source: ContextSource
    ) -> ContextSlice {
        ContextSlice(
            content: content,
            tokenCount: tokens,
            importance: importance,
            source: source,
            tier: .full,
            timestamp: .now
        )
    }
}
