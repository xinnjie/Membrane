// MembraneCore — Protocols, value types, and budget algebra for context management.

/// Rough heuristic for token estimation from text length.
/// Approximates ~4 characters per token for Latin text.
public func estimateTokenCount(from text: String) -> Int {
    max(text.count / 4, 1)
}
