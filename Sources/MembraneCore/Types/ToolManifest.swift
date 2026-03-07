public struct ToolManifest: Sendable, Equatable {
    public let name: String
    public let description: String
    public var fullSchema: String?

    public init(name: String, description: String, fullSchema: String? = nil) {
        self.name = name
        self.description = description
        self.fullSchema = fullSchema
    }

    public var estimatedTokens: Int {
        if let fullSchema {
            return estimateTokenCount(from: fullSchema)
        }

        return estimateTokenCount(from: name + description)
    }
}
