import Foundation

/// One row of the RAM tier table.
struct ModelTier: Decodable, Hashable, Identifiable {
    let id: String
    let label: String
    let subtitle: String
    let minRAMGB: Int
    let annotationModel: String
    let sopModel: String
    /// The SOP model's A/B arm (`[vlm] sop_ab = true`); empty when none.
    let abSOPModel: String

    enum CodingKeys: String, CodingKey, CaseIterable {
        case id, label, subtitle
        case minRAMGB = "min_ram_gb"
        case annotationModel = "annotation_model"
        case sopModel = "sop_model"
        case abSOPModel = "ab_sop_model"
    }
}

/// The RAM tier table, read from `tiers.json`.
///
/// The table's one source is `TIERS` in the worker's `model_profiles.py`:
/// the installer runs `python -m agenthandover_worker.model_profiles
/// write-tiers` to write it into the data directory, and this type only
/// decodes that file, so onboarding cannot name other models than the
/// worker, the CLI and the installer do.
struct ModelTierTable: Decodable {
    let schemaVersion: Int
    let minimumOllamaVersionForGemma4: String
    let embeddingModel: String
    let modelSizesGB: [String: Double]
    /// Ascending by `minRAMGB`; the first row takes any Mac.
    let tiers: [ModelTier]

    enum CodingKeys: String, CodingKey {
        case schemaVersion = "schema_version"
        case minimumOllamaVersionForGemma4 = "minimum_ollama_version_for_gemma4"
        case embeddingModel = "embedding_model"
        case modelSizesGB = "model_sizes_gb"
        case tiers
    }

    enum LoadError: LocalizedError {
        case empty
        case unavailable(String)

        var errorDescription: String? {
            switch self {
            case .empty: return "tiers.json lists no tiers"
            case .unavailable(let why): return why
            }
        }
    }

    /// `~/Library/Application Support/agenthandover/tiers.json`.
    static var installedURL: URL {
        FileManager.default.homeDirectoryForCurrentUser
            .appendingPathComponent("Library/Application Support/agenthandover/tiers.json")
    }

    static func decode(_ data: Data) throws -> ModelTierTable {
        let table = try JSONDecoder().decode(ModelTierTable.self, from: data)
        guard !table.tiers.isEmpty else { throw LoadError.empty }
        return table
    }

    static func load(from url: URL) throws -> ModelTierTable {
        try decode(Data(contentsOf: url))
    }

    /// The installed table; when the file is missing (an install that
    /// predates it), the worker's venv python writes it first.  Blocking.
    static func loadInstalled(workerPython: URL) -> Result<ModelTierTable, Error> {
        let url = installedURL
        if !FileManager.default.fileExists(atPath: url.path) {
            let process = Process()
            process.executableURL = workerPython
            process.arguments = ["-m", "agenthandover_worker.model_profiles", "write-tiers", url.path]
            do {
                try process.run()
                process.waitUntilExit()
            } catch {
                return .failure(LoadError.unavailable(
                    "tiers.json is missing and the worker could not write it (\(error.localizedDescription))"))
            }
        }
        return Result { try load(from: url) }
    }

    func tier(id: String) -> ModelTier? {
        tiers.first { $0.id == id }
    }

    /// The largest tier whose `minRAMGB` fits `ramGB`.
    func recommended(forRAMGB ramGB: Int) -> ModelTier {
        tiers.last { $0.minRAMGB <= ramGB } ?? tiers[0]
    }

    /// The tiers a Mac with `ramGB` can run.
    func available(forRAMGB ramGB: Int) -> [ModelTier] {
        tiers.filter { $0.minRAMGB <= ramGB }
    }

    /// Every model `tier` needs pulled, annotation model first, no repeats.
    func models(for tier: ModelTier) -> [String] {
        var out: [String] = []
        for m in [tier.annotationModel, tier.sopModel, embeddingModel] where !out.contains(m) {
            out.append(m)
        }
        return out
    }

    /// "3.4 GB" / "270 MB"; nil for a model the table has no size for.
    func sizeLabel(_ model: String) -> String? {
        guard let gb = modelSizesGB[model] else { return nil }
        return gb < 1 ? "\(Int((gb * 1000).rounded())) MB" : String(format: "%.1f GB", gb)
    }

    /// "~11 GB": the download a tier needs.
    func downloadLabel(for tier: ModelTier) -> String {
        let total = models(for: tier).compactMap { modelSizesGB[$0] }.reduce(0, +)
        return "~\(Int(total.rounded())) GB"
    }

    /// True when `models` include a Gemma 4 tag, which needs a recent Ollama.
    func needsNewOllama(_ models: [String]) -> Bool {
        models.contains { $0.hasPrefix("gemma4") }
    }

    // MARK: - config.toml

    /// The keys of `[vlm]` a tier choice writes, in order.
    static let managedKeys = ["mode", "tier", "annotation_model", "sop_model"]

    /// The managed `[vlm]` lines; byte-identical to the worker's
    /// `model_profiles.vlm_block` and the CLI's `tiers::vlm_block`.
    static func vlmBlock(tierID: String, annotationModel: String, sopModel: String) -> String {
        "[vlm]\nmode = \"local\"\ntier = \"\(tierID)\"\n"
            + "annotation_model = \"\(annotationModel)\"\nsop_model = \"\(sopModel)\"\n"
    }

    /// The `[vlm]` lines of a remote (API) model choice.
    static let remoteKeys = ["mode", "provider", "model", "api_key_env"]

    static func remoteVLMBlock(provider: String, model: String, apiKeyEnv: String) -> String {
        "[vlm]\nmode = \"remote\"\nprovider = \"\(provider)\"\n"
            + "model = \"\(model)\"\napi_key_env = \"\(apiKeyEnv)\"\n"
    }

    /// `config` with the managed keys of `[vlm]` replaced by `block`'s
    /// (right after the header, other `[vlm]` keys and every other section
    /// kept), or `block` appended when there is no `[vlm]` section: the
    /// same edit as `agenthandover setup` (`tiers::apply_vlm_block`).
    /// `sectionHeader` edits another section the same way.
    static func applyVLMBlock(
        _ block: String, to config: String, managedKeys: [String] = managedKeys,
        sectionHeader: String = "[vlm]"
    ) -> String {
        var lines = config.components(separatedBy: "\n").map { $0 + "\n" }
        // The piece after the last "\n" has no terminator (and is dropped when empty).
        let last = lines.removeLast().dropLast()
        if !last.isEmpty { lines.append(String(last)) }

        let trim = { (s: String) in s.trimmingCharacters(in: .whitespacesAndNewlines) }
        guard let header = lines.firstIndex(where: { trim($0) == sectionHeader }) else {
            var out = config
            if !out.isEmpty && !out.hasSuffix("\n") { out += "\n" }
            return out + "\n" + block
        }
        var out = lines[..<header].joined() + block
        var inSection = true
        for line in lines[(header + 1)...] {
            let trimmed = trim(line)
            if trimmed.hasPrefix("[") { inSection = false }
            let key = trimmed.split(separator: "=", maxSplits: 1).first.map { trim(String($0)) }
            let managed = inSection && trimmed.contains("=") && managedKeys.contains(key ?? "")
            if !managed { out += line }
        }
        return out
    }
}
