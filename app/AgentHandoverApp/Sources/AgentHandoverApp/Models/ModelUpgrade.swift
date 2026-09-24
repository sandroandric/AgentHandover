import Foundation

/// The model switch an upgrade leaves for the app (WP-F16): a config still
/// on earlier default models (v0.4.0's, or a tier's earlier pair) moves to
/// its tier's current models only once they are pulled; until then the Mac
/// keeps the models it has.  The worker's `model_profiles upgrade-plan`
/// says what the switch is, `upgrade-apply` makes it (it refuses while
/// Ollama lacks a model), and the worker makes it at startup too.
struct ModelUpgradePlan: Decodable, Equatable {
    let tier: String
    let label: String
    let current: ModelPair
    let target: ModelPair
    /// The target pair's models, annotation model first.
    let models: [String]
    let sizesGB: [String: Double]

    enum CodingKeys: String, CodingKey {
        case tier, label, current, target, models
        case sizesGB = "sizes_gb"
    }
}

/// What the app does about a plan at launch.
enum ModelUpgradeDecision: Equatable {
    /// No switch to make, Ollama cannot be asked, or the user said
    /// "Not now" for this version.
    case none
    /// Every target model is pulled: switch without asking.
    case applySilently
    /// Ask to download these models, then switch.
    case ask(missing: [String])
}

enum ModelUpgrade {

    /// UserDefaults: the app version whose ask the user declined ("Not
    /// now"); the next upgrade asks again.
    static let declinedVersionKey = "modelUpgradeDeclinedVersion"

    /// The decision at launch.  `installed` is nil when Ollama did not
    /// answer (nothing can be pulled then; the worker keeps the models).
    static func decide(
        plan: ModelUpgradePlan?, installed: Set<String>?,
        declinedVersion: String?, appVersion: String
    ) -> ModelUpgradeDecision {
        guard let plan, let installed else { return .none }
        let missing = OllamaModels.missing(plan.models, installed: installed)
        if missing.isEmpty { return .applySilently }
        if declinedVersion == appVersion { return .none }
        return .ask(missing: missing)
    }

    /// The menu row while the worker cannot process: "AI model missing:
    /// <name>. Download…", or nil.  The worker writes `missing_models` only
    /// when Ollama answered, so a stopped Ollama is not called a missing model.
    static func missingModelRow(vlmAvailable: Bool, missingModels: [String]?) -> String? {
        guard !vlmAvailable, let first = missingModels?.first else { return nil }
        return "AI model missing: \(first). Download\u{2026}"
    }

    /// "~3.4 GB" for `models` by the plan's sizes (a model without a size
    /// counts as nothing).
    static func downloadLabel(_ models: [String], sizesGB: [String: Double]) -> String {
        let total = models.compactMap { sizesGB[$0] }.reduce(0, +)
        return String(format: "~%.1f GB", total)
    }

    /// The window's question.
    static func question(plan: ModelUpgradePlan?, missing: [String]) -> String {
        let names = missing.joined(separator: " and ")
        guard let plan else {
            return "AgentHandover needs \(names) to process your recordings. Download now?"
        }
        let size = downloadLabel(missing, sizesGB: plan.sizesGB)
        let noun = missing.count == 1 ? "a faster model" : "faster models"
        return "0.5 uses \(noun) (\(names), \(size)). Download now?"
    }

    /// `{"plan": ..., "reason": ...}` from `upgrade-plan`; nil plan when
    /// there is none or the output is not that object.
    static func decodePlan(_ output: String) -> ModelUpgradePlan? {
        struct Reply: Decodable { let plan: ModelUpgradePlan? }
        guard let data = output.data(using: .utf8) else { return nil }
        return (try? JSONDecoder().decode(Reply.self, from: data))?.plan
    }

    /// This app's version (Info.plist); a declined ask holds for it.
    static var appVersion: String {
        Bundle.main.infoDictionary?["CFBundleShortVersionString"] as? String ?? ""
    }

    // MARK: - The worker's commands

    static var configURL: URL {
        FileManager.default.homeDirectoryForCurrentUser
            .appendingPathComponent("Library/Application Support/agenthandover/config.toml")
    }

    static var workerPython: URL {
        ServiceController.daemonExecutableURL.deletingLastPathComponent()
            .appendingPathComponent("venv/bin/python")
    }

    /// `model_profiles upgrade-plan` on the config.  Blocking.
    static func loadPlan() -> ModelUpgradePlan? {
        decodePlan(ServiceController.shell(
            workerPython.path,
            args: ["-m", "agenthandover_worker.model_profiles", "upgrade-plan", configURL.path]
        ))
    }

    /// `model_profiles upgrade-apply` on the config (backup first; it
    /// refuses while Ollama lacks a model), then a worker restart so it
    /// runs the new models.  Blocking.  True when the switch was made.
    @discardableResult
    static func apply() -> Bool {
        let process = Process()
        process.executableURL = workerPython
        process.arguments = ["-m", "agenthandover_worker.model_profiles", "upgrade-apply", configURL.path]
        process.standardOutput = FileHandle.nullDevice
        process.standardError = FileHandle.nullDevice
        do {
            try process.run()
        } catch {
            return false
        }
        process.waitUntilExit()
        guard process.terminationStatus == 0 else { return false }
        ServiceController.restartWorker()
        return true
    }

    /// The installed models, or nil when Ollama does not answer.  Blocking.
    static func installedModels() -> Set<String>? {
        guard let path = OllamaModels.executablePath() else { return nil }
        return try? OllamaModels.installed(ollamaPath: path).get()
    }
}
