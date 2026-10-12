import Foundation

/// `<data dir>/memory-offer.json` (`model_profiles.MEMORY_OFFER_FILE`):
/// what the upgrade offer of Memory knows and what the user chose. The
/// worker's migration writes `wanted_visual_search`; the app adds
/// `offered_at` and `choice`; every other key is kept.
struct MemoryOfferState: Equatable {
    enum Choice: String {
        /// Memory on; Skill search and the other text paths keep their model.
        case memoryOnly = "memory_only"
        /// Memory on, and the worker re-embeds everything with Memory's
        /// model in the background (WP-E9 reads this).
        case everything
        case declined
    }

    static let fileName = "memory-offer.json"

    static var fileURL: URL { MemoryAPI.dataDirectory.appendingPathComponent(fileName) }

    var choice: String?
    var wantedVisualSearch: Bool

    static func parse(_ data: Data?) -> MemoryOfferState {
        let object = data.flatMap { try? JSONSerialization.jsonObject(with: $0) as? [String: Any] } ?? [:]
        return MemoryOfferState(
            choice: object["choice"] as? String,
            wantedVisualSearch: object["wanted_visual_search"] as? Bool ?? false
        )
    }

    static func load(from url: URL = fileURL) -> MemoryOfferState {
        parse(try? Data(contentsOf: url))
    }

    /// `existing` (the file's bytes, nil when absent) with `choice` and
    /// `offered_at` set, every other key kept; sorted keys, as the worker
    /// writes the file.
    static func recording(_ choice: Choice, at date: Date, in existing: Data?) -> Data {
        var object = existing.flatMap { try? JSONSerialization.jsonObject(with: $0) as? [String: Any] } ?? [:]
        object["choice"] = choice.rawValue
        object["offered_at"] = MemoryAPI.isoString(date)
        let data = (try? JSONSerialization.data(withJSONObject: object, options: [.prettyPrinted, .sortedKeys])) ?? Data()
        return data + Data("\n".utf8)
    }

    /// Record `choice` in the file (atomic write, owner-only).
    static func record(_ choice: Choice, at date: Date = Date(), url: URL = fileURL) {
        let data = recording(choice, at: date, in: try? Data(contentsOf: url))
        try? FileManager.default.createDirectory(
            at: url.deletingLastPathComponent(), withIntermediateDirectories: true
        )
        if (try? data.write(to: url, options: .atomic)) != nil {
            try? FileManager.default.setAttributes([.posixPermissions: 0o600], ofItemAtPath: url.path)
        }
    }
}

/// What the app shows about Memory after an update.
enum MemoryOfferDecision: Equatable {
    case none
    /// Offer Memory. `visualSearch`: the user had the removed Visual search
    /// on. `ollamaTooOld`: the running Ollama's version when it cannot run
    /// Memory's model (then the offer explains how to update).
    case offer(visualSearch: Bool, ollamaTooOld: String?)
}

/// Setup's Memory choice turned into config: what a new install writes
/// and pulls once the Ollama that will run it is known.
struct MemorySetupPlan: Equatable {
    /// `[memory] enabled`.
    let memoryEnabled: Bool
    /// `[embedding] model`: Memory's model for everything when this Ollama
    /// runs it, else the text model.
    let embeddingModel: String
    /// Memory was asked for but this Ollama cannot run it: it stays off
    /// until Ollama is updated (`MemoryDecisions.deferredAction`).
    let deferred: Bool
}

/// After Ollama is updated, what the app does about a Memory that setup
/// left off because Ollama was too old.
enum MemoryDeferredAction: Equatable {
    case none
    case enable
    /// Pull this model, then turn Memory on.
    case download(String)
}

enum MemoryDecisions {
    /// UserDefaults: setup showed its Memory step on this Mac (a new
    /// install); such a Mac never sees the upgrade offer.
    static let setupAnsweredKey = "memorySetupAnswered"
    /// UserDefaults: setup's Memory was asked for but left off for an old
    /// Ollama.
    static let deferredKey = "memoryDeferredForOllama"

    /// Whether this Ollama `version` runs Memory's model; false when either
    /// is unknown.
    static func ollamaRunsMemory(_ version: String?, minimum: String?) -> Bool {
        guard let version, let minimum else { return false }
        return OllamaModels.isVersion(version, atLeast: minimum)
    }

    /// The upgrade offer at launch, once setup is complete. Shown to an
    /// updated install whose config has Memory off or absent and whose
    /// offer file holds no choice; never to a Mac whose setup asked
    /// (`setupAnswered`), nor when the installed table knows no Memory model.
    static func offer(
        setupAnswered: Bool, memoryEnabled: Bool?, offer: MemoryOfferState,
        imageEmbeddingsInConfig: Bool, ollamaVersion: String?, table: ModelTierTable?
    ) -> MemoryOfferDecision {
        guard !setupAnswered, memoryEnabled != true, offer.choice == nil,
              let table, table.memoryModel != nil,
              let minimum = table.minimumOllamaVersionForMemory else { return .none }
        let tooOld = ollamaVersion.flatMap { OllamaModels.isVersion($0, atLeast: minimum) ? nil : $0 }
        return .offer(visualSearch: offer.wantedVisualSearch || imageEmbeddingsInConfig, ollamaTooOld: tooOld)
    }

    /// Setup's choice (`wantsMemory`) on the Ollama that answers now.
    static func setupPlan(wantsMemory: Bool, ollamaVersion: String?, table: ModelTierTable) -> MemorySetupPlan {
        guard let memoryModel = table.memoryModel,
              ollamaRunsMemory(ollamaVersion, minimum: table.minimumOllamaVersionForMemory) else {
            return MemorySetupPlan(memoryEnabled: false, embeddingModel: table.embeddingModel, deferred: wantsMemory)
        }
        return MemorySetupPlan(memoryEnabled: wantsMemory, embeddingModel: memoryModel, deferred: false)
    }

    /// At launch, for a Memory setup deferred: turn it on once Ollama runs
    /// its model (pulling the model first when it is missing).
    static func deferredAction(
        deferred: Bool, memoryEnabled: Bool?, ollamaVersion: String?,
        installed: Set<String>?, table: ModelTierTable?
    ) -> MemoryDeferredAction {
        guard deferred, memoryEnabled != true, let table, let model = table.memoryModel,
              ollamaRunsMemory(ollamaVersion, minimum: table.minimumOllamaVersionForMemory),
              let installed else { return .none }
        return OllamaModels.missing([model], installed: installed).isEmpty ? .enable : .download(model)
    }

    /// The upgrade offer's sentence about Ollama, for `version` too old.
    static func ollamaTooOldText(version: String, minimum: String) -> String {
        "Memory needs Ollama \(minimum) or later; this Mac has \(version). Update the Ollama app "
            + "from ollama.com/download (or run: brew install --cask ollama-app), then turn on Memory."
    }
}
