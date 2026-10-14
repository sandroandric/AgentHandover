import Foundation

/// What `agenthandover doctor --ollama-json` reports (crates/cli/src/machine.rs):
/// the Ollama that answers, whether it is new enough, installs that compete
/// for its port, and the memory each tier needs against what is free now.
///
/// Onboarding shows these findings word for word, so the app and `doctor`
/// explain a problem, and its fix, the same way.  People run Ollama.app,
/// Homebrew's formula, both, or a remote host; the CLI detects which.
struct OllamaReport: Decodable, Equatable {
    struct Host: Decodable, Equatable {
        let url: String
        let local: Bool
        let port: Int
    }

    struct Install: Decodable, Equatable {
        let kind: String
        let path: String
        let version: String?
    }

    struct Finding: Decodable, Equatable {
        enum Level: String, Decodable {
            case pass, warn, fail
        }

        let level: Level
        let title: String
        let lines: [String]

        /// The onboarding icon, as `doctor` marks the line: FAIL or WARN.
        var symbolName: String {
            level == .fail ? "xmark.octagon.fill" : "exclamationmark.triangle.fill"
        }
    }

    struct TierNeed: Decodable, Equatable {
        let id: String
        let label: String
        let needBytes: UInt64
        let fits: Bool
    }

    struct Headroom: Decodable, Equatable {
        /// Free, inactive and speculative memory plus the models Ollama has loaded.
        let availableBytes: UInt64
        let freeBytes: UInt64
        let loadedBytes: UInt64
        let swapUsedBytes: UInt64?
        let freeDiskBytes: UInt64?
        let needBytes: UInt64
        let tiers: [TierNeed]
    }

    let host: Host
    /// `GET /api/version`; nil when nothing answers.
    let version: String?
    let minimum: String
    let installs: [Install]
    let findings: [Finding]
    /// Nil for a remote host, or when memory could not be read.
    let headroom: Headroom?

    static func decode(_ data: Data) throws -> OllamaReport {
        let decoder = JSONDecoder()
        decoder.keyDecodingStrategy = .convertFromSnakeCase
        return try decoder.decode(OllamaReport.self, from: data)
    }

    /// Warnings and failures, in the order `doctor` prints them.
    var problems: [Finding] {
        findings.filter { $0.level != .pass }
    }

    /// "Ollama 0.34.2 is running at http://localhost:11434", nil when
    /// nothing answers.
    var serviceLine: String? {
        version.map { "Ollama \($0) is running at \(host.url)" }
    }

    /// "Needs 8.5 GB · 10.5 GB free now" for the tier `id`; nil when the
    /// report has no memory figures for it.
    func memoryLabel(forTier id: String) -> String? {
        guard let headroom, let need = headroom.tiers.first(where: { $0.id == id }) else {
            return nil
        }
        return "Needs \(Self.gb(need.needBytes)) \u{00B7} \(Self.gb(headroom.availableBytes)) free now"
    }

    /// Whether the tier `id` fits in the memory free now; nil when unknown.
    func fits(tier id: String) -> Bool? {
        headroom?.tiers.first(where: { $0.id == id })?.fits
    }

    /// Decimal gigabytes, as the CLI and the tier table count.
    static func gb(_ bytes: UInt64) -> String {
        String(format: "%.1f GB", Double(bytes) / 1_000_000_000)
    }

    /// Run `<cli> doctor --ollama-json --mode <mode>`: the report for the
    /// mode setup shows, which config.toml names only once its models are
    /// pulled.  Blocking; nil when the CLI is missing, fails, or prints
    /// something else.
    static func load(cliPath: String, mode: SkillsMode) -> OllamaReport? {
        DoctorJSON.run(cliPath: cliPath, flag: "--ollama-json", extra: arguments(for: mode))
            .flatMap { try? decode($0) }
    }

    static func arguments(for mode: SkillsMode) -> [String] {
        ["--mode", mode.isMemoryOnly ? "memory-only" : "skills"]
    }
}
