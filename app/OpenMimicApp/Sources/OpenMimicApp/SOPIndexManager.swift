import Foundation
import Combine
import SwiftUI

/// A single SOP entry from the worker's sops-index.json file.
struct SOPEntry: Identifiable, Codable, Hashable {
    let sop_id: String
    let slug: String
    let title: String
    let short_title: String?
    let tags: [String]?
    let source: String
    let status: String
    let confidence: Double
    let created_at: String
    let reviewed_at: String?
    var lifecycleState: String    // "observed", "draft", "reviewed", "verified", "agent_ready", "stale", "archived"

    var id: String { sop_id }

    enum CodingKeys: String, CodingKey {
        case sop_id, slug, title, short_title, tags, source, status
        case confidence, created_at, reviewed_at
        case lifecycleState = "lifecycle_state"
    }

    init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        sop_id = try container.decode(String.self, forKey: .sop_id)
        slug = try container.decode(String.self, forKey: .slug)
        title = try container.decode(String.self, forKey: .title)
        short_title = try container.decodeIfPresent(String.self, forKey: .short_title)
        tags = try container.decodeIfPresent([String].self, forKey: .tags)
        source = try container.decode(String.self, forKey: .source)
        status = try container.decode(String.self, forKey: .status)
        confidence = try container.decode(Double.self, forKey: .confidence)
        created_at = try container.decode(String.self, forKey: .created_at)
        reviewed_at = try container.decodeIfPresent(String.self, forKey: .reviewed_at)
        lifecycleState = try container.decodeIfPresent(String.self, forKey: .lifecycleState) ?? "observed"
    }

    /// Short display title: use short_title from worker, fallback to cleaned title.
    var displayTitle: String {
        if let st = short_title, !st.isEmpty {
            return st
        }
        // Fallback: strip "The user is..." prefix, take first 6 words
        var text = title
        let noisePrefixes = ["The user is ", "User is ", "The user "]
        for pfx in noisePrefixes {
            if text.lowercased().hasPrefix(pfx.lowercased()) {
                text = String(text.dropFirst(pfx.count))
                break
            }
        }
        if let first = text.first {
            text = String(first).uppercased() + text.dropFirst()
        }
        let words = text.split(separator: " ")
        if words.count <= 6 {
            return String(text.trimmingCharacters(in: CharacterSet(charactersIn: ".")))
        }
        return words.prefix(6).joined(separator: " ")
    }

    /// Tag list, empty if none.
    var displayTags: [String] {
        tags ?? []
    }

    /// Friendly source label.
    var sourceLabel: String {
        switch source {
        case "focus": return "Focus Recording"
        case "passive": return "Auto-discovered"
        case "unknown": return "Imported"
        default: return source.capitalized
        }
    }

    /// Source icon name.
    var sourceIcon: String {
        switch source {
        case "focus": return "record.circle"
        case "passive": return "eye"
        default: return "doc"
        }
    }

    /// Parsed creation date.
    var createdDate: Date? {
        let fmt = ISO8601DateFormatter()
        fmt.formatOptions = [.withInternetDateTime, .withFractionalSeconds]
        if let d = fmt.date(from: created_at) { return d }
        // Retry without fractional seconds
        fmt.formatOptions = [.withInternetDateTime]
        return fmt.date(from: created_at)
    }

    /// Relative time string ("Just now", "2h ago", "Yesterday", "Mar 6").
    var relativeTime: String {
        guard let date = createdDate else { return "" }
        let interval = -date.timeIntervalSinceNow
        if interval < 60 { return "Just now" }
        if interval < 3600 { return "\(Int(interval / 60))m ago" }
        if interval < 86400 { return "\(Int(interval / 3600))h ago" }
        if interval < 172800 { return "Yesterday" }
        if interval < 604800 { return "\(Int(interval / 86400))d ago" }
        let fmt = DateFormatter()
        fmt.dateFormat = "MMM d"
        return fmt.string(from: date)
    }

    // MARK: - Lifecycle

    var lifecycleLabel: String {
        switch lifecycleState {
        case "agent_ready": return "Agent Ready"
        case "verified": return "Verified"
        case "reviewed": return "Reviewed"
        case "draft": return "Draft"
        case "observed": return "Observed"
        case "stale": return "Stale"
        case "archived": return "Archived"
        default: return lifecycleState.capitalized
        }
    }

    var lifecycleColor: Color {
        switch lifecycleState {
        case "agent_ready": return .green
        case "verified": return .blue
        case "reviewed": return .cyan
        case "draft": return .yellow
        case "observed": return .gray
        case "stale": return .orange
        case "archived": return .secondary
        default: return .secondary
        }
    }

    var canPromote: Bool {
        // Can promote if not already at highest or terminal states
        return ["observed", "draft", "reviewed", "verified"].contains(lifecycleState)
    }

    var nextLifecycleState: String? {
        switch lifecycleState {
        case "observed": return "draft"
        case "draft": return "reviewed"
        case "reviewed": return "verified"
        case "verified": return "agent_ready"
        default: return nil
        }
    }

    static func lifecycleLabelFor(_ state: String) -> String {
        switch state {
        case "agent_ready": return "Agent Ready"
        case "verified": return "Verified"
        case "reviewed": return "Reviewed"
        case "draft": return "Draft"
        case "observed": return "Observed"
        case "stale": return "Stale"
        case "archived": return "Archived"
        default: return state.capitalized
        }
    }
}

/// Top-level structure of sops-index.json written by the Python worker.
struct SOPIndex: Codable {
    let updated_at: String
    let sops: [SOPEntry]
    let failed_count: Int
    let draft_count: Int
    let approved_count: Int
}

/// Reads and polls the worker's ``sops-index.json`` so SwiftUI views can
/// display the workflow inbox without direct SQLite access.
@MainActor
final class SOPIndexManager: ObservableObject {
    @Published var index: SOPIndex?

    private var timer: Timer?

    private var indexPath: URL {
        FileManager.default.homeDirectoryForCurrentUser
            .appendingPathComponent("Library/Application Support/oc-apprentice/sops-index.json")
    }

    func startPolling(interval: TimeInterval = 5.0) {
        // Invalidate any existing timer to prevent stacking
        timer?.invalidate()
        timer = nil

        loadIndex()
        timer = Timer.scheduledTimer(withTimeInterval: interval, repeats: true) { [weak self] _ in
            Task { @MainActor in
                self?.loadIndex()
            }
        }
    }

    func stopPolling() {
        timer?.invalidate()
        timer = nil
    }

    func loadIndex() {
        guard let data = try? Data(contentsOf: indexPath) else { return }
        index = try? JSONDecoder().decode(SOPIndex.self, from: data)
    }

    // MARK: - Sorted & filtered views (chronological, newest first)

    /// All SOPs sorted newest first.
    var allSorted: [SOPEntry] {
        (index?.sops ?? []).sorted { ($0.created_at) > ($1.created_at) }
    }

    var drafts: [SOPEntry] { allSorted.filter { $0.status == "draft" } }
    var approved: [SOPEntry] { allSorted.filter { $0.status == "approved" } }

    var highConfidence: [SOPEntry] {
        allSorted.filter { $0.confidence >= 0.8 && $0.status == "approved" }
    }

    /// SOPs created in the last 24 hours.
    var recent: [SOPEntry] {
        let cutoff = Date().addingTimeInterval(-86400)
        return allSorted.filter { entry in
            guard let date = entry.createdDate else { return false }
            return date > cutoff
        }
    }

    // MARK: - Approve / Reject

    private var triggerDir: URL {
        FileManager.default.homeDirectoryForCurrentUser
            .appendingPathComponent("Library/Application Support/oc-apprentice")
    }

    /// Write an approval trigger file for the worker to pick up.
    func approveSOP(_ sop: SOPEntry) {
        let fmt = ISO8601DateFormatter()
        fmt.formatOptions = [.withInternetDateTime]
        let payload: [String: Any] = [
            "sop_id": sop.slug,
            "action": "approve",
            "requested_at": fmt.string(from: Date())
        ]
        writeTrigger(payload, filename: "approve-trigger.json")
    }

    /// Write a rejection trigger file for the worker to pick up.
    func rejectSOP(_ sop: SOPEntry) {
        let fmt = ISO8601DateFormatter()
        fmt.formatOptions = [.withInternetDateTime]
        let payload: [String: Any] = [
            "sop_id": sop.slug,
            "action": "reject",
            "requested_at": fmt.string(from: Date())
        ]
        writeTrigger(payload, filename: "approve-trigger.json")
    }

    /// Promote a procedure to the next lifecycle state via trigger file.
    func promoteProcedure(_ sop: SOPEntry, toState: String) {
        let trigger: [String: Any] = [
            "procedure_slug": sop.slug,
            "to_state": toState,
            "actor": "human",
            "reason": "Promoted via OpenMimic app",
            "requested_at": ISO8601DateFormatter().string(from: Date())
        ]
        writeTrigger(trigger, filename: "lifecycle-promote-trigger.json")

        // Refresh after a short delay
        DispatchQueue.main.asyncAfter(deadline: .now() + 2.0) { [weak self] in
            self?.loadIndex()
        }
    }

    /// Atomic trigger file writer. Writes payload as JSON to a named file
    /// inside the oc-apprentice state directory using tmp+rename.
    func writeTrigger(_ payload: [String: Any], filename: String) {
        guard let data = try? JSONSerialization.data(withJSONObject: payload, options: [.prettyPrinted, .sortedKeys]) else {
            return
        }

        let dir = triggerDir
        // Ensure parent directory exists
        try? FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)

        let target = dir.appendingPathComponent(filename)
        let tmp = dir.appendingPathComponent(".\(filename).tmp")

        do {
            try data.write(to: tmp, options: .atomic)
            // Rename .tmp -> final path (atomic on same filesystem)
            if FileManager.default.fileExists(atPath: target.path) {
                try FileManager.default.removeItem(at: target)
            }
            try FileManager.default.moveItem(at: tmp, to: target)
        } catch {
            // Clean up tmp file if rename failed
            try? FileManager.default.removeItem(at: tmp)
            return
        }

        // Refresh index after 2 seconds to pick up worker's response
        Timer.scheduledTimer(withTimeInterval: 2.0, repeats: false) { [weak self] _ in
            Task { @MainActor in
                self?.loadIndex()
            }
        }
    }
}
