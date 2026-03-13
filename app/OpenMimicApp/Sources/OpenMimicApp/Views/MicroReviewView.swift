import SwiftUI

/// Card-based micro-review UX for quick procedure approval.
///
/// Each card shows a procedure summary with one-tap approve/reject/detail
/// actions — designed for 5-second reviews.
struct MicroReviewView: View {
    @StateObject private var viewModel = MicroReviewViewModel()

    var body: some View {
        VStack(alignment: .leading, spacing: 16) {
            headerSection

            if viewModel.isLoading {
                loadingState
            } else if viewModel.cards.isEmpty {
                emptyState
            } else {
                cardStack
            }
        }
        .padding(24)
        .frame(minWidth: 500, minHeight: 400)
        .onAppear { viewModel.load() }
    }

    // MARK: - Header

    private var headerSection: some View {
        HStack {
            VStack(alignment: .leading, spacing: 4) {
                Text("Review Queue")
                    .font(.title2)
                    .bold()
                Text("\(viewModel.cards.count) item(s) need your review")
                    .font(.caption)
                    .foregroundColor(.secondary)
            }
            Spacer()
            Button(action: { viewModel.load() }) {
                Image(systemName: "arrow.clockwise")
                    .font(.system(size: 12))
            }
            .buttonStyle(.plain)
            .foregroundColor(.secondary)
        }
    }

    // MARK: - Card stack

    private var cardStack: some View {
        ScrollView {
            LazyVStack(spacing: 12) {
                ForEach(viewModel.cards) { card in
                    ReviewCard(card: card, onAction: { action in
                        viewModel.handleAction(card: card, action: action)
                    })
                }
            }
        }
    }

    // MARK: - States

    private var loadingState: some View {
        VStack(spacing: 12) {
            ProgressView()
            Text("Loading review items...")
                .font(.caption)
                .foregroundColor(.secondary)
        }
        .frame(maxWidth: .infinity, minHeight: 200)
    }

    private var emptyState: some View {
        VStack(spacing: 12) {
            Image(systemName: "checkmark.seal")
                .font(.system(size: 32))
                .foregroundColor(.green.opacity(0.5))
            Text("All caught up!")
                .font(.title3)
                .foregroundColor(.secondary)
            Text("No procedures need review right now.")
                .font(.caption)
                .foregroundColor(.secondary.opacity(0.7))
        }
        .frame(maxWidth: .infinity, minHeight: 200)
    }
}

// MARK: - Review Card

struct ReviewCard: View {
    let card: ReviewCardData
    let onAction: (ReviewAction) -> Void

    @State private var showingDetail = false

    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            // Title + type badge
            HStack {
                Image(systemName: card.iconName)
                    .font(.system(size: 14))
                    .foregroundColor(card.accentColor)

                Text(card.title)
                    .font(.system(size: 14, weight: .semibold))
                    .lineLimit(1)

                Spacer()

                Text(card.typeLabel)
                    .font(.system(size: 9, weight: .medium))
                    .foregroundColor(.white)
                    .padding(.horizontal, 6)
                    .padding(.vertical, 2)
                    .background(card.accentColor)
                    .cornerRadius(4)
            }

            // Metadata row
            HStack(spacing: 12) {
                metadataItem(icon: "calendar", text: card.recurrence)
                metadataItem(icon: "clock", text: card.duration)
                metadataItem(icon: "eye", text: "\(card.observations) obs")
                metadataItem(icon: "chart.bar", text: String(format: "%.0f%%", card.confidence * 100))
            }
            .font(.system(size: 10))
            .foregroundColor(.secondary)

            // Variables preview
            if !card.variables.isEmpty {
                HStack(spacing: 4) {
                    Text("Variables:")
                        .font(.system(size: 10))
                        .foregroundColor(.secondary)
                    Text(card.variables.joined(separator: ", "))
                        .font(.system(size: 10, weight: .medium))
                        .foregroundColor(.primary.opacity(0.7))
                        .lineLimit(1)
                }
            }

            // Outcome preview
            if let outcome = card.outcome {
                HStack(spacing: 4) {
                    Text("Outcome:")
                        .font(.system(size: 10))
                        .foregroundColor(.secondary)
                    Text(outcome)
                        .font(.system(size: 10, weight: .medium))
                        .foregroundColor(.primary.opacity(0.7))
                        .lineLimit(1)
                }
            }

            Divider()

            // Action buttons
            HStack(spacing: 10) {
                Button(action: { onAction(.approve) }) {
                    Label("Approve", systemImage: "checkmark")
                        .font(.system(size: 11, weight: .medium))
                }
                .buttonStyle(.borderedProminent)
                .tint(.green)
                .controlSize(.small)

                Button(action: { onAction(.reject) }) {
                    Label("Reject", systemImage: "xmark")
                        .font(.system(size: 11, weight: .medium))
                }
                .buttonStyle(.bordered)
                .tint(.red)
                .controlSize(.small)

                Spacer()

                if card.type == .trustSuggestion {
                    Button(action: { onAction(.dismiss) }) {
                        Text("Later")
                            .font(.system(size: 11))
                    }
                    .buttonStyle(.plain)
                    .foregroundColor(.secondary)
                }

                Button(action: { showingDetail.toggle() }) {
                    Text("Detail")
                        .font(.system(size: 11))
                }
                .buttonStyle(.plain)
                .foregroundColor(.accentColor)
            }

            // Expandable detail
            if showingDetail {
                VStack(alignment: .leading, spacing: 6) {
                    Divider()

                    if !card.evidenceText.isEmpty {
                        Text("Evidence")
                            .font(.system(size: 10, weight: .semibold))
                        Text(card.evidenceText)
                            .font(.system(size: 10))
                            .foregroundColor(.secondary)
                    }

                    if !card.stepsPreview.isEmpty {
                        Text("Steps")
                            .font(.system(size: 10, weight: .semibold))
                        ForEach(Array(card.stepsPreview.enumerated()), id: \.offset) { i, step in
                            Text("\(i + 1). \(step)")
                                .font(.system(size: 10))
                                .foregroundColor(.secondary)
                        }
                    }
                }
                .padding(.top, 4)
            }
        }
        .padding(14)
        .background(Color.primary.opacity(0.03))
        .cornerRadius(10)
        .overlay(
            RoundedRectangle(cornerRadius: 10)
                .stroke(Color.primary.opacity(0.08), lineWidth: 1)
        )
    }

    private func metadataItem(icon: String, text: String) -> some View {
        HStack(spacing: 3) {
            Image(systemName: icon)
                .font(.system(size: 8))
            Text(text)
        }
    }
}

// MARK: - Data Models

enum ReviewCardType: Sendable {
    case draftProcedure
    case trustSuggestion
    case staleAlert
}

enum ReviewAction: Sendable {
    case approve
    case reject
    case dismiss
}

struct ReviewCardData: Identifiable, Sendable {
    let id: String
    let type: ReviewCardType
    let title: String
    let recurrence: String
    let duration: String
    let observations: Int
    let confidence: Double
    let variables: [String]
    let outcome: String?
    let evidenceText: String
    let stepsPreview: [String]
    let slug: String

    var typeLabel: String {
        switch type {
        case .draftProcedure: return "Draft"
        case .trustSuggestion: return "Promote"
        case .staleAlert: return "Stale"
        }
    }

    var iconName: String {
        switch type {
        case .draftProcedure: return "doc.badge.gearshape"
        case .trustSuggestion: return "arrow.up.circle"
        case .staleAlert: return "exclamationmark.triangle"
        }
    }

    var accentColor: Color {
        switch type {
        case .draftProcedure: return .blue
        case .trustSuggestion: return .green
        case .staleAlert: return .orange
        }
    }
}

// MARK: - View Model

@MainActor
final class MicroReviewViewModel: ObservableObject {
    @Published var cards: [ReviewCardData] = []
    @Published var isLoading = false

    func load() {
        isLoading = true
        Task.detached(priority: .userInitiated) {
            let loaded = Self.loadReviewItems()
            await MainActor.run { [weak self] in
                self?.cards = loaded
                self?.isLoading = false
            }
        }
    }

    func handleAction(card: ReviewCardData, action: ReviewAction) {
        // Write trigger file for the worker to pick up
        let stateDir: URL = {
            let home = FileManager.default.homeDirectoryForCurrentUser
            return home.appendingPathComponent(
                "Library/Application Support/oc-apprentice"
            )
        }()

        let now = ISO8601DateFormatter().string(from: Date())
        var success = false

        switch (card.type, action) {
        // Draft SOP cards → approve-trigger.json
        case (.draftProcedure, .approve):
            success = writeTrigger(
                dir: stateDir,
                filename: "approve-trigger.json",
                payload: [
                    "sop_id": card.slug,
                    "action": "approve",
                    "requested_at": now,
                ]
            )
        case (.draftProcedure, .reject):
            success = writeTrigger(
                dir: stateDir,
                filename: "approve-trigger.json",
                payload: [
                    "sop_id": card.slug,
                    "action": "reject",
                    "requested_at": now,
                ]
            )

        // Trust suggestion cards → trust-accept / trust-dismiss triggers
        case (.trustSuggestion, .approve):
            success = writeTrigger(
                dir: stateDir,
                filename: "trust-accept-trigger.json",
                payload: [
                    "procedure_slug": card.slug,
                    "requested_at": now,
                ]
            )
        case (.trustSuggestion, .reject),
             (.trustSuggestion, .dismiss):
            success = writeTrigger(
                dir: stateDir,
                filename: "trust-dismiss-trigger.json",
                payload: [
                    "procedure_slug": card.slug,
                    "requested_at": now,
                ]
            )

        // Stale procedure cards → staleness-reviewed-trigger.json
        case (.staleAlert, .approve):
            success = writeTrigger(
                dir: stateDir,
                filename: "staleness-reviewed-trigger.json",
                payload: [
                    "procedure_slug": card.slug,
                    "action": "reviewed",
                    "requested_at": now,
                ]
            )
        case (.staleAlert, .reject):
            success = writeTrigger(
                dir: stateDir,
                filename: "staleness-reviewed-trigger.json",
                payload: [
                    "procedure_slug": card.slug,
                    "action": "archive",
                    "requested_at": now,
                ]
            )

        // Dismiss only applies to trust suggestions; ignore for other types
        default:
            return
        }

        // Only remove the card if the trigger was written successfully
        if success {
            cards.removeAll { $0.id == card.id }
        }
    }

    @discardableResult
    private func writeTrigger(dir: URL, filename: String, payload: [String: String]) -> Bool {
        do {
            try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        } catch {
            return false
        }

        let target = dir.appendingPathComponent(filename)
        let tmp = dir.appendingPathComponent(".\(filename).tmp")

        guard let data = try? JSONSerialization.data(
            withJSONObject: payload, options: [.prettyPrinted, .sortedKeys]
        ) else { return false }

        do {
            try data.write(to: tmp)
        } catch {
            return false
        }

        // Remove stale target if it exists so moveItem does not fail
        if FileManager.default.fileExists(atPath: target.path) {
            try? FileManager.default.removeItem(at: target)
        }

        do {
            try FileManager.default.moveItem(at: tmp, to: target)
            return true
        } catch {
            // Clean up the tmp file on failure
            try? FileManager.default.removeItem(at: tmp)
            return false
        }
    }

    private nonisolated static func loadReviewItems() -> [ReviewCardData] {
        let home = FileManager.default.homeDirectoryForCurrentUser
        let kbDir = home.appendingPathComponent(".openmimic/knowledge")
        var cards: [ReviewCardData] = []

        // Load draft procedures from sops-index
        let indexPath = home.appendingPathComponent(
            "Library/Application Support/oc-apprentice/sops-index.json"
        )
        if let data = try? Data(contentsOf: indexPath),
           let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
           let sops = json["sops"] as? [[String: Any]] {
            for sop in sops {
                let status = sop["status"] as? String ?? ""
                guard status == "draft" else { continue }

                let slug = sop["slug"] as? String ?? ""
                let title = sop["short_title"] as? String ?? sop["title"] as? String ?? slug

                cards.append(ReviewCardData(
                    id: sop["sop_id"] as? String ?? UUID().uuidString,
                    type: .draftProcedure,
                    title: title,
                    recurrence: "—",
                    duration: "—",
                    observations: 1,
                    confidence: sop["confidence"] as? Double ?? 0,
                    variables: [],
                    outcome: nil,
                    evidenceText: "Source: \(sop["source"] as? String ?? "unknown")",
                    stepsPreview: [],
                    slug: slug
                ))
            }
        }

        // Load trust suggestions
        let suggestionsPath = kbDir.appendingPathComponent(
            "observations/trust_suggestions.json"
        )
        if let data = try? Data(contentsOf: suggestionsPath),
           let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
           let suggestions = json["suggestions"] as? [[String: Any]] {
            for s in suggestions {
                let dismissed = s["dismissed"] as? Bool ?? false
                let accepted = s["accepted"] as? Bool ?? false
                guard !dismissed && !accepted else { continue }

                let slug = s["procedure_slug"] as? String ?? ""
                let evidence = s["evidence"] as? [String: Any] ?? [:]

                cards.append(ReviewCardData(
                    id: UUID().uuidString,
                    type: .trustSuggestion,
                    title: "Promote: \(slug)",
                    recurrence: "—",
                    duration: "—",
                    observations: evidence["observations"] as? Int ?? 0,
                    confidence: evidence["success_rate"] as? Double ?? 0,
                    variables: [],
                    outcome: nil,
                    evidenceText: s["reason"] as? String ?? "",
                    stepsPreview: [],
                    slug: slug
                ))
            }
        }

        // Load stale procedures
        let proceduresDir = kbDir.appendingPathComponent("procedures")
        if let files = try? FileManager.default.contentsOfDirectory(
            at: proceduresDir, includingPropertiesForKeys: nil
        ) {
            for file in files where file.pathExtension == "json" {
                guard let data = try? Data(contentsOf: file),
                      let proc = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
                      let staleness = proc["staleness"] as? [String: Any],
                      let status = staleness["status"] as? String,
                      status != "current" else { continue }

                let slug = proc["id"] as? String ?? file.deletingPathExtension().lastPathComponent

                cards.append(ReviewCardData(
                    id: UUID().uuidString,
                    type: .staleAlert,
                    title: "Review: \(slug)",
                    recurrence: "—",
                    duration: "—",
                    observations: 0,
                    confidence: 0,
                    variables: [],
                    outcome: nil,
                    evidenceText: "Status: \(status)",
                    stepsPreview: [],
                    slug: slug
                ))
            }
        }

        return cards
    }
}
