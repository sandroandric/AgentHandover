import SwiftUI

/// Displays the daily digest - a summary of today's activity, highlights,
/// and actionable suggestions from the knowledge base.
struct DailyDigestView: View {
    @StateObject private var viewModel = DigestViewModel()

    // Contra design tokens (system-adaptive)
    private let darkNavy = Color.primary
    private let warmOrange = Color(red: 0.92, green: 0.57, blue: 0.20)
    private let goldenYellow = Color(red: 1.0, green: 0.74, blue: 0.07)
    private let warmCream = Color(nsColor: .windowBackgroundColor)
    private let lightGray = Color(nsColor: .controlBackgroundColor)
    private let brightGreen = Color(red: 0.18, green: 0.80, blue: 0.34)
    private let cardRadius: CGFloat = 14
    private let contraBorder: CGFloat = 1.5

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 20) {
                headerSection
                Rectangle()
                    .fill(darkNavy.opacity(0.08))
                    .frame(height: 1)

                if viewModel.isLoading {
                    loadingState
                } else if let digest = viewModel.digest {
                    statsBar(digest)
                    highlightsSection(digest)
                    sectionsView(digest)
                } else {
                    emptyState
                }
            }
            .padding(24)
        }
        .frame(minWidth: 580, minHeight: 500)
        .onAppear { viewModel.load() }
    }

    // MARK: - Header

    private var headerSection: some View {
        VStack(alignment: .leading, spacing: 6) {
            HStack {
                ZStack {
                    RoundedRectangle(cornerRadius: 8)
                        .fill(warmOrange.opacity(0.12))
                        .frame(width: 32, height: 32)
                        .overlay(
                            RoundedRectangle(cornerRadius: 8)
                                .stroke(darkNavy.opacity(0.12), lineWidth: contraBorder)
                        )
                    Image(systemName: "calendar.badge.clock")
                        .font(.system(size: 15))
                        .foregroundColor(warmOrange)
                }
                Text("Daily Digest")
                    .font(.system(size: 22, weight: .bold, design: .rounded))
                    .foregroundColor(darkNavy)
                Spacer()
                Text(viewModel.dateDisplay)
                    .font(.system(size: 12))
                    .foregroundColor(darkNavy.opacity(0.5))
            }

            if let summary = viewModel.digest?.summary, !summary.isEmpty {
                Text(summary)
                    .font(.subheadline)
                    .foregroundColor(darkNavy.opacity(0.6))
                    .lineSpacing(2)
            }
        }
    }

    // MARK: - Stats

    private func statsBar(_ digest: DigestData) -> some View {
        HStack(spacing: 16) {
            statCard(
                icon: "clock",
                value: String(format: "%.1fh", digest.activeHours),
                label: "Active"
            )
            statCard(
                icon: "checkmark.circle",
                value: "\(digest.tasksCompleted)",
                label: "Tasks"
            )
            statCard(
                icon: "doc.text",
                value: "\(digest.proceduresObserved)",
                label: "Procedures"
            )
        }
    }

    private func statCard(icon: String, value: String, label: String) -> some View {
        HStack(spacing: 8) {
            Image(systemName: icon)
                .font(.system(size: 14))
                .foregroundColor(warmOrange)
            VStack(alignment: .leading, spacing: 1) {
                Text(value)
                    .font(.system(size: 16, weight: .bold, design: .rounded))
                    .foregroundColor(darkNavy)
                Text(label)
                    .font(.system(size: 10))
                    .foregroundColor(darkNavy.opacity(0.5))
            }
        }
        .padding(.horizontal, 12)
        .padding(.vertical, 8)
        .background(lightGray)
        .cornerRadius(cardRadius)
        .overlay(
            RoundedRectangle(cornerRadius: cardRadius)
                .stroke(darkNavy.opacity(0.12), lineWidth: contraBorder)
        )
    }

    // MARK: - Highlights

    private func highlightsSection(_ digest: DigestData) -> some View {
        Group {
            if !digest.highlights.isEmpty {
                VStack(alignment: .leading, spacing: 8) {
                    Text("Highlights")
                        .font(.system(size: 16, weight: .bold, design: .rounded))
                        .foregroundColor(darkNavy)

                    ForEach(digest.highlights, id: \.title) { highlight in
                        highlightRow(highlight)
                    }
                }
            }
        }
    }

    private func highlightRow(_ highlight: DigestHighlightData) -> some View {
        HStack(spacing: 10) {
            ZStack {
                RoundedRectangle(cornerRadius: 6)
                    .fill(highlight.color.opacity(0.12))
                    .frame(width: 24, height: 24)
                    .overlay(
                        RoundedRectangle(cornerRadius: 6)
                            .stroke(darkNavy.opacity(0.12), lineWidth: contraBorder)
                    )
                Image(systemName: highlight.iconName)
                    .font(.system(size: 11))
                    .foregroundColor(highlight.color)
            }

            VStack(alignment: .leading, spacing: 2) {
                Text(highlight.title)
                    .font(.system(size: 12, weight: .semibold))
                    .foregroundColor(darkNavy)
                Text(highlight.detail)
                    .font(.system(size: 11))
                    .foregroundColor(darkNavy.opacity(0.5))
                    .lineLimit(2)
            }

            Spacer()

            priorityBadge(highlight.priority)
        }
        .padding(10)
        .background(lightGray)
        .cornerRadius(cardRadius)
        .overlay(
            RoundedRectangle(cornerRadius: cardRadius)
                .stroke(darkNavy.opacity(0.12), lineWidth: contraBorder)
        )
    }

    private func priorityBadge(_ priority: Int) -> some View {
        Group {
            if priority <= 2 {
                Text(priority == 1 ? "!" : "i")
                    .font(.system(size: 9, weight: .bold))
                    .foregroundColor(priority == 1 ? .red : warmOrange)
                    .frame(width: 16, height: 16)
                    .background(
                        RoundedRectangle(cornerRadius: 4)
                            .fill(priority == 1 ? Color.red.opacity(0.12) : warmOrange.opacity(0.12))
                    )
                    .overlay(
                        RoundedRectangle(cornerRadius: 4)
                            .stroke(darkNavy.opacity(0.12), lineWidth: 1)
                    )
            }
        }
    }

    // MARK: - Sections

    private func sectionsView(_ digest: DigestData) -> some View {
        ForEach(digest.sections, id: \.title) { section in
            VStack(alignment: .leading, spacing: 8) {
                Text(section.title)
                    .font(.system(size: 16, weight: .bold, design: .rounded))
                    .foregroundColor(darkNavy)

                if section.items.isEmpty {
                    Text("Nothing to report")
                        .font(.system(size: 12))
                        .foregroundColor(darkNavy.opacity(0.4))
                        .padding(.leading, 4)
                } else {
                    ForEach(Array(section.items.enumerated()), id: \.offset) { _, item in
                        sectionItemRow(item)
                    }
                }
            }
            .padding(.top, 4)
        }
    }

    private func sectionItemRow(_ item: [String: String]) -> some View {
        HStack(spacing: 8) {
            Circle()
                .fill(warmOrange)
                .frame(width: 6, height: 6)

            VStack(alignment: .leading, spacing: 1) {
                if let title = item["title"] {
                    Text(title)
                        .font(.system(size: 12, weight: .semibold))
                        .foregroundColor(darkNavy)
                }
                if let detail = item["detail"] {
                    Text(detail)
                        .font(.system(size: 11))
                        .foregroundColor(darkNavy.opacity(0.5))
                }
            }

            Spacer()
        }
        .padding(.vertical, 4)
        .padding(.horizontal, 8)
    }

    // MARK: - States

    private var loadingState: some View {
        VStack(spacing: 12) {
            ProgressView()
                .tint(warmOrange)
            Text("Loading digest...")
                .font(.system(size: 12))
                .foregroundColor(darkNavy.opacity(0.5))
        }
        .frame(maxWidth: .infinity, minHeight: 200)
    }

    private var emptyState: some View {
        VStack(spacing: 12) {
            ZStack {
                RoundedRectangle(cornerRadius: cardRadius)
                    .fill(warmCream)
                    .frame(width: 56, height: 56)
                    .overlay(
                        RoundedRectangle(cornerRadius: cardRadius)
                            .stroke(darkNavy.opacity(0.12), lineWidth: contraBorder)
                    )
                Image(systemName: "doc.text.magnifyingglass")
                    .font(.system(size: 28))
                    .foregroundColor(warmOrange)
            }
            Text("No digest available")
                .font(.system(size: 18, weight: .bold, design: .rounded))
                .foregroundColor(darkNavy)
            Text("Digests are generated at the end of each day\nonce enough activity has been observed.")
                .font(.system(size: 13))
                .foregroundColor(darkNavy.opacity(0.5))
                .multilineTextAlignment(.center)
        }
        .frame(maxWidth: .infinity, minHeight: 200)
    }
}

// MARK: - Data Models

struct DigestData: Sendable {
    let date: String
    let summary: String
    let activeHours: Double
    let tasksCompleted: Int
    let proceduresObserved: Int
    let highlights: [DigestHighlightData]
    let sections: [DigestSectionData]
}

struct DigestHighlightData: Sendable {
    let type: String
    let title: String
    let detail: String
    let priority: Int

    var iconName: String {
        switch type {
        case "new_procedure": return "doc.badge.plus"
        case "trust_suggestion": return "arrow.up.circle"
        case "stale_alert": return "exclamationmark.triangle"
        case "pattern_detected": return "chart.line.uptrend.xyaxis"
        case "milestone": return "star"
        case "lifecycle_upgrade": return "arrow.up.square"
        case "merge_candidate": return "arrow.triangle.merge"
        case "drift_alert": return "waveform.path.ecg"
        default: return "info.circle"
        }
    }

    var color: Color {
        switch type {
        case "new_procedure": return .green
        case "trust_suggestion": return .blue
        case "stale_alert": return .orange
        case "pattern_detected": return .purple
        case "milestone": return .yellow
        case "lifecycle_upgrade": return .cyan
        case "merge_candidate": return .purple
        case "drift_alert": return .red
        default: return .secondary
        }
    }
}

struct DigestSectionData: Identifiable, Sendable {
    let title: String
    let items: [[String: String]]
    var id: String { title }
}

// MARK: - View Model

@MainActor
final class DigestViewModel: ObservableObject {
    @Published var digest: DigestData?
    @Published var isLoading = false

    var dateDisplay: String {
        // Use the loaded digest's date if available, fall back to today.
        if let dateStr = digest?.date, !dateStr.isEmpty {
            let parser = DateFormatter()
            parser.dateFormat = "yyyy-MM-dd"
            if let parsed = parser.date(from: dateStr) {
                let display = DateFormatter()
                display.dateStyle = .medium
                return display.string(from: parsed)
            }
            // Date string present but unparseable - show it as-is.
            return dateStr
        }
        let formatter = DateFormatter()
        formatter.dateStyle = .medium
        return formatter.string(from: Date())
    }

    func load() {
        isLoading = true
        Task.detached(priority: .userInitiated) {
            let data = Self.loadDigestFromDisk()
            await MainActor.run { [weak self] in
                self?.digest = data
                self?.isLoading = false
            }
        }
    }

    private nonisolated static func loadDigestFromDisk() -> DigestData? {
        let fmt = DateFormatter()
        fmt.dateFormat = "yyyy-MM-dd"
        let today = fmt.string(from: Date())
        let yesterday = fmt.string(from: Calendar.current.date(byAdding: .day, value: -1, to: Date()) ?? Date())

        // The worker generates digests at end-of-day, so the most recent
        // digest is typically yesterday's. Try yesterday first, then today.
        let datesToTry = [yesterday, today]

        let knowledgeDir = FileManager.default.homeDirectoryForCurrentUser
            .appendingPathComponent(".agenthandover/knowledge")

        // Try digest files
        for dateStr in datesToTry {
            let digestPath = knowledgeDir
                .appendingPathComponent("observations/digests/\(dateStr).json")

            if let data = try? Data(contentsOf: digestPath),
               let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any] {
                return parseDigest(json)
            }
        }

        // Fallback: build minimal digest from daily summary
        for dateStr in datesToTry {
            let summaryPath = knowledgeDir
                .appendingPathComponent("observations/daily/\(dateStr).json")

            if let data = try? Data(contentsOf: summaryPath),
               let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any] {
                return buildMinimalDigest(from: json, date: dateStr)
            }
        }

        return nil
    }

    private nonisolated static func parseDigest(_ json: [String: Any]) -> DigestData {
        let highlights = (json["highlights"] as? [[String: Any]] ?? []).map { h in
            DigestHighlightData(
                type: h["type"] as? String ?? "",
                title: h["title"] as? String ?? "",
                detail: h["detail"] as? String ?? "",
                priority: h["priority"] as? Int ?? 3
            )
        }

        let sections = (json["sections"] as? [[String: Any]] ?? []).map { s in
            // Worker emits mixed-type values (ints, arrays, strings) so cast
            // as [[String: Any]] and stringify every value for display.
            let rawItems = s["items"] as? [[String: Any]] ?? []
            let stringItems: [[String: String]] = rawItems.map { dict in
                var out: [String: String] = [:]
                for (key, val) in dict {
                    if let str = val as? String {
                        out[key] = str
                    } else if let arr = val as? [Any] {
                        out[key] = arr.map { "\($0)" }.joined(separator: ", ")
                    } else {
                        out[key] = "\(val)"
                    }
                }
                return out
            }
            return DigestSectionData(
                title: s["title"] as? String ?? "",
                items: stringItems
            )
        }

        return DigestData(
            date: json["date"] as? String ?? "",
            summary: json["summary"] as? String ?? "",
            activeHours: json["active_hours"] as? Double ?? 0,
            tasksCompleted: json["tasks_completed"] as? Int ?? 0,
            proceduresObserved: json["procedures_observed"] as? Int ?? 0,
            highlights: highlights,
            sections: sections
        )
    }

    private nonisolated static func buildMinimalDigest(
        from summary: [String: Any], date: String
    ) -> DigestData {
        DigestData(
            date: date,
            summary: "Activity recorded for \(date).",
            activeHours: summary["active_hours"] as? Double ?? 0,
            tasksCompleted: summary["task_count"] as? Int ?? 0,
            proceduresObserved: (summary["procedures_observed"] as? [Any])?.count ?? 0,
            highlights: [],
            sections: [
                DigestSectionData(
                    title: "Top Apps",
                    items: (summary["top_apps"] as? [[String: Any]] ?? []).prefix(5).map { app in
                        [
                            "title": app["app"] as? String ?? "Unknown",
                            "detail": "\(app["minutes"] as? Int ?? 0) min"
                        ]
                    }
                )
            ]
        )
    }
}
