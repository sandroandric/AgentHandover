import SwiftUI

/// Displays discovered SOPs in a master-detail layout using NavigationSplitView.
struct WorkflowInboxView: View {
    @StateObject private var sopManager = SOPIndexManager()
    @State private var filter: SOPFilter = .all
    @State private var selectedSOPID: String?

    /// Live lookup: always reflects the latest data from the polling index.
    private var selectedSOP: SOPEntry? {
        guard let id = selectedSOPID else { return nil }
        return sopManager.allSorted.first { $0.id == id }
    }

    enum SOPFilter: String, CaseIterable {
        case all = "All"
        case focus = "Focus"
        case passive = "Discovered"
        case drafts = "Drafts"
        case agentReady = "Agent Ready"
    }

    var body: some View {
        NavigationSplitView {
            sidebar
        } detail: {
            detailPane
        }
        .frame(minWidth: 750, minHeight: 500)
        .onAppear { sopManager.startPolling() }
        .onDisappear { sopManager.stopPolling() }
    }

    // MARK: - Sidebar

    private var sidebar: some View {
        VStack(alignment: .leading, spacing: 0) {
            headerBar
            Divider()

            if sopManager.index == nil {
                emptyState
            } else if filteredSOPs.isEmpty {
                noMatchState
            } else {
                sopList
            }
        }
        .navigationSplitViewColumnWidth(min: 280, ideal: 320, max: 450)
    }

    // MARK: - Detail pane

    private var detailPane: some View {
        Group {
            if let sop = selectedSOP {
                SOPDetailView(sop: sop, sopManager: sopManager)
            } else {
                VStack(spacing: 12) {
                    Image(systemName: "doc.text.magnifyingglass")
                        .font(.system(size: 40))
                        .foregroundColor(.secondary.opacity(0.4))
                    Text("Select a workflow")
                        .font(.title3)
                        .foregroundColor(.secondary)
                    Text("Choose a workflow from the sidebar to view its details.")
                        .font(.caption)
                        .foregroundColor(.secondary.opacity(0.7))
                        .multilineTextAlignment(.center)
                }
                .frame(maxWidth: .infinity, maxHeight: .infinity)
            }
        }
    }

    // MARK: - Header

    private var headerBar: some View {
        VStack(spacing: 10) {
            HStack {
                Text("Workflows")
                    .font(.title2)
                    .bold()
                Spacer()
                if let index = sopManager.index {
                    HStack(spacing: 12) {
                        StatPill(
                            count: index.approved_count,
                            label: "approved",
                            color: .secondary
                        )
                        if index.draft_count > 0 {
                            StatPill(
                                count: index.draft_count,
                                label: "drafts",
                                color: .secondary
                            )
                        }
                    }
                }
            }

            // Filter tabs
            HStack(spacing: 2) {
                ForEach(SOPFilter.allCases, id: \.self) { tab in
                    Button(action: { withAnimation(.easeInOut(duration: 0.15)) { filter = tab } }) {
                        Text(tab.rawValue)
                            .font(.caption)
                            .fontWeight(filter == tab ? .semibold : .regular)
                            .padding(.horizontal, 10)
                            .padding(.vertical, 5)
                            .background(filter == tab ? Color.accentColor.opacity(0.12) : Color.clear)
                            .foregroundColor(filter == tab ? .accentColor : .secondary)
                            .cornerRadius(6)
                    }
                    .buttonStyle(.plain)
                }
                Spacer()
            }
        }
        .padding(.horizontal, 16)
        .padding(.top, 14)
        .padding(.bottom, 8)
    }

    // MARK: - List

    private var sopList: some View {
        ScrollView {
            LazyVStack(spacing: 1) {
                ForEach(filteredSOPs) { sop in
                    SOPRow(sop: sop, isSelected: selectedSOPID == sop.id)
                        .contentShape(Rectangle())
                        .onTapGesture {
                            selectedSOPID = sop.id
                        }
                }
            }
            .padding(.vertical, 4)
        }
    }

    // MARK: - Empty states

    private var emptyState: some View {
        VStack(spacing: 10) {
            Image(systemName: "tray")
                .font(.system(size: 36))
                .foregroundColor(.secondary.opacity(0.5))
            Text("No workflows yet")
                .font(.headline)
                .foregroundColor(.secondary)
            Text("AgentHandover discovers workflows as you work.\nUse Record Workflow for instant capture.")
                .font(.caption)
                .foregroundColor(.secondary.opacity(0.7))
                .multilineTextAlignment(.center)
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .padding()
    }

    private var noMatchState: some View {
        VStack(spacing: 8) {
            Image(systemName: "line.3.horizontal.decrease.circle")
                .font(.title2)
                .foregroundColor(.secondary.opacity(0.5))
            Text("No \(filter.rawValue.lowercased()) workflows")
                .font(.subheadline)
                .foregroundColor(.secondary)
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .padding()
    }

    // MARK: - Filtering

    private var filteredSOPs: [SOPEntry] {
        switch filter {
        case .all: return sopManager.allSorted
        case .focus: return sopManager.allSorted.filter { $0.source == "focus" }
        case .passive: return sopManager.allSorted.filter { $0.source == "passive" }
        case .drafts: return sopManager.drafts
        case .agentReady: return sopManager.allSorted.filter { $0.lifecycleState == "agent_ready" }
        }
    }
}

// MARK: - Row

struct SOPRow: View {
    let sop: SOPEntry
    let isSelected: Bool

    @State private var isHovered = false

    var body: some View {
        VStack(alignment: .leading, spacing: 4) {
            // Title
            Text(sop.displayTitle)
                .font(.system(size: 13, weight: .medium))
                .lineLimit(1)
                .foregroundColor(.primary)

            // Tags + metadata row
            HStack(spacing: 6) {
                // Tags
                ForEach(sop.displayTags, id: \.self) { tag in
                    Text(tag)
                        .font(.system(size: 9, weight: .medium))
                        .foregroundColor(.secondary)
                        .padding(.horizontal, 5)
                        .padding(.vertical, 2)
                        .background(Color.primary.opacity(0.05))
                        .cornerRadius(3)
                }

                if !sop.displayTags.isEmpty {
                    Text("·")
                        .font(.system(size: 9))
                        .foregroundColor(.secondary.opacity(0.4))
                }

                Text(sop.sourceLabel)
                    .font(.system(size: 10))
                    .foregroundColor(.secondary.opacity(0.6))

                // Lifecycle state pill
                Text(sop.lifecycleLabel)
                    .font(.caption2)
                    .padding(.horizontal, 4)
                    .padding(.vertical, 1)
                    .background(sop.lifecycleColor.opacity(0.12))
                    .foregroundColor(sop.lifecycleColor)
                    .clipShape(Capsule())

                Spacer()

                Text(sop.relativeTime)
                    .font(.system(size: 10))
                    .foregroundColor(.secondary.opacity(0.5))
            }
        }
        .padding(.horizontal, 14)
        .padding(.vertical, 8)
        .background(
            isSelected
                ? Color.accentColor.opacity(0.12)
                : (isHovered ? Color.primary.opacity(0.03) : Color.clear)
        )
        .cornerRadius(6)
        .padding(.horizontal, 6)
        .onHover { isHovered = $0 }
    }
}

// MARK: - Confidence bar

struct ConfidenceBar: View {
    let value: Double

    var body: some View {
        GeometryReader { _ in
            ZStack(alignment: .leading) {
                RoundedRectangle(cornerRadius: 2)
                    .fill(Color.primary.opacity(0.08))
                RoundedRectangle(cornerRadius: 2)
                    .fill(Color.primary.opacity(value >= 0.8 ? 0.4 : 0.2))
                    .frame(width: CGFloat(value) * 30)
            }
        }
        .frame(width: 30, height: 4)
    }
}

// MARK: - Status Badge

struct StatusBadge: View {
    let status: String

    var body: some View {
        Text(status.capitalized)
            .font(.system(size: 9, weight: .semibold))
            .padding(.horizontal, 6)
            .padding(.vertical, 3)
            .background(backgroundColor.opacity(0.12))
            .foregroundColor(backgroundColor)
            .cornerRadius(4)
    }

    private var backgroundColor: Color {
        .secondary
    }
}

// MARK: - Stat Pill (header)

struct StatPill: View {
    let count: Int
    let label: String
    let color: Color

    var body: some View {
        HStack(spacing: 3) {
            Text("\(count)")
                .font(.system(size: 11, weight: .bold, design: .rounded))
            Text(label)
                .font(.system(size: 10))
        }
        .foregroundColor(color)
        .padding(.horizontal, 8)
        .padding(.vertical, 3)
        .background(color.opacity(0.1))
        .cornerRadius(10)
    }
}
