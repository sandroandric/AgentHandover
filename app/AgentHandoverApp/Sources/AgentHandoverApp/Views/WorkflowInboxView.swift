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
        case agentReady = "Ready"
    }

    // Contra design tokens
    private let darkNavy = Color(red: 0.09, green: 0.10, blue: 0.12)
    private let warmOrange = Color(red: 0.92, green: 0.57, blue: 0.20)
    private let goldenYellow = Color(red: 1.0, green: 0.74, blue: 0.07)
    private let warmCream = Color(red: 1.0, green: 0.96, blue: 0.88)
    private let lightGray = Color(red: 0.96, green: 0.96, blue: 0.96)
    private let brightGreen = Color(red: 0.18, green: 0.80, blue: 0.34)
    private let cardRadius: CGFloat = 14
    private let contraBorder: CGFloat = 1.5

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
            Rectangle()
                .fill(darkNavy.opacity(0.08))
                .frame(height: 1)

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
                VStack(spacing: 14) {
                    ZStack {
                        RoundedRectangle(cornerRadius: cardRadius)
                            .fill(warmCream)
                            .frame(width: 64, height: 64)
                            .overlay(
                                RoundedRectangle(cornerRadius: cardRadius)
                                    .stroke(darkNavy.opacity(0.12), lineWidth: contraBorder)
                            )
                        Image(systemName: "doc.text.magnifyingglass")
                            .font(.system(size: 28))
                            .foregroundColor(darkNavy.opacity(0.4))
                    }
                    Text("Select a workflow")
                        .font(.system(size: 17, weight: .bold, design: .rounded))
                        .foregroundColor(darkNavy)
                    Text("Choose a workflow from the sidebar to view its details.")
                        .font(.system(size: 13))
                        .foregroundColor(darkNavy.opacity(0.5))
                        .multilineTextAlignment(.center)
                }
                .frame(maxWidth: .infinity, maxHeight: .infinity)
            }
        }
    }

    // MARK: - Header

    private var headerBar: some View {
        VStack(spacing: 12) {
            HStack {
                Text("Workflows")
                    .font(.system(size: 24, weight: .bold, design: .rounded))
                    .foregroundColor(darkNavy)
                Spacer()
                if let index = sopManager.index {
                    HStack(spacing: 10) {
                        StatPill(
                            count: index.approved_count,
                            label: "approved",
                            darkNavy: darkNavy,
                            brightGreen: brightGreen
                        )
                        if index.draft_count > 0 {
                            StatPill(
                                count: index.draft_count,
                                label: "drafts",
                                darkNavy: darkNavy,
                                warmOrange: warmOrange
                            )
                        }
                    }
                }
            }

            // Filter tabs - Contra segmented style, scrollable
            ScrollView(.horizontal, showsIndicators: false) {
                HStack(spacing: 4) {
                    ForEach(SOPFilter.allCases, id: \.self) { tab in
                        Button(action: { withAnimation(.spring(response: 0.3, dampingFraction: 0.8)) { filter = tab } }) {
                            Text(tab.rawValue)
                                .font(.system(size: 11, weight: filter == tab ? .bold : .medium))
                                .lineLimit(1)
                                .fixedSize()
                                .padding(.horizontal, 10)
                                .padding(.vertical, 6)
                                .background(
                                    filter == tab
                                        ? RoundedRectangle(cornerRadius: 8)
                                            .fill(goldenYellow)
                                        : nil
                                )
                                .overlay(
                                    filter == tab
                                        ? RoundedRectangle(cornerRadius: 8)
                                            .stroke(darkNavy, lineWidth: 1.5)
                                        : nil
                                )
                                .foregroundColor(darkNavy)
                        }
                        .buttonStyle(.plain)
                    }
                }
            }
            .padding(3)
            .background(
                RoundedRectangle(cornerRadius: 10)
                    .fill(lightGray)
            )
        }
        .padding(.horizontal, 16)
        .padding(.top, 16)
        .padding(.bottom, 10)
    }

    // MARK: - List

    private var sopList: some View {
        ScrollView {
            LazyVStack(spacing: 2) {
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
        VStack(spacing: 12) {
            ZStack {
                RoundedRectangle(cornerRadius: cardRadius)
                    .fill(warmCream)
                    .frame(width: 56, height: 56)
                    .overlay(
                        RoundedRectangle(cornerRadius: cardRadius)
                            .stroke(darkNavy.opacity(0.12), lineWidth: contraBorder)
                    )
                Image(systemName: "tray")
                    .font(.system(size: 26))
                    .foregroundColor(warmOrange)
            }
            Text("No workflows yet")
                .font(.system(size: 15, weight: .bold, design: .rounded))
                .foregroundColor(darkNavy)
            Text("AgentHandover discovers workflows as you work.\nUse Record Workflow for instant capture.")
                .font(.system(size: 12))
                .foregroundColor(darkNavy.opacity(0.5))
                .multilineTextAlignment(.center)
                .lineSpacing(3)
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .padding()
    }

    private var noMatchState: some View {
        VStack(spacing: 10) {
            Image(systemName: "line.3.horizontal.decrease.circle")
                .font(.system(size: 22))
                .foregroundColor(warmOrange)
            Text("No \(filter.rawValue.lowercased()) workflows")
                .font(.system(size: 14, weight: .semibold, design: .rounded))
                .foregroundColor(darkNavy)
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

    // Contra design tokens
    private let darkNavy = Color(red: 0.09, green: 0.10, blue: 0.12)
    private let warmOrange = Color(red: 0.92, green: 0.57, blue: 0.20)
    private let goldenYellow = Color(red: 1.0, green: 0.74, blue: 0.07)
    private let warmCream = Color(red: 1.0, green: 0.96, blue: 0.88)
    private let lightGray = Color(red: 0.96, green: 0.96, blue: 0.96)
    private let cardRadius: CGFloat = 14
    private let contraBorder: CGFloat = 1.5

    var body: some View {
        HStack(spacing: 12) {
            // Source icon with Contra border
            ZStack {
                RoundedRectangle(cornerRadius: 9)
                    .fill(
                        isSelected
                            ? goldenYellow.opacity(0.15)
                            : lightGray
                    )
                    .frame(width: 38, height: 38)
                    .overlay(
                        RoundedRectangle(cornerRadius: 9)
                            .stroke(darkNavy.opacity(isSelected ? 0.3 : 0.12), lineWidth: contraBorder)
                    )
                Image(systemName: sop.sourceIcon)
                    .font(.system(size: 15))
                    .foregroundColor(isSelected ? warmOrange : darkNavy.opacity(0.5))
            }

            VStack(alignment: .leading, spacing: 5) {
                // Title
                Text(sop.displayTitle)
                    .font(.system(size: 13, weight: .medium))
                    .lineLimit(2)
                    .foregroundColor(darkNavy)

                // Metadata row
                HStack(spacing: 6) {
                    // Lifecycle state pill
                    Text(sop.lifecycleLabel)
                        .font(.system(size: 9, weight: .bold))
                        .padding(.horizontal, 7)
                        .padding(.vertical, 2)
                        .background(sop.lifecycleColor.opacity(0.1))
                        .foregroundColor(sop.lifecycleColor)
                        .clipShape(Capsule())
                        .overlay(
                            Capsule()
                                .stroke(darkNavy.opacity(0.08), lineWidth: 1)
                        )

                    if sop.confidence > 0 {
                        Text(String(format: "%.0f%%", sop.confidence * 100))
                            .font(.system(size: 10))
                            .foregroundColor(darkNavy.opacity(0.5))
                    }

                    Spacer()

                    Text(sop.relativeTime)
                        .font(.system(size: 10))
                        .foregroundColor(darkNavy.opacity(0.35))
                }
            }
        }
        .padding(.horizontal, 12)
        .padding(.vertical, 10)
        .background(
            RoundedRectangle(cornerRadius: 10)
                .fill(
                    isSelected
                        ? goldenYellow.opacity(0.15)
                        : (isHovered ? lightGray : Color.clear)
                )
        )
        .overlay(
            RoundedRectangle(cornerRadius: 10)
                .stroke(
                    isSelected ? darkNavy.opacity(0.15) : Color.clear,
                    lineWidth: contraBorder
                )
        )
        .padding(.horizontal, 6)
        .onHover { isHovered = $0 }
    }
}

// MARK: - Confidence bar

struct ConfidenceBar: View {
    let value: Double

    private let darkNavy = Color(red: 0.09, green: 0.10, blue: 0.12)
    private let warmOrange = Color(red: 0.92, green: 0.57, blue: 0.20)

    var body: some View {
        GeometryReader { _ in
            ZStack(alignment: .leading) {
                RoundedRectangle(cornerRadius: 2)
                    .fill(darkNavy.opacity(0.08))
                RoundedRectangle(cornerRadius: 2)
                    .fill(warmOrange.opacity(value >= 0.8 ? 0.7 : 0.4))
                    .frame(width: CGFloat(value) * 30)
            }
        }
        .frame(width: 30, height: 4)
    }
}

// MARK: - Status Badge

struct StatusBadge: View {
    let status: String

    private let darkNavy = Color(red: 0.09, green: 0.10, blue: 0.12)

    var body: some View {
        Text(status.capitalized)
            .font(.system(size: 9, weight: .bold))
            .padding(.horizontal, 7)
            .padding(.vertical, 3)
            .background(darkNavy.opacity(0.06))
            .foregroundColor(darkNavy.opacity(0.6))
            .clipShape(Capsule())
            .overlay(
                Capsule()
                    .stroke(darkNavy.opacity(0.12), lineWidth: 1)
            )
    }
}

// MARK: - Stat Pill (header)

struct StatPill: View {
    let count: Int
    let label: String

    // Contra tokens passed from parent for flexibility
    private let navyColor: Color
    private let accentColor: Color

    /// Approved-style pill (green accent).
    init(count: Int, label: String, darkNavy: Color, brightGreen: Color) {
        self.count = count
        self.label = label
        self.navyColor = darkNavy
        self.accentColor = brightGreen
    }

    /// Draft/warning-style pill (orange accent).
    init(count: Int, label: String, darkNavy: Color, warmOrange: Color) {
        self.count = count
        self.label = label
        self.navyColor = darkNavy
        self.accentColor = warmOrange
    }

    var body: some View {
        HStack(spacing: 3) {
            Text("\(count)")
                .font(.system(size: 10, weight: .bold, design: .rounded))
            Text(label)
                .font(.system(size: 9, weight: .medium))
        }
        .lineLimit(1)
        .fixedSize()
        .foregroundColor(navyColor)
        .padding(.horizontal, 8)
        .padding(.vertical, 4)
        .background(accentColor.opacity(0.12))
        .clipShape(Capsule())
        .overlay(
            Capsule()
                .stroke(navyColor.opacity(0.12), lineWidth: 1.5)
        )
    }
}
