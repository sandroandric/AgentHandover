import AppKit
import SwiftUI
import UniformTypeIdentifiers

/// The Commonplace window: one search over everything AgentHandover knows
/// on this Mac, your Skills, Memory (what you saw) and your days, grouped
/// by kind.  Keyboard first: the search field has focus when the window
/// opens, the arrows move through the results across groups, Return opens
/// the selected one (a moment here, a Skill in Workflows, a day in the
/// Daily Digest), Command-1 to 4 pick the scope, Command-Delete forgets a
/// moment after a confirmation.
struct CommonplaceView: View {
    @EnvironmentObject private var delegate: AppDelegate
    @ObservedObject private var center = MemoryCenter.shared
    @StateObject private var model = CommonplaceWindowModel()
    @StateObject private var thumbnails = MemoryThumbnailStore()
    @FocusState private var searchFocused: Bool
    @State private var openMoment: MemoryMoment?
    /// `[skills] enabled = false`: no new Days or Skills are made.
    @State private var memoryOnly = false
    @Environment(\.openWindow) private var openWindow

    private let format = MemoryFormat()

    var body: some View {
        VStack(spacing: 0) {
            searchBar
            scopeBar
            Divider()
            agentsLine
            banner
            if memoryOnly, model.scope == .days || model.scope == .skills {
                memoryOnlyLine
            }
            content
                .frame(maxWidth: .infinity, maxHeight: .infinity)
            Divider()
            footer
        }
        .frame(minWidth: 780, minHeight: 500)
        .background(Color(nsColor: .windowBackgroundColor))
        .task {
            searchFocused = true
            memoryOnly = await Task.detached { SkillsMode.load().isMemoryOnly }.value
            await center.refresh()
            await model.checkWaiting(status: center.status)
            // Nothing typed yet: the newest of each kind.
            model.search(force: true)
        }
        .onChange(of: center.forgetCount) { _, _ in model.search(force: true) }
        .sheet(item: $openMoment) { moment in
            MomentSheet(moment: moment, thumbnails: thumbnails, forget: { forget(moment) })
        }
    }

    // MARK: - Search

    private var memoryOnlyLine: some View {
        HStack(spacing: 6) {
            Image(systemName: "info.circle")
            Text(SkillsMode.memoryOnlyNote)
            Spacer()
            Button("Learn Skills\u{2026}") { openWindow(id: "memory-settings") }
                .buttonStyle(.link)
        }
        .font(.system(size: 11))
        .foregroundStyle(.secondary)
        .padding(.horizontal, 16)
        .padding(.vertical, 6)
    }

    private var searchBar: some View {
        HStack(spacing: 10) {
            Image(systemName: "magnifyingglass")
                .font(.system(size: 17, weight: .regular))
                .foregroundStyle(.secondary)
            TextField(model.scope.placeholder, text: $model.query)
                .textFieldStyle(.plain)
                .font(.system(size: 18))
                .focused($searchFocused)
                .onSubmit {
                    if model.isShowingResults(for: model.query), let selected = model.selected {
                        open(selected)
                    } else {
                        model.search(force: true)
                    }
                }
                .onKeyPress(.downArrow) { model.moveSelection(1); return .handled }
                .onKeyPress(.upArrow) { model.moveSelection(-1); return .handled }
                .onChange(of: model.query) { _, _ in model.searchSoon() }
            if model.searching {
                ProgressView().controlSize(.small)
            } else if !model.query.isEmpty {
                Button {
                    model.query = ""
                    searchFocused = true
                } label: {
                    Image(systemName: "xmark.circle.fill").foregroundStyle(.tertiary)
                }
                .buttonStyle(.plain)
                .help("Clear")
            }
        }
        .padding(.horizontal, 18)
        .padding(.top, 16)
        .padding(.bottom, 12)
    }

    /// The scope chips, and the filters the scope applies.
    private var scopeBar: some View {
        HStack(spacing: 6) {
            ForEach(Array(CommonplaceScope.allCases.enumerated()), id: \.element) { index, scope in
                ScopeChip(title: scope.rawValue, selected: model.scope == scope) {
                    model.scope = scope
                }
                .keyboardShortcut(KeyEquivalent(Character(String(index + 1))), modifiers: .command)
                .help("\(scope.rawValue) (Command-\(index + 1))")
            }
            Spacer(minLength: 12)
            if model.scope.showsTimeFilter {
                Picker("Time", selection: $model.range) {
                    ForEach(MemoryRange.allCases) { Text($0.rawValue).tag($0) }
                }
                .pickerStyle(.segmented)
                .labelsHidden()
                .frame(width: 240)
            }
            if model.scope.showsAppFilter {
                Picker("App", selection: $model.app) {
                    Text("All Apps").tag(String?.none)
                    if !model.knownApps.isEmpty { Divider() }
                    ForEach(model.knownApps, id: \.self) { app in
                        Text(app).tag(Optional(app))
                    }
                }
                .labelsHidden()
                .frame(width: 160)
            }
        }
        .padding(.horizontal, 18)
        .padding(.bottom, 12)
        .onChange(of: model.scope) { _, _ in model.scopeChanged() }
        .onChange(of: model.range) { _, _ in model.search(force: false) }
        .onChange(of: model.app) { _, _ in model.search(force: false) }
    }

    // MARK: - What agents see

    private var agentsLine: some View {
        let line = AgentsLine(status: center.status)
        return HStack(spacing: 6) {
            Text(line.text)
                .font(.system(size: 11))
                .foregroundStyle(.secondary)
            if let title = line.actionTitle {
                Text("\u{00B7}").font(.system(size: 11)).foregroundStyle(.tertiary)
                Button(title) { perform(line.action) }
                    .buttonStyle(.link)
                    .font(.system(size: 11))
            }
            Spacer()
        }
        .padding(.horizontal, 18)
        .padding(.vertical, 7)
        .overlay(alignment: .bottom) { Divider() }
    }

    private func perform(_ action: AgentsLine.Action?) {
        switch action {
        case .allow?:
            NSApp.activate(ignoringOtherApps: true)
            let alert = NSAlert()
            alert.messageText = "Let agents search your Memory?"
            alert.informativeText = "Agents connected to AgentHandover can then search the descriptions, apps, window titles and times of what was on your screen. They never get screenshots. You can turn this off here or in Memory Settings."
            alert.addButton(withTitle: "Let Agents Search")
            alert.addButton(withTitle: "Cancel")
            guard alert.runModal() == .alertFirstButtonReturn else { return }
            Task { await center.setAgentsAccess(true) }
        case .disallow?:
            Task { await center.setAgentsAccess(false) }
        case nil:
            break
        }
    }

    // MARK: - Banner

    @ViewBuilder
    private var banner: some View {
        if model.scope.kinds.contains(.memory) {
            if case .paused(let until) = center.menuState {
                bannerRow(
                    icon: "pause.circle",
                    text: "Memory is paused" + (until.map { " until \(format.pointInTime($0, now: Date()))" } ?? "")
                        + ". Search still finds what was kept before.",
                    action: "Resume"
                ) { Task { await center.resume() } }
            } else if case .problem(let problem) = model.waiting {
                bannerRow(icon: "hourglass", text: problem.message,
                          action: actionTitle(for: problem)) { perform(problem) }
            }
        }
        if let notice = center.notice {
            bannerRow(icon: "info.circle", text: notice, action: "Dismiss") { center.notice = nil }
        }
    }

    private func bannerRow(icon: String, text: String, action: String?, perform: @escaping () -> Void) -> some View {
        HStack(spacing: 10) {
            Image(systemName: icon).foregroundStyle(.secondary)
            Text(text)
                .font(.system(size: 12))
                .fixedSize(horizontal: false, vertical: true)
            Spacer()
            if let action {
                Button(action, action: perform).controlSize(.small)
            }
        }
        .padding(.horizontal, 18)
        .padding(.vertical, 8)
        .background(Color.primary.opacity(0.04))
        .overlay(alignment: .bottom) { Divider() }
    }

    // MARK: - Content

    @ViewBuilder
    private var content: some View {
        if workerDown {
            stateView(icon: "bolt.horizontal.circle",
                      text: "AgentHandover's worker is not running, so Commonplace cannot be searched.",
                      action: "Start Worker") { perform(.workerNotRunning) }
        } else if model.isBrowsing, !model.hasSomethingToBrowse, singleScopeFailure == nil {
            // Nothing typed and nothing kept yet (or still loading): what to try.
            if model.scope == .all {
                examples
            } else {
                stateView(icon: "text.magnifyingglass", text: introText, action: nil, perform: {})
            }
        } else if let only = singleScopeFailure {
            failureState(only)
        } else if model.phase == .results && !model.isBrowsing && model.results.isEmpty(model.scope) {
            stateView(
                icon: "magnifyingglass",
                text: "Nothing in \(model.scope == .all ? "Commonplace" : model.scope.rawValue) matches \u{201C}\(model.lastQuery)\u{201D}"
                    + (model.hasFilters ? " in this time\(model.request.app == nil ? "" : " and app")." : "."),
                action: model.hasFilters ? "Clear Filters" : nil
            ) { model.clearFilters() }
        } else {
            HSplitView {
                resultList
                    .frame(minWidth: 340, idealWidth: 400)
                detail
                    .frame(minWidth: 320, maxWidth: .infinity, maxHeight: .infinity)
            }
        }
    }

    private var workerDown: Bool {
        if let error = center.statusError, error == .workerNotRunning || error == .noToken { return true }
        return model.phase == .results && model.results.workerUnreachable
    }

    private var introText: String {
        switch model.scope {
        case .all:
            return "Search everything AgentHandover keeps on this Mac: what you saw, your days and your Skills."
        case .skills:
            return "Find a Skill by what it does: \u{201C}send the weekly report\u{201D}."
        case .memory:
            return "Search what you saw by meaning: \u{201C}the invoice from Acme\u{201D}, \u{201C}that chart about churn\u{201D}."
        case .days:
            return "Find a day by what you did: \u{201C}the day I fixed the banner\u{201D}."
        }
    }

    /// The empty All window: searches to try and requests for an agent.
    private var examples: some View {
        ScrollView {
            VStack(spacing: 22) {
                VStack(spacing: 10) {
                    Image(systemName: "text.magnifyingglass")
                        .font(.system(size: 30, weight: .light))
                        .foregroundStyle(.tertiary)
                    Text(introText)
                        .font(.system(size: 13))
                        .foregroundStyle(.secondary)
                        .multilineTextAlignment(.center)
                        .fixedSize(horizontal: false, vertical: true)
                }
                HStack(alignment: .top, spacing: 28) {
                    exampleColumn("Try a search") {
                        ForEach(CommonplaceExamples.searches, id: \.self) { query in
                            exampleButton(query, icon: "magnifyingglass", help: "Search for this") {
                                model.query = query
                                searchFocused = true
                            }
                        }
                    }
                    exampleColumn("Ask your agent") {
                        ForEach(CommonplaceExamples.agentRequests, id: \.self) { request in
                            exampleButton(request, icon: "doc.on.doc", help: "Copy, then paste it to Claude Code, Codex or any connected agent") {
                                NSPasteboard.general.clearContents()
                                NSPasteboard.general.setString(request, forType: .string)
                                center.notice = "Copied. Paste it to Claude Code, Codex or any agent connected to AgentHandover."
                            }
                        }
                    }
                }
                Text("Agents search your Skills, days and profile. They see Memory only when you allow it above.")
                    .font(.system(size: 11))
                    .foregroundStyle(.tertiary)
                    .multilineTextAlignment(.center)
            }
            .frame(maxWidth: 640)
            .padding(.horizontal, 28)
            .padding(.vertical, 32)
            .frame(maxWidth: .infinity)
        }
    }

    private func exampleColumn<Content: View>(_ title: String, @ViewBuilder content: () -> Content) -> some View {
        VStack(alignment: .leading, spacing: 6) {
            Text(title)
                .font(.system(size: 11, weight: .semibold))
                .foregroundStyle(.secondary)
                .padding(.bottom, 2)
            content()
        }
        .frame(maxWidth: .infinity, alignment: .leading)
    }

    private func exampleButton(_ text: String, icon: String, help: String, action: @escaping () -> Void) -> some View {
        Button(action: action) {
            HStack(alignment: .firstTextBaseline, spacing: 8) {
                Image(systemName: icon)
                    .font(.system(size: 11))
                    .foregroundStyle(.tertiary)
                Text(text)
                    .font(.system(size: 13))
                    .foregroundStyle(.primary)
                    .multilineTextAlignment(.leading)
                    .fixedSize(horizontal: false, vertical: true)
                Spacer(minLength: 0)
            }
            .padding(.horizontal, 10)
            .padding(.vertical, 7)
            .background(RoundedRectangle(cornerRadius: 7, style: .continuous).fill(Color.primary.opacity(0.05)))
            .contentShape(Rectangle())
        }
        .buttonStyle(.plain)
        .help(help)
    }

    /// A single-kind scope whose search failed: the whole window says why.
    private var singleScopeFailure: CommonplaceSection? {
        guard model.scope != .all, model.phase == .results,
              let section = model.sections.first, section.error != nil else { return nil }
        return section
    }

    @ViewBuilder
    private func failureState(_ section: CommonplaceSection) -> some View {
        if section.kind == .memory, let problem = model.memoryProblem {
            stateView(icon: icon(for: problem), text: problem.message,
                      action: actionTitle(for: problem)) { perform(problem) }
        } else {
            stateView(icon: "exclamationmark.triangle", text: section.errorMessage, action: "Try Again") {
                model.search(force: true)
            }
        }
    }

    private func stateView(icon: String, text: String, action: String?, perform: @escaping () -> Void) -> some View {
        VStack(spacing: 14) {
            Image(systemName: icon)
                .font(.system(size: 30, weight: .light))
                .foregroundStyle(.tertiary)
            Text(text)
                .font(.system(size: 13))
                .foregroundStyle(.secondary)
                .multilineTextAlignment(.center)
                .frame(maxWidth: 420)
                .fixedSize(horizontal: false, vertical: true)
            if let action {
                Button(action, action: perform)
                    .controlSize(.large)
            }
        }
        .padding(40)
    }

    private func icon(for problem: MemoryProblem) -> String {
        switch problem {
        case .workerNotRunning: return "bolt.horizontal.circle"
        case .off: return "clock.arrow.circlepath"
        case .ollamaNotRunning, .ollamaTooOld: return "cpu"
        case .modelMissing: return "arrow.down.circle"
        case .waiting: return "hourglass"
        }
    }

    private func actionTitle(for problem: MemoryProblem) -> String? {
        switch problem {
        case .workerNotRunning: return "Start Worker"
        case .off: return "Turn On Memory"
        case .ollamaNotRunning: return "Open Ollama"
        case .ollamaTooOld: return "Download Ollama"
        case .modelMissing: return "Download\u{2026}"
        // It clears by itself; the worker retries on its own.
        case .waiting: return nil
        }
    }

    private func perform(_ problem: MemoryProblem) {
        switch problem {
        case .workerNotRunning:
            Task {
                await Task.detached { _ = ServiceController.startWorker() }.value
                try? await Task.sleep(nanoseconds: 2_000_000_000)
                await center.refresh()
                model.search(force: true)
            }
        case .off:
            Task {
                await center.turnOn()
                model.search(force: true)
            }
        case .ollamaNotRunning:
            let app = URL(fileURLWithPath: "/Applications/Ollama.app")
            if FileManager.default.fileExists(atPath: app.path) {
                NSWorkspace.shared.openApplication(at: app, configuration: NSWorkspace.OpenConfiguration())
            } else if let url = URL(string: "https://ollama.com/download/mac") {
                NSWorkspace.shared.open(url)
            }
        case .ollamaTooOld:
            if let url = URL(string: "https://ollama.com/download/mac") { NSWorkspace.shared.open(url) }
        case .modelMissing(let name):
            delegate.showMemoryModelDownload(model: name, turnOn: false)
        case .waiting:
            Task {
                await center.refresh()
                await model.checkWaiting(status: center.status)
                model.search(force: true)
            }
        }
    }

    // MARK: - Results

    private var resultList: some View {
        List(selection: $model.selectedID) {
            ForEach(model.sections) { section in
                Section {
                    sectionBody(section)
                } header: {
                    sectionHeader(section)
                }
            }
        }
        .listStyle(.inset)
        .onKeyPress(.return) {
            guard let selected = model.selected else { return .ignored }
            open(selected)
            return .handled
        }
    }

    private func sectionHeader(_ section: CommonplaceSection) -> some View {
        HStack(spacing: 6) {
            Text(section.kind.title)
                .font(.system(size: 11, weight: .semibold))
                .foregroundStyle(.secondary)
            if section.searched, section.error == nil {
                Text(section.countLabel)
                    .font(.system(size: 11))
                    .foregroundStyle(.tertiary)
                    .monospacedDigit()
            }
            if model.isBrowsing, section.searched, !section.items.isEmpty {
                Text("\u{00B7} newest first")
                    .font(.system(size: 11))
                    .foregroundStyle(.tertiary)
            }
            if let reason = section.byWordsReason {
                Text("\u{00B7} searched by words")
                    .font(.system(size: 11))
                    .foregroundStyle(.tertiary)
                    .help(reason.isEmpty ? "Searched by the words of your query." : reason)
            }
            Spacer()
        }
    }

    @ViewBuilder
    private func sectionBody(_ section: CommonplaceSection) -> some View {
        if section.error != nil {
            sectionFailure(section)
        } else if section.items.isEmpty {
            Text(section.searched ? (model.isBrowsing ? section.kind.noneYet : section.kind.noMatches) : " ")
                .font(.system(size: 12))
                .foregroundStyle(.tertiary)
                .selectionDisabled()
        } else {
            ForEach(section.items) { item in
                row(item)
                    .tag(item.id)
                    .contextMenu {
                        Button("Open") { open(item) }
                        if let moment = item.moment {
                            Button("Forget This Moment\u{2026}") { forget(moment) }
                        }
                    }
            }
            if section.hiddenCount > 0 {
                Button(section.showAllLabel) { model.scope = CommonplaceScope(section.kind) }
                    .buttonStyle(.link)
                    .font(.system(size: 12))
                    .selectionDisabled()
            }
        }
    }

    @ViewBuilder
    private func sectionFailure(_ section: CommonplaceSection) -> some View {
        HStack(spacing: 8) {
            if section.kind == .memory, let problem = model.memoryProblem {
                Text(problem.message)
                    .font(.system(size: 12))
                    .foregroundStyle(.secondary)
                    .fixedSize(horizontal: false, vertical: true)
                Spacer()
                if let action = actionTitle(for: problem) {
                    Button(action) { perform(problem) }.controlSize(.small)
                }
            } else {
                Text(section.errorMessage)
                    .font(.system(size: 12))
                    .foregroundStyle(.secondary)
                    .fixedSize(horizontal: false, vertical: true)
                Spacer()
                Button("Try Again") { model.search(force: true) }.controlSize(.small)
            }
        }
        .padding(.vertical, 2)
        .selectionDisabled()
    }

    @ViewBuilder
    private func row(_ item: CommonplaceItem) -> some View {
        switch item {
        case .skill(let skill): SkillRow(skill: skill)
        case .moment(let moment): MomentRow(moment: moment, thumbnails: thumbnails, format: format)
        case .day(let day): DayRow(day: day)
        }
    }

    @ViewBuilder
    private var detail: some View {
        switch model.selected {
        case .moment(let moment)?:
            ScrollView {
                VStack(alignment: .leading, spacing: 14) {
                    ThumbnailView(moment: moment, thumbnails: thumbnails, maxHeight: 260)
                    MomentFacts(moment: moment, format: format)
                    HStack {
                        Button("Open") { openMoment = moment }
                            .help("Return")
                        Spacer()
                        Button("Forget This Moment\u{2026}", role: .destructive) { forget(moment) }
                            .keyboardShortcut(.delete, modifiers: .command)
                            .help("Command-Delete")
                    }
                }
                .padding(18)
            }
        case .skill(let skill)?:
            ScrollView {
                VStack(alignment: .leading, spacing: 10) {
                    SkillFacts(skill: skill)
                    Button("Open in Workflows") { open(.skill(skill)) }
                        .help("Return")
                        .padding(.top, 6)
                }
                .frame(maxWidth: .infinity, alignment: .leading)
                .padding(18)
            }
        case .day(let day)?:
            ScrollView {
                VStack(alignment: .leading, spacing: 10) {
                    DayFacts(day: day)
                    Button("Open in Daily Digest") { open(.day(day)) }
                        .help("Return")
                        .padding(.top, 6)
                }
                .frame(maxWidth: .infinity, alignment: .leading)
                .padding(18)
            }
        case nil:
            Text("Select a result")
                .foregroundStyle(.tertiary)
                .frame(maxWidth: .infinity, maxHeight: .infinity)
        }
    }

    private func open(_ item: CommonplaceItem) {
        switch item.open(navigator: AppNavigator.shared) {
        case .moment(let moment):
            openMoment = moment
        case .window(let id):
            NSApp.setActivationPolicy(.regular)
            openWindow(id: id)
            NSApp.activate(ignoringOtherApps: true)
        }
    }

    private func forget(_ moment: MemoryMoment) {
        Task {
            if await center.forget(.moment(moment)) {
                openMoment = nil
                thumbnails.forget(moment.thumbID)
            }
        }
    }

    // MARK: - Footer

    private var footer: some View {
        HStack(spacing: 12) {
            Text(center.status.map { format.statusLine($0, now: Date()) } ?? " ")
                .font(.system(size: 11))
                .foregroundStyle(.secondary)
            Spacer()
            Button("Memory Settings\u{2026}") {
                NSApp.setActivationPolicy(.regular)
                openWindow(id: "memory-settings")
            }
            .buttonStyle(.link)
            .font(.system(size: 11))
        }
        .padding(.horizontal, 18)
        .padding(.vertical, 8)
    }
}

/// A scope chip: quiet when not chosen, filled when chosen.
private struct ScopeChip: View {
    let title: String
    let selected: Bool
    let action: () -> Void

    var body: some View {
        Button(action: action) {
            Text(title)
                .font(.system(size: 12, weight: selected ? .semibold : .regular))
                .foregroundStyle(selected ? .primary : .secondary)
                .padding(.horizontal, 11)
                .padding(.vertical, 4)
                .background(Capsule().fill(Color.primary.opacity(selected ? 0.1 : 0)))
                .overlay(Capsule().stroke(Color.primary.opacity(selected ? 0 : 0.12)))
                .contentShape(Capsule())
        }
        .buttonStyle(.plain)
        .accessibilityAddTraits(selected ? .isSelected : [])
    }
}

// MARK: - The window's model

@MainActor
final class CommonplaceWindowModel: ObservableObject {
    enum Phase: Equatable {
        case idle
        case results
    }

    enum Waiting: Equatable {
        case none
        case problem(MemoryProblem)
    }

    @Published var query = ""
    @Published var scope: CommonplaceScope = .all
    @Published var range: MemoryRange = .anyTime
    @Published var app: String?
    @Published var selectedID: String?
    @Published private(set) var results = CommonplaceResults.empty
    @Published private(set) var knownApps: [String] = []
    @Published private(set) var phase: Phase = .idle
    @Published private(set) var searching = false
    /// Why new screens are not being embedded (status `waiting`).
    @Published private(set) var waiting: Waiting = .none
    /// Why the Memory search failed, diagnosed (Ollama, its model, off).
    @Published private(set) var memoryProblem: MemoryProblem?
    /// The query the shown results answer.
    @Published private(set) var lastQuery = ""

    private let api: MemoryAPI
    private let diagnose: (String) async -> MemoryProblem
    /// The search under way (debounced or not).
    private(set) var pending: Task<Void, Never>?
    /// The request the shown results answer.
    private(set) var lastRequest: CommonplaceRequest?

    init(api: MemoryAPI = MemoryAPI(), diagnose: @escaping (String) async -> MemoryProblem = CommonplaceWindowModel.diagnose) {
        self.api = api
        self.diagnose = diagnose
    }

    /// What the window asks for now.
    var request: CommonplaceRequest {
        CommonplaceRequest(query: query, scope: scope, range: range, app: app)
    }

    var sections: [CommonplaceSection] { results.sections(scope) }
    var visibleItems: [CommonplaceItem] { results.visibleItems(scope) }
    var selected: CommonplaceItem? { visibleItems.first { $0.id == selectedID } }
    var hasFilters: Bool { request.range != .anyTime || request.app != nil }

    /// Nothing typed: the window lists the newest of each kind (the
    /// worker's answer to an empty query) instead of search results.
    var isBrowsing: Bool { request.query.isEmpty }

    /// Browsing, and the newest-first answer holds something to show.
    var hasSomethingToBrowse: Bool {
        isBrowsing && phase == .results && lastQuery.isEmpty && !results.isEmpty(scope)
    }

    func isShowingResults(for query: String) -> Bool {
        phase == .results && lastQuery == query.trimmingCharacters(in: .whitespacesAndNewlines)
    }

    func clearFilters() {
        range = .anyTime
        app = nil
        search(force: true)
    }

    /// Move the selection through every shown result, group after group.
    func moveSelection(_ step: Int) {
        let items = visibleItems
        guard !items.isEmpty else { return }
        let index = items.firstIndex { $0.id == selectedID } ?? (step > 0 ? -1 : items.count)
        selectedID = items[min(max(index + step, 0), items.count - 1)].id
    }

    /// A new scope: the app filter belongs to Memory alone; the shown
    /// answer is kept when it already covers the scope.
    func scopeChanged() {
        if !scope.showsAppFilter { app = nil }
        search(force: false)
    }

    /// Search after a short pause in typing.
    func searchSoon() {
        pending?.cancel()
        pending = Task { [weak self] in
            try? await Task.sleep(nanoseconds: 350_000_000)
            guard !Task.isCancelled else { return }
            await self?.run(force: false)
        }
    }

    func search(force: Bool) {
        pending?.cancel()
        pending = Task { [weak self] in await self?.run(force: force) }
    }

    /// Search now (unless the shown answer already covers the request and
    /// `force` is false).  An empty query asks for the newest of each kind.
    func run(force: Bool) async {
        let request = self.request
        if !force, phase == .results, let last = lastRequest, last.covers(request) {
            keepSelectionVisible()
            return
        }
        searching = true
        defer { searching = false }
        let found = await CommonplaceResults.search(request, api: api)
        guard !Task.isCancelled else { return }
        let problem = await problem(for: found.memory?.error)
        guard !Task.isCancelled else { return }
        results = found
        memoryProblem = problem
        lastRequest = request
        lastQuery = request.query
        phase = .results
        let apps = Set(knownApps).union((found.memory?.value ?? []).compactMap(\.app).filter { !$0.isEmpty })
        knownApps = apps.sorted { $0.localizedCaseInsensitiveCompare($1) == .orderedAscending }
        keepSelectionVisible()
    }

    /// The selection stays on a shown result: the first when it is not.
    private func keepSelectionVisible() {
        if selected == nil { selectedID = visibleItems.first?.id }
    }

    /// What a failed Memory search means for the user.
    func problem(for error: MemoryAPIError?) async -> MemoryProblem? {
        switch error {
        case .workerNotRunning?, .noToken?: return .workerNotRunning
        case .memoryOff?: return .off
        case .modelUnavailable(let detail)?: return await diagnose(detail)
        default: return nil
        }
    }

    /// The window opens on a Memory that is waiting: say why.
    func checkWaiting(status: MemoryStatus?) async {
        guard let status, status.enabled, let why = status.waiting else {
            waiting = .none
            return
        }
        waiting = .problem(await diagnose(why))
    }

    /// What stands behind `detail`: the local Ollama, its models and the
    /// installed table (`MemoryProblem.diagnose`).
    nonisolated static func diagnose(_ detail: String) async -> MemoryProblem {
        await Task.detached(priority: .userInitiated) {
            let version = OllamaModels.serverVersion()
            let installed = version == nil ? nil : ModelUpgrade.installedModels()
            let table = try? ModelTierTable.load(from: ModelTierTable.installedURL)
            return MemoryProblem.diagnose(
                detail: detail, ollamaVersion: version, installed: installed,
                memoryModel: table?.memoryModel, minimum: table?.minimumOllamaVersionForMemory
            )
        }.value
    }
}

// MARK: - Thumbnails and icons

/// Decrypted thumbnails, opened off the main thread and kept for the
/// window's life.
@MainActor
final class MemoryThumbnailStore: ObservableObject {
    @Published private var images: [String: NSImage] = [:]
    private var requested: Set<String> = []

    func image(for thumbID: String?) -> NSImage? {
        thumbID.flatMap { images[$0] }
    }

    func load(_ thumbID: String?) {
        guard let thumbID, requested.insert(thumbID).inserted else { return }
        Task {
            let image = await Task.detached(priority: .utility) {
                MemoryThumbnail.decrypt(MemoryThumbnail.fileURL(eventID: thumbID))
            }.value
            if let image { images[thumbID] = image }
        }
    }

    func forget(_ thumbID: String?) {
        guard let thumbID else { return }
        images[thumbID] = nil
    }
}

/// An app's icon by its name (or bundle id), resolved once.
@MainActor
enum AppIcons {
    private static var cache: [String: NSImage] = [:]

    static func icon(for app: String?) -> NSImage {
        let name = app ?? ""
        if let cached = cache[name] { return cached }
        let icon = resolve(name) ?? NSWorkspace.shared.icon(for: .applicationBundle)
        cache[name] = icon
        return icon
    }

    private static func resolve(_ name: String) -> NSImage? {
        guard !name.isEmpty else { return nil }
        if let running = NSWorkspace.shared.runningApplications.first(where: {
            $0.localizedName == name || $0.bundleIdentifier == name
        }), let icon = running.icon {
            return icon
        }
        if name.contains("."), let url = NSWorkspace.shared.urlForApplication(withBundleIdentifier: name) {
            return NSWorkspace.shared.icon(forFile: url.path)
        }
        for dir in ["/Applications", "/System/Applications", "/System/Applications/Utilities", "/Applications/Utilities"] {
            let path = "\(dir)/\(name).app"
            if FileManager.default.fileExists(atPath: path) {
                return NSWorkspace.shared.icon(forFile: path)
            }
        }
        return nil
    }
}

// MARK: - Rows and details

private struct ThumbnailView: View {
    let moment: MemoryMoment
    @ObservedObject var thumbnails: MemoryThumbnailStore
    let maxHeight: CGFloat

    var body: some View {
        Group {
            if let image = thumbnails.image(for: moment.thumbID) {
                Image(nsImage: image)
                    .resizable()
                    .aspectRatio(contentMode: .fit)
            } else {
                ZStack {
                    Rectangle().fill(Color.primary.opacity(0.05))
                    Image(nsImage: AppIcons.icon(for: moment.app))
                        .resizable()
                        .frame(width: min(48, maxHeight / 2), height: min(48, maxHeight / 2))
                        .opacity(0.8)
                }
                .aspectRatio(16 / 10, contentMode: .fit)
            }
        }
        .frame(maxHeight: maxHeight)
        .clipShape(RoundedRectangle(cornerRadius: 6, style: .continuous))
        .overlay(RoundedRectangle(cornerRadius: 6, style: .continuous).stroke(Color.primary.opacity(0.08)))
        .onAppear { thumbnails.load(moment.thumbID) }
    }
}

private struct MomentRow: View {
    let moment: MemoryMoment
    @ObservedObject var thumbnails: MemoryThumbnailStore
    let format: MemoryFormat

    var body: some View {
        HStack(alignment: .top, spacing: 12) {
            ThumbnailView(moment: moment, thumbnails: thumbnails, maxHeight: 54)
                .frame(width: 86, height: 54)
            VStack(alignment: .leading, spacing: 3) {
                HStack(spacing: 6) {
                    Image(nsImage: AppIcons.icon(for: moment.app))
                        .resizable()
                        .frame(width: 14, height: 14)
                    Text(moment.windowTitle?.isEmpty == false ? moment.windowTitle! : (moment.app ?? "Untitled"))
                        .font(.system(size: 13, weight: .semibold))
                        .lineLimit(1)
                }
                if let description = moment.rowDescription {
                    Text(description)
                        .font(.system(size: 12))
                        .foregroundStyle(.secondary)
                        .lineLimit(2)
                }
                Text(timeLine)
                    .font(.system(size: 11))
                    .foregroundStyle(.tertiary)
                    .lineLimit(1)
            }
        }
        .padding(.vertical, 4)
    }

    private var timeLine: String {
        let when = moment.firstDate.map { date in
            format.relative(date, now: Date()) ?? format.pointInTime(date, now: Date())
        } ?? moment.firstTS
        return [when, format.span(moment)].compactMap { $0 }.joined(separator: " \u{00B7} ")
    }
}

private struct MomentFacts: View {
    let moment: MemoryMoment
    let format: MemoryFormat

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text(moment.windowTitle?.isEmpty == false ? moment.windowTitle! : (moment.app ?? "Untitled"))
                .font(.system(size: 15, weight: .semibold))
                .textSelection(.enabled)
            HStack(spacing: 6) {
                Image(nsImage: AppIcons.icon(for: moment.app))
                    .resizable()
                    .frame(width: 16, height: 16)
                Text([moment.app, moment.urlHost].compactMap { $0 }.filter { !$0.isEmpty }.joined(separator: " \u{00B7} "))
                    .font(.system(size: 12))
                    .foregroundStyle(.secondary)
            }
            Text(format.when(moment, now: Date()))
                .font(.system(size: 12))
                .foregroundStyle(.secondary)
            if let span = format.span(moment) {
                Text(span)
                    .font(.system(size: 12))
                    .foregroundStyle(.secondary)
            }
            Text(moment.description?.isEmpty == false ? moment.description! : "Not described yet.")
                .font(.system(size: 13))
                .foregroundStyle(moment.description?.isEmpty == false ? .primary : .tertiary)
                .textSelection(.enabled)
                .fixedSize(horizontal: false, vertical: true)
                .padding(.top, 4)
        }
    }
}

/// The opened moment: the thumbnail as large as it was kept, and
/// everything Memory knows about it.
private struct MomentSheet: View {
    let moment: MemoryMoment
    @ObservedObject var thumbnails: MemoryThumbnailStore
    let forget: () -> Void
    @Environment(\.dismiss) private var dismiss

    var body: some View {
        VStack(alignment: .leading, spacing: 16) {
            ThumbnailView(moment: moment, thumbnails: thumbnails, maxHeight: 420)
                .frame(maxWidth: .infinity)
            MomentFacts(moment: moment, format: MemoryFormat())
            HStack {
                Button("Forget This Moment\u{2026}", role: .destructive) { forget() }
                    .keyboardShortcut(.delete, modifiers: .command)
                Spacer()
                Button("Close") { dismiss() }
                    .keyboardShortcut(.cancelAction)
            }
        }
        .padding(22)
        .frame(width: 680)
    }
}

// MARK: - Skills and days

private struct SkillRow: View {
    let skill: CommonplaceSkill

    var body: some View {
        HStack(alignment: .top, spacing: 12) {
            Image(systemName: "list.bullet.rectangle")
                .font(.system(size: 18, weight: .light))
                .foregroundStyle(.secondary)
                .frame(width: 28, height: 28)
            VStack(alignment: .leading, spacing: 3) {
                Text(skill.title.isEmpty ? skill.slug : skill.title)
                    .font(.system(size: 13, weight: .semibold))
                    .lineLimit(1)
                if !skill.description.isEmpty {
                    Text(skill.description)
                        .font(.system(size: 12))
                        .foregroundStyle(.secondary)
                        .lineLimit(2)
                }
                Text(skill.handoffLine)
                    .font(.system(size: 11))
                    .foregroundStyle(.tertiary)
                    .lineLimit(1)
            }
        }
        .padding(.vertical, 4)
    }
}

private struct SkillFacts: View {
    let skill: CommonplaceSkill

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text(skill.title.isEmpty ? skill.slug : skill.title)
                .font(.system(size: 15, weight: .semibold))
                .textSelection(.enabled)
            Text(skill.handoffLine)
                .font(.system(size: 12))
                .foregroundStyle(.secondary)
            ForEach(skill.reasons ?? [], id: \.self) { reason in
                Text(reason)
                    .font(.system(size: 12))
                    .foregroundStyle(.secondary)
                    .fixedSize(horizontal: false, vertical: true)
            }
            Text(skill.description.isEmpty ? "No description yet." : skill.description)
                .font(.system(size: 13))
                .foregroundStyle(skill.description.isEmpty ? .tertiary : .primary)
                .textSelection(.enabled)
                .fixedSize(horizontal: false, vertical: true)
                .padding(.top, 4)
        }
    }
}

private struct DayRow: View {
    let day: CommonplaceDay

    var body: some View {
        HStack(alignment: .top, spacing: 12) {
            Image(systemName: "calendar")
                .font(.system(size: 18, weight: .light))
                .foregroundStyle(.secondary)
                .frame(width: 28, height: 28)
            VStack(alignment: .leading, spacing: 3) {
                Text(MemoryFormat().dayTitle(day))
                    .font(.system(size: 13, weight: .semibold))
                    .lineLimit(1)
                if !day.excerpt.isEmpty {
                    Text(day.excerpt)
                        .font(.system(size: 12))
                        .foregroundStyle(.secondary)
                        .lineLimit(2)
                }
                if !day.factsLine.isEmpty {
                    Text(day.factsLine)
                        .font(.system(size: 11))
                        .foregroundStyle(.tertiary)
                        .lineLimit(1)
                }
            }
        }
        .padding(.vertical, 4)
    }
}

private struct DayFacts: View {
    let day: CommonplaceDay

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text(MemoryFormat().dayTitle(day))
                .font(.system(size: 15, weight: .semibold))
            if !day.factsLine.isEmpty {
                Text(day.factsLine)
                    .font(.system(size: 12))
                    .foregroundStyle(.secondary)
            }
            Text(day.excerpt.isEmpty ? "No tasks recorded for this day." : day.excerpt)
                .font(.system(size: 13))
                .foregroundStyle(day.excerpt.isEmpty ? .tertiary : .primary)
                .textSelection(.enabled)
                .fixedSize(horizontal: false, vertical: true)
                .padding(.top, 4)
        }
    }
}
