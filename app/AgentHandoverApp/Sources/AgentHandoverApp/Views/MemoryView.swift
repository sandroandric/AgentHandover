import AppKit
import SwiftUI
import UniformTypeIdentifiers

/// The Memory window: search what you saw by describing it.  Keyboard
/// first: the search field has focus when the window opens, the arrows
/// move through the moments, Return opens the selected one, Command-Delete
/// forgets it after a confirmation.
struct MemoryView: View {
    @EnvironmentObject private var delegate: AppDelegate
    @ObservedObject private var center = MemoryCenter.shared
    @StateObject private var model = MemoryWindowModel()
    @StateObject private var thumbnails = MemoryThumbnailStore()
    @FocusState private var searchFocused: Bool
    @State private var openMoment: MemoryMoment?
    @Environment(\.openWindow) private var openWindow

    private let format = MemoryFormat()

    var body: some View {
        VStack(spacing: 0) {
            searchBar
            filterBar
            Divider()
            banner
            content
                .frame(maxWidth: .infinity, maxHeight: .infinity)
            Divider()
            footer
        }
        .frame(minWidth: 780, minHeight: 500)
        .background(Color(nsColor: .windowBackgroundColor))
        .task {
            searchFocused = true
            await center.refresh()
            await model.checkWaiting(status: center.status)
        }
        .onChange(of: center.forgetCount) { _, _ in model.search(force: true) }
        .sheet(item: $openMoment) { moment in
            MomentSheet(moment: moment, thumbnails: thumbnails, forget: { forget(moment) })
        }
    }

    // MARK: - Search

    private var searchBar: some View {
        HStack(spacing: 10) {
            Image(systemName: "magnifyingglass")
                .font(.system(size: 17, weight: .regular))
                .foregroundStyle(.secondary)
            TextField("Describe what you saw", text: $model.query)
                .textFieldStyle(.plain)
                .font(.system(size: 18))
                .focused($searchFocused)
                .onSubmit {
                    if model.isShowingResults(for: model.query), let selected = model.selected {
                        openMoment = selected
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

    private var filterBar: some View {
        HStack(spacing: 12) {
            Picker("Time", selection: $model.range) {
                ForEach(MemoryRange.allCases) { Text($0.rawValue).tag($0) }
            }
            .pickerStyle(.segmented)
            .labelsHidden()
            .frame(width: 270)
            .onChange(of: model.range) { _, _ in model.search(force: true) }

            Picker("App", selection: $model.app) {
                Text("All Apps").tag(String?.none)
                if !model.knownApps.isEmpty { Divider() }
                ForEach(model.knownApps, id: \.self) { app in
                    Text(app).tag(Optional(app))
                }
            }
            .labelsHidden()
            .frame(width: 180)
            .onChange(of: model.app) { _, _ in model.search(force: true) }

            Spacer()
        }
        .padding(.horizontal, 18)
        .padding(.bottom, 12)
    }

    // MARK: - Banner

    @ViewBuilder
    private var banner: some View {
        if case .paused(let until) = center.menuState {
            bannerRow(
                icon: "pause.circle",
                text: "Memory is paused" + (until.map { " until \(format.pointInTime($0, now: Date()))" } ?? "")
                    + ". Search still finds what was kept before.",
                action: "Resume"
            ) { Task { await center.resume() } }
        } else if case .problem(let problem) = model.waiting {
            bannerRow(icon: "hourglass", text: problem.message + " New screens wait until then.",
                      action: actionTitle(for: problem)) { perform(problem) }
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
        if let problem = blockingProblem {
            stateView(icon: icon(for: problem), text: problem.message,
                      action: actionTitle(for: problem)) { perform(problem) }
        } else if case .failed(let message) = model.phase {
            stateView(icon: "exclamationmark.triangle", text: message, action: "Try Again") {
                model.search(force: true)
            }
        } else if model.query.trimmingCharacters(in: .whitespaces).isEmpty {
            stateView(
                icon: "text.magnifyingglass",
                text: "Search what you saw by meaning: \u{201C}the invoice from Acme\u{201D}, \u{201C}that chart about churn\u{201D}.",
                action: nil, perform: {}
            )
        } else if model.results.isEmpty && model.phase == .results {
            stateView(
                icon: "magnifyingglass",
                text: "No moments match \u{201C}\(model.lastQuery)\u{201D}" + (model.hasFilters ? " in this time and app." : "."),
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

    /// A problem that leaves nothing to search: the worker, Memory off, or
    /// a search the worker could not embed.
    private var blockingProblem: MemoryProblem? {
        if let error = center.statusError, error == .workerNotRunning || error == .noToken {
            return .workerNotRunning
        }
        if center.status?.enabled == false { return .off }
        if case .problem(let problem) = model.phase { return problem }
        return nil
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
        case .waiting: return "Try Again"
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
            ForEach(model.results) { moment in
                MomentRow(moment: moment, thumbnails: thumbnails, format: format)
                    .tag(moment.id)
                    .contextMenu {
                        Button("Open") { openMoment = moment }
                        Button("Forget This Moment\u{2026}") { forget(moment) }
                    }
            }
        }
        .listStyle(.inset)
        .onKeyPress(.return) {
            guard let selected = model.selected else { return .ignored }
            openMoment = selected
            return .handled
        }
    }

    @ViewBuilder
    private var detail: some View {
        if let moment = model.selected {
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
        } else {
            Text("Select a moment")
                .foregroundStyle(.tertiary)
                .frame(maxWidth: .infinity, maxHeight: .infinity)
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
            Button("Settings\u{2026}") {
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

// MARK: - The window's model

@MainActor
final class MemoryWindowModel: ObservableObject {
    enum Phase: Equatable {
        case idle
        case results
        case problem(MemoryProblem)
        case failed(String)
    }

    enum Waiting: Equatable {
        case none
        case problem(MemoryProblem)
    }

    @Published var query = ""
    @Published var range: MemoryRange = .anyTime
    @Published var app: String?
    @Published var selectedID: String?
    @Published private(set) var results: [MemoryMoment] = []
    @Published private(set) var knownApps: [String] = []
    @Published private(set) var phase: Phase = .idle
    @Published private(set) var searching = false
    /// Why new screens are not being embedded (status `waiting`).
    @Published private(set) var waiting: Waiting = .none
    /// The query the shown results answer.
    @Published private(set) var lastQuery = ""

    private let api: MemoryAPI
    private var pending: Task<Void, Never>?
    private var lastRequest: (String, MemoryRange, String?)?

    init(api: MemoryAPI = MemoryAPI()) {
        self.api = api
    }

    var selected: MemoryMoment? { results.first { $0.id == selectedID } }
    var hasFilters: Bool { range != .anyTime || app != nil }

    func isShowingResults(for query: String) -> Bool {
        phase == .results && lastQuery == query.trimmingCharacters(in: .whitespacesAndNewlines)
    }

    func clearFilters() {
        range = .anyTime
        app = nil
        search(force: true)
    }

    func moveSelection(_ step: Int) {
        guard !results.isEmpty else { return }
        let index = results.firstIndex { $0.id == selectedID } ?? (step > 0 ? -1 : results.count)
        selectedID = results[min(max(index + step, 0), results.count - 1)].id
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

    private func run(force: Bool) async {
        let text = query.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !text.isEmpty else {
            results = []
            phase = .idle
            lastQuery = ""
            return
        }
        if !force, let last = lastRequest, last == (text, range, app) { return }
        lastRequest = (text, range, app)
        searching = true
        defer { searching = false }
        do {
            let answer = try await api.search(query: text, since: range.since(now: Date()), app: app, limit: 50)
            guard !Task.isCancelled else { return }
            results = answer.moments
            lastQuery = text
            phase = .results
            let apps = Set(knownApps).union(answer.moments.compactMap(\.app).filter { !$0.isEmpty })
            knownApps = apps.sorted { $0.localizedCaseInsensitiveCompare($1) == .orderedAscending }
            if selected == nil { selectedID = results.first?.id }
        } catch let error as MemoryAPIError {
            guard !Task.isCancelled else { return }
            results = []
            switch error {
            case .workerNotRunning, .noToken: phase = .problem(.workerNotRunning)
            case .memoryOff: phase = .problem(.off)
            case .modelUnavailable(let detail): phase = .problem(await Self.diagnose(detail))
            case .badRequest(let reason): phase = .failed(reason)
            case .busy, .momentGone: phase = .failed(MemoryCenter.describe(error))
            case .unexpected(let status, _): phase = .failed("The worker answered \(status). Try again.")
            }
        } catch {
            guard !Task.isCancelled else { return }
            phase = .failed(error.localizedDescription)
        }
    }

    /// The window opens on a Memory that is waiting: say why.
    func checkWaiting(status: MemoryStatus?) async {
        guard let status, status.enabled, let why = status.waiting else {
            waiting = .none
            return
        }
        waiting = .problem(await Self.diagnose(why))
    }

    /// What stands behind `detail`: the local Ollama, its models and the
    /// installed table (`MemoryProblem.diagnose`).
    static func diagnose(_ detail: String) async -> MemoryProblem {
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
                if let description = moment.description, !description.isEmpty {
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
