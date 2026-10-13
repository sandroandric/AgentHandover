import Foundation

// Commonplace: everything AgentHandover knows locally (Skills, Memory
// moments, daily summaries), searched from one window.  Skills and days
// come from the worker's `POST /commonplace/search`; Memory comes from
// `POST /memory/search`, the user's own search, which does not depend on
// whether agents may search Memory (`[memory] agents_access` gates only
// what agents get).

/// A Skill as `/commonplace/search` returns it.
struct CommonplaceSkill: Decodable, Equatable, Hashable {
    let slug: String
    let title: String
    let description: String
    let score: Double
    /// `ready` (an agent may run it), `held_back` (approved, but a check
    /// fails: `reasons`) or `not_approved`.
    let handoff: String
    let reasons: [String]?

    /// What the hand-off gate says, in one line.
    var handoffLine: String {
        switch handoff {
        case "ready": return "Ready for agents"
        case "held_back": return "Approved, held back from agents"
        case "not_approved": return "Not approved for agents yet"
        default: return handoff
        }
    }
}

/// A day (its daily summary) as `/commonplace/search` returns it.
struct CommonplaceDay: Decodable, Equatable, Hashable {
    /// "2026-10-12", the local day.
    let date: String
    let activeHours: Double?
    let taskCount: Int?
    let apps: [String]
    let excerpt: String
    let score: Double

    enum CodingKeys: String, CodingKey {
        case date, apps, excerpt, score
        case activeHours = "active_hours"
        case taskCount = "task_count"
    }

    /// The day at local midnight; nil when the date does not parse.
    func day(calendar: Calendar = .current) -> Date? {
        let parts = date.split(separator: "-").compactMap { Int($0) }
        guard parts.count == 3 else { return nil }
        return calendar.date(from: DateComponents(year: parts[0], month: parts[1], day: parts[2]))
    }

    /// "2.5 h active · 4 tasks · Safari, Slack": what is known of the day.
    var factsLine: String {
        var facts: [String] = []
        if let hours = activeHours {
            facts.append(String(format: "%.1f h active", hours))
        }
        if let tasks = taskCount {
            facts.append(tasks == 1 ? "1 task" : "\(tasks) tasks")
        }
        if !apps.isEmpty { facts.append(apps.joined(separator: ", ")) }
        return facts.joined(separator: " \u{00B7} ")
    }
}

/// One kind's answer: how it was searched and what it found.
struct CommonplaceGroup<Item: Decodable & Equatable>: Decodable, Equatable {
    /// `semantic` (by meaning), `keyword_fallback` (by words: `reason`
    /// says why) or `not_allowed`.
    let mode: String
    let reason: String
    let results: [Item]

    var searchedByWords: Bool { mode == "keyword_fallback" }
}

/// `POST /commonplace/search`'s answer for the kinds the window asks for.
struct CommonplaceSearchResponse: Decodable, Equatable {
    let skills: CommonplaceGroup<CommonplaceSkill>?
    let days: CommonplaceGroup<CommonplaceDay>?
}

extension MemoryAPI {
    /// `POST /commonplace/search` for `kinds` ("skills", "days"; the
    /// window searches Memory through `search`).
    func commonplaceSearch(
        query: String, kinds: [CommonplaceKind], since: Date? = nil, limit: Int? = nil
    ) async throws -> CommonplaceSearchResponse {
        var body: [String: Any] = ["query": query, "kinds": kinds.map(\.rawValue)]
        if let since { body["since"] = Self.isoString(since) }
        if let limit { body["limit"] = limit }
        do {
            return try await send("POST", "/commonplace/search", body: body)
        } catch MemoryAPIError.momentGone {
            // A 404 here means a worker without the endpoint, not a moment.
            throw MemoryAPIError.unexpected(status: 404, body: "no /commonplace/search")
        }
    }
}

// MARK: - Scope and request

/// The kinds the window searches, in the order it shows them: what you
/// saw first (this is where you look for it), then your days, then Skills.
/// Scores of different kinds are not comparable, so the order is fixed.
enum CommonplaceKind: String, CaseIterable, Identifiable {
    case memory, days, skills

    var id: Self { self }

    var title: String {
        switch self {
        case .skills: return "Skills"
        case .memory: return "Memory"
        case .days: return "Days"
        }
    }

    /// A group of All that found nothing.
    var noMatches: String {
        switch self {
        case .skills: return "No Skills match."
        case .memory: return "No moments match."
        case .days: return "No days match."
        }
    }
}

/// The scope chips: everything, or one kind.
enum CommonplaceScope: String, CaseIterable, Identifiable {
    case all = "All"
    case memory = "Memory"
    case days = "Days"
    case skills = "Skills"

    var id: Self { self }

    init(_ kind: CommonplaceKind) {
        switch kind {
        case .skills: self = .skills
        case .memory: self = .memory
        case .days: self = .days
        }
    }

    var kinds: [CommonplaceKind] {
        switch self {
        case .all: return CommonplaceKind.allCases
        case .skills: return [.skills]
        case .memory: return [.memory]
        case .days: return [.days]
        }
    }

    /// Skills are not tied to a time; Memory and days are.
    var showsTimeFilter: Bool { self != .skills }
    /// Only Memory filters by app.
    var showsAppFilter: Bool { self == .memory }

    var placeholder: String {
        switch self {
        case .all: return "Search what you saw, your days and your Skills"
        case .skills: return "Find a Skill"
        case .memory: return "Describe what you saw"
        case .days: return "Find a day"
        }
    }
}

/// One search: the query, the kinds it covers and the filters that apply
/// to them (a filter the scope does not show is not applied).
struct CommonplaceRequest: Equatable {
    /// Each kind returns at most this many (the worker's maximum); All
    /// shows the first `CommonplaceResults.previewCount` of each.
    static let limit = 50

    let query: String
    let kinds: Set<CommonplaceKind>
    let range: MemoryRange
    let app: String?

    init(query: String, scope: CommonplaceScope, range: MemoryRange, app: String?) {
        self.query = query.trimmingCharacters(in: .whitespacesAndNewlines)
        self.kinds = Set(scope.kinds)
        self.range = scope.showsTimeFilter ? range : .anyTime
        self.app = scope.showsAppFilter ? app : nil
    }

    /// Whether the answer to this request already holds `other`'s: the same
    /// query and filters for every kind `other` asks for.
    func covers(_ other: CommonplaceRequest) -> Bool {
        guard query == other.query, other.kinds.isSubset(of: kinds) else { return false }
        if other.kinds != [.skills], range != other.range { return false }
        if other.kinds.contains(.memory), app != other.app { return false }
        return true
    }
}

// MARK: - Results

/// What a search found, kind by kind; a kind that was not searched is nil.
/// A kind can fail on its own (Memory off, its model down) while the
/// others answer.
struct CommonplaceResults: Equatable {
    enum Outcome<Value: Equatable>: Equatable {
        case found(Value)
        case failed(MemoryAPIError)

        var value: Value? {
            if case .found(let value) = self { return value }
            return nil
        }

        var error: MemoryAPIError? {
            if case .failed(let error) = self { return error }
            return nil
        }
    }

    /// How many of each kind All shows before "Show all".
    static let previewCount = 5

    var skills: Outcome<CommonplaceGroup<CommonplaceSkill>>?
    var memory: Outcome<[MemoryMoment]>?
    var days: Outcome<CommonplaceGroup<CommonplaceDay>>?

    static let empty = CommonplaceResults()

    /// Every item of `kind` that was found, best first.
    func items(_ kind: CommonplaceKind) -> [CommonplaceItem] {
        switch kind {
        case .skills: return skills?.value?.results.map(CommonplaceItem.skill) ?? []
        case .memory: return memory?.value?.map(CommonplaceItem.moment) ?? []
        case .days: return days?.value?.results.map(CommonplaceItem.day) ?? []
        }
    }

    /// Whether `kind` was searched (found something, nothing, or failed).
    func searched(_ kind: CommonplaceKind) -> Bool {
        switch kind {
        case .skills: return skills != nil
        case .memory: return memory != nil
        case .days: return days != nil
        }
    }

    func error(_ kind: CommonplaceKind) -> MemoryAPIError? {
        switch kind {
        case .skills: return skills?.error
        case .memory: return memory?.error
        case .days: return days?.error
        }
    }

    /// Why `kind` was searched by words instead of by meaning; nil when it
    /// was not.
    func byWordsReason(_ kind: CommonplaceKind) -> String? {
        let group: (mode: Bool, reason: String)?
        switch kind {
        case .skills: group = skills?.value.map { ($0.searchedByWords, $0.reason) }
        case .days: group = days?.value.map { ($0.searchedByWords, $0.reason) }
        case .memory: group = nil
        }
        guard let group, group.mode else { return nil }
        return group.reason
    }

    /// The groups `scope` shows, in order: All shows up to
    /// `previewCount` items of each kind, a single scope all of its own.
    func sections(_ scope: CommonplaceScope) -> [CommonplaceSection] {
        scope.kinds.map { kind in
            let all = items(kind)
            let shown = scope == .all ? Array(all.prefix(Self.previewCount)) : all
            return CommonplaceSection(
                kind: kind, items: shown, total: all.count, searched: searched(kind),
                error: error(kind), byWordsReason: byWordsReason(kind)
            )
        }
    }

    /// The items `scope` shows, top to bottom: what the arrow keys move
    /// through, across groups.
    func visibleItems(_ scope: CommonplaceScope) -> [CommonplaceItem] {
        sections(scope).flatMap(\.items)
    }

    /// Whether every kind of `scope` was searched and found nothing (and
    /// nothing failed).
    func isEmpty(_ scope: CommonplaceScope) -> Bool {
        sections(scope).allSatisfy { $0.searched && $0.items.isEmpty && $0.error == nil }
    }

    /// Whether every searched kind failed because the worker is not there.
    var workerUnreachable: Bool {
        let errors = [skills?.error, memory?.error, days?.error]
        let searched = [skills != nil, memory != nil, days != nil]
        guard searched.contains(true) else { return false }
        return zip(errors, searched).allSatisfy { error, wasSearched in
            !wasSearched || error == .workerNotRunning || error == .noToken
        }
    }

    // MARK: - Searching

    /// Run `request` against the worker: `/commonplace/search` for Skills
    /// and days and `/memory/search` for Memory, at the same time.
    static func search(_ request: CommonplaceRequest, api: MemoryAPI, now: Date = Date()) async -> CommonplaceResults {
        let since = request.range.since(now: now)
        let textKinds = CommonplaceKind.allCases.filter { $0 != .memory && request.kinds.contains($0) }
        async let text: Outcome<CommonplaceSearchResponse>? = textKinds.isEmpty ? nil : outcome {
            try await api.commonplaceSearch(
                query: request.query, kinds: textKinds, since: since, limit: CommonplaceRequest.limit
            )
        }
        async let memory: Outcome<[MemoryMoment]>? = request.kinds.contains(.memory) ? outcome {
            try await api.search(
                query: request.query, since: since, app: request.app, limit: CommonplaceRequest.limit
            ).moments
        } : nil

        var results = CommonplaceResults()
        results.memory = await memory
        switch await text {
        case .found(let answer)?:
            if request.kinds.contains(.skills) {
                results.skills = answer.skills.map { .found($0) } ?? .failed(.unexpected(status: 200, body: "no skills group"))
            }
            if request.kinds.contains(.days) {
                results.days = answer.days.map { .found($0) } ?? .failed(.unexpected(status: 200, body: "no days group"))
            }
        case .failed(let error)?:
            if request.kinds.contains(.skills) { results.skills = .failed(error) }
            if request.kinds.contains(.days) { results.days = .failed(error) }
        case nil:
            break
        }
        return results
    }

    private static func outcome<Value>(_ call: () async throws -> Value) async -> Outcome<Value> {
        do {
            return .found(try await call())
        } catch let error as MemoryAPIError {
            return .failed(error)
        } catch {
            return .failed(.unexpected(status: 0, body: error.localizedDescription))
        }
    }
}

/// One group of the result list.
struct CommonplaceSection: Equatable, Identifiable {
    let kind: CommonplaceKind
    let items: [CommonplaceItem]
    /// How many the search found (All shows fewer).
    let total: Int
    /// False while the kind has not been searched for the shown query.
    let searched: Bool
    let error: MemoryAPIError?
    /// Set when the kind was searched by words, with the worker's reason.
    let byWordsReason: String?

    var id: CommonplaceKind { kind }
    /// "Show all 12" when All shows only some.
    var hiddenCount: Int { total - items.count }
    /// The search asks for at most CommonplaceRequest.limit of a kind: a
    /// group that reached it may have more.
    var reachedLimit: Bool { total >= CommonplaceRequest.limit }
    /// "12", or "50+" when the group reached the limit.
    var countLabel: String { reachedLimit ? "\(total)+" : "\(total)" }
    /// "Show All 12", or "Show More" when the group reached the limit.
    var showAllLabel: String { reachedLimit ? "Show More" : "Show All \(total)" }

    /// Why the kind could not be searched, in one sentence.
    var errorMessage: String {
        switch error {
        case .unexpected(0, let body)?:
            return body
        case .unexpected(let status, _)?:
            return "The worker answered \(status). Try again."
        case let error?:
            return MemoryCenter.describe(error)
        case nil:
            return ""
        }
    }
}

/// One result, of any kind.
enum CommonplaceItem: Identifiable, Hashable {
    case skill(CommonplaceSkill)
    case moment(MemoryMoment)
    case day(CommonplaceDay)

    /// Unique across kinds (a list selection).
    var id: String {
        switch self {
        case .skill(let skill): return "skill:" + skill.slug
        case .moment(let moment): return "moment:" + moment.id
        case .day(let day): return "day:" + day.date
        }
    }

    var kind: CommonplaceKind {
        switch self {
        case .skill: return .skills
        case .moment: return .memory
        case .day: return .days
        }
    }

    var moment: MemoryMoment? {
        if case .moment(let moment) = self { return moment }
        return nil
    }

    /// What opening it does: a moment opens in the window's sheet; a Skill
    /// selects it in the Workflows window, a day shows it in the Daily
    /// Digest (`navigator` carries the request to that window).
    @MainActor
    func open(navigator: AppNavigator) -> CommonplaceOpening {
        switch self {
        case .moment(let moment):
            return .moment(moment)
        case .skill(let skill):
            navigator.skillSlug = skill.slug
            return .window(id: "workflows")
        case .day(let day):
            navigator.digestDate = day.date
            return .window(id: "daily-digest")
        }
    }
}

enum CommonplaceOpening: Equatable {
    case moment(MemoryMoment)
    case window(id: String)
}

/// Requests from one window to another: which Skill the Workflows window
/// selects and which day the Daily Digest shows.  The target window takes
/// the request (`take…`) when it can act on it.
@MainActor
final class AppNavigator: ObservableObject {
    static let shared = AppNavigator()

    @Published var skillSlug: String?
    @Published var digestDate: String?

    func takeDigestDate() -> String? {
        defer { digestDate = nil }
        return digestDate
    }
}

// MARK: - What agents see

/// The window's top line: what connected agents can search, and the switch
/// for Memory (`[memory] agents_access`).
struct AgentsLine: Equatable {
    enum Action: Equatable {
        /// Let agents search Memory (asks first).
        case allow
        /// Stop agents searching Memory.
        case disallow
    }

    let text: String
    let action: Action?

    static let base = "Agents see your Skills, daily summaries and profile."

    init(status: MemoryStatus?) {
        guard let status else {
            // Unknown: the worker did not answer, so neither does the switch.
            text = Self.base
            action = nil
            return
        }
        if !status.agentsAccess {
            text = Self.base + " Memory: off"
            action = .allow
        } else if status.enabled {
            text = Self.base + " Memory: on"
            action = .disallow
        } else {
            text = Self.base + " Memory: allowed, but Memory is off"
            action = .disallow
        }
    }

    var actionTitle: String? {
        switch action {
        case .allow?: return "Turn On\u{2026}"
        case .disallow?: return "Turn Off"
        case nil: return nil
        }
    }
}


// MARK: - Examples

/// What the empty window offers: searches to click (they run) and requests
/// to give an agent (they are copied). The README and the website list the
/// same ones.
enum CommonplaceExamples {
    /// Each finds something in a different way: by meaning, by look, in
    /// your days, in your Skills.
    static let searches = [
        "the invoice from Acme",
        "that page with the orange header",
        "the error in the terminal",
        "a slide with a chart about churn",
        "how I deploy to staging",
        "the day I spent in Figma",
    ]

    /// Requests a connected agent answers with search_commonplace.
    static let agentRequests = [
        "Write my weekly update from what I actually did this week.",
        "Find the pricing article I read yesterday and summarise it.",
        "Pick up where I left off on the onboarding doc.",
        "Before you ask me how I release, check my Skills.",
        "Roast my week: where did my hours really go?",
    ]
}
