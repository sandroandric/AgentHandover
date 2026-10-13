import XCTest
@testable import AgentHandoverApp

/// WP-E13: the Commonplace window.  Its client of `/commonplace/search`
/// (requests and each group's decoding, fixtures in the worker's own
/// shapes from test_commonplace.py), the concurrent search that merges
/// Skills and days with the user's own Memory search, the scope chips, the
/// grouping, the agents line, opening a Skill or a day, and the keyboard
/// model across groups.
@MainActor
final class CommonplaceTests: XCTestCase {

    // MARK: - A fake worker

    /// Answers each path with its own status and body; records requests.
    /// With `waitForBoth`, no request is answered until both endpoints
    /// were asked (a sequential client would time out).
    private final class FakeWorker: @unchecked Sendable {
        private let lock = NSLock()
        private var _requests: [URLRequest] = []
        var answers: [String: (Int, String)] = [:]
        var waitForBoth = false

        var requests: [URLRequest] { lock.withLock { _requests } }
        func requests(to path: String) -> [URLRequest] { requests.filter { $0.url?.path == path } }

        func transport(_ request: URLRequest) async throws -> (Data, URLResponse) {
            lock.withLock { _requests.append(request) }
            if waitForBoth {
                let deadline = Date().addingTimeInterval(3)
                while Set(requests.compactMap { $0.url?.path }).count < 2 {
                    if Date() > deadline { throw URLError(.timedOut) }
                    try await Task.sleep(nanoseconds: 5_000_000)
                }
            }
            let path = request.url?.path ?? ""
            let (status, body) = answers[path] ?? (404, #"{"error": "Not found"}"#)
            let response = HTTPURLResponse(url: request.url!, statusCode: status, httpVersion: "HTTP/1.1", headerFields: nil)!
            return (Data(body.utf8), response)
        }
    }

    private var worker = FakeWorker()

    private var api: MemoryAPI {
        let worker = worker
        return MemoryAPI(token: { "tok-123" }, transport: { try await worker.transport($0) })
    }

    private func body(_ request: URLRequest?) -> String? {
        request?.httpBody.flatMap { String(data: $0, encoding: .utf8) }
    }

    // MARK: - Fixtures (the worker's shapes)

    private static let skillsAndDays = """
        {"query": "domain",
         "skills": {"mode": "semantic", "reason": "", "results": [
           {"slug": "check-domain", "title": "Check Domain Availability",
            "description": "Check Domain Availability, as the user does it", "score": 1.0, "handoff": "ready"},
           {"slug": "old-domain-report", "title": "Old Domain Report", "description": "", "score": 0.8,
            "handoff": "held_back", "reasons": ["trust level is 'observe' (must be above 'observe')"]},
           {"slug": "deploy-staging", "title": "Deploy Staging", "description": "Deploy", "score": 0.5,
            "handoff": "not_approved"}]},
         "days": {"mode": "keyword_fallback", "reason": "embedding model unavailable: connection refused", "results": [
           {"date": "2026-10-12", "active_hours": 2.5, "task_count": 2, "apps": ["Safari", "Slack"],
            "excerpt": "Paid the Acme invoice; Reviewed the banner", "score": 1.0},
           {"date": "2026-10-10", "active_hours": null, "task_count": null, "apps": [], "excerpt": "", "score": 0.5}]}}
        """

    private static let moments = """
        {"moments": [
          {"id": "ev-1", "first_ts": "2026-10-12T13:00:00Z", "last_ts": "2026-10-12T13:12:00Z", "frames": 34,
           "app": "Safari", "window_title": "Invoice 4411 - Acme", "url_host": "billing.acme.com",
           "description": "An invoice from Acme.", "score": 0.82, "thumb_id": "ev-1"},
          {"id": "ev-2", "first_ts": "2026-10-11T09:00:00Z", "last_ts": "2026-10-11T09:00:00Z", "frames": 1,
           "app": "Mail", "window_title": null, "url_host": null, "description": null, "score": 0.41, "thumb_id": null}
        ], "model": "embeddinggemma-2:440m"}
        """

    private func answerEverything() {
        worker.answers["/commonplace/search"] = (200, Self.skillsAndDays)
        worker.answers["/memory/search"] = (200, Self.moments)
    }

    private static func status(enabled: Bool, agentsAccess: Bool) -> MemoryStatus {
        MemoryStatus(
            enabled: enabled, paused: false, pausedUntil: nil, agentsAccess: agentsAccess,
            model: "embeddinggemma-2:440m", moments: 3, frames: 9, oldest: nil, newest: nil,
            thumbnails: 0, diskBytes: 0, waiting: nil
        )
    }

    private static func skill(_ slug: String) -> CommonplaceSkill {
        CommonplaceSkill(slug: slug, title: slug, description: "", score: 1, handoff: "ready", reasons: nil)
    }

    private static func day(_ date: String) -> CommonplaceDay {
        CommonplaceDay(date: date, activeHours: nil, taskCount: nil, apps: [], excerpt: "", score: 1)
    }

    // MARK: - The client

    func testCommonplaceSearchPostsTheKindsWithTheToken() async throws {
        answerEverything()
        let since = Date(timeIntervalSince1970: 1_791_763_200)  // 2026-10-12T00:00:00Z
        _ = try await api.commonplaceSearch(query: "domain", kinds: [.skills, .days], since: since, limit: 50)
        let request = try XCTUnwrap(worker.requests.first)
        XCTAssertEqual(request.url?.absoluteString, "http://127.0.0.1:9477/commonplace/search")
        XCTAssertEqual(request.httpMethod, "POST")
        XCTAssertEqual(request.value(forHTTPHeaderField: "X-AgentHandover-Token"), "tok-123")
        XCTAssertEqual(body(request),
                       #"{"kinds":["skills","days"],"limit":50,"query":"domain","since":"2026-10-12T00:00:00Z"}"#)
    }

    func testSkillsDecodeWithTheHandoffGate() async throws {
        answerEverything()
        let answer = try await api.commonplaceSearch(query: "domain", kinds: [.skills, .days])
        let skills = try XCTUnwrap(answer.skills)
        XCTAssertEqual(skills.mode, "semantic")
        XCTAssertFalse(skills.searchedByWords)
        XCTAssertEqual(skills.results.map(\.slug), ["check-domain", "old-domain-report", "deploy-staging"])
        XCTAssertEqual(skills.results[0].handoffLine, "Ready for agents")
        XCTAssertNil(skills.results[0].reasons)
        XCTAssertEqual(skills.results[1].handoffLine, "Approved, held back from agents")
        XCTAssertEqual(skills.results[1].reasons, ["trust level is 'observe' (must be above 'observe')"])
        XCTAssertEqual(skills.results[2].handoffLine, "Not approved for agents yet")
    }

    func testDaysDecodeWithNullsAndSaySearchedByWords() async throws {
        answerEverything()
        let answer = try await api.commonplaceSearch(query: "domain", kinds: [.days])
        let days = try XCTUnwrap(answer.days)
        XCTAssertTrue(days.searchedByWords)
        XCTAssertEqual(days.reason, "embedding model unavailable: connection refused")
        XCTAssertEqual(days.results[0], CommonplaceDay(
            date: "2026-10-12", activeHours: 2.5, taskCount: 2, apps: ["Safari", "Slack"],
            excerpt: "Paid the Acme invoice; Reviewed the banner", score: 1.0
        ))
        XCTAssertEqual(days.results[0].factsLine, "2.5 h active \u{00B7} 2 tasks \u{00B7} Safari, Slack")
        XCTAssertNil(days.results[1].activeHours)
        XCTAssertNil(days.results[1].taskCount)
        XCTAssertEqual(days.results[1].factsLine, "")
    }

    func testAMissingGroupDecodesAsNil() async throws {
        worker.answers["/commonplace/search"] = (200, #"{"query": "x", "skills": {"mode": "keyword_fallback", "reason": "no Skill has been embedded yet", "results": []}}"#)
        let answer = try await api.commonplaceSearch(query: "x", kinds: [.skills])
        XCTAssertEqual(answer.skills?.results, [])
        XCTAssertNil(answer.days)
    }

    func testRefusalsAreTyped() async {
        worker.answers["/commonplace/search"] = (400, #"{"error": "'limit' must be an integer from 1 to 50"}"#)
        await assertThrows(.badRequest("'limit' must be an integer from 1 to 50"))
        worker.answers["/commonplace/search"] = (501, #"{"error": "Commonplace search is not available in this worker"}"#)
        await assertThrows(.unexpected(status: 501, body: #"{"error": "Commonplace search is not available in this worker"}"#))
        // A worker without the endpoint: not "that moment is already gone".
        worker.answers["/commonplace/search"] = (404, #"{"error": "Not found: /commonplace/search"}"#)
        await assertThrows(.unexpected(status: 404, body: "no /commonplace/search"))
    }

    private func assertThrows(_ expected: MemoryAPIError, line: UInt = #line) async {
        do {
            _ = try await api.commonplaceSearch(query: "x", kinds: [.skills])
            XCTFail("no error", line: line)
        } catch {
            XCTAssertEqual(error as? MemoryAPIError, expected, line: line)
        }
    }

    // MARK: - Searching: two endpoints at once, merged by kind

    func testAllAsksBothEndpointsConcurrentlyAndMergesTheGroups() async throws {
        answerEverything()
        worker.waitForBoth = true
        let request = CommonplaceRequest(query: " domain ", scope: .all, range: .anyTime, app: "Safari")
        let results = await CommonplaceResults.search(request, api: api)

        let text = try XCTUnwrap(worker.requests(to: "/commonplace/search").first)
        XCTAssertEqual(body(text), #"{"kinds":["days","skills"],"limit":50,"query":"domain"}"#)
        let memory = try XCTUnwrap(worker.requests(to: "/memory/search").first)
        // The app filter is Memory's own: not applied in All.
        XCTAssertEqual(body(memory), #"{"limit":50,"query":"domain"}"#)

        XCTAssertEqual(results.items(.skills).map(\.id), ["skill:check-domain", "skill:old-domain-report", "skill:deploy-staging"])
        XCTAssertEqual(results.items(.memory).map(\.id), ["moment:ev-1", "moment:ev-2"])
        XCTAssertEqual(results.items(.days).map(\.id), ["day:2026-10-12", "day:2026-10-10"])
        XCTAssertNil(results.byWordsReason(.skills))
        XCTAssertEqual(results.byWordsReason(.days), "embedding model unavailable: connection refused")
    }

    func testTheUsersMemorySearchDoesNotDependOnTheAgentsSwitch() async throws {
        // The worker's Commonplace answer would say "not_allowed" for
        // Memory; the window never asks it for Memory.
        answerEverything()
        _ = await CommonplaceResults.search(
            CommonplaceRequest(query: "acme", scope: .all, range: .anyTime, app: nil), api: api
        )
        let text = try XCTUnwrap(worker.requests(to: "/commonplace/search").first)
        XCTAssertFalse(try XCTUnwrap(body(text)).contains("memory"))
        XCTAssertEqual(worker.requests(to: "/memory/search").count, 1)
    }

    func testOneKindFailsWhileTheOthersAnswer() async {
        worker.answers["/commonplace/search"] = (200, Self.skillsAndDays)
        worker.answers["/memory/search"] = (503, #"{"error": "memory_off"}"#)
        let results = await CommonplaceResults.search(
            CommonplaceRequest(query: "acme", scope: .all, range: .anyTime, app: nil), api: api
        )
        XCTAssertEqual(results.memory, .failed(.memoryOff))
        XCTAssertEqual(results.items(.skills).count, 3)
        XCTAssertEqual(results.items(.days).count, 2)
        XCTAssertFalse(results.workerUnreachable)
        XCTAssertFalse(results.isEmpty(.all))
        XCTAssertEqual(results.sections(.all).map(\.error), [.memoryOff, nil, nil])
    }

    func testAFailedCommonplaceSearchFailsSkillsAndDays() async {
        worker.answers["/commonplace/search"] = (400, #"{"error": "'since' is after 'until'"}"#)
        worker.answers["/memory/search"] = (200, Self.moments)
        let results = await CommonplaceResults.search(
            CommonplaceRequest(query: "acme", scope: .all, range: .anyTime, app: nil), api: api
        )
        XCTAssertEqual(results.skills, .failed(.badRequest("'since' is after 'until'")))
        XCTAssertEqual(results.days, .failed(.badRequest("'since' is after 'until'")))
        XCTAssertEqual(results.items(.memory).count, 2)
    }

    func testNobodyListeningMeansTheWorkerIsUnreachable() async {
        let results = await CommonplaceResults.search(
            CommonplaceRequest(query: "acme", scope: .all, range: .anyTime, app: nil),
            api: MemoryAPI(token: { "t" }, transport: { _ in throw URLError(.cannotConnectToHost) })
        )
        XCTAssertTrue(results.workerUnreachable)
        XCTAssertFalse(CommonplaceResults.empty.workerUnreachable)
    }

    // MARK: - Scopes

    func testEachScopeAsksOnlyForItsKindWithItsFilters() async throws {
        answerEverything()
        let now = Date(timeIntervalSince1970: 1_791_806_400)  // 2026-10-12T12:00:00Z
        // MemoryRange starts the day in the Mac's own calendar.
        let startOfToday = MemoryAPI.isoString(Calendar.current.startOfDay(for: now))

        _ = await CommonplaceResults.search(
            CommonplaceRequest(query: "q", scope: .skills, range: .today, app: "Safari"), api: api, now: now
        )
        XCTAssertEqual(worker.requests.count, 1)
        // Skills are not tied to a time: no since.
        XCTAssertEqual(body(worker.requests(to: "/commonplace/search").last), #"{"kinds":["skills"],"limit":50,"query":"q"}"#)

        _ = await CommonplaceResults.search(
            CommonplaceRequest(query: "q", scope: .days, range: .today, app: "Safari"), api: api, now: now
        )
        XCTAssertEqual(body(worker.requests(to: "/commonplace/search").last),
                       #"{"kinds":["days"],"limit":50,"query":"q","since":"\#(startOfToday)"}"#)

        _ = await CommonplaceResults.search(
            CommonplaceRequest(query: "q", scope: .memory, range: .today, app: "Safari"), api: api, now: now
        )
        XCTAssertEqual(body(worker.requests(to: "/memory/search").last),
                       #"{"app":"Safari","limit":50,"query":"q","since":"\#(startOfToday)"}"#)
        XCTAssertEqual(worker.requests.count, 3)
    }

    func testScopeFilters() {
        XCTAssertEqual(CommonplaceScope.allCases.map(\.rawValue), ["All", "Memory", "Days", "Skills"])
        XCTAssertEqual(CommonplaceScope.allCases.map(\.showsTimeFilter), [true, true, true, false])
        XCTAssertEqual(CommonplaceScope.allCases.map(\.showsAppFilter), [false, true, false, false])
        XCTAssertEqual(CommonplaceKind.allCases.map(CommonplaceScope.init), [.memory, .days, .skills])
    }

    func testARequestCoversNarrowerScopesWithTheSameQueryAndFilters() {
        let all = CommonplaceRequest(query: "q", scope: .all, range: .today, app: nil)
        XCTAssertTrue(all.covers(CommonplaceRequest(query: "q", scope: .skills, range: .today, app: nil)))
        XCTAssertTrue(all.covers(CommonplaceRequest(query: "q", scope: .days, range: .today, app: nil)))
        XCTAssertTrue(all.covers(CommonplaceRequest(query: "q", scope: .memory, range: .today, app: nil)))
        XCTAssertTrue(all.covers(all))
        // Skills ignore the time, so any range of All covers them.
        XCTAssertEqual(CommonplaceRequest(query: "q", scope: .skills, range: .today, app: "X").range, .anyTime)
        // A different query, range or app filter does not.
        XCTAssertFalse(all.covers(CommonplaceRequest(query: "r", scope: .skills, range: .today, app: nil)))
        XCTAssertFalse(all.covers(CommonplaceRequest(query: "q", scope: .days, range: .anyTime, app: nil)))
        XCTAssertFalse(all.covers(CommonplaceRequest(query: "q", scope: .memory, range: .today, app: "Safari")))
        // A single kind does not cover All.
        let memory = CommonplaceRequest(query: "q", scope: .memory, range: .today, app: nil)
        XCTAssertFalse(memory.covers(all))
    }

    // MARK: - Grouping

    private func manyResults() -> CommonplaceResults {
        var results = CommonplaceResults()
        results.skills = .found(CommonplaceGroup(
            mode: "semantic", reason: "", results: (1...7).map { Self.skill("s\($0)") }
        ))
        results.memory = .found([])
        results.days = .found(CommonplaceGroup(
            mode: "keyword_fallback", reason: "no daily summary has been embedded yet",
            results: ["2026-10-12", "2026-10-11"].map(Self.day)
        ))
        return results
    }

    func testAllShowsFiveOfEachKindWithShowAll() {
        let sections = manyResults().sections(.all)
        // What you saw first, then your days, then Skills.
        XCTAssertEqual(sections.map(\.kind), [.memory, .days, .skills])
        XCTAssertEqual(sections[2].items.map(\.id), ["skill:s1", "skill:s2", "skill:s3", "skill:s4", "skill:s5"])
        XCTAssertEqual(sections[2].total, 7)
        XCTAssertEqual(sections[2].hiddenCount, 2)
        XCTAssertEqual(sections[0].items, [])
        XCTAssertEqual(sections[0].hiddenCount, 0)
        XCTAssertEqual(sections[1].byWordsReason, "no daily summary has been embedded yet")
        XCTAssertEqual(sections[1].hiddenCount, 0)
    }

    func testASingleScopeShowsAllOfItsKindOnly() {
        let sections = manyResults().sections(.skills)
        XCTAssertEqual(sections.map(\.kind), [.skills])
        XCTAssertEqual(sections[0].items.count, 7)
        XCTAssertEqual(sections[0].hiddenCount, 0)
    }

    func testVisibleItemsRunAcrossGroupsInOrder() {
        XCTAssertEqual(manyResults().visibleItems(.all).map(\.id), [
            "day:2026-10-12", "day:2026-10-11", "skill:s1", "skill:s2", "skill:s3", "skill:s4", "skill:s5",
        ])
        XCTAssertEqual(manyResults().visibleItems(.days).map(\.id), ["day:2026-10-12", "day:2026-10-11"])
    }

    func testEmptyMeansNothingFoundAndNothingFailed() {
        var results = CommonplaceResults()
        results.skills = .found(CommonplaceGroup(mode: "semantic", reason: "", results: []))
        results.memory = .found([])
        results.days = .found(CommonplaceGroup(mode: "semantic", reason: "", results: []))
        XCTAssertTrue(results.isEmpty(.all))
        results.memory = .failed(.memoryOff)
        XCTAssertFalse(results.isEmpty(.all))
        XCTAssertTrue(results.isEmpty(.skills))
        // A kind not searched yet (a new scope, its search under way) is
        // not "nothing found".
        results.skills = nil
        XCTAssertFalse(results.isEmpty(.skills))
        XCTAssertEqual(results.sections(.skills).map(\.searched), [false])
    }

    func testGroupMessages() {
        func section(_ error: MemoryAPIError?) -> CommonplaceSection {
            CommonplaceSection(kind: .skills, items: [], total: 0, searched: true, error: error, byWordsReason: nil)
        }
        XCTAssertEqual(section(.unexpected(status: 0, body: "The request timed out.")).errorMessage, "The request timed out.")
        XCTAssertEqual(section(.unexpected(status: 501, body: "x")).errorMessage, "The worker answered 501. Try again.")
        XCTAssertEqual(section(.badRequest("'query' must be a non-empty string")).errorMessage, "'query' must be a non-empty string")
        XCTAssertEqual(section(.workerNotRunning).errorMessage, "The worker is not running.")
        XCTAssertEqual(CommonplaceKind.allCases.map(\.noMatches), ["No moments match.", "No days match.", "No Skills match."])
    }

    func testDayTitleReadsTheLocalDay() {
        var calendar = Calendar(identifier: .gregorian)
        calendar.timeZone = TimeZone(identifier: "America/Los_Angeles")!
        let format = MemoryFormat(calendar: calendar, locale: Locale(identifier: "en_US"))
        XCTAssertEqual(format.dayTitle(Self.day("2026-10-12")), "Monday, Oct 12, 2026")
        XCTAssertEqual(format.dayTitle(Self.day("someday")), "someday")
    }

    // MARK: - What agents see

    func testAgentsLineForEachState() {
        let unknown = AgentsLine(status: nil)
        XCTAssertEqual(unknown.text, "Agents see your Skills, daily summaries and profile.")
        XCTAssertNil(unknown.action)
        XCTAssertNil(unknown.actionTitle)

        let off = AgentsLine(status: Self.status(enabled: true, agentsAccess: false))
        XCTAssertEqual(off.text, "Agents see your Skills, daily summaries and profile. Memory: off")
        XCTAssertEqual(off.action, .allow)
        XCTAssertEqual(off.actionTitle, "Turn On\u{2026}")

        let on = AgentsLine(status: Self.status(enabled: true, agentsAccess: true))
        XCTAssertEqual(on.text, "Agents see your Skills, daily summaries and profile. Memory: on")
        XCTAssertEqual(on.action, .disallow)
        XCTAssertEqual(on.actionTitle, "Turn Off")

        // Allowed, but there is no Memory to search: agents get nothing.
        let memoryOff = AgentsLine(status: Self.status(enabled: false, agentsAccess: true))
        XCTAssertEqual(memoryOff.text, "Agents see your Skills, daily summaries and profile. Memory: allowed, but Memory is off")
        XCTAssertEqual(memoryOff.action, .disallow)

        // Memory off and not allowed: the switch still says off.
        XCTAssertEqual(AgentsLine(status: Self.status(enabled: false, agentsAccess: false)).action, .allow)
    }

    func testTheAgentsSwitchIsWrittenWithoutARestart() {
        XCTAssertFalse(MemorySettings.Key.agentsAccess.needsRestart)
        var settings = MemorySettings.defaults
        settings.agentsAccess = true
        let config = "[memory]\nenabled = true\n"
        let updated = settings.apply([.agentsAccess], to: config)
        XCTAssertEqual(ConfigTOML.value(section: "memory", key: "agents_access", in: updated), "true")
        XCTAssertEqual(ConfigTOML.value(section: "memory", key: "enabled", in: updated), "true")
    }

    // MARK: - Opening

    func testOpeningASkillSelectsItInWorkflows() {
        let navigator = AppNavigator()
        let opening = CommonplaceItem.skill(Self.skill("check-domain")).open(navigator: navigator)
        XCTAssertEqual(opening, .window(id: "workflows"))
        XCTAssertEqual(navigator.skillSlug, "check-domain")
        XCTAssertNil(navigator.digestDate)
    }

    func testOpeningADayShowsItInTheDigest() {
        let navigator = AppNavigator()
        let opening = CommonplaceItem.day(Self.day("2026-10-12")).open(navigator: navigator)
        XCTAssertEqual(opening, .window(id: "daily-digest"))
        XCTAssertEqual(navigator.digestDate, "2026-10-12")
        XCTAssertEqual(navigator.takeDigestDate(), "2026-10-12")
        XCTAssertNil(navigator.digestDate)
        XCTAssertNil(navigator.skillSlug)
    }

    func testOpeningAMomentOpensItHere() throws {
        let moment = try JSONDecoder().decode(MemorySearchResponse.self, from: Data(Self.moments.utf8)).moments[0]
        let navigator = AppNavigator()
        XCTAssertEqual(CommonplaceItem.moment(moment).open(navigator: navigator), .moment(moment))
        XCTAssertNil(navigator.skillSlug)
        XCTAssertNil(navigator.digestDate)
    }

    func testTheWindowIdsExist() {
        // CommonplaceItem.open names these Window scenes (AgentHandoverApp).
        let source = try? String(contentsOf: Self.appSource, encoding: .utf8)
        XCTAssertNotNil(source)
        for id in ["workflows", "daily-digest", "memory"] {
            XCTAssertTrue(source?.contains("id: \"\(id)\"") == true, id)
        }
        XCTAssertTrue(source?.contains("Window(\"Commonplace\", id: \"memory\")") == true)
    }

    private static var appSource: URL {
        URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
            .appendingPathComponent("Sources/AgentHandoverApp/AgentHandoverApp.swift")
    }

    func testWorkflowsFindsTheSkillBySlug() throws {
        let json = """
            [{"sop_id": "sop-1", "slug": "check-email", "title": "Check Email", "source": "passive",
              "status": "approved", "confidence": 0.85, "created_at": "2026-03-10T10:00:00Z"},
             {"sop_id": "sop-2", "slug": "check-domain", "title": "Check Domain", "source": "focus",
              "status": "draft", "confidence": 0.7, "created_at": "2026-03-11T10:00:00Z"}]
            """
        let entries = try JSONDecoder().decode([SOPEntry].self, from: Data(json.utf8))
        XCTAssertEqual(SOPIndexManager.entry(slug: "check-domain", in: entries)?.id, "sop-2")
        XCTAssertNil(SOPIndexManager.entry(slug: "missing", in: entries))
    }

    func testTheDigestLoadsTheDayAskedForOrTheLatest() {
        var calendar = Calendar(identifier: .gregorian)
        calendar.timeZone = TimeZone(identifier: "UTC")!
        let now = Date(timeIntervalSince1970: 1_791_806_400)  // 2026-10-12T12:00:00Z
        XCTAssertEqual(DigestViewModel.datesToTry(requested: "2026-10-03", now: now, calendar: calendar), ["2026-10-03"])
        XCTAssertEqual(DigestViewModel.datesToTry(requested: nil, now: now, calendar: calendar), ["2026-10-11", "2026-10-12"])
    }

    // MARK: - The window's model

    private func model() -> CommonplaceWindowModel {
        CommonplaceWindowModel(api: api, diagnose: { .waiting($0) })
    }

    func testArrowsMoveAcrossGroupsAndStopAtTheEnds() async {
        answerEverything()
        let model = model()
        model.query = "domain"
        await model.run(force: true)
        XCTAssertEqual(model.phase, .results)
        XCTAssertEqual(model.lastQuery, "domain")
        // The first result is selected: Memory comes first.
        XCTAssertEqual(model.selectedID, "moment:ev-1")
        XCTAssertEqual(model.selected?.moment?.id, "ev-1")
        model.moveSelection(1)
        model.moveSelection(1)  // into Days
        XCTAssertEqual(model.selectedID, "day:2026-10-12")
        model.moveSelection(1)
        model.moveSelection(1)  // into Skills
        XCTAssertEqual(model.selectedID, "skill:check-domain")
        model.moveSelection(5)
        XCTAssertEqual(model.selectedID, "skill:deploy-staging")
        model.moveSelection(-100)
        XCTAssertEqual(model.selectedID, "moment:ev-1")
        XCTAssertEqual(model.knownApps, ["Mail", "Safari"])
    }

    func testANarrowerScopeKeepsTheAnswerAndTheAppFilterIsMemorysOwn() async {
        answerEverything()
        let model = model()
        model.query = "domain"
        await model.run(force: true)
        XCTAssertEqual(worker.requests.count, 2)

        // Show all Skills: answered by what All found, no new search.
        model.selectedID = "day:2026-10-12"
        model.scope = .skills
        model.scopeChanged()
        await model.pending?.value
        XCTAssertEqual(worker.requests.count, 2)
        XCTAssertEqual(model.sections.map(\.kind), [.skills])
        // The selection moves to a shown result.
        XCTAssertEqual(model.selectedID, "skill:check-domain")

        // An app filter in Memory searches Memory again.
        model.scope = .memory
        model.scopeChanged()
        await model.pending?.value
        XCTAssertEqual(worker.requests.count, 2)
        model.app = "Safari"
        model.search(force: false)
        await model.pending?.value
        XCTAssertEqual(worker.requests(to: "/memory/search").count, 2)
        XCTAssertEqual(body(worker.requests(to: "/memory/search").last), #"{"app":"Safari","limit":50,"query":"domain"}"#)

        // Back to All: the app filter goes, and All is searched again.
        model.scope = .all
        model.scopeChanged()
        XCTAssertNil(model.app)
        await model.pending?.value
        XCTAssertEqual(worker.requests.count, 5)
        XCTAssertFalse(model.hasFilters)
    }

    func testAFailedMemorySearchIsDiagnosed() async {
        worker.answers["/commonplace/search"] = (200, Self.skillsAndDays)
        worker.answers["/memory/search"] = (503, #"{"error": "model_unavailable", "detail": "connection refused"}"#)
        let model = model()
        model.query = "acme"
        await model.run(force: true)
        XCTAssertEqual(model.memoryProblem, .waiting("connection refused"))
        XCTAssertEqual(model.results.memory, .failed(.modelUnavailable("connection refused")))
        // The other groups still answer and the arrows skip the failed one.
        XCTAssertEqual(model.visibleItems.count, 5)

        worker.answers["/memory/search"] = (503, #"{"error": "memory_off"}"#)
        await model.run(force: true)
        XCTAssertEqual(model.memoryProblem, .off)
    }

    func testAnEmptyQueryClearsTheResults() async {
        answerEverything()
        let model = model()
        model.query = "domain"
        await model.run(force: true)
        model.query = "   "
        await model.run(force: false)
        XCTAssertEqual(model.phase, .idle)
        XCTAssertEqual(model.results, .empty)
        XCTAssertEqual(model.lastQuery, "")
        XCTAssertFalse(model.isShowingResults(for: "domain"))
    }

    // MARK: - Polish

    func testARowLeavesOutTheAppAndPlaceItAlreadyShows() throws {
        func moment(_ description: String?, app: String? = "Finder") throws -> MemoryMoment {
            var object: [String: Any] = ["id": "e", "first_ts": "2026-10-13T10:00:00Z", "last_ts": "2026-10-13T10:00:00Z",
                                         "frames": 1, "score": 0.5]
            if let app { object["app"] = app }
            if let description { object["description"] = description }
            return try JSONDecoder().decode(MemoryMoment.self, from: JSONSerialization.data(withJSONObject: object))
        }
        XCTAssertEqual(try moment("viewing invoice files (Finder, Macintosh HD/Users/invoices-2026/)").rowDescription,
                       "viewing invoice files")
        // A parenthesis that is not the app's is kept.
        XCTAssertEqual(try moment("reading about cats (and dogs)").rowDescription, "reading about cats (and dogs)")
        XCTAssertEqual(try moment("reading (Finder notes)", app: nil).rowDescription, "reading (Finder notes)")
        XCTAssertNil(try moment(nil).rowDescription)
        XCTAssertNil(try moment("  ").rowDescription)
    }

    func testAGroupAtTheLimitSaysThereMayBeMore() {
        func section(_ total: Int) -> CommonplaceSection {
            CommonplaceSection(kind: .memory, items: [], total: total, searched: true, error: nil, byWordsReason: nil)
        }
        XCTAssertEqual(section(12).countLabel, "12")
        XCTAssertEqual(section(12).showAllLabel, "Show All 12")
        XCTAssertEqual(section(CommonplaceRequest.limit).countLabel, "\(CommonplaceRequest.limit)+")
        XCTAssertEqual(section(CommonplaceRequest.limit).showAllLabel, "Show More")
    }

    func testTheExamplesCoverEveryKindAndAreDistinct() {
        XCTAssertEqual(Set(CommonplaceExamples.searches).count, CommonplaceExamples.searches.count)
        XCTAssertEqual(Set(CommonplaceExamples.agentRequests).count, CommonplaceExamples.agentRequests.count)
        XCTAssertGreaterThanOrEqual(CommonplaceExamples.searches.count, 5)
        XCTAssertGreaterThanOrEqual(CommonplaceExamples.agentRequests.count, 4)
        XCTAssertEqual(CommonplaceScope.all.placeholder, "Search what you saw, your days and your Skills")
    }

    func testWaitingSaysItClearsByItself() {
        XCTAssertEqual(MemoryProblem.waiting("macOS reports memory pressure (warning)").message,
                       "Search still works. New screens are added once this clears: macOS reports memory pressure (warning).")
    }
}
