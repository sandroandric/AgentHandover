import XCTest
@testable import AgentHandoverApp

/// WP-E8: the app's client of the worker's Memory endpoints, against the
/// shared contract: the exact requests, the token header, and every answer
/// and error the contract names.
final class MemoryAPITests: XCTestCase {

    /// Records each request and answers with `status` and `body`.
    private final class FakeWorker: @unchecked Sendable {
        var requests: [URLRequest] = []
        var status = 200
        var body = Data()
        var failure: URLError?

        func transport(_ request: URLRequest) async throws -> (Data, URLResponse) {
            requests.append(request)
            if let failure { throw failure }
            let response = HTTPURLResponse(url: request.url!, statusCode: status, httpVersion: "HTTP/1.1", headerFields: nil)!
            return (body, response)
        }

        func answer(_ status: Int, _ json: String) {
            self.status = status
            body = Data(json.utf8)
        }
    }

    private var worker = FakeWorker()

    private var api: MemoryAPI {
        let worker = worker
        return MemoryAPI(token: { "tok-123" }, transport: { try await worker.transport($0) })
    }

    private static let statusJSON = """
        {"enabled": true, "paused": true, "paused_until": "2026-10-12T15:40:00Z", "agents_access": false,
         "model": "embeddinggemma-2:440m", "moments": 1204, "frames": 5120, "oldest": "2026-10-03T08:00:00Z",
         "newest": "2026-10-12T13:39:12.123456+00:00", "thumbnails": 310, "disk_bytes": 38000000, "waiting": null}
        """

    private func bodyText(_ request: URLRequest?) -> String? {
        request?.httpBody.flatMap { String(data: $0, encoding: .utf8) }
    }

    // MARK: - Requests

    func testSearchSendsTheQueryWithTheTokenToTheLoopbackPort() async throws {
        worker.answer(200, #"{"moments": [], "model": "embeddinggemma-2:440m"}"#)
        let since = Date(timeIntervalSince1970: 1_791_763_200)  // 2026-10-12T00:00:00Z
        _ = try await api.search(query: "invoice from acme", since: since, app: "Safari", limit: 50)
        let request = try XCTUnwrap(worker.requests.first)
        XCTAssertEqual(request.url?.absoluteString, "http://127.0.0.1:9477/memory/search")
        XCTAssertEqual(request.httpMethod, "POST")
        XCTAssertEqual(request.value(forHTTPHeaderField: "X-AgentHandover-Token"), "tok-123")
        XCTAssertEqual(request.value(forHTTPHeaderField: "Content-Type"), "application/json")
        XCTAssertEqual(bodyText(request),
                       #"{"app":"Safari","limit":50,"query":"invoice from acme","since":"2026-10-12T00:00:00Z"}"#)
    }

    func testSearchLeavesOutTheFiltersItIsNotGiven() async throws {
        worker.answer(200, #"{"moments": [], "model": "m"}"#)
        _ = try await api.search(query: "x")
        XCTAssertEqual(bodyText(worker.requests.first), #"{"query":"x"}"#)
    }

    func testStatusIsAGetWithoutABody() async throws {
        worker.answer(200, Self.statusJSON)
        _ = try await api.status()
        let request = try XCTUnwrap(worker.requests.first)
        XCTAssertEqual(request.url?.absoluteString, "http://127.0.0.1:9477/memory/status")
        XCTAssertEqual(request.httpMethod, "GET")
        XCTAssertNil(request.httpBody)
        XCTAssertEqual(request.value(forHTTPHeaderField: "X-AgentHandover-Token"), "tok-123")
    }

    func testPauseSendsMinutesOrNullForUntilResumed() async throws {
        worker.answer(200, Self.statusJSON)
        _ = try await api.pause(minutes: 60)
        _ = try await api.pause(minutes: nil)
        XCTAssertEqual(worker.requests.map(\.url?.path), ["/memory/pause", "/memory/pause"])
        XCTAssertEqual(bodyText(worker.requests[0]), #"{"minutes":60}"#)
        XCTAssertEqual(bodyText(worker.requests[1]), #"{"minutes":null}"#)
    }

    func testResumePostsAnEmptyObject() async throws {
        worker.answer(200, Self.statusJSON)
        _ = try await api.resume()
        XCTAssertEqual(worker.requests.first?.url?.path, "/memory/resume")
        XCTAssertEqual(worker.requests.first?.httpMethod, "POST")
        XCTAssertEqual(bodyText(worker.requests.first), "{}")
    }

    func testForgetSendsExactlyOneSelector() async throws {
        worker.answer(200, #"{"events": 1, "memory_moments": 1, "thumbnails": 1, "screenshots": 0, "vectors": 1, "daily_summaries": 0}"#)
        let since = Date(timeIntervalSince1970: 1_791_763_200)
        let until = since.addingTimeInterval(3600)
        _ = try await api.forget(.moment(id: "ev-1"))
        _ = try await api.forget(.all)
        _ = try await api.forget(.range(since: since, until: until, app: nil))
        _ = try await api.forget(.range(since: since, until: until, app: "Mail"))
        _ = try await api.forget(.app("Mail"))
        XCTAssertEqual(worker.requests.map { bodyText($0) }, [
            #"{"moment_id":"ev-1"}"#,
            #"{"all":true}"#,
            #"{"since":"2026-10-12T00:00:00Z","until":"2026-10-12T01:00:00Z"}"#,
            #"{"app":"Mail","since":"2026-10-12T00:00:00Z","until":"2026-10-12T01:00:00Z"}"#,
            #"{"app":"Mail"}"#,
        ])
        XCTAssertTrue(worker.requests.allSatisfy { $0.url?.path == "/memory/forget" && $0.httpMethod == "POST" })
    }

    func testNoTokenSendsNothing() async {
        let worker = worker
        let api = MemoryAPI(token: { nil }, transport: { try await worker.transport($0) })
        do {
            _ = try await api.status()
            XCTFail("expected noToken")
        } catch {
            XCTAssertEqual(error as? MemoryAPIError, .noToken)
        }
        XCTAssertTrue(worker.requests.isEmpty)
    }

    func testTheTokenIsReadTrimmedFromTheDataDirectory() throws {
        let dir = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: dir) }
        XCTAssertNil(MemoryAPI.readToken(dataDirectory: dir))
        try "  abc\n".write(to: dir.appendingPathComponent("query-api-token"), atomically: true, encoding: .utf8)
        XCTAssertEqual(MemoryAPI.readToken(dataDirectory: dir), "abc")
        try "\n".write(to: dir.appendingPathComponent("query-api-token"), atomically: true, encoding: .utf8)
        XCTAssertNil(MemoryAPI.readToken(dataDirectory: dir))
    }

    // MARK: - Answers

    func testSearchDecodesMomentsWithNulls() async throws {
        worker.answer(200, """
            {"moments": [
              {"id": "ev-1", "first_ts": "2026-10-12T13:00:00Z", "last_ts": "2026-10-12T13:12:00.250Z", "frames": 34,
               "app": "Safari", "window_title": "Invoice 4411 - Acme", "url_host": "billing.acme.com",
               "description": "An invoice from Acme for October.", "score": 0.82, "thumb_id": "ev-1"},
              {"id": "ev-2", "first_ts": "2026-10-11T09:00:00.123456+00:00", "last_ts": "2026-10-11T09:00:00.123456+00:00",
               "frames": 1, "app": null, "window_title": null, "url_host": null, "description": null,
               "score": 0.41, "thumb_id": null}
            ], "model": "embeddinggemma-2:440m"}
            """)
        let result = try await api.search(query: "acme invoice")
        XCTAssertEqual(result.model, "embeddinggemma-2:440m")
        XCTAssertEqual(result.moments.count, 2)
        let first = result.moments[0]
        XCTAssertEqual(first.id, "ev-1")
        XCTAssertEqual(first.frames, 34)
        XCTAssertEqual(first.app, "Safari")
        XCTAssertEqual(first.windowTitle, "Invoice 4411 - Acme")
        XCTAssertEqual(first.urlHost, "billing.acme.com")
        XCTAssertEqual(first.description, "An invoice from Acme for October.")
        XCTAssertEqual(first.score, 0.82)
        XCTAssertEqual(first.thumbID, "ev-1")
        XCTAssertEqual(first.firstDate, Date(timeIntervalSince1970: 1_791_810_000))
        XCTAssertEqual(first.lastDate!.timeIntervalSince1970, 1_791_810_720.25, accuracy: 0.001)
        let second = result.moments[1]
        XCTAssertNil(second.app)
        XCTAssertNil(second.description)
        XCTAssertNil(second.thumbID)
        XCTAssertEqual(second.firstDate!.timeIntervalSince1970, 1_791_709_200.123, accuracy: 0.001)
    }

    func testStatusDecodesEveryField() async throws {
        worker.answer(200, Self.statusJSON)
        let status = try await api.status()
        XCTAssertTrue(status.enabled)
        XCTAssertTrue(status.paused)
        XCTAssertEqual(status.pausedUntilDate, Date(timeIntervalSince1970: 1_791_819_600))
        XCTAssertFalse(status.agentsAccess)
        XCTAssertEqual(status.model, "embeddinggemma-2:440m")
        XCTAssertEqual(status.moments, 1204)
        XCTAssertEqual(status.frames, 5120)
        XCTAssertEqual(status.oldestDate, Date(timeIntervalSince1970: 1_791_014_400))
        XCTAssertNotNil(status.newest.flatMap(MemoryAPI.parseDate))
        XCTAssertEqual(status.thumbnails, 310)
        XCTAssertEqual(status.diskBytes, 38_000_000)
        XCTAssertNil(status.waiting)
    }

    func testStatusWithNothingKeptAndAReasonToWait() async throws {
        worker.answer(200, """
            {"enabled": true, "paused": false, "paused_until": null, "agents_access": true, "model": "embeddinggemma-2:440m",
             "moments": 0, "frames": 0, "oldest": null, "newest": null, "thumbnails": 0, "disk_bytes": 0,
             "waiting": "embeddinggemma-2:440m is not pulled"}
            """)
        let status = try await api.status()
        XCTAssertNil(status.pausedUntilDate)
        XCTAssertNil(status.oldestDate)
        XCTAssertEqual(status.waiting, "embeddinggemma-2:440m is not pulled")
    }

    func testForgetDecodesTheCounts() async throws {
        worker.answer(200, #"{"events": 12, "memory_moments": 3, "thumbnails": 3, "screenshots": 2, "vectors": 3, "daily_summaries": 1}"#)
        let counts = try await api.forget(.all)
        XCTAssertEqual(counts, MemoryForgetCounts(events: 12, memoryMoments: 3, thumbnails: 3, screenshots: 2, vectors: 3, dailySummaries: 1))
    }

    // MARK: - Errors

    private func error(_ status: Int, _ json: String) async -> MemoryAPIError? {
        worker.answer(status, json)
        do {
            _ = try await api.search(query: "q")
            return nil
        } catch {
            return error as? MemoryAPIError
        }
    }

    func testEveryContractErrorIsTyped() async {
        let off = await error(503, #"{"error": "memory_off"}"#)
        XCTAssertEqual(off, .memoryOff)
        let model = await error(503, #"{"error": "model_unavailable", "detail": "Ollama 0.34.2 cannot run embeddinggemma-2:440m"}"#)
        XCTAssertEqual(model, .modelUnavailable("Ollama 0.34.2 cannot run embeddinggemma-2:440m"))
        let bad = await error(400, #"{"error": "query must be a non-empty string"}"#)
        XCTAssertEqual(bad, .badRequest("query must be a non-empty string"))
        let refused = await error(403, #"{"error": "forbidden"}"#)
        XCTAssertEqual(refused, .unexpected(status: 403, body: #"{"error": "forbidden"}"#))
        let garbage = await error(200, "not json")
        XCTAssertEqual(garbage, .unexpected(status: 200, body: "not json"))
        // The worker's forget: busy after waiting for other work; an
        // unknown moment.
        let busy = await error(503, #"{"error": "busy"}"#)
        XCTAssertEqual(busy, .busy)
        let gone = await error(404, #"{"error": "no Memory moment m1"}"#)
        XCTAssertEqual(gone, .momentGone)
        XCTAssertEqual(MemoryCenter.describe(MemoryAPIError.busy),
                       "AgentHandover is busy with other work. Try again in a minute.")
    }

    func testNothingListeningMeansTheWorkerIsNotRunning() async {
        worker.failure = URLError(.cannotConnectToHost)
        do {
            _ = try await api.status()
            XCTFail("expected workerNotRunning")
        } catch {
            XCTAssertEqual(error as? MemoryAPIError, .workerNotRunning)
        }
    }

    func testDatesSentAreUTCSeconds() {
        XCTAssertEqual(MemoryAPI.isoString(Date(timeIntervalSince1970: 1_791_819_600.7)), "2026-10-12T15:40:00Z")
        XCTAssertEqual(MemoryAPI.parseDate("2026-10-12T15:40:00"), Date(timeIntervalSince1970: 1_791_819_600))
        XCTAssertEqual(MemoryAPI.parseDate("2026-10-12T17:40:00+02:00"), Date(timeIntervalSince1970: 1_791_819_600))
        XCTAssertNil(MemoryAPI.parseDate("yesterday"))
    }
}
