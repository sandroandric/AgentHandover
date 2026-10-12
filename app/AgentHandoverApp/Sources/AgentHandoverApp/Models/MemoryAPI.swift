import Foundation

/// One moment of Memory: a run of near-identical screens of one window,
/// as `POST /memory/search` returns it.
struct MemoryMoment: Decodable, Equatable, Identifiable, Hashable {
    let id: String
    let firstTS: String
    let lastTS: String
    let frames: Int
    let app: String?
    let windowTitle: String?
    let urlHost: String?
    /// nil until the worker has described the screen.
    let description: String?
    let score: Double
    /// The event id the thumbnail is named by; nil when none was kept.
    let thumbID: String?

    enum CodingKeys: String, CodingKey {
        case id, frames, app, description, score
        case firstTS = "first_ts"
        case lastTS = "last_ts"
        case windowTitle = "window_title"
        case urlHost = "url_host"
        case thumbID = "thumb_id"
    }

    var firstDate: Date? { MemoryAPI.parseDate(firstTS) }
    var lastDate: Date? { MemoryAPI.parseDate(lastTS) }
}

struct MemorySearchResponse: Decodable, Equatable {
    let moments: [MemoryMoment]
    let model: String
}

/// `GET /memory/status` (and the answer of pause and resume).
struct MemoryStatus: Decodable, Equatable {
    let enabled: Bool
    let paused: Bool
    let pausedUntil: String?
    let agentsAccess: Bool
    let model: String
    let moments: Int
    let frames: Int
    let oldest: String?
    let newest: String?
    let thumbnails: Int
    let diskBytes: Int64
    /// Why embedding is waiting (Ollama down, model not pulled, ...); nil
    /// when it is not.
    let waiting: String?

    enum CodingKeys: String, CodingKey {
        case enabled, paused, model, moments, frames, oldest, newest, thumbnails, waiting
        case pausedUntil = "paused_until"
        case agentsAccess = "agents_access"
        case diskBytes = "disk_bytes"
    }

    var pausedUntilDate: Date? { pausedUntil.flatMap(MemoryAPI.parseDate) }
    var oldestDate: Date? { oldest.flatMap(MemoryAPI.parseDate) }
}

/// What `POST /memory/forget` deleted.
struct MemoryForgetCounts: Decodable, Equatable {
    let events: Int
    let memoryMoments: Int
    let thumbnails: Int
    let screenshots: Int
    let vectors: Int
    let dailySummaries: Int

    enum CodingKeys: String, CodingKey {
        case events, thumbnails, screenshots, vectors
        case memoryMoments = "memory_moments"
        case dailySummaries = "daily_summaries"
    }
}

/// What to forget: exactly one of these selectors per request.
enum MemoryForgetSelector: Equatable {
    case moment(id: String)
    case all
    case range(since: Date, until: Date, app: String?)
    case app(String)
}

enum MemoryAPIError: Error, Equatable {
    /// Nothing listens on the query API's port: the worker is not running.
    case workerNotRunning
    /// `<data dir>/query-api-token` is missing or empty (the worker writes
    /// it when it starts).
    case noToken
    /// 503 `memory_off`: `[memory] enabled = false`.
    case memoryOff
    /// 503 `model_unavailable`: the query could not be embedded.
    case modelUnavailable(String)
    /// 400: the request was refused, with the worker's reason.
    case badRequest(String)
    /// 503 `busy`: a forget waited for other work and gave up; try again.
    case busy
    /// 404: the moment to forget is gone (already forgotten or purged).
    case momentGone
    /// Any other answer.
    case unexpected(status: Int, body: String)
}

/// The worker's Memory endpoints on the query API (127.0.0.1:9477, every
/// request carries `X-AgentHandover-Token`). The transport is injected so
/// tests see the exact requests and answer them.
struct MemoryAPI {
    typealias Transport = (URLRequest) async throws -> (Data, URLResponse)

    static let defaultPort = 9477
    static let tokenHeader = "X-AgentHandover-Token"

    static var dataDirectory: URL {
        FileManager.default.homeDirectoryForCurrentUser
            .appendingPathComponent("Library/Application Support/agenthandover")
    }

    var baseURL: URL
    /// The token, read for each request (the worker writes it at start).
    var token: () -> String?
    var transport: Transport

    init(
        baseURL: URL = URL(string: "http://127.0.0.1:\(MemoryAPI.defaultPort)")!,
        token: @escaping () -> String? = { MemoryAPI.readToken() },
        transport: @escaping Transport = { try await URLSession.shared.data(for: $0) }
    ) {
        self.baseURL = baseURL
        self.token = token
        self.transport = transport
    }

    /// `<data dir>/query-api-token`, trimmed; nil when missing or empty.
    static func readToken(dataDirectory: URL = MemoryAPI.dataDirectory) -> String? {
        let url = dataDirectory.appendingPathComponent("query-api-token")
        guard let raw = try? String(contentsOf: url, encoding: .utf8) else { return nil }
        let token = raw.trimmingCharacters(in: .whitespacesAndNewlines)
        return token.isEmpty ? nil : token
    }

    // MARK: - Endpoints

    func search(
        query: String, since: Date? = nil, until: Date? = nil, app: String? = nil, limit: Int? = nil
    ) async throws -> MemorySearchResponse {
        var body: [String: Any] = ["query": query]
        if let since { body["since"] = Self.isoString(since) }
        if let until { body["until"] = Self.isoString(until) }
        if let app { body["app"] = app }
        if let limit { body["limit"] = limit }
        return try await send("POST", "/memory/search", body: body)
    }

    func status() async throws -> MemoryStatus {
        try await send("GET", "/memory/status", body: nil)
    }

    /// Pause for `minutes`, or until resumed when nil.
    func pause(minutes: Int?) async throws -> MemoryStatus {
        let body: [String: Any] = ["minutes": minutes.map { $0 as Any } ?? NSNull()]
        return try await send("POST", "/memory/pause", body: body)
    }

    func resume() async throws -> MemoryStatus {
        try await send("POST", "/memory/resume", body: [:])
    }

    func forget(_ selector: MemoryForgetSelector) async throws -> MemoryForgetCounts {
        try await send("POST", "/memory/forget", body: Self.forgetBody(selector))
    }

    static func forgetBody(_ selector: MemoryForgetSelector) -> [String: Any] {
        switch selector {
        case .moment(let id): return ["moment_id": id]
        case .all: return ["all": true]
        case .app(let app): return ["app": app]
        case .range(let since, let until, let app):
            var body: [String: Any] = ["since": isoString(since), "until": isoString(until)]
            if let app { body["app"] = app }
            return body
        }
    }

    // MARK: - Requests

    /// The request for `method path`, with the token header and, for a
    /// body, JSON with sorted keys.
    func request(_ method: String, _ path: String, body: [String: Any]?) throws -> URLRequest {
        guard let token = token() else { throw MemoryAPIError.noToken }
        var request = URLRequest(url: baseURL.appendingPathComponent(path))
        request.httpMethod = method
        request.timeoutInterval = 15
        request.setValue(token, forHTTPHeaderField: Self.tokenHeader)
        if let body {
            request.setValue("application/json", forHTTPHeaderField: "Content-Type")
            request.httpBody = try JSONSerialization.data(withJSONObject: body, options: [.sortedKeys])
        }
        return request
    }

    private func send<T: Decodable>(_ method: String, _ path: String, body: [String: Any]?) async throws -> T {
        let request = try request(method, path, body: body)
        let data: Data
        let response: URLResponse
        do {
            (data, response) = try await transport(request)
        } catch let error as URLError where Self.meansNotRunning(error) {
            throw MemoryAPIError.workerNotRunning
        }
        let status = (response as? HTTPURLResponse)?.statusCode ?? 0
        return try Self.decode(status: status, data: data)
    }

    static func meansNotRunning(_ error: URLError) -> Bool {
        [.cannotConnectToHost, .networkConnectionLost, .cannotFindHost, .timedOut, .notConnectedToInternet]
            .contains(error.code)
    }

    /// The answer `data` with HTTP `status` as `T`, or the contract's error.
    static func decode<T: Decodable>(status: Int, data: Data) throws -> T {
        if status == 200, let value = try? JSONDecoder().decode(T.self, from: data) {
            return value
        }
        let text = String(data: data, encoding: .utf8) ?? ""
        let error = try? JSONDecoder().decode(ErrorBody.self, from: data)
        switch (status, error?.error) {
        case (503, "memory_off"):
            throw MemoryAPIError.memoryOff
        case (503, "model_unavailable"):
            throw MemoryAPIError.modelUnavailable(error?.detail ?? "")
        case (400, let reason?):
            throw MemoryAPIError.badRequest(reason)
        case (503, "busy"):
            throw MemoryAPIError.busy
        case (404, _):
            throw MemoryAPIError.momentGone
        default:
            throw MemoryAPIError.unexpected(status: status, body: text)
        }
    }

    /// The body of a refused request: `{"error": ..., "detail"?: ...}`.
    private struct ErrorBody: Decodable {
        let error: String
        let detail: String?
    }

    // MARK: - Dates

    /// "2026-10-12T13:40:00Z": what the app sends.
    static func isoString(_ date: Date) -> String {
        let formatter = ISO8601DateFormatter()
        formatter.formatOptions = [.withInternetDateTime]
        return formatter.string(from: date)
    }

    /// An ISO-8601 time as the worker writes them: with or without
    /// fractional seconds, `Z` or an offset; a time without a zone is UTC.
    static func parseDate(_ text: String) -> Date? {
        let plain = ISO8601DateFormatter()
        plain.formatOptions = [.withInternetDateTime]
        let fractional = ISO8601DateFormatter()
        fractional.formatOptions = [.withInternetDateTime, .withFractionalSeconds]
        for candidate in [text, text + "Z"] {
            if let date = plain.date(from: candidate) ?? fractional.date(from: candidate) {
                return date
            }
        }
        // Python's isoformat writes microseconds; keep milliseconds.
        if let dot = text.firstIndex(of: ".") {
            let digits = text[text.index(after: dot)...].prefix { $0.isNumber }
            if digits.count > 3 {
                let zone = text[text.index(after: dot)...].dropFirst(digits.count)
                let trimmed = String(text[..<dot]) + "." + digits.prefix(3) + zone
                return parseDate(trimmed)
            }
        }
        return nil
    }
}
