import XCTest
@testable import AgentHandoverApp

/// WP-E8: the menu's Memory state and pause choices, and how moments,
/// spans, sizes and the status line read.
final class MemoryFormatTests: XCTestCase {
    private var calendar: Calendar = {
        var calendar = Calendar(identifier: .gregorian)
        calendar.timeZone = TimeZone(identifier: "UTC")!
        return calendar
    }()

    private var format: MemoryFormat {
        MemoryFormat(calendar: calendar, locale: Locale(identifier: "en_US"))
    }

    /// 2026-10-12T13:40:00Z, a Monday.
    private let now = Date(timeIntervalSince1970: 1_791_812_400)

    private func status(enabled: Bool = true, paused: Bool = false, until: String? = nil,
                        moments: Int = 0, bytes: Int64 = 0, oldest: String? = nil,
                        waiting: String? = nil) -> MemoryStatus {
        MemoryStatus(enabled: enabled, paused: paused, pausedUntil: until, agentsAccess: false,
                     model: "embeddinggemma-2:440m", moments: moments, frames: moments, oldest: oldest,
                     newest: nil, thumbnails: 0, diskBytes: bytes, waiting: waiting)
    }

    private func moment(_ first: String, _ last: String, frames: Int) -> MemoryMoment {
        MemoryMoment(id: "ev", firstTS: first, lastTS: last, frames: frames, app: "Safari", windowTitle: "t",
                     urlHost: nil, description: nil, score: 0.5, thumbID: nil)
    }

    // MARK: - Menu state

    func testTheMenuStateFollowsTheStatus() {
        XCTAssertEqual(MemoryMenuState(status: nil), .unavailable)
        XCTAssertEqual(MemoryMenuState(status: status(enabled: false)), .off)
        XCTAssertEqual(MemoryMenuState(status: status(enabled: false, paused: true)), .off)
        XCTAssertEqual(MemoryMenuState(status: status()), .recording)
        XCTAssertEqual(MemoryMenuState(status: status(paused: true)), .paused(until: nil))
        XCTAssertEqual(MemoryMenuState(status: status(paused: true, until: "2026-10-12T15:40:00Z")),
                       .paused(until: Date(timeIntervalSince1970: 1_791_819_600)))
        XCTAssertTrue(MemoryMenuState.paused(until: nil).isPaused)
        XCTAssertFalse(MemoryMenuState.recording.isPaused)
        // On, but the worker cannot index yet: say so instead of "Recording".
        let waiting = status(waiting: "macOS reports memory pressure (warning)")
        XCTAssertEqual(MemoryMenuState(status: waiting), .waiting)
        XCTAssertEqual(MemoryMenuState.waiting.label(now: now, format: format), "Recording; indexing waits")
        XCTAssertTrue(MemoryMenuState.waiting.isOn)
        XCTAssertEqual(MemoryMenuState(status: status(paused: true, waiting: "x")), .paused(until: nil))
    }

    func testTheMenuLabels() {
        XCTAssertEqual(MemoryMenuState.recording.label(now: now, format: format), "Recording")
        XCTAssertEqual(MemoryMenuState.off.label(now: now, format: format), "Off")
        XCTAssertEqual(MemoryMenuState.unavailable.label(now: now, format: format), "Not available")
        XCTAssertEqual(MemoryMenuState.paused(until: nil).label(now: now, format: format), "Paused until you resume")
        let later = Date(timeIntervalSince1970: 1_791_819_600)  // 15:40 today
        XCTAssertEqual(MemoryMenuState.paused(until: later).label(now: now, format: format), "Paused until 3:40\u{202F}PM")
        let midnight = Date(timeIntervalSince1970: 1_791_849_600)  // 2026-10-13T00:00Z
        XCTAssertEqual(MemoryMenuState.paused(until: midnight).label(now: now, format: format),
                       "Paused until tomorrow, 12:00\u{202F}AM")
    }

    func testPauseChoicesSendTheirMinutes() {
        XCTAssertEqual(MemoryPauseOption.oneHour.minutes(now: now, calendar: calendar), 60)
        XCTAssertNil(MemoryPauseOption.untilResumed.minutes(now: now, calendar: calendar))
        // 13:40 to midnight.
        XCTAssertEqual(MemoryPauseOption.untilTomorrow.minutes(now: now, calendar: calendar), 620)
        let lateNight = Date(timeIntervalSince1970: 1_791_849_590)  // 23:59:50
        XCTAssertEqual(MemoryPauseOption.untilTomorrow.minutes(now: lateNight, calendar: calendar), 1)
        XCTAssertEqual(MemoryPauseOption.allCases.map(\.title), ["For 1 Hour", "Until Tomorrow", "Until I Resume"])
    }

    // MARK: - Moments

    func testRelativeTimes() {
        XCTAssertEqual(format.relative(now.addingTimeInterval(-20), now: now), "just now")
        XCTAssertEqual(format.relative(now.addingTimeInterval(-12 * 60), now: now), "12 min ago")
        XCTAssertEqual(format.relative(now.addingTimeInterval(-3 * 3600 - 60), now: now), "3 h ago")
        XCTAssertEqual(format.relative(now.addingTimeInterval(-20 * 3600), now: now), "yesterday")
        XCTAssertNil(format.relative(now.addingTimeInterval(-3 * 86400), now: now))
    }

    func testAMomentsTimeIsRelativeAndAbsolute() {
        let recent = moment("2026-10-12T13:28:00Z", "2026-10-12T13:28:00Z", frames: 1)
        XCTAssertEqual(format.when(recent, now: now), "12 min ago \u{00B7} Mon, Oct 12, 2026 at 1:28\u{202F}PM")
        let old = moment("2026-10-01T09:05:00Z", "2026-10-01T09:05:00Z", frames: 1)
        XCTAssertEqual(format.when(old, now: now), "Thu, Oct 1, 2026 at 9:05\u{202F}AM")
    }

    func testFramesAndSpan() {
        XCTAssertNil(format.span(moment("2026-10-12T13:00:00Z", "2026-10-12T13:00:00Z", frames: 1)))
        XCTAssertEqual(format.span(moment("2026-10-12T13:00:00Z", "2026-10-12T13:12:30Z", frames: 34)),
                       "12 min \u{00B7} 34 screens")
        XCTAssertEqual(format.span(moment("2026-10-12T13:00:00Z", "2026-10-12T13:00:20Z", frames: 2)),
                       "under a minute \u{00B7} 2 screens")
        XCTAssertEqual(MemoryFormat.duration(3900), "1 h 5 min")
        XCTAssertEqual(MemoryFormat.duration(7200), "2 h")
    }

    func testSizes() {
        XCTAssertEqual(MemoryFormat.size(0), "0 KB")
        XCTAssertEqual(MemoryFormat.size(820_400), "820 KB")
        XCTAssertEqual(MemoryFormat.size(4_200_000), "4.2 MB")
        XCTAssertEqual(MemoryFormat.size(38_000_000), "38 MB")
        XCTAssertEqual(MemoryFormat.size(1_300_000_000), "1.3 GB")
    }

    func testTheStatusLine() {
        XCTAssertEqual(format.statusLine(status(), now: now), "No moments yet")
        XCTAssertEqual(format.statusLine(status(moments: 1, bytes: 120_000, oldest: "2026-10-12T08:00:00Z"), now: now),
                       "1 moment \u{00B7} 120 KB \u{00B7} since Oct 12")
        XCTAssertEqual(format.statusLine(status(moments: 1204, bytes: 38_000_000, oldest: "2025-11-03T08:00:00Z"), now: now),
                       "1,204 moments \u{00B7} 38 MB \u{00B7} since Nov 3, 2025")
    }

    // MARK: - Why Memory cannot search

    func testTheProblemBehindAnUnavailableModel() {
        let model = "embeddinggemma-2:440m"
        func diagnose(_ version: String?, _ installed: Set<String>?) -> MemoryProblem {
            MemoryProblem.diagnose(detail: "embed failed", ollamaVersion: version, installed: installed,
                                   memoryModel: model, minimum: "0.40.0")
        }
        XCTAssertEqual(diagnose(nil, nil), .ollamaNotRunning)
        XCTAssertEqual(diagnose("0.34.2", [model]), .ollamaTooOld(have: "0.34.2", need: "0.40.0"))
        XCTAssertEqual(diagnose("0.40.3", ["qwen3.5:4b"]), .modelMissing(model))
        XCTAssertEqual(diagnose("0.40.3", [model]), .waiting("embed failed"))
        XCTAssertEqual(diagnose("0.40.3", nil), .waiting("embed failed"))
        XCTAssertEqual(MemoryProblem.modelMissing(model).message,
                       "Memory needs the embeddinggemma-2:440m model, which is not downloaded yet.")
    }

    func testTheWorkerStatusCarriesTheUpgradeProgress() throws {
        let json = #"{"pid": 1, "version": "2.0.0", "started_at": "", "heartbeat": "", "events_processed_today": 0, "sops_generated": 0, "consecutive_errors": 0, "vlm_available": true, "embedding_upgrade": {"state": "running", "done": 120, "total": 400, "dropped": 2, "error": null}}"#
        let status = try JSONDecoder().decode(WorkerStatusFile.self, from: Data(json.utf8))
        XCTAssertEqual(status.embedding_upgrade, EmbeddingUpgradeStatus(state: "running", done: 120, total: 400, dropped: 2, error: nil))
        XCTAssertEqual(status.embedding_upgrade?.line,
                       "Re-indexing your Skills with EmbeddingGemma 2: 120 of 400. Search keeps working meanwhile.")
        let old = #"{"pid": 1, "version": "0.5.0", "started_at": "", "heartbeat": "", "events_processed_today": 0, "sops_generated": 0, "consecutive_errors": 0, "vlm_available": true}"#
        XCTAssertNil(try JSONDecoder().decode(WorkerStatusFile.self, from: Data(old.utf8)).embedding_upgrade)
        let failed = EmbeddingUpgradeStatus(state: "failed", done: 3, total: 9, dropped: 0, error: "Ollama 0.34.2 is older than 0.40")
        XCTAssertEqual(failed.line, "Re-indexing stopped: Ollama 0.34.2 is older than 0.40. Your Skills still work with the previous model.")
    }
}
