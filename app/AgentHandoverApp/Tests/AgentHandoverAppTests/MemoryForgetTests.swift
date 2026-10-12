import XCTest
@testable import AgentHandoverApp

/// WP-E8: what each "Forget …" sends, and that its confirmation says what
/// goes and that Skills stay; the window's time filter.
final class MemoryForgetTests: XCTestCase {
    private var calendar: Calendar = {
        var calendar = Calendar(identifier: .gregorian)
        calendar.timeZone = TimeZone(identifier: "UTC")!
        calendar.firstWeekday = 2  // Monday
        return calendar
    }()

    private var format: MemoryFormat { MemoryFormat(calendar: calendar, locale: Locale(identifier: "en_US")) }

    /// 2026-10-12T13:40:00Z, a Monday.
    private let now = Date(timeIntervalSince1970: 1_791_812_400)
    private let midnight = Date(timeIntervalSince1970: 1_791_763_200)

    private let moment = MemoryMoment(
        id: "ev-9", firstTS: "2026-10-12T13:00:00Z", lastTS: "2026-10-12T13:12:00Z", frames: 34,
        app: "Safari", windowTitle: "Invoice 4411", urlHost: "billing.acme.com",
        description: "An invoice.", score: 0.8, thumbID: "ev-9")

    func testSelectors() {
        XCTAssertEqual(MemoryForgetRequest.lastHour.selector(now: now, calendar: calendar),
                       .range(since: now.addingTimeInterval(-3600), until: now, app: nil))
        XCTAssertEqual(MemoryForgetRequest.today.selector(now: now, calendar: calendar),
                       .range(since: midnight, until: now, app: nil))
        XCTAssertEqual(MemoryForgetRequest.moment(moment).selector(now: now), .moment(id: "ev-9"))
        XCTAssertEqual(MemoryForgetRequest.everything.selector(now: now), .all)
    }

    func testEveryConfirmationSaysWhatGoesAndThatSkillsStay() {
        let requests: [MemoryForgetRequest] = [.lastHour, .today, .moment(moment), .everything]
        for request in requests {
            let text = request.confirmation(now: now, format: format)
            XCTAssertTrue(text.message.hasSuffix("Your approved Skills stay."), text.message)
            XCTAssertTrue(text.message.contains("screenshots"), text.message)
            XCTAssertTrue(text.title.hasSuffix("?"))
        }
        let hour = MemoryForgetRequest.lastHour.confirmation(now: now, format: format)
        XCTAssertEqual(hour.title, "Forget the last hour?")
        XCTAssertTrue(hour.message.contains("since 12:40\u{202F}PM"), hour.message)
        XCTAssertEqual(hour.button, "Forget Last Hour")
        let one = MemoryForgetRequest.moment(moment).confirmation(now: now, format: format)
        XCTAssertTrue(one.message.hasPrefix("This deletes \u{201C}Safari \u{2014} Invoice 4411\u{201D} (1:00\u{202F}PM):"), one.message)
        XCTAssertTrue(MemoryForgetRequest.everything.confirmation(now: now, format: format).message
            .contains("cannot be undone"))
    }

    func testTheSummaryCountsMoments() {
        func counts(_ n: Int) -> MemoryForgetCounts {
            MemoryForgetCounts(events: 5, memoryMoments: n, thumbnails: n, screenshots: 0, vectors: n, dailySummaries: 0)
        }
        XCTAssertEqual(MemoryForgetRequest.summary(counts(0)), "There was nothing to forget.")
        XCTAssertEqual(MemoryForgetRequest.summary(counts(1)), "Forgot 1 moment.")
        XCTAssertEqual(MemoryForgetRequest.summary(counts(12)), "Forgot 12 moments.")
    }

    func testTheTimeFilter() {
        XCTAssertNil(MemoryRange.anyTime.since(now: now, calendar: calendar))
        XCTAssertEqual(MemoryRange.today.since(now: now, calendar: calendar), midnight)
        // Monday is the week's first day here: this week starts today.
        XCTAssertEqual(MemoryRange.thisWeek.since(now: now, calendar: calendar), midnight)
        let thursday = now.addingTimeInterval(3 * 86400)
        XCTAssertEqual(MemoryRange.thisWeek.since(now: thursday, calendar: calendar), midnight)
    }

    func testRetentionLabels() {
        XCTAssertEqual(MemorySettingsView.retentionLabel(30), "1 month")
        XCTAssertEqual(MemorySettingsView.retentionLabel(365), "1 year")
        XCTAssertEqual(MemorySettingsView.retentionLabel(3650), "10 years")
        XCTAssertEqual(MemorySettingsView.retentionLabel(400), "400 days")
    }
}
