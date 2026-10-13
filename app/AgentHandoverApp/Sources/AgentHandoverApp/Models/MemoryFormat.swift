import Foundation

/// Memory's state as the menu shows it.
enum MemoryMenuState: Equatable {
    case recording
    /// On and capturing, but the worker cannot index yet (the Mac is short
    /// of memory, Skills are being written, Ollama is down...).
    case waiting
    /// Paused until a time, or until resumed (nil).
    case paused(until: Date?)
    case off
    /// The worker did not answer.
    case unavailable

    init(status: MemoryStatus?) {
        guard let status else { self = .unavailable; return }
        if !status.enabled {
            self = .off
        } else if status.paused {
            self = .paused(until: status.pausedUntilDate)
        } else if status.waiting != nil {
            self = .waiting
        } else {
            self = .recording
        }
    }

    /// Recording or waiting: pause and forget apply.
    var isOn: Bool { self == .recording || self == .waiting }

    var isPaused: Bool {
        if case .paused = self { return true }
        return false
    }

    /// "Recording", "Paused until 3:40 PM", "Paused until tomorrow, 12:00 AM",
    /// "Paused until you resume", "Off", "Not available".
    func label(now: Date, format: MemoryFormat = MemoryFormat()) -> String {
        switch self {
        case .recording: return "Recording"
        case .waiting: return "Recording; indexing waits"
        case .off: return "Off"
        case .unavailable: return "Not available"
        case .paused(nil): return "Paused until you resume"
        case .paused(let until?): return "Paused until \(format.pointInTime(until, now: now))"
        }
    }
}

/// The menu's "Pause Memory" choices.
enum MemoryPauseOption: CaseIterable, Identifiable {
    case oneHour
    case untilTomorrow
    case untilResumed

    var id: Self { self }

    var title: String {
        switch self {
        case .oneHour: return "For 1 Hour"
        case .untilTomorrow: return "Until Tomorrow"
        case .untilResumed: return "Until I Resume"
        }
    }

    /// The `minutes` of `POST /memory/pause`: nil means until resumed.
    /// "Until tomorrow" runs to the start of the next local day.
    func minutes(now: Date, calendar: Calendar = .current) -> Int? {
        switch self {
        case .oneHour: return 60
        case .untilResumed: return nil
        case .untilTomorrow:
            let tomorrow = calendar.startOfDay(for: calendar.date(byAdding: .day, value: 1, to: now)!)
            return max(1, Int((tomorrow.timeIntervalSince(now) / 60).rounded(.up)))
        }
    }
}

/// The Commonplace window's time filter (Memory and days).
enum MemoryRange: String, CaseIterable, Identifiable {
    case anyTime = "Any Time"
    case today = "Today"
    case thisWeek = "This Week"

    var id: Self { self }

    /// The search's `since`: nil for any time, the start of the local day,
    /// or the start of the calendar's week.
    func since(now: Date, calendar: Calendar = .current) -> Date? {
        switch self {
        case .anyTime: return nil
        case .today: return calendar.startOfDay(for: now)
        case .thisWeek: return calendar.dateInterval(of: .weekOfYear, for: now)?.start
        }
    }
}

/// How Memory's times, spans and sizes read. The calendar and locale are
/// injected so tests read the same words on every Mac.
struct MemoryFormat {
    var calendar: Calendar = .current
    var locale: Locale = .current

    private func formatter(_ template: String) -> DateFormatter {
        let formatter = DateFormatter()
        formatter.calendar = calendar
        formatter.timeZone = calendar.timeZone
        formatter.locale = locale
        formatter.setLocalizedDateFormatFromTemplate(template)
        return formatter
    }

    /// "3:40 PM" today, "tomorrow, 9:00 AM", else "Wed, Oct 14 at 3:40 PM".
    func pointInTime(_ date: Date, now: Date) -> String {
        let time = formatter("jmm").string(from: date)
        if calendar.isDate(date, inSameDayAs: now) { return time }
        if let tomorrow = calendar.date(byAdding: .day, value: 1, to: now),
           calendar.isDate(date, inSameDayAs: tomorrow) {
            return "tomorrow, \(time)"
        }
        return formatter("EEEMMMdjmm").string(from: date)
    }

    /// "just now", "12 min ago", "3 h ago", "yesterday"; nil beyond (the
    /// absolute time says it).
    func relative(_ date: Date, now: Date) -> String? {
        let seconds = now.timeIntervalSince(date)
        if seconds < 60 { return "just now" }
        if seconds < 3600 { return "\(Int(seconds / 60)) min ago" }
        if calendar.isDate(date, inSameDayAs: now) { return "\(Int(seconds / 3600)) h ago" }
        if let yesterday = calendar.date(byAdding: .day, value: -1, to: now),
           calendar.isDate(date, inSameDayAs: yesterday) {
            return "yesterday"
        }
        return nil
    }

    /// "Mon, Oct 12, 2026 at 3:40 PM".
    func absolute(_ date: Date) -> String {
        formatter("EEEyMMMdjmm").string(from: date)
    }

    /// "12 min ago · Mon, Oct 12, 2026, 3:40 PM" for a moment's start.
    func when(_ moment: MemoryMoment, now: Date) -> String {
        guard let first = moment.firstDate else { return moment.firstTS }
        let absolute = absolute(first)
        return relative(first, now: now).map { "\($0) \u{00B7} \(absolute)" } ?? absolute
    }

    /// "12 min · 34 screens" for a moment of more than one screen, nil
    /// for one.
    func span(_ moment: MemoryMoment) -> String? {
        guard moment.frames > 1 else { return nil }
        let screens = "\(moment.frames) screens"
        guard let first = moment.firstDate, let last = moment.lastDate else { return screens }
        return "\(Self.duration(last.timeIntervalSince(first))) \u{00B7} \(screens)"
    }

    /// "under a minute", "12 min", "1 h 5 min", "3 h".
    static func duration(_ seconds: TimeInterval) -> String {
        let minutes = Int(max(0, seconds) / 60)
        if minutes < 1 { return "under a minute" }
        if minutes < 60 { return "\(minutes) min" }
        let rest = minutes % 60
        return rest == 0 ? "\(minutes / 60) h" : "\(minutes / 60) h \(rest) min"
    }

    /// Decimal sizes: "820 KB", "4.2 MB", "38 MB", "1.3 GB".
    static func size(_ bytes: Int64) -> String {
        let value = Double(bytes)
        if value < 1_000_000 { return "\(Int((value / 1_000).rounded())) KB" }
        if value < 1_000_000_000 {
            let mb = value / 1_000_000
            return mb < 10 ? String(format: "%.1f MB", mb) : "\(Int(mb.rounded())) MB"
        }
        return String(format: "%.1f GB", value / 1_000_000_000)
    }

    /// "1,204 moments · 38 MB · since Oct 3" (the year when not this one);
    /// "No moments yet" for none.
    func statusLine(_ status: MemoryStatus, now: Date) -> String {
        guard status.moments > 0 else { return "No moments yet" }
        let count = NumberFormatter.localizedString(from: NSNumber(value: status.moments), number: .decimal)
        var parts = ["\(count) \(status.moments == 1 ? "moment" : "moments")", Self.size(status.diskBytes)]
        if let oldest = status.oldestDate {
            let sameYear = calendar.component(.year, from: oldest) == calendar.component(.year, from: now)
            parts.append("since " + formatter(sameYear ? "MMMd" : "yMMMd").string(from: oldest))
        }
        return parts.joined(separator: " \u{00B7} ")
    }

    /// A Commonplace day: "Monday, Oct 12, 2026"; its date as written when
    /// it does not parse.
    func dayTitle(_ day: CommonplaceDay) -> String {
        guard let date = day.day(calendar: calendar) else { return day.date }
        return formatter("EEEEyMMMd").string(from: date)
    }
}

/// Why Memory cannot search, and what fixes it: the window's empty and
/// error states.
enum MemoryProblem: Equatable {
    case workerNotRunning
    case off
    case ollamaNotRunning
    case ollamaTooOld(have: String, need: String)
    case modelMissing(String)
    /// The worker's own reason (`model_unavailable`'s detail, or status
    /// `waiting`).
    case waiting(String)

    /// Why a search answered `model_unavailable`: from the Ollama that
    /// answers (`ollamaVersion` nil when none does), its pulled models
    /// (nil when unknown) and the table's Memory model and minimum.
    static func diagnose(
        detail: String, ollamaVersion: String?, installed: Set<String>?,
        memoryModel: String?, minimum: String?
    ) -> MemoryProblem {
        guard let version = ollamaVersion else { return .ollamaNotRunning }
        if let minimum, !OllamaModels.isVersion(version, atLeast: minimum) {
            return .ollamaTooOld(have: version, need: minimum)
        }
        if let memoryModel, let installed, !OllamaModels.missing([memoryModel], installed: installed).isEmpty {
            return .modelMissing(memoryModel)
        }
        return .waiting(detail)
    }

    /// The one plain sentence the window shows.
    var message: String {
        switch self {
        case .workerNotRunning:
            return "AgentHandover's worker is not running, so Memory cannot be searched."
        case .off:
            return "Memory is off. Turn it on to keep a private, searchable record of what you see."
        case .ollamaNotRunning:
            return "Ollama is not running. Open Ollama, then search again."
        case .ollamaTooOld(let have, let need):
            return MemoryDecisions.ollamaTooOldText(version: have, minimum: need)
        case .modelMissing(let model):
            return "Memory needs the \(model) model, which is not downloaded yet."
        case .waiting(let why):
            return why.isEmpty ? "Memory is waiting for Ollama." : "Memory is waiting: \(why)"
        }
    }
}
