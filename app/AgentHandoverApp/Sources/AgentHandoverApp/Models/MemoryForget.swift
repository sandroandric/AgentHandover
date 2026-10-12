import Foundation

/// A "Forget …" the user asked for: what is sent, and the confirmation
/// that says exactly what is deleted (and that Skills stay).
enum MemoryForgetRequest: Equatable {
    case lastHour
    case today
    case moment(MemoryMoment)
    case everything

    func selector(now: Date, calendar: Calendar = .current) -> MemoryForgetSelector {
        switch self {
        case .lastHour: return .range(since: now.addingTimeInterval(-3600), until: now, app: nil)
        case .today: return .range(since: calendar.startOfDay(for: now), until: now, app: nil)
        case .moment(let moment): return .moment(id: moment.id)
        case .everything: return .all
        }
    }

    struct Confirmation: Equatable {
        let title: String
        let message: String
        let button: String
    }

    private static let whatGoes = "the screenshots, descriptions, thumbnails and search entries AgentHandover kept"
    private static let skillsStay = "Your approved Skills stay."

    func confirmation(now: Date, format: MemoryFormat = MemoryFormat()) -> Confirmation {
        switch self {
        case .lastHour:
            let since = format.pointInTime(now.addingTimeInterval(-3600), now: now)
            return Confirmation(
                title: "Forget the last hour?",
                message: "This deletes \(Self.whatGoes) since \(since), and the activity recorded in that time. "
                    + "A moment that started earlier is forgotten whole. \(Self.skillsStay)",
                button: "Forget Last Hour")
        case .today:
            return Confirmation(
                title: "Forget today?",
                message: "This deletes \(Self.whatGoes) since midnight, and the activity recorded today. "
                    + "A moment that started before midnight is forgotten whole. \(Self.skillsStay)",
                button: "Forget Today")
        case .moment(let moment):
            let what = [moment.app, moment.windowTitle].compactMap { $0 }.filter { !$0.isEmpty }
                .joined(separator: " \u{2014} ")
            let when = moment.firstDate.map { " (\(format.pointInTime($0, now: now)))" } ?? ""
            return Confirmation(
                title: "Forget this moment?",
                message: "This deletes \(what.isEmpty ? "this moment" : "\u{201C}\(what)\u{201D}")\(when): "
                    + "its screenshots, description, thumbnail and search entry. \(Self.skillsStay)",
                button: "Forget Moment")
        case .everything:
            return Confirmation(
                title: "Forget everything in Memory?",
                message: "This deletes every moment Memory kept on this Mac: \(Self.whatGoes), "
                    + "and the recorded activity behind them. It cannot be undone. \(Self.skillsStay)",
                button: "Forget Everything")
        }
    }

    /// "Forgot 3 moments." once the worker answered.
    static func summary(_ counts: MemoryForgetCounts) -> String {
        switch counts.memoryMoments {
        case 0: return "There was nothing to forget."
        case 1: return "Forgot 1 moment."
        default: return "Forgot \(counts.memoryMoments) moments."
        }
    }
}
