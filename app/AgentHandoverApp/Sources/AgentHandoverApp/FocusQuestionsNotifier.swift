import AppKit
@preconcurrency import UserNotifications

/// What the "questions are ready" notification says, and when it is sent:
/// once per focus session.  Pure, so the tests pin it.
enum FocusQuestionsNotice {
    static let lastNotifiedKey = "focusQuestionsLastNotifiedSession"

    /// "3 questions about 'Look up the IANA reserved domains page'": the
    /// session title as the user typed it.
    static func title(for file: FocusQuestionsFile) -> String {
        let count = file.questions.count
        let noun = count == 1 ? "question" : "questions"
        return "\(count) \(noun) about '\(FocusQAView.displayTitle(for: file))'"
    }

    static let body = "Answer them so your agent can run this workflow without guessing."

    /// True the first time a session's questions are seen, false for the
    /// same session again (a status refresh, an app relaunch).
    static func isNew(_ file: FocusQuestionsFile, lastNotified: String?) -> Bool {
        !file.session_id.isEmpty && file.session_id != lastNotified && !file.questions.isEmpty
    }
}

/// Tells the user their focus recording's questions are ready.
///
/// With notifications allowed (the user turns them on from the Q&A window,
/// which says what they are for, not at launch), one notification per
/// session; clicking it opens the Q&A window.  Without them (never asked or
/// denied, as the user's settings say), the Q&A window opens by itself, as
/// before, once per session.
@MainActor
final class FocusQuestionsNotifier: NSObject, UNUserNotificationCenterDelegate {
    static let shared = FocusQuestionsNotifier()

    /// Opens the Q&A window (set by the app delegate).
    var openQuestions: () -> Void = {}
    /// Opens the Review Queue (set by the app delegate).
    var openReviewQueue: () -> Void = {}

    private let defaults = UserDefaults.standard

    /// Called on every read of a pending focus-questions.json.
    func questionsReady(_ file: FocusQuestionsFile) {
        guard FocusQuestionsNotice.isNew(
            file, lastNotified: defaults.string(forKey: FocusQuestionsNotice.lastNotifiedKey)
        ) else { return }
        defaults.set(file.session_id, forKey: FocusQuestionsNotice.lastNotifiedKey)

        UNUserNotificationCenter.current().getNotificationSettings { settings in
            let allowed = [.authorized, .provisional].contains(settings.authorizationStatus)
                && settings.alertSetting != .disabled
            Task { @MainActor in
                if allowed {
                    self.post(file)
                } else {
                    self.openQuestions()
                }
            }
        }
    }

    /// Called when micro-questions.json changed: one notification for new
    /// review-card questions, when the user allowed notifications (the
    /// cards wait in the Review Queue and the day's digest either way).
    func microQuestionsReady(_ questions: [MicroQuestion]) {
        let last = defaults.string(forKey: MicroQuestionsNotice.lastNotifiedKey)
        guard MicroQuestionsNotice.isNew(questions, lastNotified: last) else { return }
        defaults.set(MicroQuestionsNotice.key(for: questions), forKey: MicroQuestionsNotice.lastNotifiedKey)

        UNUserNotificationCenter.current().getNotificationSettings { settings in
            let allowed = [.authorized, .provisional].contains(settings.authorizationStatus)
                && settings.alertSetting != .disabled
            guard allowed else { return }
            Task { @MainActor in
                let content = UNMutableNotificationContent()
                content.title = MicroQuestionsNotice.title(for: questions)
                content.body = MicroQuestionsNotice.body
                content.sound = .default
                let request = UNNotificationRequest(
                    identifier: "\(Self.microIdentifierPrefix)\(UUID().uuidString)",
                    content: content, trigger: nil
                )
                try? await UNUserNotificationCenter.current().add(request)
            }
        }
    }

    nonisolated static let microIdentifierPrefix = "micro-questions-"

    /// The Q&A window's "Notify me" button: the one place the app asks.
    func requestAuthorization(completion: @escaping @MainActor (Bool) -> Void) {
        UNUserNotificationCenter.current().requestAuthorization(options: [.alert, .sound]) { granted, _ in
            Task { @MainActor in completion(granted) }
        }
    }

    /// Whether the Q&A window should offer notifications (never asked yet).
    func canAsk(completion: @escaping @MainActor (Bool) -> Void) {
        UNUserNotificationCenter.current().getNotificationSettings { settings in
            let notDetermined = settings.authorizationStatus == .notDetermined
            Task { @MainActor in completion(notDetermined) }
        }
    }

    func start() {
        UNUserNotificationCenter.current().delegate = self
    }

    private func post(_ file: FocusQuestionsFile) {
        let content = UNMutableNotificationContent()
        content.title = FocusQuestionsNotice.title(for: file)
        content.body = FocusQuestionsNotice.body
        content.sound = .default
        let request = UNNotificationRequest(
            identifier: "focus-questions-\(file.session_id)", content: content, trigger: nil
        )
        UNUserNotificationCenter.current().add(request)
    }

    // A click on the notification opens the questions.
    nonisolated func userNotificationCenter(
        _ center: UNUserNotificationCenter,
        didReceive response: UNNotificationResponse,
        withCompletionHandler completionHandler: @escaping () -> Void
    ) {
        let identifier = response.notification.request.identifier
        Task { @MainActor in
            if identifier.hasPrefix(Self.microIdentifierPrefix) {
                self.openReviewQueue()
            } else {
                self.openQuestions()
            }
            completionHandler()
        }
    }

    // Shown even while the app is frontmost (its menu is open).
    nonisolated func userNotificationCenter(
        _ center: UNUserNotificationCenter,
        willPresent notification: UNNotification,
        withCompletionHandler completionHandler: @escaping (UNNotificationPresentationOptions) -> Void
    ) {
        completionHandler([.banner, .sound])
    }
}
