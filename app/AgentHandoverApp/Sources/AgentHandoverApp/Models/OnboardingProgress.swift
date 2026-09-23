import Foundation

/// Where the setup window opens, and the step it was on when the app went
/// away mid-setup.
///
/// Granting Screen Recording makes macOS offer "Quit & Reopen"; after the
/// relaunch setup has to come back at the step it was on, even for an
/// upgrade where `hasCompletedOnboarding` is already true from the old
/// install. An upgrade that lost a grant (the installer resets grants when
/// the app's code signature changes) opens setup at that permission's step
/// rather than at the welcome screen.
enum OnboardingProgress {
    static let stepKey = "onboardingStep"
    static let completedKey = "hasCompletedOnboarding"

    static let accessibilityStep = 4
    static let screenRecordingStep = 5

    /// The step to open setup at, or nil when there is nothing to set up.
    static func startStep(
        savedStep: Int?,
        completed: Bool,
        accessibilityGranted: Bool,
        screenRecordingGranted: Bool
    ) -> Int? {
        if completed && accessibilityGranted && screenRecordingGranted {
            return nil
        }
        if let savedStep {
            return savedStep
        }
        if !completed {
            return 0
        }
        return accessibilityGranted ? screenRecordingStep : accessibilityStep
    }

    static func startStep(
        defaults: UserDefaults,
        accessibilityGranted: Bool,
        screenRecordingGranted: Bool
    ) -> Int? {
        startStep(
            savedStep: savedStep(defaults),
            completed: defaults.bool(forKey: completedKey),
            accessibilityGranted: accessibilityGranted,
            screenRecordingGranted: screenRecordingGranted
        )
    }

    static func savedStep(_ defaults: UserDefaults) -> Int? {
        defaults.object(forKey: stepKey) as? Int
    }

    static func save(step: Int, in defaults: UserDefaults) {
        defaults.set(step, forKey: stepKey)
    }

    /// Setup finished: nothing to resume.
    static func finish(in defaults: UserDefaults) {
        defaults.set(true, forKey: completedKey)
        defaults.removeObject(forKey: stepKey)
    }
}
