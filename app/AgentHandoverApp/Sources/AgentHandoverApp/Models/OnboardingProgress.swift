import Foundation

/// Where the setup window opens, what a launch does when setup is not
/// needed, and what finishing setup does to the user's observing choice.
///
/// Granting Screen Recording makes macOS offer "Quit & Reopen"; after the
/// relaunch a first install's setup comes back where it was. An existing
/// install (`hasCompletedOnboarding` already true, an upgrade included)
/// opens setup only for a grant it lost (the installer resets grants when
/// the app's code signature changes), at that grant's step, and once the
/// grants are back it resumes what the user had: observing or paused.
enum OnboardingProgress {
    static let stepKey = "onboardingStep"
    static let completedKey = "hasCompletedOnboarding"
    /// The user's choice in the menu (Observe Me / Stop Observing).
    static let pausedKey = "observingPaused"

    static let accessibilityStep = 4
    static let screenRecordingStep = 5

    /// The first permission step whose grant is missing, or nil.
    static func missingPermissionStep(accessibilityGranted: Bool, screenRecordingGranted: Bool) -> Int? {
        if !accessibilityGranted { return accessibilityStep }
        if !screenRecordingGranted { return screenRecordingStep }
        return nil
    }

    /// The step to open setup at, or nil when there is nothing to set up.
    static func startStep(
        savedStep: Int?,
        completed: Bool,
        accessibilityGranted: Bool,
        screenRecordingGranted: Bool
    ) -> Int? {
        if completed {
            // Only a missing grant brings setup back; a step saved by an
            // earlier, interrupted setup is not resumed.
            return missingPermissionStep(
                accessibilityGranted: accessibilityGranted,
                screenRecordingGranted: screenRecordingGranted
            )
        }
        let step = savedStep ?? 0
        // A first install relaunched by Quit & Reopen after granting
        // Accessibility goes on to Screen Recording. The Screen Recording
        // step itself is kept even when granted: its Next takes the test
        // capture that explains macOS's second prompt.
        if step == accessibilityStep && accessibilityGranted {
            return screenRecordingStep
        }
        return step
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

    /// What a launch does.
    enum LaunchAction: Equatable {
        case setup(step: Int)
        /// Setup is done and the user was observing: start capture.
        case observe
        /// Setup is done and the user had paused: start nothing.
        case stayPaused
    }

    static func launchAction(
        defaults: UserDefaults,
        accessibilityGranted: Bool,
        screenRecordingGranted: Bool
    ) -> LaunchAction {
        if let step = startStep(
            defaults: defaults,
            accessibilityGranted: accessibilityGranted,
            screenRecordingGranted: screenRecordingGranted
        ) {
            return .setup(step: step)
        }
        return defaults.bool(forKey: pausedKey) ? .stayPaused : .observe
    }

    /// What the app does when both grants come back while it runs (the
    /// user switched them on in System Settings).
    enum RegrantAction: Equatable {
        /// Setup was reopened for an existing install's lost grants: close
        /// it, nothing else is missing.
        case finishSetup
        /// No setup open, the user was observing: start capture.
        case observe
        case nothing
    }

    static func regrantAction(
        defaults: UserDefaults,
        setupOpen: Bool,
        setupForExistingInstall: Bool,
        observing: Bool
    ) -> RegrantAction {
        if setupOpen {
            // A first install carries on through the rest of setup.
            return setupForExistingInstall ? .finishSetup : .nothing
        }
        guard defaults.bool(forKey: completedKey),
              !defaults.bool(forKey: pausedKey),
              !observing else { return .nothing }
        return .observe
    }

    static func savedStep(_ defaults: UserDefaults) -> Int? {
        defaults.object(forKey: stepKey) as? Int
    }

    static func save(step: Int, in defaults: UserDefaults) {
        defaults.set(step, forKey: stepKey)
    }

    /// Setup finished: nothing to resume. A first install starts paused
    /// (the user chooses "Observe Me" or "Record" explicitly); an existing
    /// install keeps the user's choice. Returns whether observation is
    /// paused now.
    @discardableResult
    static func finish(in defaults: UserDefaults) -> Bool {
        if !defaults.bool(forKey: completedKey) {
            defaults.set(true, forKey: pausedKey)
        }
        defaults.set(true, forKey: completedKey)
        defaults.removeObject(forKey: stepKey)
        return defaults.bool(forKey: pausedKey)
    }
}
