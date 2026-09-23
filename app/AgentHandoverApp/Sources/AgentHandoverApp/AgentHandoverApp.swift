import SwiftUI
import Sparkle
import ServiceManagement

@main
struct AgentHandoverApp: App {
    @NSApplicationDelegateAdaptor(AppDelegate.self) var delegate
    @AppStorage("hasCompletedOnboarding") private var hasCompletedOnboarding = false

    var body: some Scene {
        MenuBarExtra {
            MenuBarView()
                .environmentObject(delegate.sharedAppState)
                .environmentObject(delegate.observationToggle)
                .environmentObject(delegate)
                .onAppear {
                    if !delegate.hasTriggeredOnboarding
                        && (!delegate.sharedAppState.accessibilityGranted
                            || !delegate.sharedAppState.screenRecordingGranted)
                    {
                        delegate.hasTriggeredOnboarding = true
                        delegate.showOnboarding()
                    }
                }
        } label: {
            Image(systemName: delegate.sharedAppState.menuBarIcon)
        }
        .menuBarExtraStyle(.window)

        // These Window scenes are opened via openWindow(id:) from MenuBarView
        Window("Workflows", id: "workflows") {
            WorkflowInboxView()
        }
        .defaultSize(width: 900, height: 620)
        .windowResizability(.contentMinSize)

        Window("Daily Digest", id: "daily-digest") {
            DailyDigestView()
        }
        .defaultSize(width: 640, height: 680)
        .windowResizability(.contentMinSize)

        Window("Review Queue", id: "micro-review") {
            MicroReviewView()
        }
        .defaultSize(width: 640, height: 680)
        .windowResizability(.contentMinSize)

        Window("FAQ", id: "faq") {
            FAQView()
        }
        .defaultSize(width: 560, height: 700)
        .windowResizability(.contentMinSize)
    }
}

/// App delegate — handles first-launch onboarding window directly via NSWindow.
///
/// Owns the single shared `AppState` so both onboarding (opened from
/// applicationDidFinishLaunching) and the menu bar use the same instance.
final class AppDelegate: NSObject, NSApplicationDelegate, NSWindowDelegate, ObservableObject {
    @Published var pendingOnboarding = false
    var hasTriggeredOnboarding = false
    private var onboardingWindow: NSWindow?
    private var focusQAWindow: NSWindow?

    /// Single shared AppState used by onboarding, menu bar, and all windows.
    @MainActor let sharedAppState = AppState()

    /// Starts/stops services off the main thread; shared so the menu shows
    /// "Starting…" for the launch-time start too.
    @MainActor let observationToggle = ObservationToggle()

    /// Sparkle auto-updater — starts on launch per SUEnableAutomaticChecks in Info.plist.
    let updaterController = SPUStandardUpdaterController(
        startingUpdater: true,
        updaterDelegate: nil,
        userDriverDelegate: nil
    )

    /// Screenshot capture server — serves pixels to the daemon via Unix socket.
    private let captureServer = ScreenCaptureServer()
    /// Observation server — serves AX/window/display metadata to the daemon.
    private let observationServer = ObservationServer()
    private var runtimeBridgesStarted = false

    /// The setup step to open at, or nil when setup is complete (see
    /// `OnboardingProgress`).
    private func onboardingStartStep() -> Int? {
        OnboardingProgress.startStep(
            defaults: .standard,
            accessibilityGranted: PermissionChecker.isAccessibilityGranted(),
            screenRecordingGranted: PermissionChecker.isScreenRecordingGranted()
        )
    }

    func applicationDidFinishLaunching(_ notification: Notification) {
        // Always start capture + observation servers so the daemon can
        // request screenshots at any point — during onboarding or after.
        startRuntimeBridges()
        // Focus questions: a notification opens them (FocusQuestionsNotifier).
        MainActor.assumeIsolated {
            FocusQuestionsNotifier.shared.openQuestions = { [weak self] in self?.showFocusQA() }
            FocusQuestionsNotifier.shared.start()
        }
        // [privacy] read_app_state: macOS asks per app, from here, the
        // first time the user switches to one (see AppStateReading).
        AppStateConsentAsker.shared.start()

        // The worker serves approvals, search and MCP reporting whether or
        // not capture is on, so it is loaded on every launch (a paused
        // relaunch included); only capture follows the pause.
        DispatchQueue.global(qos: .utility).async {
            ServiceController.startWorker()
        }

        if let step = onboardingStartStep() {
            hasTriggeredOnboarding = true

            // Show in dock during onboarding
            NSApp.setActivationPolicy(.regular)

            // Clean up stale helper state on a background thread
            // so it doesn't block the onboarding window.
            DispatchQueue.global(qos: .utility).async {
                ServiceController.prepareForAppOwnedPermissionRequest()
                ServiceController.removeNativeMessagingHostManifest()
            }

            // Show onboarding immediately
            DispatchQueue.main.async { [weak self] in
                self?.showOnboarding(at: step)
            }
        } else {
            // Keep the bundle a normal app principal for TCC purposes, but
            // present it as a menu bar app once onboarding is already done.
            ServiceController.installNativeMessagingHostManifest()
            hideFromDock()

            registerLoginItemOnce()

            // Onboarding done — start services if user hasn't paused.
            // Off the main thread: startAll waits for the daemon.
            let userPaused = UserDefaults.standard.bool(forKey: "observingPaused")
            if !userPaused {
                MainActor.assumeIsolated {
                    sharedAppState.servicesStarting()
                    observationToggle.start { [weak self] _ in
                        self?.sharedAppState.refreshStatus()
                    }
                }
            }
        }
    }

    /// Open AgentHandover at login so a reboot restores capture (the app
    /// spawns the daemon; nothing else would). Registered once, after
    /// onboarding: if the user later removes it in System Settings > General
    /// > Login Items, we respect that instead of re-adding it every launch.
    func registerLoginItemOnce() {
        let key = "loginItemRegistered"
        guard !UserDefaults.standard.bool(forKey: key) else { return }
        do {
            if SMAppService.mainApp.status != .enabled {
                try SMAppService.mainApp.register()
            }
            UserDefaults.standard.set(true, forKey: key)
        } catch {
            NSLog("AgentHandover: could not register as a login item (%@)", error.localizedDescription)
        }
    }

    /// Open setup at `step`, or where `OnboardingProgress` says it should
    /// open (the welcome screen when there is nothing left to set up and
    /// the user asked for it anyway).
    @MainActor
    func showOnboarding(at step: Int? = nil) {
        // Don't create duplicates; bring a closed or hidden one back.
        if let window = onboardingWindow {
            NSApp.activate(ignoringOtherApps: true)
            window.makeKeyAndOrderFront(nil)
            return
        }

        let onboardingView = OnboardingView(initialStep: step ?? onboardingStartStep() ?? 0, onComplete: { [weak self] in
            OnboardingProgress.finish(in: .standard)
            // Default to paused — user chooses "Observe Me" or "Record" explicitly
            UserDefaults.standard.set(true, forKey: "observingPaused")
            self?.sharedAppState.userStopped = true
            self?.onboardingWindow?.close()
            self?.onboardingWindow = nil
            self?.hideFromDock()
            self?.registerLoginItemOnce()
            // Install NM manifest on background (no services started yet)
            DispatchQueue.global(qos: .utility).async {
                ServiceController.installNativeMessagingHostManifest()
            }
        })
        .environmentObject(sharedAppState)
        .frame(width: 660, height: 820)

        let hostingController = NSHostingController(rootView: onboardingView)
        let window = NSWindow(contentViewController: hostingController)
        window.title = "AgentHandover Setup"
        window.styleMask = [.titled, .closable, .miniaturizable, .resizable]
        window.setContentSize(NSSize(width: 660, height: 820))
        window.minSize = NSSize(width: 620, height: 720)
        window.center()
        window.isReleasedWhenClosed = false

        self.onboardingWindow = window

        NSApp.activate(ignoringOtherApps: true)
        window.makeKeyAndOrderFront(nil)
        window.orderFrontRegardless()
    }

    func hideFromDock() {
        NSApp.setActivationPolicy(.accessory)
    }

    /// The Q&A window for a focus recording's questions, from the menu,
    /// the notification or the questions appearing.  A menu bar app's
    /// window does not take the keyboard until the app is a regular,
    /// active app: without that, clicking an answer field did nothing.
    @MainActor
    func showFocusQA() {
        NSApp.setActivationPolicy(.regular)
        NSApp.activate(ignoringOtherApps: true)
        if let window = focusQAWindow {
            window.makeKeyAndOrderFront(nil)
            return
        }
        let view = FocusQAView(onDone: { [weak self] in self?.focusQAWindow?.close() })
            .environmentObject(sharedAppState)
        let window = NSWindow(contentViewController: NSHostingController(rootView: view))
        window.title = "Focus Q&A"
        window.styleMask = [.titled, .closable, .miniaturizable, .resizable]
        window.setContentSize(NSSize(width: 560, height: 620))
        window.center()
        window.isReleasedWhenClosed = false
        window.delegate = self
        focusQAWindow = window
        window.makeKeyAndOrderFront(nil)
        window.orderFrontRegardless()
    }

    /// The Q&A window closed: back to a menu bar app unless another window
    /// is open.
    func windowWillClose(_ notification: Notification) {
        guard let window = notification.object as? NSWindow, window === focusQAWindow else { return }
        focusQAWindow = nil
        let others = NSApp.windows.contains {
            $0 !== window && $0.isVisible && $0.styleMask.contains(.titled)
        }
        if !others { hideFromDock() }
    }

    func applicationWillTerminate(_ notification: Notification) {
        stopRuntimeBridges()
    }

    private func startRuntimeBridges() {
        guard !runtimeBridgesStarted else { return }
        captureServer.start()
        observationServer.start()
        runtimeBridgesStarted = true
    }

    /// Quitting stops capture only: the worker (launchd) keeps serving
    /// approvals, search and MCP reporting while the app is not running.
    private func stopRuntimeBridges() {
        ServiceController.stopCapture()
        captureServer.stop()
        observationServer.stop()
        runtimeBridgesStarted = false
    }
}
