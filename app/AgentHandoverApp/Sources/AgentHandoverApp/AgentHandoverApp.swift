import SwiftUI

@main
struct AgentHandoverApp: App {
    @NSApplicationDelegateAdaptor(AppDelegate.self) var delegate
    @AppStorage("hasCompletedOnboarding") private var hasCompletedOnboarding = false

    var body: some Scene {
        MenuBarExtra(isInserted: $hasCompletedOnboarding) {
            MenuBarView()
                .environmentObject(delegate.sharedAppState)
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

        Window("Focus Q&A", id: "focus-qa") {
            FocusQAView()
                .environmentObject(delegate.sharedAppState)
        }
        .defaultSize(width: 560, height: 620)
        .windowResizability(.contentMinSize)
    }
}

/// App delegate — handles first-launch onboarding window directly via NSWindow.
///
/// Owns the single shared `AppState` so both onboarding (opened from
/// applicationDidFinishLaunching) and the menu bar use the same instance.
final class AppDelegate: NSObject, NSApplicationDelegate, ObservableObject {
    @Published var pendingOnboarding = false
    var hasTriggeredOnboarding = false
    private var onboardingWindow: NSWindow?

    /// Single shared AppState used by onboarding, menu bar, and all windows.
    @MainActor let sharedAppState = AppState()

    /// Screenshot capture server — serves pixels to the daemon via Unix socket.
    private let captureServer = ScreenCaptureServer()
    /// Observation server — serves AX/window/display metadata to the daemon.
    private let observationServer = ObservationServer()
    private var runtimeBridgesStarted = false

    private var onboardingNeeded: Bool {
        !UserDefaults.standard.bool(forKey: "hasCompletedOnboarding")
            || !PermissionChecker.allPermissionsGranted()
    }

    func applicationDidFinishLaunching(_ notification: Notification) {
        if onboardingNeeded {
            hasTriggeredOnboarding = true

            // Keep onboarding isolated from any stale helper runtime so macOS
            // always attributes permission prompts to AgentHandover itself.
            ServiceController.stopAll()
            ServiceController.prepareForAppOwnedPermissionRequest()
            ServiceController.removeNativeMessagingHostManifest()

            // Show in dock during onboarding
            NSApp.setActivationPolicy(.regular)

            // Create the onboarding window directly — don't wait for SwiftUI
            DispatchQueue.main.asyncAfter(deadline: .now() + 0.3) {
                Task { @MainActor [weak self] in
                    self?.showOnboarding()
                }
            }
        } else {
            // Keep the bundle a normal app principal for TCC purposes, but
            // present it as a menu bar app once onboarding is already done.
            ServiceController.installNativeMessagingHostManifest()
            startRuntimeBridges()
            hideFromDock()

            // Onboarding done — start services if user hasn't paused
            let userPaused = UserDefaults.standard.bool(forKey: "observingPaused")
            if !userPaused {
                ServiceController.startAll()
            }
        }
    }

    @MainActor
    func showOnboarding() {
        // Don't create duplicates
        if onboardingWindow != nil { return }

        // Re-entering onboarding should also clear any stale daemon process so
        // the permission steps stay app-owned even after upgrades or crashes.
        ServiceController.stopAll()
        ServiceController.prepareForAppOwnedPermissionRequest()
        ServiceController.removeNativeMessagingHostManifest()

        let onboardingView = OnboardingView(onComplete: { [weak self] in
            UserDefaults.standard.set(true, forKey: "hasCompletedOnboarding")
            ServiceController.installNativeMessagingHostManifest()
            self?.startRuntimeBridges()
            let userPaused = UserDefaults.standard.bool(forKey: "observingPaused")
            if !userPaused {
                ServiceController.startAll()
            }
            self?.onboardingWindow?.close()
            self?.onboardingWindow = nil
            self?.hideFromDock()
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

    func applicationWillTerminate(_ notification: Notification) {
        stopRuntimeBridges()
    }

    private func startRuntimeBridges() {
        guard !runtimeBridgesStarted else { return }
        captureServer.start()
        observationServer.start()
        runtimeBridgesStarted = true
    }

    private func stopRuntimeBridges() {
        ServiceController.stopAll()
        captureServer.stop()
        observationServer.stop()
        runtimeBridgesStarted = false
    }
}
