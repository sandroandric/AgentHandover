import SwiftUI

@main
struct AgentHandoverApp: App {
    @NSApplicationDelegateAdaptor(AppDelegate.self) var delegate
    @StateObject private var appState = AppState()
    @AppStorage("hasCompletedOnboarding") private var hasCompletedOnboarding = false

    var body: some Scene {
        MenuBarExtra {
            MenuBarView()
                .environmentObject(appState)
                .environmentObject(delegate)
                .onAppear {
                    if !hasCompletedOnboarding && !delegate.hasTriggeredOnboarding {
                        delegate.hasTriggeredOnboarding = true
                        delegate.showOnboarding(appState: appState)
                    }
                }
        } label: {
            Image(systemName: appState.menuBarIcon)
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
                .environmentObject(appState)
        }
        .defaultSize(width: 560, height: 620)
        .windowResizability(.contentMinSize)
    }
}

/// App delegate — handles first-launch onboarding window directly via NSWindow.
///
/// SwiftUI `Window` scenes are lazy and can't be opened from AppDelegate.
/// Instead, we create the onboarding window ourselves using NSHostingController,
/// which guarantees it appears immediately on first launch.
final class AppDelegate: NSObject, NSApplicationDelegate, ObservableObject {
    @Published var pendingOnboarding = false
    var hasTriggeredOnboarding = false
    private var onboardingWindow: NSWindow?

    func applicationDidFinishLaunching(_ notification: Notification) {
        if !UserDefaults.standard.bool(forKey: "hasCompletedOnboarding") {
            hasTriggeredOnboarding = true

            // Show in dock during onboarding
            NSApp.setActivationPolicy(.regular)

            // Create the onboarding window directly — don't wait for SwiftUI
            DispatchQueue.main.asyncAfter(deadline: .now() + 0.3) {
                Task { @MainActor [weak self] in
                    self?.showOnboarding(appState: nil)
                }
            }
        }
    }

    @MainActor
    func showOnboarding(appState: AppState?) {
        // Don't create duplicates
        if onboardingWindow != nil { return }

        let state = appState ?? AppState()
        let onboardingView = OnboardingView(onComplete: { [weak self] in
            UserDefaults.standard.set(true, forKey: "hasCompletedOnboarding")
            self?.onboardingWindow?.close()
            self?.onboardingWindow = nil
            self?.hideFromDock()
        })
        .environmentObject(state)
        .frame(width: 620, height: 720)

        let hostingController = NSHostingController(rootView: onboardingView)
        let window = NSWindow(contentViewController: hostingController)
        window.title = "AgentHandover Setup"
        window.styleMask = [.titled, .closable, .miniaturizable]
        window.setContentSize(NSSize(width: 620, height: 720))
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
}
