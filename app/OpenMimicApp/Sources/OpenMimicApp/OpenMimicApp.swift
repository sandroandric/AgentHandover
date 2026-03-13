import SwiftUI

@main
struct OpenMimicApp: App {
    @NSApplicationDelegateAdaptor(AppDelegate.self) var delegate
    @StateObject private var appState = AppState()
    @AppStorage("hasCompletedOnboarding") private var hasCompletedOnboarding = false

    var body: some Scene {
        MenuBarExtra {
            MenuBarView()
                .environmentObject(appState)
                .environmentObject(delegate)
                .onAppear {
                    // When user first clicks the menu bar icon, auto-open
                    // onboarding if not completed.  `.onAppear` on MenuBarExtra
                    // content fires on first click, not at app launch.  The
                    // AppDelegate handles the true first-launch case below.
                    if !hasCompletedOnboarding && !delegate.hasTriggeredOnboarding {
                        delegate.hasTriggeredOnboarding = true
                        delegate.pendingOnboarding = true
                    }
                }
        } label: {
            Label {
                Text("OpenMimic")
            } icon: {
                Image(systemName: appState.menuBarIcon)
                    .symbolRenderingMode(.palette)
            }
        }
        .menuBarExtraStyle(.window)

        // Onboarding window (shown on first launch or when permissions missing)
        Window("OpenMimic Setup", id: "onboarding") {
            OnboardingView(onComplete: {
                hasCompletedOnboarding = true
            })
                .environmentObject(appState)
                .frame(width: 520, height: 520)
        }
        .windowResizability(.contentSize)

        // Workflow inbox window
        Window("Workflows", id: "workflows") {
            WorkflowInboxView()
        }
        .defaultSize(width: 700, height: 550)
        .windowResizability(.contentMinSize)

        // Daily digest window
        Window("Daily Digest", id: "daily-digest") {
            DailyDigestView()
        }
        .defaultSize(width: 560, height: 600)
        .windowResizability(.contentMinSize)

        // Micro-review window
        Window("Review Queue", id: "micro-review") {
            MicroReviewView()
        }
        .defaultSize(width: 560, height: 600)
        .windowResizability(.contentMinSize)
    }
}

/// App delegate for handling first-launch auto-open of onboarding.
///
/// SwiftUI `MenuBarExtra` doesn't provide a hook at app launch to open
/// secondary windows.  The delegate bridges this gap by signalling the
/// MenuBarView to open the onboarding window on its first appearance.
final class AppDelegate: NSObject, NSApplicationDelegate, ObservableObject {
    @Published var pendingOnboarding = false
    var hasTriggeredOnboarding = false

    func applicationDidFinishLaunching(_ notification: Notification) {
        if !UserDefaults.standard.bool(forKey: "hasCompletedOnboarding") {
            // Signal that onboarding should open.  The actual `openWindow`
            // call happens in MenuBarView (which owns @Environment(\.openWindow)).
            pendingOnboarding = true
            hasTriggeredOnboarding = true

            // Activate the app so the menu bar popover auto-opens, which
            // in turn triggers MenuBarView's `.onAppear` → onboarding.
            DispatchQueue.main.asyncAfter(deadline: .now() + 0.5) {
                NSApp.activate(ignoringOtherApps: true)
                // Simulate a click on the menu bar extra to open the popover.
                // This is the only reliable way to trigger `.onAppear` on the
                // MenuBarExtra content at launch.
                if let button = NSApp.windows
                    .compactMap({ $0.contentView?.subviews.first as? NSStatusBarButton })
                    .first {
                    button.performClick(nil)
                }
            }
        }
    }
}
