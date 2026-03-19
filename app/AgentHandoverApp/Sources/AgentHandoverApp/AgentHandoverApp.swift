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
                Text("AgentHandover")
            } icon: {
                if let nsImage = NSImage(named: "MenuBarIcon") {
                    Image(nsImage: nsImage)
                } else {
                    Image(systemName: appState.menuBarIcon)
                }
            }
        }
        .menuBarExtraStyle(.window)

        // Onboarding window (shown on first launch or when permissions missing)
        Window("AgentHandover Setup", id: "onboarding") {
            OnboardingView(onComplete: {
                hasCompletedOnboarding = true
            })
                .environmentObject(appState)
                .frame(width: 600, height: 640)
        }
        .windowResizability(.contentSize)

        // Workflow inbox window
        Window("Workflows", id: "workflows") {
            WorkflowInboxView()
        }
        .defaultSize(width: 900, height: 620)
        .windowResizability(.contentMinSize)

        // Daily digest window
        Window("Daily Digest", id: "daily-digest") {
            DailyDigestView()
        }
        .defaultSize(width: 640, height: 680)
        .windowResizability(.contentMinSize)

        // Micro-review window
        Window("Review Queue", id: "micro-review") {
            MicroReviewView()
        }
        .defaultSize(width: 640, height: 680)
        .windowResizability(.contentMinSize)

        // Focus Q&A window (questions from worker after focus recording)
        Window("Focus Q&A", id: "focus-qa") {
            FocusQAView()
                .environmentObject(appState)
        }
        .defaultSize(width: 480, height: 400)
        .windowResizability(.contentSize)
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

            // Activate the app and bring onboarding to front after a short delay
            // to allow SwiftUI to create the window scenes.
            DispatchQueue.main.asyncAfter(deadline: .now() + 0.8) {
                NSApp.activate(ignoringOtherApps: true)

                // Bring the onboarding window to front if it exists
                for window in NSApp.windows {
                    if window.title == "AgentHandover Setup" {
                        window.makeKeyAndOrderFront(nil)
                        window.orderFrontRegardless()
                        return
                    }
                }

                // Fallback: simulate a click on the menu bar to trigger .onAppear
                if let button = NSApp.windows
                    .compactMap({ $0.contentView?.subviews.first as? NSStatusBarButton })
                    .first {
                    button.performClick(nil)
                }
            }
        }
    }
}
