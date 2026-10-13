import SwiftUI
import Sparkle
import ServiceManagement
import os

/// Launch and capture-start decisions, in the unified log
/// (`log show --predicate 'subsystem == "com.agenthandover.app"'`).
let launchLog = Logger(subsystem: "com.agenthandover.app", category: "launch")

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
                .environmentObject(delegate.sharedAppState)
        }
        .defaultSize(width: 560, height: 700)
        .windowResizability(.contentMinSize)

        // The Commonplace window keeps the id of the Memory window it
        // replaced, so its saved frame carries over.
        Window("Commonplace", id: "memory") {
            CommonplaceView()
                .environmentObject(delegate)
        }
        .defaultSize(width: 960, height: 640)
        .windowResizability(.contentMinSize)

        Window("Memory Settings", id: "memory-settings") {
            MemorySettingsView()
                .environmentObject(delegate.sharedAppState)
        }
        .windowResizability(.contentSize)
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
    /// The open setup was for an install that had finished setup before
    /// (an upgrade that lost a grant): it closes itself once the grants
    /// are back.
    private var setupForExistingInstall = false
    private var focusQAWindow: NSWindow?
    private var modelDownloadWindow: NSWindow?
    private var memoryOfferWindow: NSWindow?
    /// Set by the offer's "Remind Me Later": that close records no choice.
    private var memoryOfferRemindLater = false

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
            FocusQuestionsNotifier.shared.openReviewQueue = { [weak self] in self?.showReviewQueue() }
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
            // Cloud mode's API keys (0.4.x and earlier) are read by nothing now.
            LegacyAPIKeys.removeOnce()
        }

        // Both grants back while the app runs: finish a setup that was only
        // waiting for them, or resume observing.
        MainActor.assumeIsolated {
            sharedAppState.onPermissionsGranted = { [weak self] in self?.permissionsGranted() }
        }

        let accessibility = PermissionChecker.isAccessibilityGranted()
        let screenRecording = PermissionChecker.isScreenRecordingGranted()
        let launch = OnboardingProgress.launchAction(
            defaults: .standard,
            accessibilityGranted: accessibility,
            screenRecordingGranted: screenRecording
        )
        // The grants' next reading is compared with this one: a launch that
        // opened setup for a grant it read as missing (right after an
        // upgrade macOS can report one missing for a moment) resumes
        // observing when the next reading has both.
        MainActor.assumeIsolated {
            sharedAppState.seedGrants(accessibility: accessibility, screenRecording: screenRecording)
        }
        launchLog.notice("launch: \(String(describing: launch), privacy: .public) (accessibility \(accessibility, privacy: .public), screen recording \(screenRecording, privacy: .public))")
        if case .setup(let step) = launch {
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

            // Onboarding done: start services unless the user paused.
            if launch == .observe {
                MainActor.assumeIsolated { startObserving() }
            }
            checkModelUpgrade()
        }
    }

    /// Start capture off the main thread (startAll waits for the daemon).
    @MainActor
    private func startObserving() {
        launchLog.notice("start observing (a start already in flight: \(self.observationToggle.isTransitioning, privacy: .public))")
        sharedAppState.userStopped = false
        sharedAppState.servicesStarting()
        observationToggle.start { [weak self] _ in
            self?.sharedAppState.refreshStatus()
        }
    }

    /// Both grants came back while the app runs (see
    /// `OnboardingProgress.regrantAction`).
    @MainActor
    private func permissionsGranted() {
        switch OnboardingProgress.regrantAction(
            defaults: .standard,
            setupOpen: onboardingWindow != nil,
            setupForExistingInstall: setupForExistingInstall,
            observing: sharedAppState.daemonRunning || observationToggle.isTransitioning
        ) {
        case .finishSetup: finishSetup()
        case .observe: startObserving()
        case .nothing: break
        }
    }

    /// Setup finished (its last screen) or has nothing left to do (an
    /// existing install's grants came back). A first install starts
    /// paused; an existing install resumes what the user had.
    @MainActor
    private func finishSetup() {
        let paused = OnboardingProgress.finish(in: .standard)
        sharedAppState.userStopped = paused
        onboardingWindow?.close()
        onboardingWindow = nil
        hideFromDock()
        registerLoginItemOnce()
        DispatchQueue.global(qos: .utility).async {
            ServiceController.installNativeMessagingHostManifest()
        }
        // Capture without both grants would record nothing; the grants
        // coming back starts it (permissionsGranted).
        let granted = PermissionChecker.isAccessibilityGranted()
            && PermissionChecker.isScreenRecordingGranted()
        if !paused && granted && !sharedAppState.daemonRunning {
            startObserving()
        }
        sharedAppState.refreshStatus()
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

        setupForExistingInstall = UserDefaults.standard.bool(forKey: OnboardingProgress.completedKey)
        let onboardingView = OnboardingView(initialStep: step ?? onboardingStartStep() ?? 0, onComplete: { [weak self] in
            self?.finishSetup()
        })
        .environmentObject(sharedAppState)
        .frame(width: 660, height: 820)

        let hostingController = NSHostingController(rootView: onboardingView)
        let window = NSWindow(contentViewController: hostingController)
        window.title = "AgentHandover Setup"
        // Setup's palette is light only (white and cream screens, navy
        // text). In Dark Mode, system colours and controls inside it (a
        // card's controlBackgroundColor, text fields) turned dark and left
        // navy text on a dark fill; pinned to Aqua they match the palette.
        window.appearance = NSAppearance(named: .aqua)
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

    /// The Review Queue, from a micro-question notification: the open one,
    /// else a new window hosting it.
    func showReviewQueue() {
        NSApp.setActivationPolicy(.regular)
        NSApp.activate(ignoringOtherApps: true)
        if let window = NSApp.windows.first(where: { $0.title == "Review Queue" }) {
            window.makeKeyAndOrderFront(nil)
            return
        }
        let window = NSWindow(contentViewController: NSHostingController(rootView: MicroReviewView()))
        window.title = "Review Queue"
        window.styleMask = [.titled, .closable, .miniaturizable, .resizable]
        window.setContentSize(NSSize(width: 640, height: 680))
        window.center()
        window.isReleasedWhenClosed = false
        window.makeKeyAndOrderFront(nil)
        window.orderFrontRegardless()
    }

    /// After an upgrade, a config on earlier default models (see
    /// `ModelUpgrade`): switch when the new models are pulled, else ask to
    /// download them unless the user said "Not now" for this version.
    private func checkModelUpgrade() {
        DispatchQueue.global(qos: .utility).async { [weak self] in
            let plan = ModelUpgrade.loadPlan()
            let decision = ModelUpgrade.decide(
                plan: plan,
                installed: plan == nil ? nil : ModelUpgrade.installedModels(),
                declinedVersion: UserDefaults.standard.string(forKey: ModelUpgrade.declinedVersionKey),
                appVersion: ModelUpgrade.appVersion
            )
            launchLog.notice("model upgrade: \(String(describing: decision), privacy: .public)")
            switch decision {
            case .none:
                break
            case .applySilently:
                ModelUpgrade.apply()
            case .ask(let missing):
                DispatchQueue.main.async {
                    MainActor.assumeIsolated { self?.showModelDownload(plan: plan, missing: missing) }
                }
                // One question per launch: Memory's waits for the next.
                return
            }
            self?.checkMemory()
        }
    }

    /// After the model upgrade, Memory: the once-only offer to an updated
    /// install (`MemoryDecisions.offer`), or turning on a Memory setup left
    /// off for an Ollama that has since been updated
    /// (`MemoryDecisions.deferredAction`).  Blocking; off the main thread.
    private func checkMemory() {
        let defaults = UserDefaults.standard
        let config = MemorySettings.readConfig()
        let enabled = MemorySettings.enabledValue(in: config)
        let table = try? ModelTierTable.load(from: ModelTierTable.installedURL)
        let version = OllamaModels.serverVersion()
        let deferred = MemoryDecisions.deferredAction(
            deferred: defaults.bool(forKey: MemoryDecisions.deferredKey), memoryEnabled: enabled,
            ollamaVersion: version, installed: version == nil ? nil : ModelUpgrade.installedModels(), table: table
        )
        launchLog.notice("memory deferred: \(String(describing: deferred), privacy: .public)")
        switch deferred {
        case .enable:
            MemoryDecisions.enableDeferredMemory()
            return
        case .download(let model):
            DispatchQueue.main.async {
                MainActor.assumeIsolated { self.showMemoryModelDownload(model: model, turnOn: true) }
            }
            return
        case .none:
            break
        }
        let decision = MemoryDecisions.offer(
            setupAnswered: defaults.bool(forKey: MemoryDecisions.setupAnsweredKey),
            memoryEnabled: enabled, offer: MemoryOfferState.load(),
            imageEmbeddingsInConfig: ConfigTOML.value(section: "embedding", key: "image_embeddings", in: config) == "true",
            ollamaVersion: version, table: table
        )
        launchLog.notice("memory offer: \(String(describing: decision), privacy: .public)")
        guard case .offer(let visualSearch, let tooOld) = decision, let table,
              let model = table.memoryModel, let minimum = table.minimumOllamaVersionForMemory else { return }
        DispatchQueue.main.async {
            MainActor.assumeIsolated {
                self.showMemoryOffer(MemoryOfferView(
                    visualSearch: visualSearch, ollamaTooOld: tooOld, minimumOllama: minimum,
                    memoryModel: model, memoryModelSize: table.sizeLabel(model),
                    close: { [weak self] in self?.memoryOfferWindow?.close() },
                    remindLater: { [weak self] in
                        self?.memoryOfferRemindLater = true
                        self?.memoryOfferWindow?.close()
                    }
                ))
            }
        }
    }

    @MainActor
    private func showMemoryOffer(_ view: MemoryOfferView) {
        NSApp.setActivationPolicy(.regular)
        NSApp.activate(ignoringOtherApps: true)
        if let window = memoryOfferWindow {
            window.makeKeyAndOrderFront(nil)
            return
        }
        let controller = NSHostingController(rootView: view)
        // Follow SwiftUI's size: the window opened before the wrapped text
        // was measured and clipped the last line of each paragraph.
        controller.sizingOptions = [.preferredContentSize]
        let window = NSWindow(contentViewController: controller)
        window.title = "Memory"
        window.styleMask = [.titled, .closable]
        window.setContentSize(controller.view.fittingSize)
        window.center()
        window.isReleasedWhenClosed = false
        window.delegate = self
        memoryOfferWindow = window
        window.makeKeyAndOrderFront(nil)
        window.orderFrontRegardless()
    }

    /// Download Memory's model (the Commonplace window's "Download…", or a
    /// Memory setup left off until Ollama was updated: then `turnOn`).
    @MainActor
    func showMemoryModelDownload(model: String, turnOn: Bool) {
        let size = (try? ModelTierTable.load(from: ModelTierTable.installedURL))?.sizeLabel(model)
        let view = ModelDownloadView(
            question: turnOn
                ? "Ollama can run Memory now. Download \(model)\(size.map { " (\($0))" } ?? "") and turn Memory on?"
                : "Memory needs \(model)\(size.map { " (\($0))" } ?? ""). Download now?",
            detail: "Memory picks up the screens that waited for it.",
            missing: [model],
            afterDownload: {
                if turnOn {
                    MemoryDecisions.enableDeferredMemory()
                } else {
                    MemorySettings.restartServices()
                }
                return .success("Downloaded. Memory is on.")
            },
            // Not now: Memory stays off; Memory Settings turns it on later.
            notNow: { if turnOn { UserDefaults.standard.set(false, forKey: MemoryDecisions.deferredKey) } },
            close: { [weak self] in
                self?.modelDownloadWindow?.close()
                Task { await MemoryCenter.shared.refresh() }
            }
        )
        presentModelDownload(view)
    }

    /// The menu's "AI model missing" row: the upgrade's models when a
    /// switch waits for them, else the configured models the worker
    /// reported missing.  Called on the main thread (the menu).
    func showModelDownload() {
        let reported = MainActor.assumeIsolated { sharedAppState.workerStatus?.missing_models ?? [] }
        DispatchQueue.global(qos: .userInitiated).async { [weak self] in
            let plan = ModelUpgrade.loadPlan()
            let upgradeMissing = plan.flatMap { plan in
                ModelUpgrade.installedModels().map { OllamaModels.missing(plan.models, installed: $0) }
            } ?? []
            DispatchQueue.main.async {
                MainActor.assumeIsolated {
                    if !upgradeMissing.isEmpty {
                        self?.showModelDownload(plan: plan, missing: upgradeMissing)
                    } else if !reported.isEmpty {
                        self?.showModelDownload(plan: nil, missing: reported)
                    }
                }
            }
        }
    }

    @MainActor
    private func showModelDownload(plan: ModelUpgradePlan?, missing: [String]) {
        presentModelDownload(ModelDownloadView(plan: plan, missing: missing, close: { [weak self] in
            self?.modelDownloadWindow?.close()
            self?.sharedAppState.refreshStatus()
        }))
    }

    /// The models window, hosting `view` (one at a time).
    @MainActor
    private func presentModelDownload(_ view: ModelDownloadView) {
        NSApp.setActivationPolicy(.regular)
        NSApp.activate(ignoringOtherApps: true)
        if let window = modelDownloadWindow {
            window.makeKeyAndOrderFront(nil)
            return
        }
        let window = NSWindow(contentViewController: NSHostingController(rootView: view))
        window.title = "AgentHandover Models"
        window.styleMask = [.titled, .closable]
        window.center()
        window.isReleasedWhenClosed = false
        window.delegate = self
        modelDownloadWindow = window
        window.makeKeyAndOrderFront(nil)
        window.orderFrontRegardless()
    }

    /// The Q&A or the models window closed: back to a menu bar app unless
    /// another window is open.
    func windowWillClose(_ notification: Notification) {
        guard let window = notification.object as? NSWindow,
              window === focusQAWindow || window === modelDownloadWindow || window === memoryOfferWindow
        else { return }
        if window === focusQAWindow {
            focusQAWindow = nil
        } else if window === modelDownloadWindow {
            modelDownloadWindow = nil
        } else {
            memoryOfferWindow = nil
            // Closed with the title bar's button: the offer is shown once,
            // so that is "Not now" (not for "Remind Me Later").
            if let choice = MemoryOfferState.choiceOnClose(
                remindLater: memoryOfferRemindLater, recorded: MemoryOfferState.load().choice
            ) {
                MemoryOfferState.record(choice)
            }
            memoryOfferRemindLater = false
        }
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
