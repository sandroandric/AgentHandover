import AppKit
import CoreServices

/// `[privacy] read_app_state` (off by default): with it on, the daemon
/// reads the open document, page or message of a few Apple apps through
/// AppleScript.  macOS gates that per app ("AgentHandover wants access to
/// control Safari"), and its dialog says "control" although the daemon
/// only reads.
///
/// The daemon never triggers that dialog: before each read it checks the
/// consent without prompting and skips an app that is not granted
/// (crates/daemon/src/platform/applescript_bridge.rs).  The app asks, as
/// onboarding explains: for the supported apps that are open when the user
/// turns the option on, and for any other one the first time the user
/// switches to it while the option is on.
enum AppStateReading {
    /// The apps the daemon reads: `KNOWN_APPS` in applescript_bridge.rs
    /// (a test pins the two lists together).
    static let supportedApps: [String: String] = [
        "com.apple.finder": "Finder",
        "com.apple.mail": "Mail",
        "com.apple.Notes": "Notes",
        "com.apple.iCal": "Calendar",
        "com.apple.MobileSMS": "Messages",
        "com.apple.Safari": "Safari",
        "com.apple.Preview": "Preview",
        "com.apple.TextEdit": "TextEdit",
        "com.apple.reminders": "Reminders",
        "com.apple.AddressBook": "Contacts",
    ]

    static let managedKeys = ["read_app_state"]

    static var configURL: URL {
        FileManager.default.homeDirectoryForCurrentUser
            .appendingPathComponent("Library/Application Support/agenthandover/config.toml")
    }

    /// Whether `config` turns the reads on (`read_app_state = true` in `[privacy]`).
    static func isEnabled(in config: String) -> Bool {
        var inPrivacy = false
        for raw in config.components(separatedBy: "\n") {
            let line = raw.components(separatedBy: "#")[0]
                .trimmingCharacters(in: .whitespaces)
            if line.hasPrefix("[") {
                inPrivacy = line == "[privacy]"
                continue
            }
            let parts = line.split(separator: "=", maxSplits: 1)
                .map { $0.trimmingCharacters(in: .whitespaces) }
            if inPrivacy, parts.count == 2, parts[0] == "read_app_state" {
                return parts[1] == "true"
            }
        }
        return false
    }

    /// `config` with `read_app_state` set in `[privacy]` (other keys kept).
    static func apply(enabled: Bool, to config: String) -> String {
        ModelTierTable.applyVLMBlock(
            "[privacy]\nread_app_state = \(enabled)\n",
            to: config, managedKeys: managedKeys, sectionHeader: "[privacy]"
        )
    }

    static func readConfig() -> String {
        (try? String(contentsOf: configURL, encoding: .utf8)) ?? ""
    }

    static var isEnabled: Bool { isEnabled(in: readConfig()) }

    /// Write the choice into config.toml.  The daemon reads it at start.
    static func setEnabled(_ enabled: Bool) {
        let current = readConfig()
        let updated = apply(enabled: enabled, to: current)
        guard updated != current else { return }
        try? FileManager.default.createDirectory(
            at: configURL.deletingLastPathComponent(), withIntermediateDirectories: true
        )
        try? updated.write(to: configURL, atomically: true, encoding: .utf8)
    }

    // MARK: - Consent

    enum Consent: Equatable {
        case granted, denied, notAsked, notRunning, unknown

        init(status: OSStatus) {
            switch Int(status) {
            case Int(noErr): self = .granted
            case errAEEventNotPermitted: self = .denied
            case errAEEventWouldRequireUserConsent: self = .notAsked
            case procNotFound: self = .notRunning
            default: self = .unknown
            }
        }
    }

    /// macOS's Automation consent for this app to query `bundleID`.  With
    /// `ask`, macOS shows its dialog when the user has not answered yet
    /// (it blocks until they do: never call it on the main thread).
    static func consent(for bundleID: String, ask: Bool) -> Consent {
        var target = AEAddressDesc()
        let status: OSStatus = bundleID.withCString { bytes in
            let created = AECreateDesc(
                DescType(typeApplicationBundleID), bytes, strlen(bytes), &target
            )
            guard created == noErr else { return OSStatus(created) }
            defer { AEDisposeDesc(&target) }
            return AEDeterminePermissionToAutomateTarget(
                &target, AEEventClass(typeWildCard), AEEventID(typeWildCard), ask
            )
        }
        return Consent(status: status)
    }

    /// Ask for `bundleID` if macOS has not asked yet (in the background).
    static func askIfNotAsked(_ bundleID: String) {
        guard supportedApps[bundleID] != nil else { return }
        DispatchQueue.global(qos: .userInitiated).async {
            if consent(for: bundleID, ask: false) == .notAsked {
                _ = consent(for: bundleID, ask: true)
            }
        }
    }

    /// Ask for every supported app that is open now.
    static func askForOpenApps() {
        for app in NSWorkspace.shared.runningApplications {
            if let id = app.bundleIdentifier { askIfNotAsked(id) }
        }
    }
}

/// Asks for a supported app's consent the first time the user switches to
/// it while `[privacy] read_app_state` is on.  Started once at launch.
final class AppStateConsentAsker {
    static let shared = AppStateConsentAsker()
    private var observer: NSObjectProtocol?

    func start() {
        guard observer == nil else { return }
        observer = NSWorkspace.shared.notificationCenter.addObserver(
            forName: NSWorkspace.didActivateApplicationNotification, object: nil, queue: .main
        ) { note in
            let app = note.userInfo?[NSWorkspace.applicationUserInfoKey] as? NSRunningApplication
            guard let id = app?.bundleIdentifier,
                  AppStateReading.supportedApps[id] != nil,
                  AppStateReading.isEnabled else { return }
            AppStateReading.askIfNotAsked(id)
        }
    }
}
