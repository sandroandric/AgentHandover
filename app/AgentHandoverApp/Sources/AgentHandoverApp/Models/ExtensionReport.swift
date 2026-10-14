import Foundation

/// What `agenthandover doctor --extension-json` reports
/// (crates/cli/src/browser_extension.rs): the folder to load unpacked and
/// the browser profiles that have the extension. `doctor` passes its
/// "Browser extension" check on the same data, so setup and doctor agree.
struct ExtensionReport: Decodable, Equatable {
    struct Install: Decodable, Equatable {
        let browser: String
        let profile: String
        let enabled: Bool
    }

    let extensionId: String
    let buildPath: String?
    let installedIn: [Install]

    static func decode(_ data: Data) throws -> ExtensionReport {
        let decoder = JSONDecoder()
        decoder.keyDecodingStrategy = .convertFromSnakeCase
        return try decoder.decode(ExtensionReport.self, from: data)
    }

    /// What the setup screen shows.
    enum Status: Equatable {
        /// Loaded and switched on; the browsers, e.g. "Google Chrome".
        case installed(browsers: [String])
        /// Loaded but switched off in `browser`'s extensions page.
        case turnedOff(browser: String)
        /// Built at `path`, not loaded in any browser yet.
        case notLoaded(path: String)
        /// No build on disk.
        case notBuilt
    }

    var status: Status {
        let enabled = installedIn.filter(\.enabled)
        if !enabled.isEmpty {
            var browsers: [String] = []
            for install in enabled where !browsers.contains(install.browser) {
                browsers.append(install.browser)
            }
            return .installed(browsers: browsers)
        }
        if let off = installedIn.first {
            return .turnedOff(browser: off.browser)
        }
        if let buildPath, !buildPath.isEmpty {
            return .notLoaded(path: buildPath)
        }
        return .notBuilt
    }

    /// Run `<cli> doctor --extension-json`. Blocking; nil when the CLI is
    /// missing, fails, or prints something else.
    static func load(cliPath: String) -> ExtensionReport? {
        DoctorJSON.run(cliPath: cliPath, flag: "--extension-json").flatMap { try? decode($0) }
    }
}

/// Runs `<cli> doctor <flag>` and returns what it printed.
enum DoctorJSON {
    static func run(cliPath: String, flag: String, extra: [String] = []) -> Data? {
        let process = Process()
        process.executableURL = URL(fileURLWithPath: cliPath)
        process.arguments = ["doctor", flag] + extra
        let out = Pipe()
        process.standardOutput = out
        process.standardError = FileHandle.nullDevice
        do {
            try process.run()
        } catch {
            return nil
        }
        let data = out.fileHandleForReading.readDataToEndOfFile()
        process.waitUntilExit()
        return process.terminationStatus == 0 ? data : nil
    }
}
