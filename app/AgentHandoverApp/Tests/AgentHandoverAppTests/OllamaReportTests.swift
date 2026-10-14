import XCTest
@testable import AgentHandoverApp

/// Onboarding shows what `agenthandover doctor --ollama-json` reports.  The
/// fixture is the CLI's own output for the test Mac's case (Ollama.app
/// 0.30.6 serving ahead of Homebrew's 0.34.2, with heavy swap); the CLI's
/// `machine::tests::the_report_matches_the_apps_fixture` writes and pins it.
final class OllamaReportTests: XCTestCase {
    private static let fixtures = URL(fileURLWithPath: #filePath)
        .deletingLastPathComponent()  // AgentHandoverAppTests
        .deletingLastPathComponent()  // Tests
        .deletingLastPathComponent()  // AgentHandoverApp
        .deletingLastPathComponent()  // app
        .deletingLastPathComponent()  // repo root
        .appendingPathComponent("tests/fixtures/ollama_report")

    private func report(_ name: String = "app_serving_ahead_of_formula.json") throws -> OllamaReport {
        try OllamaReport.decode(Data(contentsOf: Self.fixtures.appendingPathComponent(name)))
    }

    func testDecodesTheCLIReport() throws {
        let r = try report()
        XCTAssertEqual(r.version, "0.30.6")
        XCTAssertEqual(r.minimum, "0.34.0")
        XCTAssertTrue(r.host.local)
        XCTAssertEqual(r.installs.map(\.kind), ["Ollama.app", "Homebrew"])
        XCTAssertEqual(r.headroom?.availableBytes, 7_100_000_000)
        XCTAssertEqual(r.headroom?.swapUsedBytes, 44_000_000_000)
    }

    func testTheOllamaStepShowsDoctorsProblemsWordForWord() throws {
        let r = try report()
        XCTAssertEqual(r.serviceLine, "Ollama 0.30.6 is running at http://localhost:11434")
        XCTAssertEqual(r.problems.map(\.level), [.fail, .warn, .warn, .warn])
        let tooOld = r.problems[0]
        XCTAssertEqual(tooOld.title, "Ollama 0.30.6 is older than 0.34.0")
        XCTAssertTrue(tooOld.lines.contains {
            $0.hasPrefix("Homebrew 0.34.2 is installed, but Ollama.app 0.30.6 (pid 812")
        })
        XCTAssertTrue(tooOld.lines.last?.hasPrefix("Fix: To keep Homebrew 0.34.2: quit Ollama.app") == true)
        XCTAssertEqual(r.problems[1].title, "Ollama is installed 2 times")
        // Passing checks are not repeated as problems.
        XCTAssertFalse(r.problems.contains { $0.level == .pass })
        XCTAssertEqual(tooOld.symbolName, "xmark.octagon.fill")
    }

    /// An Ollama older than the Gemma 4 minimum on a tier without Gemma 4
    /// (the CLI's `machine::tests::the_warning_report_matches_the_apps_fixture`):
    /// doctor warns, and onboarding shows it as a warning, not a failure.
    func testAnOldOllamaWithoutGemma4IsShownAsAWarning() throws {
        let r = try report("old_ollama_without_gemma4.json")
        XCTAssertEqual(r.problems.map(\.level), [.warn])
        let tooOld = r.problems[0]
        XCTAssertEqual(tooOld.title, "Ollama 0.30.6 is older than 0.34.0")
        XCTAssertEqual(tooOld.symbolName, "exclamationmark.triangle.fill")
        XCTAssertTrue(tooOld.lines.contains {
            $0.hasPrefix("Only the Gemma 4 models require it: the configured models (qwen3.5:4b)")
        })
        XCTAssertTrue(tooOld.lines.last?.hasPrefix("Fix: update Ollama.app") == true)
    }

    func testTheTierStepShowsNeedAgainstFreeMemory() throws {
        let r = try report()
        XCTAssertEqual(r.memoryLabel(forTier: "standard"), "Needs 5.4 GB \u{00B7} 7.1 GB free now")
        XCTAssertEqual(r.memoryLabel(forTier: "recommended"), "Needs 8.5 GB \u{00B7} 7.1 GB free now")
        XCTAssertEqual(r.fits(tier: "standard"), true)
        XCTAssertEqual(r.fits(tier: "recommended"), false)
        // A tier the Mac's RAM does not offer has no figures.
        XCTAssertNil(r.memoryLabel(forTier: "max_quality"))
        XCTAssertNil(r.fits(tier: "max_quality"))
    }

    func testARemoteHostHasNoMemoryFiguresAndNothingAnsweringHasNoServiceLine() throws {
        let json = #"""
        {"host": {"url": "http://gpu-box.lan:11434", "local": false, "port": 11434},
         "version": null, "minimum": "0.34.0", "installs": [], "launch_items": [],
         "serving": null,
         "findings": [{"level": "fail", "title": "Ollama service",
                       "lines": ["Nothing answered at http://gpu-box.lan:11434."]}],
         "headroom": null}
        """#
        let r = try OllamaReport.decode(Data(json.utf8))
        XCTAssertNil(r.serviceLine)
        XCTAssertNil(r.memoryLabel(forTier: "standard"))
        XCTAssertEqual(r.problems.first?.lines, ["Nothing answered at http://gpu-box.lan:11434."])
    }

    func testAMissingCLIIsNoReport() {
        XCTAssertNil(OllamaReport.load(cliPath: "/nonexistent/agenthandover", mode: .skillsAndMemory))
        // A program that prints something else is no report either.
        XCTAssertNil(OllamaReport.load(cliPath: "/bin/echo", mode: .memoryOnly))
    }

    /// Setup asks for the report of the mode the person picked (config.toml
    /// names it only after the pull): `doctor --ollama-json --mode ...`.
    func testTheReportIsAskedForThePickedMode() {
        XCTAssertEqual(OllamaReport.arguments(for: .memoryOnly), ["--mode", "memory-only"])
        XCTAssertEqual(OllamaReport.arguments(for: .skillsAndMemory), ["--mode", "skills"])
    }
}
