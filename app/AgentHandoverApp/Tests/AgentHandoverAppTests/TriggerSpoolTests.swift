import XCTest
import Foundation
@testable import AgentHandoverApp

final class TriggerSpoolTests: XCTestCase {
    private var stateDir: URL!

    override func setUpWithError() throws {
        stateDir = FileManager.default.temporaryDirectory
            .appendingPathComponent("ah-spool-\(UUID().uuidString)")
    }

    override func tearDownWithError() throws {
        try? FileManager.default.removeItem(at: stateDir)
    }

    private func spoolFiles() throws -> [URL] {
        let dir = stateDir.appendingPathComponent(TriggerSpool.directoryName)
        return try FileManager.default.contentsOfDirectory(at: dir, includingPropertiesForKeys: nil)
    }

    private func draft(_ slug: String, state: String) throws -> SOPEntry {
        let json = """
        {"sop_id": "id-\(slug)", "slug": "\(slug)", "title": "T", "source": "focus",
         "status": "draft", "confidence": 0.9, "created_at": "2026-09-01T10:00:00Z",
         "lifecycle_state": "\(state)"}
        """.data(using: .utf8)!
        return try JSONDecoder().decode(SOPEntry.self, from: json)
    }

    @MainActor
    func testApproveForAgentsWritesOneLifecyclePromoteTrigger() throws {
        let manager = SOPIndexManager()
        manager.stateDir = stateDir

        manager.approveForAgents(try draft("check-domain", state: "draft"))

        let files = try spoolFiles()
        XCTAssertEqual(files.count, 1, "one request, no staggered chain: \(files)")
        let file = try XCTUnwrap(files.first)
        XCTAssertTrue(file.lastPathComponent.hasPrefix("lifecycle-promote-"))
        XCTAssertEqual(file.pathExtension, "json")

        let payload = try XCTUnwrap(
            JSONSerialization.jsonObject(with: Data(contentsOf: file)) as? [String: Any]
        )
        XCTAssertEqual(payload["kind"] as? String, "lifecycle-promote")
        XCTAssertEqual(payload["slug"] as? String, "check-domain")
        XCTAssertEqual(payload["to_state"] as? String, "agent_ready")
    }

    @MainActor
    func testApproveForAgentsIsTheSameSingleWriteFromAnyState() throws {
        let manager = SOPIndexManager()
        manager.stateDir = stateDir

        manager.approveForAgents(try draft("a", state: "observed"))
        manager.approveForAgents(try draft("b", state: "verified"))

        XCTAssertEqual(try spoolFiles().count, 2)
    }

    func testTwoRequestsOfOneKindDoNotOverwriteEachOther() throws {
        let first = TriggerSpool.write(kind: "approve", payload: ["sop_id": "a"], stateDir: stateDir)
        let second = TriggerSpool.write(kind: "approve", payload: ["sop_id": "b"], stateDir: stateDir)

        XCTAssertNotNil(first)
        XCTAssertNotNil(second)
        XCTAssertNotEqual(first, second)
        let names = try spoolFiles().map(\.lastPathComponent)
        XCTAssertEqual(names.count, 2)
        XCTAssertFalse(names.contains { $0.hasSuffix(".tmp") })
    }
}
