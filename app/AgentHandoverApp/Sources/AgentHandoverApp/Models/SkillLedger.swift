import Foundation

/// What AgentHandover has learned about a Skill, slot by slot: the
/// procedure JSON's `knowledge` object, written by the worker's
/// `ledger.py` on every save (`{slot: {status, value_ref, evidence,
/// source, updated_at}}`).  Each slot reads as one line: known from N
/// runs, you told me, or still unknown.
struct SkillLedger: Equatable {
    struct Slot: Equatable {
        /// The worker's slot key (`done_criteria`).
        let key: String
        /// "Done criteria".
        let label: String
        /// `known`, `hypothesis` or `unknown`.
        let status: String
        /// "Known from 2 runs" / "You told me" / "Seen in 1 run, not confirmed" / "Still unknown".
        let line: String
    }

    /// The worker's slot order (`ledger.SLOTS`); `voice` and `context` are optional.
    static let slotOrder = [
        "goal", "done_criteria", "trigger", "inputs", "steps", "decisions",
        "exceptions", "guardrails", "verification", "voice", "context",
    ]
    static let optionalSlots: Set<String> = ["voice", "context"]

    let slots: [Slot]

    /// nil when the procedure has no ledger yet (saved by an older worker).
    init?(procedure json: [String: Any]) {
        guard let knowledge = json["knowledge"] as? [String: Any], !knowledge.isEmpty else {
            return nil
        }
        slots = Self.slotOrder.compactMap { key in
            guard let entry = knowledge[key] as? [String: Any] else { return nil }
            let status = entry["status"] as? String ?? "unknown"
            let evidence = (entry["evidence"] as? [Any] ?? []).compactMap { $0 as? [String: Any] }
            return Slot(
                key: key,
                label: key.replacingOccurrences(of: "_", with: " ").sentenceCased,
                status: status,
                line: Self.line(status: status, source: entry["source"] as? String, evidence: evidence)
            )
        }
    }

    /// "4 of 9 known" over the tracked (non-optional) slots.
    var headline: String {
        let tracked = slots.filter { !Self.optionalSlots.contains($0.key) }
        let known = tracked.filter { $0.status == "known" }.count
        return "\(known) of \(tracked.count) known"
    }

    private static func line(status: String, source: String?, evidence: [[String: Any]]) -> String {
        if status == "unknown" { return "Still unknown" }
        if source == "user" { return "You told me" }
        let runs = Set(evidence.compactMap { item -> String? in
            let kind = item["kind"] as? String
            guard kind == "instance" || kind == "run" else { return nil }
            return item["id"] as? String
        }).count
        let seen: String
        if runs > 0 {
            seen = runs == 1 ? "1 run" : "\(runs) runs"
        } else {
            seen = "the recording"
        }
        return status == "known" ? "Known from \(seen)" : "Seen in \(seen), not confirmed"
    }
}

private extension String {
    /// "Done criteria": only the first letter upper-cased.
    var sentenceCased: String {
        guard let first = first else { return self }
        return first.uppercased() + dropFirst()
    }
}
