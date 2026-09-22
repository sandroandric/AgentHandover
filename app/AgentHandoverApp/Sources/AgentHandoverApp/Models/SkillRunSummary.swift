import Foundation

/// What agents' runs of a Skill say, read from the procedure JSON the
/// worker keeps in the knowledge base (`execution_stats` and
/// `observed_alternatives`, written when an agent reports
/// `report_execution_complete` over MCP).
struct SkillRunSummary: Equatable {
    /// A step where agents did something other than the Skill says: the
    /// user decides whether the Skill should change.
    struct OpenQuestion: Equatable {
        let stepId: String
        /// 1-based position of the step, or nil when the Skill no longer has it.
        let stepNumber: Int?
        let stepAction: String
        let observedAction: String
        let count: Int

        /// "Step 2: Export the table as CSV" (the step id when the Skill no longer has it).
        var stepLabel: String {
            let name = stepNumber.map { "Step \($0)" } ?? stepId
            return stepAction.isEmpty ? name : "\(name): \(stepAction)"
        }

        /// "An agent did: … (2 runs). Should the Skill change to match?"
        var prompt: String {
            let runs = count > 1 ? " (\(count) runs)" : ""
            return "An agent did: \(observedAction)\(runs). Should the Skill change to match?"
        }
    }

    let runs: Int
    let completedRuns: Int
    let deviatedRuns: Int
    let failedRuns: Int
    /// Deviated steps over all runs.
    let deviations: Int
    let lastRun: String?
    let openQuestions: [OpenQuestion]

    /// nil when no agent has reported a run of this Skill.
    init?(procedure json: [String: Any]) {
        let stats = json["execution_stats"] as? [String: Any] ?? [:]
        let alternatives = json["observed_alternatives"] as? [[String: Any]] ?? []
        let runs = Self.int(stats["total"])
        guard runs > 0 || !alternatives.isEmpty else { return nil }

        self.runs = runs
        self.completedRuns = Self.int(stats["completed"])
        self.deviatedRuns = Self.int(stats["deviated"])
        self.failedRuns = Self.int(stats["failed"])
        self.deviations = stats["deviations"] != nil ? Self.int(stats["deviations"]) : alternatives.count
        self.lastRun = stats["last_run"] as? String

        // Step ids as the packet prints them: `step_id`, else `step_<n>`.
        let steps = (json["steps"] as? [Any] ?? []).compactMap { $0 as? [String: Any] }
        var stepIndex: [String: (Int, String)] = [:]
        for (i, step) in steps.enumerated() {
            let sid = (step["step_id"] as? String).flatMap { $0.isEmpty ? nil : $0 } ?? "step_\(i + 1)"
            let action = step["action"] as? String ?? step["step"] as? String ?? ""
            stepIndex[sid] = (i + 1, action)
        }

        var order: [String] = []
        var grouped: [String: OpenQuestion] = [:]
        for alt in alternatives {
            guard let sid = alt["step_id"] as? String, !sid.isEmpty else { continue }
            let observed = (alt["observed_action"] as? String ?? alt["detail"] as? String ?? "")
                .trimmingCharacters(in: .whitespacesAndNewlines)
            let key = sid + "\u{1F}" + observed.lowercased()
            if let existing = grouped[key] {
                grouped[key] = OpenQuestion(
                    stepId: existing.stepId, stepNumber: existing.stepNumber,
                    stepAction: existing.stepAction, observedAction: existing.observedAction,
                    count: existing.count + 1
                )
            } else {
                order.append(key)
                grouped[key] = OpenQuestion(
                    stepId: sid, stepNumber: stepIndex[sid]?.0,
                    stepAction: stepIndex[sid]?.1 ?? "", observedAction: observed, count: 1
                )
            }
        }
        // Most frequent first; then in step order; then first seen.
        self.openQuestions = order.enumerated()
            .compactMap { seen, key in grouped[key].map { (seen, $0) } }
            .sorted { a, b in
                if a.1.count != b.1.count { return a.1.count > b.1.count }
                let na = a.1.stepNumber ?? Int.max, nb = b.1.stepNumber ?? Int.max
                return na != nb ? na < nb : a.0 < b.0
            }
            .map { $0.1 }
    }

    /// "Ran 3 times · 2 deviations" (the header line of the runs card).
    var headline: String {
        let times = runs == 1 ? "once" : "\(runs) times"
        let devs = deviations == 1 ? "1 deviation" : "\(deviations) deviations"
        return "Ran \(times) · \(devs)"
    }

    private static func int(_ value: Any?) -> Int {
        if let n = value as? Int { return n }
        if let d = value as? Double { return Int(d) }
        if let n = value as? NSNumber { return n.intValue }
        return 0
    }
}
