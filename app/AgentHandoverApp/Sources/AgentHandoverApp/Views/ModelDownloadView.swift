import SwiftUI

/// "0.5 uses a faster model (<name>, <size>). Download now?": downloads the
/// missing models with a progress bar, then makes the model switch (with a
/// plan) or restarts the worker (a model the config names went missing).
/// Until the download finishes the Mac keeps the models it has.
struct ModelDownloadView: View {
    let plan: ModelUpgradePlan?
    let missing: [String]
    let close: () -> Void

    private enum Phase: Equatable {
        case asking
        case downloading(status: String, fraction: Double)
        case done(String)
        case failed(String)
    }

    @State private var phase: Phase = .asking

    var body: some View {
        VStack(alignment: .leading, spacing: 14) {
            Text(ModelUpgrade.question(plan: plan, missing: missing))
                .font(.system(size: 13, weight: .semibold))
                .fixedSize(horizontal: false, vertical: true)
            if let plan {
                Text("Until the download finishes, AgentHandover keeps running \(plan.current.annotationModel)"
                     + (plan.current.sopModel == plan.current.annotationModel ? "" : " and \(plan.current.sopModel)")
                     + ". Then it switches to the \(plan.label) models.")
                    .font(.system(size: 11))
                    .foregroundColor(.secondary)
                    .fixedSize(horizontal: false, vertical: true)
            }
            switch phase {
            case .asking:
                HStack {
                    Spacer()
                    Button("Not now") { notNow() }
                        .keyboardShortcut(.cancelAction)
                    Button("Download") { download() }
                        .keyboardShortcut(.defaultAction)
                }
            case .downloading(let status, let fraction):
                ProgressView(value: fraction)
                Text(status)
                    .font(.system(size: 11, design: .monospaced))
                    .foregroundColor(.secondary)
                    .lineLimit(1)
            case .done(let message):
                Text(message).font(.system(size: 11))
                HStack { Spacer(); Button("Done") { close() }.keyboardShortcut(.defaultAction) }
            case .failed(let message):
                Text(message).font(.system(size: 11)).foregroundColor(.red)
                    .fixedSize(horizontal: false, vertical: true)
                HStack {
                    Spacer()
                    Button("Close") { close() }.keyboardShortcut(.cancelAction)
                    Button("Try again") { download() }.keyboardShortcut(.defaultAction)
                }
            }
        }
        .padding(20)
        .frame(width: 420)
    }

    /// Keep the current models; ask again after the next upgrade.
    private func notNow() {
        if plan != nil {
            UserDefaults.standard.set(ModelUpgrade.appVersion, forKey: ModelUpgrade.declinedVersionKey)
        }
        close()
    }

    private func download() {
        guard let ollamaPath = OllamaModels.executablePath() else {
            phase = .failed("Ollama is not installed. Install it from ollama.com/download, then try again.")
            return
        }
        let models = missing
        let plan = plan
        phase = .downloading(status: "Starting download\u{2026}", fraction: 0)
        DispatchQueue.global(qos: .userInitiated).async {
            for (index, model) in models.enumerated() {
                let failure = OllamaModels.pull(model, ollamaPath: ollamaPath) { line in
                    let part = OllamaModels.fraction(in: line) ?? 0
                    DispatchQueue.main.async {
                        phase = .downloading(
                            status: "\(model): \(line.prefix(60))",
                            fraction: (Double(index) + part) / Double(models.count)
                        )
                    }
                }
                if let failure {
                    DispatchQueue.main.async { phase = .failed(failure) }
                    return
                }
            }
            let result: Phase
            if let plan {
                result = ModelUpgrade.apply()
                    ? .done("AgentHandover now runs the \(plan.label) models.")
                    : .failed("Downloaded, but the switch did not happen. Run: agenthandover doctor")
            } else {
                ServiceController.restartWorker()
                result = .done("Downloaded. AgentHandover processes your recordings again.")
            }
            DispatchQueue.main.async { phase = result }
        }
    }
}
