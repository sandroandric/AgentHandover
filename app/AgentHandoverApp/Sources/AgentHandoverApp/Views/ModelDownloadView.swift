import SwiftUI

/// "0.5 uses a faster model (<name>, <size>). Download now?": downloads the
/// missing models with a progress bar, then runs `afterDownload` (the model
/// switch with a plan, a worker restart when a model the config names went
/// missing, turning Memory on).  Until the download finishes the Mac keeps
/// the models it has.
struct ModelDownloadView: View {
    let question: String
    let detail: String?
    let missing: [String]
    /// Off the main thread once every model is pulled: the message to show,
    /// or why that last step failed.
    let afterDownload: () -> Result<String, ModelDownloadFailure>
    /// "Not now" (the window closes after it).
    let notNow: () -> Void
    let close: () -> Void

    /// The model upgrade (WP-F16) and the menu's "AI model missing" row.
    init(plan: ModelUpgradePlan?, missing: [String], close: @escaping () -> Void) {
        self.question = ModelUpgrade.question(plan: plan, missing: missing)
        self.detail = plan.map { plan in
            "Until the download finishes, AgentHandover keeps running \(plan.current.annotationModel)"
                + (plan.current.sopModel == plan.current.annotationModel ? "" : " and \(plan.current.sopModel)")
                + ". Then it switches to the \(plan.label) models."
        }
        self.missing = missing
        self.afterDownload = {
            if let plan {
                return ModelUpgrade.apply()
                    ? .success("AgentHandover now runs the \(plan.label) models.")
                    : .failure(ModelDownloadFailure("Downloaded, but the switch did not happen. Run: agenthandover doctor"))
            }
            ServiceController.restartWorker()
            return .success("Downloaded. AgentHandover processes your recordings again.")
        }
        // Keep the current models; ask again after the next upgrade.
        self.notNow = {
            if plan != nil {
                UserDefaults.standard.set(ModelUpgrade.appVersion, forKey: ModelUpgrade.declinedVersionKey)
            }
        }
        self.close = close
    }

    init(
        question: String, detail: String?, missing: [String],
        afterDownload: @escaping () -> Result<String, ModelDownloadFailure>,
        notNow: @escaping () -> Void = {}, close: @escaping () -> Void
    ) {
        self.question = question
        self.detail = detail
        self.missing = missing
        self.afterDownload = afterDownload
        self.notNow = notNow
        self.close = close
    }

    private enum Phase: Equatable {
        case asking
        case downloading(status: String, fraction: Double)
        case done(String)
        case failed(String)
    }

    @State private var phase: Phase = .asking

    var body: some View {
        VStack(alignment: .leading, spacing: 14) {
            Text(question)
                .font(.system(size: 13, weight: .semibold))
                .fixedSize(horizontal: false, vertical: true)
            if let detail {
                Text(detail)
                    .font(.system(size: 11))
                    .foregroundColor(.secondary)
                    .fixedSize(horizontal: false, vertical: true)
            }
            switch phase {
            case .asking:
                HStack {
                    Spacer()
                    Button("Not now") {
                        notNow()
                        close()
                    }
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

    private func download() {
        guard let ollamaPath = OllamaModels.executablePath() else {
            phase = .failed("Ollama is not installed. Install it from ollama.com/download, then try again.")
            return
        }
        let models = missing
        let afterDownload = afterDownload
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
            switch afterDownload() {
            case .success(let message): result = .done(message)
            case .failure(let failure): result = .failed(failure.message)
            }
            DispatchQueue.main.async { phase = result }
        }
    }
}

/// Why the step after a download failed, in words for the window.
struct ModelDownloadFailure: Error, Equatable {
    let message: String
    init(_ message: String) { self.message = message }
}
