import SwiftUI

/// Detailed status view accessible from the menu bar.
struct StatusDetailView: View {
    @EnvironmentObject var appState: AppState

    var body: some View {
        VStack(alignment: .leading, spacing: 16) {
            // Daemon section
            GroupBox("Daemon") {
                if let daemon = appState.daemonStatus {
                    VStack(alignment: .leading, spacing: 4) {
                        DetailRow(label: "PID", value: "\(daemon.pid)")
                        DetailRow(label: "Version", value: daemon.version)
                        DetailRow(label: "Uptime", value: formatUptime(daemon.uptime_seconds))
                        DetailRow(label: "Events Today", value: "\(daemon.events_today)")
                        DetailRow(label: "DB Path", value: daemon.db_path)

                        HStack(spacing: 8) {
                            PermissionBadge(
                                name: "Accessibility",
                                granted: appState.accessibilityGranted
                            )
                            PermissionBadge(
                                name: "Screen Recording",
                                granted: appState.screenRecordingGranted
                            )
                        }
                        .padding(.top, 4)
                    }
                } else {
                    Text("Not running")
                        .foregroundColor(.secondary)
                }
            }

            // Worker section
            GroupBox("Worker") {
                if let worker = appState.workerStatus {
                    VStack(alignment: .leading, spacing: 4) {
                        DetailRow(label: "PID", value: "\(worker.pid)")
                        DetailRow(label: "Version", value: worker.version)
                        DetailRow(label: "Events Processed", value: "\(worker.events_processed_today)")
                        DetailRow(label: "SOPs Generated", value: "\(worker.sops_generated)")
                        DetailRow(label: "Errors", value: "\(worker.consecutive_errors)")

                        if let ms = worker.last_pipeline_duration_ms {
                            DetailRow(label: "Last Pipeline", value: "\(ms)ms")
                        }

                        HStack(spacing: 8) {
                            FeatureBadge(name: "VLM", available: worker.vlm_available)
                            FeatureBadge(name: "SOP Inducer", available: worker.sop_inducer_available)
                        }
                        .padding(.top, 4)
                    }
                } else {
                    Text("Not running")
                        .foregroundColor(.secondary)
                }
            }
        }
        .padding()
    }

    private func formatUptime(_ seconds: UInt64) -> String {
        let hours = seconds / 3600
        let minutes = (seconds % 3600) / 60
        if hours > 0 {
            return "\(hours)h \(minutes)m"
        }
        return "\(minutes)m"
    }
}

struct DetailRow: View {
    let label: String
    let value: String

    var body: some View {
        HStack {
            Text(label)
                .font(.caption)
                .foregroundColor(.secondary)
                .frame(width: 120, alignment: .trailing)
            Text(value)
                .font(.caption)
                .fontWeight(.medium)
                .lineLimit(1)
                .truncationMode(.middle)
            Spacer()
        }
    }
}

struct PermissionBadge: View {
    let name: String
    let granted: Bool

    var body: some View {
        HStack(spacing: 3) {
            Image(systemName: granted ? "checkmark.circle.fill" : "xmark.circle.fill")
                .foregroundColor(granted ? .green : .red)
                .font(.caption2)
            Text(name)
                .font(.caption2)
        }
    }
}

struct FeatureBadge: View {
    let name: String
    let available: Bool

    var body: some View {
        HStack(spacing: 3) {
            Image(systemName: available ? "checkmark.circle.fill" : "minus.circle")
                .foregroundColor(available ? .green : .secondary)
                .font(.caption2)
            Text(name)
                .font(.caption2)
        }
    }
}
