import SwiftUI
import AppKit

/// Displays the full detail of a selected SOP, including parsed SKILL.md content.
struct SOPDetailView: View {
    let sop: SOPEntry
    @ObservedObject var sopManager: SOPIndexManager

    @State private var skillContent: String?
    @State private var fileExists = false
    @State private var resolvedPath: URL?

    // Contra design tokens
    private let darkNavy = Color(red: 0.09, green: 0.10, blue: 0.12)
    private let warmOrange = Color(red: 0.92, green: 0.57, blue: 0.20)
    private let goldenYellow = Color(red: 1.0, green: 0.74, blue: 0.07)
    private let warmCream = Color(red: 1.0, green: 0.96, blue: 0.88)
    private let lightGray = Color(red: 0.96, green: 0.96, blue: 0.96)
    private let brightGreen = Color(red: 0.18, green: 0.80, blue: 0.34)
    private let cardRadius: CGFloat = 14
    private let contraBorder: CGFloat = 1.5

    /// Returns candidate export paths in priority order.
    /// The first existing file wins in `loadSkillFile()`.
    private func candidatePaths() -> [URL] {
        let home = FileManager.default.homeDirectoryForCurrentUser
        return [
            home.appendingPathComponent(".agenthandover/knowledge/procedures/\(sop.slug).json"),
            home.appendingPathComponent(".openclaw/workspace/memory/apprentice/sops/sop.\(sop.slug).md"),
            home.appendingPathComponent(".claude/skills/\(sop.slug)/SKILL.md"),
        ]
    }

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 0) {
                headerSection
                Rectangle()
                    .fill(darkNavy.opacity(0.08))
                    .frame(height: 1)
                    .padding(.vertical, 20)

                if fileExists, let content = skillContent {
                    if resolvedPath?.pathExtension == "json" {
                        jsonContent(content)
                    } else {
                        parsedContent(content)
                    }
                } else {
                    notExportedView
                }

                footerSection
            }
            .padding(28)
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .topLeading)
        .onAppear { loadSkillFile() }
        .onChange(of: sop.slug) { _ in loadSkillFile() }
        .onChange(of: sop.sop_id) { _ in loadSkillFile() }
        .id(sop.slug)  // Force view recreation when slug changes
    }

    // MARK: - Header

    private var headerSection: some View {
        VStack(alignment: .leading, spacing: 14) {
            // Short title
            Text(sop.displayTitle)
                .font(.system(size: 22, weight: .bold, design: .rounded))
                .foregroundColor(darkNavy)
                .textSelection(.enabled)

            // Full description from title field
            if sop.title != sop.displayTitle {
                Text(sop.title)
                    .font(.system(size: 13))
                    .foregroundColor(darkNavy.opacity(0.6))
                    .textSelection(.enabled)
                    .fixedSize(horizontal: false, vertical: true)
                    .lineSpacing(3)
            }

            // Tags
            if !sop.displayTags.isEmpty {
                HStack(spacing: 6) {
                    ForEach(sop.displayTags, id: \.self) { tag in
                        Text(tag)
                            .font(.system(size: 10, weight: .medium))
                            .foregroundColor(darkNavy.opacity(0.6))
                            .padding(.horizontal, 8)
                            .padding(.vertical, 3)
                            .background(
                                RoundedRectangle(cornerRadius: 6)
                                    .fill(lightGray)
                            )
                            .overlay(
                                RoundedRectangle(cornerRadius: 6)
                                    .stroke(darkNavy.opacity(0.12), lineWidth: 1)
                            )
                    }
                }
            }

            // Metadata row
            HStack(spacing: 10) {
                StatusBadge(status: sop.status)

                // Lifecycle state badge
                Text(sop.lifecycleLabel)
                    .font(.system(size: 10, weight: .bold))
                    .padding(.horizontal, 8)
                    .padding(.vertical, 3)
                    .background(sop.lifecycleColor.opacity(0.1))
                    .foregroundColor(sop.lifecycleColor)
                    .clipShape(Capsule())
                    .overlay(
                        Capsule()
                            .stroke(darkNavy.opacity(0.08), lineWidth: 1)
                    )

                if sop.confidence > 0 {
                    Text(String(format: "%.0f%%", sop.confidence * 100))
                        .font(.system(size: 11))
                        .foregroundColor(darkNavy.opacity(0.5))
                }

                Text("\u{00B7}")
                    .foregroundColor(darkNavy.opacity(0.2))

                HStack(spacing: 4) {
                    Image(systemName: sop.sourceIcon)
                        .font(.system(size: 10))
                    Text(sop.sourceLabel)
                        .font(.system(size: 11))
                }
                .foregroundColor(darkNavy.opacity(0.5))

                Spacer()

                Text(sop.relativeTime)
                    .font(.system(size: 11))
                    .foregroundColor(darkNavy.opacity(0.35))
            }

            // Action bar
            actionBar
        }
    }

    private var isAgentReady: Bool {
        sop.lifecycleState == "agent_ready" && sop.status == "approved"
    }

    private var actionBar: some View {
        HStack(spacing: 10) {
            if isAgentReady {
                // Agent Ready badge
                HStack(spacing: 5) {
                    Image(systemName: "checkmark.seal.fill")
                        .font(.system(size: 12))
                    Text("Agent Ready")
                        .font(.system(size: 12, weight: .bold))
                }
                .padding(.horizontal, 14)
                .padding(.vertical, 7)
                .background(
                    RoundedRectangle(cornerRadius: 8)
                        .fill(brightGreen)
                )
                .foregroundColor(.white)
                .overlay(
                    RoundedRectangle(cornerRadius: 8)
                        .stroke(brightGreen, lineWidth: contraBorder)
                )
            } else {
                // Approve for Agents - green filled
                Button(action: {
                    sopManager.approveForAgents(sop)
                }) {
                    HStack(spacing: 5) {
                        Image(systemName: "checkmark.circle.fill")
                            .font(.system(size: 12))
                        Text("Approve for Agents")
                            .font(.system(size: 12, weight: .bold))
                    }
                    .padding(.horizontal, 14)
                    .padding(.vertical, 7)
                    .background(
                        RoundedRectangle(cornerRadius: 8)
                            .fill(brightGreen)
                    )
                    .foregroundColor(.white)
                    .overlay(
                        RoundedRectangle(cornerRadius: 8)
                            .stroke(brightGreen, lineWidth: contraBorder)
                    )
                }
                .buttonStyle(.plain)
            }

            // Reject button - red outline
            Button(action: {
                sopManager.rejectSOP(sop)
            }) {
                HStack(spacing: 5) {
                    Image(systemName: "xmark.circle")
                        .font(.system(size: 12))
                    Text("Reject")
                        .font(.system(size: 12, weight: .medium))
                }
                .padding(.horizontal, 14)
                .padding(.vertical, 7)
                .foregroundColor(sop.status == "rejected" ? Color.red.opacity(0.4) : .red)
                .overlay(
                    RoundedRectangle(cornerRadius: 8)
                        .stroke(Color.red.opacity(0.3), lineWidth: contraBorder)
                )
            }
            .buttonStyle(.plain)
            .disabled(sop.status == "rejected")

            // Open in Editor - subtle
            Button(action: openInEditor) {
                HStack(spacing: 5) {
                    Image(systemName: "square.and.pencil")
                        .font(.system(size: 12))
                    Text("Open in Editor")
                        .font(.system(size: 12, weight: .medium))
                }
                .padding(.horizontal, 12)
                .padding(.vertical, 7)
                .foregroundColor(darkNavy.opacity(0.5))
                .overlay(
                    RoundedRectangle(cornerRadius: 8)
                        .stroke(darkNavy.opacity(0.12), lineWidth: contraBorder)
                )
            }
            .buttonStyle(.plain)
            .disabled(!fileExists)

            Spacer()
        }
        .padding(.top, 4)
    }

    // MARK: - Parsed Content

    /// Renders a JSON SOP by extracting key fields and displaying them.
    private func jsonContent(_ raw: String) -> some View {
        let parsed: [String: Any]? = {
            guard let data = raw.data(using: .utf8),
                  let obj = try? JSONSerialization.jsonObject(with: data) as? [String: Any]
            else { return nil }
            return obj
        }()

        return VStack(alignment: .leading, spacing: 20) {
            if let json = parsed {
                // Description / goal
                if let desc = json["description"] as? String ?? json["goal"] as? String, !desc.isEmpty {
                    Text(desc)
                        .font(.system(size: 13))
                        .foregroundColor(darkNavy.opacity(0.8))
                        .textSelection(.enabled)
                        .fixedSize(horizontal: false, vertical: true)
                        .lineSpacing(4)
                }

                // Steps
                if let steps = json["steps"] as? [[String: Any]], !steps.isEmpty {
                    VStack(alignment: .leading, spacing: 10) {
                        sectionHeader("Steps", icon: "list.number", color: warmOrange)
                        VStack(alignment: .leading, spacing: 6) {
                            ForEach(Array(steps.enumerated()), id: \.offset) { index, step in
                                HStack(alignment: .top, spacing: 10) {
                                    // Numbered circle - Contra style
                                    Text("\(index + 1)")
                                        .font(.system(size: 11, weight: .bold, design: .rounded))
                                        .foregroundColor(.white)
                                        .frame(width: 24, height: 24)
                                        .background(
                                            Circle()
                                                .fill(warmOrange)
                                        )

                                    VStack(alignment: .leading, spacing: 3) {
                                        if let app = step["app"] as? String, !app.trimmingCharacters(in: .whitespaces).isEmpty {
                                            Text(app)
                                                .font(.system(size: 10, weight: .medium))
                                                .foregroundColor(darkNavy.opacity(0.5))
                                                .padding(.horizontal, 7)
                                                .padding(.vertical, 2)
                                                .background(lightGray)
                                                .cornerRadius(5)
                                        }
                                        Text(step["action"] as? String ?? step["step"] as? String ?? step["description"] as? String ?? "")
                                            .font(.system(size: 12))
                                            .foregroundColor(darkNavy.opacity(0.85))
                                            .textSelection(.enabled)
                                            .fixedSize(horizontal: false, vertical: true)
                                            .lineSpacing(3)
                                    }
                                }
                            }
                        }
                        .padding(.horizontal, 14)
                        .padding(.vertical, 12)
                        .frame(maxWidth: .infinity, alignment: .leading)
                        .background(
                            RoundedRectangle(cornerRadius: cardRadius)
                                .fill(Color.white)
                        )
                        .overlay(
                            RoundedRectangle(cornerRadius: cardRadius)
                                .stroke(darkNavy.opacity(0.12), lineWidth: contraBorder)
                        )
                    }
                }

                // Preconditions / prerequisites
                if let preconds = json["preconditions"] as? [String] ?? json["prerequisites"] as? [String], !preconds.isEmpty {
                    VStack(alignment: .leading, spacing: 10) {
                        sectionHeader("Prerequisites", icon: "checkmark.circle", color: brightGreen)
                        VStack(alignment: .leading, spacing: 6) {
                            ForEach(preconds, id: \.self) { item in
                                HStack(alignment: .top, spacing: 8) {
                                    Image(systemName: "checkmark")
                                        .font(.system(size: 9, weight: .semibold))
                                        .foregroundColor(brightGreen)
                                        .frame(width: 16, height: 16)
                                    Text(item)
                                        .font(.system(size: 12))
                                        .foregroundColor(darkNavy.opacity(0.8))
                                        .textSelection(.enabled)
                                }
                            }
                        }
                        .padding(.horizontal, 14)
                        .padding(.vertical, 12)
                        .background(
                            RoundedRectangle(cornerRadius: cardRadius)
                                .fill(lightGray)
                        )
                        .overlay(
                            RoundedRectangle(cornerRadius: cardRadius)
                                .stroke(darkNavy.opacity(0.12), lineWidth: contraBorder)
                        )
                    }
                }

                // Metadata fields
                let metaKeys = ["trigger", "frequency", "confidence", "version"]
                let metaItems = metaKeys.compactMap { key -> (String, String)? in
                    if let val = json[key] {
                        return (key.capitalized, "\(val)")
                    }
                    return nil
                }
                if !metaItems.isEmpty {
                    VStack(alignment: .leading, spacing: 8) {
                        sectionHeader("Metadata", icon: "info.circle", color: darkNavy.opacity(0.5))
                        ForEach(metaItems, id: \.0) { label, value in
                            metadataRow(label: label, value: value)
                        }
                    }
                    .padding(14)
                    .background(
                        RoundedRectangle(cornerRadius: cardRadius)
                            .fill(lightGray)
                    )
                    .overlay(
                        RoundedRectangle(cornerRadius: cardRadius)
                            .stroke(darkNavy.opacity(0.12), lineWidth: contraBorder)
                    )
                }
            } else {
                // JSON parse failed - show raw content
                Text(raw)
                    .font(.system(size: 12, design: .monospaced))
                    .foregroundColor(darkNavy.opacity(0.7))
                    .textSelection(.enabled)
                    .fixedSize(horizontal: false, vertical: true)
            }
        }
    }

    private func parsedContent(_ content: String) -> some View {
        let sections = parseSections(content)
        return VStack(alignment: .leading, spacing: 24) {
            // Description (text before the first ## section, after frontmatter)
            if let desc = extractDescription(content), !desc.isEmpty {
                Text(desc)
                    .font(.system(size: 13))
                    .foregroundColor(darkNavy.opacity(0.8))
                    .textSelection(.enabled)
                    .fixedSize(horizontal: false, vertical: true)
                    .lineSpacing(4)
            }

            ForEach(sections, id: \.title) { section in
                renderSection(section)
            }
        }
    }

    @ViewBuilder
    private func renderSection(_ section: ParsedSection) -> some View {
        switch section.kind {
        case .arguments:
            argumentsSection(section)
        case .prerequisites:
            bulletSection(title: section.title, items: section.lines, icon: "checkmark.circle", color: brightGreen)
        case .steps:
            stepsSection(section)
        case .successCriteria:
            bulletSection(title: section.title, items: section.lines, icon: "target", color: warmOrange)
        case .commonErrors:
            errorSection(section)
        case .other:
            genericSection(section)
        }
    }

    // MARK: - Section renderers

    private func argumentsSection(_ section: ParsedSection) -> some View {
        VStack(alignment: .leading, spacing: 10) {
            sectionHeader(section.title, icon: "slider.horizontal.3", color: .purple)

            VStack(alignment: .leading, spacing: 5) {
                ForEach(section.lines, id: \.self) { line in
                    let cleaned = cleanMarkdown(line
                        .trimmingCharacters(in: .whitespaces)
                        .replacingOccurrences(of: "- ", with: "")
                        .replacingOccurrences(of: "* ", with: ""))
                    HStack(alignment: .top, spacing: 8) {
                        Image(systemName: "chevron.right")
                            .font(.system(size: 8, weight: .bold))
                            .foregroundColor(.purple.opacity(0.6))
                            .frame(width: 12)
                            .padding(.top, 4)
                        Text(cleaned)
                            .font(.system(size: 12))
                            .foregroundColor(darkNavy.opacity(0.8))
                            .textSelection(.enabled)
                    }
                }
            }
            .padding(.horizontal, 14)
            .padding(.vertical, 12)
            .background(
                RoundedRectangle(cornerRadius: cardRadius)
                    .fill(lightGray)
            )
            .overlay(
                RoundedRectangle(cornerRadius: cardRadius)
                    .stroke(darkNavy.opacity(0.12), lineWidth: contraBorder)
            )
        }
    }

    private func bulletSection(title: String, items: [String], icon: String, color: Color) -> some View {
        VStack(alignment: .leading, spacing: 10) {
            sectionHeader(title, icon: icon, color: color)

            VStack(alignment: .leading, spacing: 6) {
                ForEach(items, id: \.self) { line in
                    let cleaned = cleanMarkdown(line
                        .trimmingCharacters(in: .whitespaces)
                        .replacingOccurrences(of: "- ", with: "")
                        .replacingOccurrences(of: "* ", with: ""))
                    HStack(alignment: .top, spacing: 8) {
                        Image(systemName: "checkmark")
                            .font(.system(size: 9, weight: .semibold))
                            .foregroundColor(color)
                            .frame(width: 16, height: 16)
                        Text(cleaned)
                            .font(.system(size: 12))
                            .foregroundColor(darkNavy.opacity(0.8))
                            .textSelection(.enabled)
                    }
                }
            }
            .padding(.horizontal, 14)
            .padding(.vertical, 12)
            .background(
                RoundedRectangle(cornerRadius: cardRadius)
                    .fill(lightGray)
            )
            .overlay(
                RoundedRectangle(cornerRadius: cardRadius)
                    .stroke(darkNavy.opacity(0.12), lineWidth: contraBorder)
            )
        }
    }

    private func errorSection(_ section: ParsedSection) -> some View {
        VStack(alignment: .leading, spacing: 10) {
            sectionHeader(section.title, icon: "exclamationmark.triangle", color: .red)

            VStack(alignment: .leading, spacing: 6) {
                ForEach(section.lines, id: \.self) { line in
                    let cleaned = cleanErrorLine(line)
                    if !cleaned.isEmpty {
                        HStack(alignment: .top, spacing: 8) {
                            Image(systemName: "exclamationmark.circle")
                                .font(.system(size: 10))
                                .foregroundColor(.red.opacity(0.6))
                                .frame(width: 16, height: 16)
                            Text(cleaned)
                                .font(.system(size: 12))
                                .foregroundColor(darkNavy.opacity(0.8))
                                .textSelection(.enabled)
                        }
                    }
                }
            }
            .padding(.horizontal, 14)
            .padding(.vertical, 12)
            .background(
                RoundedRectangle(cornerRadius: cardRadius)
                    .fill(Color.red.opacity(0.03))
            )
            .overlay(
                RoundedRectangle(cornerRadius: cardRadius)
                    .stroke(Color.red.opacity(0.15), lineWidth: contraBorder)
            )
        }
    }

    private func stepsSection(_ section: ParsedSection) -> some View {
        VStack(alignment: .leading, spacing: 10) {
            sectionHeader(section.title, icon: "list.number", color: warmOrange)

            VStack(alignment: .leading, spacing: 6) {
                let steps = parseSteps(section.lines)
                ForEach(Array(steps.enumerated()), id: \.offset) { index, step in
                    HStack(alignment: .top, spacing: 10) {
                        // Numbered circle - Contra style
                        Text("\(index + 1)")
                            .font(.system(size: 11, weight: .bold, design: .rounded))
                            .foregroundColor(.white)
                            .frame(width: 24, height: 24)
                            .background(
                                Circle()
                                    .fill(warmOrange)
                            )

                        VStack(alignment: .leading, spacing: 3) {
                            // App badge - only show if non-empty
                            if let app = step.app, !app.trimmingCharacters(in: .whitespaces).isEmpty {
                                Text(app)
                                    .font(.system(size: 10, weight: .medium))
                                    .foregroundColor(darkNavy.opacity(0.5))
                                    .padding(.horizontal, 7)
                                    .padding(.vertical, 2)
                                    .background(lightGray)
                                    .cornerRadius(5)
                            }

                            Text(step.action)
                                .font(.system(size: 12))
                                .foregroundColor(darkNavy.opacity(0.85))
                                .textSelection(.enabled)
                                .fixedSize(horizontal: false, vertical: true)
                                .lineSpacing(3)

                            if let verify = step.verify {
                                Text(verify)
                                    .font(.system(size: 11))
                                    .italic()
                                    .foregroundColor(darkNavy.opacity(0.45))
                                    .textSelection(.enabled)
                                    .padding(.top, 1)
                            }
                        }
                    }
                }
            }
            .padding(.horizontal, 14)
            .padding(.vertical, 12)
            .frame(maxWidth: .infinity, alignment: .leading)
            .background(
                RoundedRectangle(cornerRadius: cardRadius)
                    .fill(Color.white)
            )
            .overlay(
                RoundedRectangle(cornerRadius: cardRadius)
                    .stroke(darkNavy.opacity(0.12), lineWidth: contraBorder)
            )
        }
    }

    private func genericSection(_ section: ParsedSection) -> some View {
        VStack(alignment: .leading, spacing: 10) {
            sectionHeader(section.title, icon: "doc.text", color: darkNavy.opacity(0.5))

            VStack(alignment: .leading, spacing: 5) {
                ForEach(section.lines, id: \.self) { line in
                    Text(cleanMarkdown(line))
                        .font(.system(size: 12))
                        .foregroundColor(darkNavy.opacity(0.8))
                        .textSelection(.enabled)
                }
            }
            .padding(.horizontal, 14)
            .padding(.vertical, 12)
            .background(
                RoundedRectangle(cornerRadius: cardRadius)
                    .fill(lightGray)
            )
            .overlay(
                RoundedRectangle(cornerRadius: cardRadius)
                    .stroke(darkNavy.opacity(0.12), lineWidth: contraBorder)
            )
        }
    }

    private func sectionHeader(_ title: String, icon: String, color: Color) -> some View {
        HStack(spacing: 8) {
            ZStack {
                RoundedRectangle(cornerRadius: 6)
                    .fill(color.opacity(0.12))
                    .frame(width: 24, height: 24)
                    .overlay(
                        RoundedRectangle(cornerRadius: 6)
                            .stroke(darkNavy.opacity(0.12), lineWidth: contraBorder)
                    )
                Image(systemName: icon)
                    .font(.system(size: 11))
                    .foregroundColor(color)
            }
            Text(title)
                .font(.system(size: 14, weight: .bold, design: .rounded))
                .foregroundColor(darkNavy.opacity(0.7))
                .tracking(0.3)
        }
    }

    // MARK: - Not exported fallback

    private var notExportedView: some View {
        VStack(alignment: .leading, spacing: 14) {
            HStack(spacing: 10) {
                ZStack {
                    RoundedRectangle(cornerRadius: 8)
                        .fill(warmCream)
                        .frame(width: 32, height: 32)
                        .overlay(
                            RoundedRectangle(cornerRadius: 8)
                                .stroke(darkNavy.opacity(0.12), lineWidth: contraBorder)
                        )
                    Image(systemName: "doc.questionmark")
                        .font(.system(size: 16))
                        .foregroundColor(warmOrange)
                }
                Text("SOP not yet exported")
                    .font(.system(size: 15, weight: .bold, design: .rounded))
                    .foregroundColor(darkNavy)
            }

            Text("This SOP has not been exported yet. Approve it to trigger export.")
                .font(.system(size: 13))
                .foregroundColor(darkNavy.opacity(0.5))
                .lineSpacing(3)

            Rectangle()
                .fill(darkNavy.opacity(0.08))
                .frame(height: 1)
                .padding(.vertical, 4)

            VStack(alignment: .leading, spacing: 8) {
                metadataRow(label: "SOP ID", value: sop.sop_id)
                metadataRow(label: "Slug", value: sop.slug)
                metadataRow(label: "Source", value: sop.sourceLabel)
                metadataRow(label: "Status", value: sop.status.capitalized)
                if sop.confidence > 0 {
                    metadataRow(label: "Confidence", value: String(format: "%.0f%%", sop.confidence * 100))
                }
                metadataRow(label: "Created", value: sop.created_at)
                if let reviewed = sop.reviewed_at {
                    metadataRow(label: "Reviewed", value: reviewed)
                }
            }
            .padding(14)
            .background(
                RoundedRectangle(cornerRadius: cardRadius)
                    .fill(lightGray)
            )
            .overlay(
                RoundedRectangle(cornerRadius: cardRadius)
                    .stroke(darkNavy.opacity(0.12), lineWidth: contraBorder)
            )
        }
    }

    private func metadataRow(label: String, value: String) -> some View {
        HStack(alignment: .top) {
            Text(label)
                .font(.system(size: 11, weight: .medium))
                .foregroundColor(darkNavy.opacity(0.5))
                .frame(width: 80, alignment: .trailing)
            Text(value)
                .font(.system(size: 11, design: .monospaced))
                .foregroundColor(darkNavy.opacity(0.7))
                .textSelection(.enabled)
        }
    }

    // MARK: - Footer

    private var footerSection: some View {
        VStack {
            Rectangle()
                .fill(darkNavy.opacity(0.08))
                .frame(height: 1)
                .padding(.top, 28)
            Text("Generated by AgentHandover")
                .font(.system(size: 10))
                .foregroundColor(darkNavy.opacity(0.3))
                .padding(.top, 8)
        }
    }

    // MARK: - File loading

    private func loadSkillFile() {
        // Clear stale state first to prevent showing old content
        fileExists = false
        resolvedPath = nil
        skillContent = nil

        let fm = FileManager.default
        for candidate in candidatePaths() {
            if fm.fileExists(atPath: candidate.path) {
                fileExists = true
                resolvedPath = candidate
                skillContent = try? String(contentsOf: candidate, encoding: .utf8)
                print("[SOPDetail] Loaded \(sop.slug) from \(candidate.lastPathComponent)")
                return
            }
        }
    }

    private func openInEditor() {
        guard let path = resolvedPath else { return }
        NSWorkspace.shared.open(path)
    }

    // MARK: - Text cleaning helpers

    /// Strip markdown syntax for plain text display.
    private func cleanMarkdown(_ text: String) -> String {
        var result = text
        // Remove bold markers
        result = result.replacingOccurrences(of: "**", with: "")
        // Remove inline code backticks
        result = result.replacingOccurrences(of: "`", with: "")
        // Remove italic underscores (single _ at word boundaries)
        // Only strip leading/trailing single underscores that wrap text
        if result.hasPrefix("_") && result.hasSuffix("_") && result.count > 2 {
            result = String(result.dropFirst().dropLast())
        }
        return result.trimmingCharacters(in: .whitespaces)
    }

    /// Clean common error lines that may contain raw Python dict syntax.
    private func cleanErrorLine(_ line: String) -> String {
        var text = line.trimmingCharacters(in: .whitespaces)
        // Remove bullet markers
        if text.hasPrefix("- ") { text = String(text.dropFirst(2)) }
        if text.hasPrefix("* ") { text = String(text.dropFirst(2)) }
        // Skip separator lines
        if text == "---" || text.isEmpty { return "" }

        // Detect raw Python dict format: **{'type'**: 'error_type', 'message': 'Some message'}
        // Extract just the message value
        if text.contains("'message':") || text.contains("\"message\":") {
            // Try to extract the message value
            if let msgRange = text.range(of: #"'message':\s*['\"](.+?)['\"]"#, options: .regularExpression) {
                let msgPart = text[msgRange]
                // Extract the value between quotes after 'message':
                if let valueStart = msgPart.range(of: #":\s*['\"]"#, options: .regularExpression) {
                    var value = String(msgPart[valueStart.upperBound...])
                    // Remove trailing quote
                    if value.hasSuffix("'") || value.hasSuffix("\"") {
                        value = String(value.dropLast())
                    }
                    // Also try to get the type
                    var errorType = ""
                    if let typeRange = text.range(of: #"'type'[^:]*:\s*['\"]([^'\"]+)['\"]"#, options: .regularExpression) {
                        let typePart = String(text[typeRange])
                        if let typeValueStart = typePart.range(of: #":\s*['\"]"#, options: .regularExpression) {
                            errorType = String(typePart[typeValueStart.upperBound...])
                            if errorType.hasSuffix("'") || errorType.hasSuffix("\"") {
                                errorType = String(errorType.dropLast())
                            }
                            errorType = errorType
                                .replacingOccurrences(of: "_", with: " ")
                                .capitalized
                        }
                    }
                    if !errorType.isEmpty {
                        return "\(errorType): \(value)"
                    }
                    return value
                }
            }
        }

        // Fallback: strip markdown
        return cleanMarkdown(text)
    }

    // MARK: - Parsing helpers

    struct ParsedSection: Identifiable {
        let title: String
        let lines: [String]
        let kind: SectionKind
        var id: String { title }
    }

    enum SectionKind {
        case arguments
        case prerequisites
        case steps
        case successCriteria
        case commonErrors
        case other
    }

    struct StepItem {
        let app: String?
        let action: String
        let verify: String?
    }

    private func classifySection(_ title: String) -> SectionKind {
        let lower = title.lowercased()
        if lower.contains("argument") || lower.contains("variable") || lower.contains("input") {
            return .arguments
        }
        if lower.contains("prerequisit") || lower.contains("before you start") || lower.contains("requirement") {
            return .prerequisites
        }
        if lower.contains("step") || lower.contains("procedure") || lower.contains("instruction") {
            return .steps
        }
        if lower.contains("success") || lower.contains("criteria") || lower.contains("outcome") || lower.contains("expected result") {
            return .successCriteria
        }
        if lower.contains("error") || lower.contains("troubleshoot") || lower.contains("common issue") || lower.contains("warning") {
            return .commonErrors
        }
        return .other
    }

    private func parseSections(_ content: String) -> [ParsedSection] {
        var sections: [ParsedSection] = []
        var currentTitle: String?
        var currentLines: [String] = []
        var pastFrontmatter = false
        var inFrontmatter = false

        for line in content.components(separatedBy: "\n") {
            if line.trimmingCharacters(in: .whitespaces) == "---" {
                if !pastFrontmatter && !inFrontmatter {
                    inFrontmatter = true
                    continue
                } else if inFrontmatter {
                    inFrontmatter = false
                    pastFrontmatter = true
                    continue
                }
            }
            if inFrontmatter { continue }

            if line.hasPrefix("## ") {
                if let title = currentTitle {
                    let filtered = currentLines.filter { !$0.trimmingCharacters(in: .whitespaces).isEmpty }
                    if !filtered.isEmpty {
                        sections.append(ParsedSection(
                            title: title,
                            lines: filtered,
                            kind: classifySection(title)
                        ))
                    }
                }
                currentTitle = String(line.dropFirst(3)).trimmingCharacters(in: .whitespaces)
                currentLines = []
            } else if line.hasPrefix("# ") {
                continue
            } else if currentTitle != nil {
                currentLines.append(line)
            }
        }

        if let title = currentTitle {
            let filtered = currentLines.filter { !$0.trimmingCharacters(in: .whitespaces).isEmpty }
            if !filtered.isEmpty {
                sections.append(ParsedSection(
                    title: title,
                    lines: filtered,
                    kind: classifySection(title)
                ))
            }
        }

        return sections
    }

    private func extractDescription(_ content: String) -> String? {
        var pastFrontmatter = false
        var inFrontmatter = false
        var descriptionLines: [String] = []
        var pastTitle = false

        for line in content.components(separatedBy: "\n") {
            if line.trimmingCharacters(in: .whitespaces) == "---" {
                if !pastFrontmatter && !inFrontmatter {
                    inFrontmatter = true
                    continue
                } else if inFrontmatter {
                    inFrontmatter = false
                    pastFrontmatter = true
                    continue
                }
            }
            if inFrontmatter { continue }

            if line.hasPrefix("# ") && !pastTitle {
                pastTitle = true
                continue
            }

            if line.hasPrefix("## ") { break }

            if pastFrontmatter || pastTitle {
                // Skip the **Arguments:** line - it's handled in the arguments section
                let trimmed = line.trimmingCharacters(in: .whitespaces)
                if trimmed.hasPrefix("**Arguments") || trimmed.hasPrefix("Arguments:") {
                    continue
                }
                descriptionLines.append(line)
            }
        }

        let result = descriptionLines
            .joined(separator: "\n")
            .trimmingCharacters(in: .whitespacesAndNewlines)
        return result.isEmpty ? nil : cleanMarkdown(result)
    }

    private func parseSteps(_ lines: [String]) -> [StepItem] {
        var steps: [StepItem] = []
        var currentAction: String?
        var currentVerify: String?
        var currentApp: String?

        for line in lines {
            let trimmed = line.trimmingCharacters(in: .whitespaces)

            let isNumberedStep = trimmed.range(of: #"^\d+[\.\)]\s+"#, options: .regularExpression) != nil
            let isBulletStep = trimmed.hasPrefix("- ") || trimmed.hasPrefix("* ")

            if isNumberedStep || isBulletStep {
                // Save previous step
                if let action = currentAction {
                    steps.append(StepItem(app: currentApp, action: cleanMarkdown(action), verify: currentVerify.map { cleanMarkdown($0) }))
                }
                // Start new step
                var rawAction: String
                if isNumberedStep {
                    rawAction = trimmed.replacingOccurrences(
                        of: #"^\d+[\.\)]\s+"#, with: "",
                        options: .regularExpression
                    )
                } else {
                    rawAction = String(trimmed.dropFirst(2))
                }

                // Extract app name from "In **AppName**, ..." pattern
                currentApp = nil
                if let appMatch = rawAction.range(of: #"^In \*\*([^*]+)\*\*,?\s*"#, options: .regularExpression) {
                    let appPart = rawAction[appMatch]
                    // Extract just the app name between ** **
                    if let starStart = appPart.range(of: "**"),
                       let starEnd = appPart[starStart.upperBound...].range(of: "**") {
                        currentApp = String(appPart[starStart.upperBound..<starEnd.lowerBound])
                    }
                    rawAction = String(rawAction[appMatch.upperBound...])
                }

                // Note: "open `location`." lines are kept as-is (cleaned later via cleanMarkdown)

                currentAction = rawAction
                currentVerify = nil
            } else if trimmed.hasPrefix("_") && trimmed.hasSuffix("_") && currentAction != nil {
                currentVerify = String(trimmed.dropFirst().dropLast())
            } else if trimmed.lowercased().hasPrefix("verify:") && currentAction != nil {
                currentVerify = String(trimmed.dropFirst(7)).trimmingCharacters(in: .whitespaces)
            } else if !trimmed.isEmpty, currentAction != nil {
                currentAction = (currentAction ?? "") + " " + trimmed
            }
        }

        if let action = currentAction {
            steps.append(StepItem(app: currentApp, action: cleanMarkdown(action), verify: currentVerify.map { cleanMarkdown($0) }))
        }

        return steps
    }
}
