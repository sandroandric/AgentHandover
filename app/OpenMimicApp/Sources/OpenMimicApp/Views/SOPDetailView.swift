import SwiftUI
import AppKit

/// Displays the full detail of a selected SOP, including parsed SKILL.md content.
struct SOPDetailView: View {
    let sop: SOPEntry
    @ObservedObject var sopManager: SOPIndexManager

    @State private var skillContent: String?
    @State private var fileExists = false
    @State private var resolvedPath: URL?

    /// Returns candidate export paths in priority order.
    /// The first existing file wins in `loadSkillFile()`.
    private func candidatePaths() -> [URL] {
        let home = FileManager.default.homeDirectoryForCurrentUser
        return [
            home.appendingPathComponent(".openmimic/knowledge/procedures/\(sop.slug).json"),
            home.appendingPathComponent(".openclaw/workspace/memory/apprentice/sops/sop.\(sop.slug).md"),
            home.appendingPathComponent(".claude/skills/\(sop.slug)/SKILL.md"),
        ]
    }

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 0) {
                headerSection
                Divider().padding(.vertical, 16)

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
    }

    // MARK: - Header

    private var headerSection: some View {
        VStack(alignment: .leading, spacing: 12) {
            // Short title
            Text(sop.displayTitle)
                .font(.system(size: 20, weight: .semibold))
                .foregroundColor(.primary)
                .textSelection(.enabled)

            // Full description from title field
            if sop.title != sop.displayTitle {
                Text(sop.title)
                    .font(.system(size: 13))
                    .foregroundColor(.secondary)
                    .textSelection(.enabled)
                    .fixedSize(horizontal: false, vertical: true)
                    .lineSpacing(2)
            }

            // Tags
            if !sop.displayTags.isEmpty {
                HStack(spacing: 6) {
                    ForEach(sop.displayTags, id: \.self) { tag in
                        Text(tag)
                            .font(.system(size: 10, weight: .medium))
                            .foregroundColor(.secondary)
                            .padding(.horizontal, 7)
                            .padding(.vertical, 3)
                            .background(Color.primary.opacity(0.05))
                            .cornerRadius(4)
                    }
                }
            }

            // Metadata row
            HStack(spacing: 10) {
                StatusBadge(status: sop.status)

                // Lifecycle state badge
                Text(sop.lifecycleLabel)
                    .font(.caption2)
                    .fontWeight(.semibold)
                    .padding(.horizontal, 6)
                    .padding(.vertical, 2)
                    .background(sop.lifecycleColor.opacity(0.15))
                    .foregroundColor(sop.lifecycleColor)
                    .clipShape(Capsule())

                if sop.confidence > 0 {
                    Text(String(format: "%.0f%%", sop.confidence * 100))
                        .font(.system(size: 11))
                        .foregroundColor(.secondary)
                }

                Text("·")
                    .foregroundColor(.secondary.opacity(0.3))

                HStack(spacing: 3) {
                    Image(systemName: sop.sourceIcon)
                        .font(.system(size: 10))
                    Text(sop.sourceLabel)
                        .font(.system(size: 11))
                }
                .foregroundColor(.secondary)

                Spacer()

                Text(sop.relativeTime)
                    .font(.system(size: 11))
                    .foregroundColor(.secondary.opacity(0.5))
            }

            // Action bar
            actionBar
        }
    }

    private var actionBar: some View {
        HStack(spacing: 8) {
            Button(action: {
                sopManager.approveSOP(sop)
            }) {
                Label("Approve", systemImage: "checkmark.circle")
                    .font(.system(size: 12, weight: .medium))
            }
            .buttonStyle(.bordered)
            .controlSize(.small)
            .disabled(sop.status == "approved")

            // Promote lifecycle button
            if sop.canPromote, let nextState = sop.nextLifecycleState {
                Button(action: {
                    sopManager.promoteProcedure(sop, toState: nextState)
                }) {
                    Label("Promote to \(SOPEntry.lifecycleLabelFor(nextState))", systemImage: "arrow.up.circle")
                        .font(.system(size: 12, weight: .medium))
                }
                .buttonStyle(.borderedProminent)
                .tint(.blue)
                .controlSize(.small)
            }

            Button(action: {
                sopManager.rejectSOP(sop)
            }) {
                Label("Reject", systemImage: "xmark.circle")
                    .font(.system(size: 12, weight: .medium))
            }
            .buttonStyle(.bordered)
            .controlSize(.small)
            .disabled(sop.status == "rejected")

            Button(action: openInEditor) {
                Label("Open in Editor", systemImage: "square.and.pencil")
                    .font(.system(size: 12, weight: .medium))
            }
            .buttonStyle(.bordered)
            .controlSize(.small)
            .disabled(!fileExists)

            Spacer()
        }
        .padding(.top, 2)
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

        return VStack(alignment: .leading, spacing: 16) {
            if let json = parsed {
                // Description / goal
                if let desc = json["description"] as? String ?? json["goal"] as? String, !desc.isEmpty {
                    Text(desc)
                        .font(.system(size: 13))
                        .foregroundColor(.primary.opacity(0.8))
                        .textSelection(.enabled)
                        .fixedSize(horizontal: false, vertical: true)
                        .lineSpacing(3)
                }

                // Steps
                if let steps = json["steps"] as? [[String: Any]], !steps.isEmpty {
                    VStack(alignment: .leading, spacing: 10) {
                        sectionHeader("Steps", icon: "list.number")
                        ForEach(Array(steps.enumerated()), id: \.offset) { index, step in
                            HStack(alignment: .top, spacing: 10) {
                                Text("\(index + 1)")
                                    .font(.system(size: 11, weight: .semibold, design: .rounded))
                                    .foregroundColor(.secondary)
                                    .frame(width: 22, height: 22)
                                    .background(Color.primary.opacity(0.06))
                                    .cornerRadius(11)

                                VStack(alignment: .leading, spacing: 4) {
                                    if let app = step["app"] as? String {
                                        Text(app)
                                            .font(.system(size: 10, weight: .medium))
                                            .foregroundColor(.secondary)
                                            .padding(.horizontal, 6)
                                            .padding(.vertical, 2)
                                            .background(Color.primary.opacity(0.04))
                                            .cornerRadius(4)
                                    }
                                    Text(step["action"] as? String ?? step["description"] as? String ?? "")
                                        .font(.system(size: 12))
                                        .foregroundColor(.primary.opacity(0.85))
                                        .textSelection(.enabled)
                                        .fixedSize(horizontal: false, vertical: true)
                                        .lineSpacing(2)
                                }
                            }
                            .padding(.horizontal, 12)
                            .padding(.vertical, 10)
                            .frame(maxWidth: .infinity, alignment: .leading)
                            .background(Color.primary.opacity(0.02))
                            .cornerRadius(8)
                            .overlay(
                                RoundedRectangle(cornerRadius: 8)
                                    .stroke(Color.primary.opacity(0.05), lineWidth: 1)
                            )
                        }
                    }
                }

                // Preconditions / prerequisites
                if let preconds = json["preconditions"] as? [String] ?? json["prerequisites"] as? [String], !preconds.isEmpty {
                    VStack(alignment: .leading, spacing: 8) {
                        sectionHeader("Prerequisites", icon: "checkmark.circle")
                        VStack(alignment: .leading, spacing: 5) {
                            ForEach(preconds, id: \.self) { item in
                                HStack(alignment: .top, spacing: 8) {
                                    Circle()
                                        .fill(Color.primary.opacity(0.2))
                                        .frame(width: 5, height: 5)
                                        .padding(.top, 5)
                                    Text(item)
                                        .font(.system(size: 12))
                                        .foregroundColor(.primary.opacity(0.8))
                                        .textSelection(.enabled)
                                }
                            }
                        }
                        .padding(.horizontal, 14)
                        .padding(.vertical, 10)
                        .background(Color.primary.opacity(0.025))
                        .cornerRadius(8)
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
                    VStack(alignment: .leading, spacing: 6) {
                        sectionHeader("Metadata", icon: "info.circle")
                        ForEach(metaItems, id: \.0) { label, value in
                            metadataRow(label: label, value: value)
                        }
                    }
                    .padding(12)
                    .background(Color.primary.opacity(0.025))
                    .cornerRadius(8)
                }
            } else {
                // JSON parse failed — show raw content
                Text(raw)
                    .font(.system(size: 12, design: .monospaced))
                    .foregroundColor(.primary.opacity(0.7))
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
                    .foregroundColor(.primary.opacity(0.8))
                    .textSelection(.enabled)
                    .fixedSize(horizontal: false, vertical: true)
                    .lineSpacing(3)
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
            bulletSection(title: section.title, items: section.lines, icon: "checkmark.circle")
        case .steps:
            stepsSection(section)
        case .successCriteria:
            bulletSection(title: section.title, items: section.lines, icon: "target")
        case .commonErrors:
            errorSection(section)
        case .other:
            genericSection(section)
        }
    }

    // MARK: - Section renderers

    private func argumentsSection(_ section: ParsedSection) -> some View {
        VStack(alignment: .leading, spacing: 8) {
            sectionHeader(section.title, icon: "slider.horizontal.3")

            VStack(alignment: .leading, spacing: 4) {
                ForEach(section.lines, id: \.self) { line in
                    let cleaned = cleanMarkdown(line
                        .trimmingCharacters(in: .whitespaces)
                        .replacingOccurrences(of: "- ", with: "")
                        .replacingOccurrences(of: "* ", with: ""))
                    HStack(alignment: .top, spacing: 8) {
                        Text("·")
                            .font(.system(size: 13, weight: .bold))
                            .foregroundColor(.secondary.opacity(0.5))
                            .padding(.top, 0)
                        Text(cleaned)
                            .font(.system(size: 12))
                            .foregroundColor(.primary.opacity(0.8))
                            .textSelection(.enabled)
                    }
                }
            }
            .padding(.horizontal, 14)
            .padding(.vertical, 10)
            .background(Color.primary.opacity(0.025))
            .cornerRadius(8)
        }
    }

    private func bulletSection(title: String, items: [String], icon: String) -> some View {
        VStack(alignment: .leading, spacing: 8) {
            sectionHeader(title, icon: icon)

            VStack(alignment: .leading, spacing: 5) {
                ForEach(items, id: \.self) { line in
                    let cleaned = cleanMarkdown(line
                        .trimmingCharacters(in: .whitespaces)
                        .replacingOccurrences(of: "- ", with: "")
                        .replacingOccurrences(of: "* ", with: ""))
                    HStack(alignment: .top, spacing: 8) {
                        Circle()
                            .fill(Color.primary.opacity(0.2))
                            .frame(width: 5, height: 5)
                            .padding(.top, 5)
                        Text(cleaned)
                            .font(.system(size: 12))
                            .foregroundColor(.primary.opacity(0.8))
                            .textSelection(.enabled)
                    }
                }
            }
            .padding(.horizontal, 14)
            .padding(.vertical, 10)
            .background(Color.primary.opacity(0.025))
            .cornerRadius(8)
        }
    }

    private func errorSection(_ section: ParsedSection) -> some View {
        VStack(alignment: .leading, spacing: 8) {
            sectionHeader(section.title, icon: "exclamationmark.triangle")

            VStack(alignment: .leading, spacing: 5) {
                ForEach(section.lines, id: \.self) { line in
                    let cleaned = cleanErrorLine(line)
                    if !cleaned.isEmpty {
                        HStack(alignment: .top, spacing: 8) {
                            Circle()
                                .fill(Color.primary.opacity(0.2))
                                .frame(width: 5, height: 5)
                                .padding(.top, 5)
                            Text(cleaned)
                                .font(.system(size: 12))
                                .foregroundColor(.primary.opacity(0.8))
                                .textSelection(.enabled)
                        }
                    }
                }
            }
            .padding(.horizontal, 14)
            .padding(.vertical, 10)
            .background(Color.primary.opacity(0.025))
            .cornerRadius(8)
        }
    }

    private func stepsSection(_ section: ParsedSection) -> some View {
        VStack(alignment: .leading, spacing: 10) {
            sectionHeader(section.title, icon: "list.number")

            VStack(alignment: .leading, spacing: 6) {
                let steps = parseSteps(section.lines)
                ForEach(Array(steps.enumerated()), id: \.offset) { index, step in
                    HStack(alignment: .top, spacing: 10) {
                        // Step number
                        Text("\(index + 1)")
                            .font(.system(size: 11, weight: .semibold, design: .rounded))
                            .foregroundColor(.secondary)
                            .frame(width: 22, height: 22)
                            .background(Color.primary.opacity(0.06))
                            .cornerRadius(11)

                        VStack(alignment: .leading, spacing: 4) {
                            // App badge if present
                            if let app = step.app {
                                Text(app)
                                    .font(.system(size: 10, weight: .medium))
                                    .foregroundColor(.secondary)
                                    .padding(.horizontal, 6)
                                    .padding(.vertical, 2)
                                    .background(Color.primary.opacity(0.04))
                                    .cornerRadius(4)
                            }

                            Text(step.action)
                                .font(.system(size: 12))
                                .foregroundColor(.primary.opacity(0.85))
                                .textSelection(.enabled)
                                .fixedSize(horizontal: false, vertical: true)
                                .lineSpacing(2)

                            if let verify = step.verify {
                                Text(verify)
                                    .font(.system(size: 11))
                                    .italic()
                                    .foregroundColor(.secondary.opacity(0.7))
                                    .textSelection(.enabled)
                                    .padding(.top, 2)
                            }
                        }
                    }
                    .padding(.horizontal, 12)
                    .padding(.vertical, 10)
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .background(Color.primary.opacity(0.02))
                    .cornerRadius(8)
                    .overlay(
                        RoundedRectangle(cornerRadius: 8)
                            .stroke(Color.primary.opacity(0.05), lineWidth: 1)
                    )
                }
            }
        }
    }

    private func genericSection(_ section: ParsedSection) -> some View {
        VStack(alignment: .leading, spacing: 8) {
            sectionHeader(section.title, icon: "doc.text")

            VStack(alignment: .leading, spacing: 4) {
                ForEach(section.lines, id: \.self) { line in
                    Text(cleanMarkdown(line))
                        .font(.system(size: 12))
                        .foregroundColor(.primary.opacity(0.8))
                        .textSelection(.enabled)
                }
            }
            .padding(.horizontal, 14)
            .padding(.vertical, 10)
            .background(Color.primary.opacity(0.025))
            .cornerRadius(8)
        }
    }

    private func sectionHeader(_ title: String, icon: String) -> some View {
        HStack(spacing: 6) {
            Image(systemName: icon)
                .font(.system(size: 12))
                .foregroundColor(.secondary)
            Text(title)
                .font(.system(size: 13, weight: .semibold))
                .foregroundColor(.primary.opacity(0.7))
        }
    }

    // MARK: - Not exported fallback

    private var notExportedView: some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack(spacing: 8) {
                Image(systemName: "doc.questionmark")
                    .font(.system(size: 18))
                    .foregroundColor(.secondary.opacity(0.4))
                Text("SOP not yet exported")
                    .font(.system(size: 14, weight: .medium))
                    .foregroundColor(.secondary)
            }

            Text("This SOP has not been exported yet. Approve it to trigger export.")
                .font(.system(size: 12))
                .foregroundColor(.secondary.opacity(0.6))

            Divider().padding(.vertical, 4)

            VStack(alignment: .leading, spacing: 6) {
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
            .padding(12)
            .background(Color.primary.opacity(0.025))
            .cornerRadius(8)
        }
    }

    private func metadataRow(label: String, value: String) -> some View {
        HStack(alignment: .top) {
            Text(label)
                .font(.system(size: 11, weight: .medium))
                .foregroundColor(.secondary)
                .frame(width: 70, alignment: .trailing)
            Text(value)
                .font(.system(size: 11, design: .monospaced))
                .foregroundColor(.primary.opacity(0.7))
                .textSelection(.enabled)
        }
    }

    // MARK: - Footer

    private var footerSection: some View {
        VStack {
            Divider().padding(.top, 24)
            Text("Generated by OpenMimic")
                .font(.system(size: 10))
                .foregroundColor(.secondary.opacity(0.4))
                .padding(.top, 8)
        }
    }

    // MARK: - File loading

    private func loadSkillFile() {
        let fm = FileManager.default
        for candidate in candidatePaths() {
            if fm.fileExists(atPath: candidate.path) {
                fileExists = true
                resolvedPath = candidate
                skillContent = try? String(contentsOf: candidate, encoding: .utf8)
                return
            }
        }
        fileExists = false
        resolvedPath = nil
        skillContent = nil
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
                // Skip the **Arguments:** line — it's handled in the arguments section
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
