import Foundation

/// The local Ollama's models through its CLI: where it is, which models
/// it has pulled, and pulling one.  Onboarding's download and the model
/// upgrade window share these.
enum OllamaModels {

    /// The `ollama` executable: a known install location, else `which`.
    static func executablePath() -> String? {
        let paths = [
            "/usr/local/bin/ollama",
            "/opt/homebrew/bin/ollama",
            "/Applications/Ollama.app/Contents/Resources/ollama",
        ]
        if let knownPath = paths.first(where: { FileManager.default.fileExists(atPath: $0) }) {
            return knownPath
        }
        let output = ServiceController.shell("/usr/bin/which", args: ["ollama"])
            .trimmingCharacters(in: .whitespacesAndNewlines)
        return output.isEmpty ? nil : output
    }

    /// The models `ollama list` shows, by `canonical` name; a failure when
    /// the service does not answer.  Blocking.
    static func installed(ollamaPath: String) -> Result<Set<String>, Error> {
        let process = Process()
        process.executableURL = URL(fileURLWithPath: ollamaPath)
        process.arguments = ["list"]
        let outputPipe = Pipe()
        process.standardOutput = outputPipe
        process.standardError = outputPipe
        do {
            try process.run()
            // Read before waiting: a full pipe blocks the child.
            let output = String(data: outputPipe.fileHandleForReading.readDataToEndOfFile(), encoding: .utf8) ?? ""
            process.waitUntilExit()
            guard process.terminationStatus == 0 else {
                return .failure(NSError(domain: "OllamaList", code: Int(process.terminationStatus)))
            }
            return .success(parseList(output))
        } catch {
            return .failure(error)
        }
    }

    /// The model names of `ollama list`'s output (its first column, after
    /// the header), by `canonical` name.
    static func parseList(_ output: String) -> Set<String> {
        Set(
            output.components(separatedBy: "\n").dropFirst().compactMap { line -> String? in
                let trimmed = line.trimmingCharacters(in: .whitespacesAndNewlines)
                guard let name = trimmed.components(separatedBy: .whitespaces).first,
                      !name.isEmpty else { return nil }
                return canonical(name)
            }
        )
    }

    /// `name` without a `:latest` suffix (how `ollama list` and a config
    /// may name the same model).
    static func canonical(_ name: String) -> String {
        let trimmed = name.trimmingCharacters(in: .whitespacesAndNewlines)
        return trimmed.hasSuffix(":latest") ? String(trimmed.dropLast(":latest".count)) : trimmed
    }

    /// The `models` not in `installed` (by `canonical` name), in order.
    static func missing(_ models: [String], installed: Set<String>) -> [String] {
        models.filter { !installed.contains(canonical($0)) }
    }

    /// `ollama pull model`, handing each chunk's last output line to
    /// `progress`.  Blocking.  Returns nil on success, else why it failed.
    static func pull(
        _ model: String, ollamaPath: String, progress: @escaping (String) -> Void
    ) -> String? {
        let process = Process()
        process.executableURL = URL(fileURLWithPath: ollamaPath)
        process.arguments = ["pull", model]
        let pipe = Pipe()
        process.standardOutput = pipe
        process.standardError = pipe
        do {
            try process.run()
        } catch {
            return "Failed to run ollama: \(error.localizedDescription)"
        }
        // Read until the child closes its end, then wait for it.
        while true {
            let data = pipe.fileHandleForReading.availableData
            if data.isEmpty { break }
            if let output = String(data: data, encoding: .utf8),
               let last = lastLine(output) {
                progress(last)
            }
        }
        process.waitUntilExit()
        return process.terminationStatus == 0
            ? nil : "Failed to pull \(model). Make sure Ollama is running."
    }

    /// The last non-empty line of a chunk of `ollama pull` output (its
    /// progress redraws with carriage returns).
    static func lastLine(_ chunk: String) -> String? {
        chunk.components(separatedBy: CharacterSet(charactersIn: "\r\n"))
            .map { $0.trimmingCharacters(in: .whitespaces) }
            .last { !$0.isEmpty }
    }

    /// The fraction a progress line of `ollama pull` reports ("pulling
    /// 8eeb52dfb3bb:  45% ▕████ ▏ 1.5 GB/3.4 GB" is 0.45), or nil.
    static func fraction(in line: String) -> Double? {
        guard let range = line.range(of: #"(\d{1,3})%"#, options: .regularExpression) else {
            return nil
        }
        guard let percent = Double(line[range].dropLast()), percent <= 100 else { return nil }
        return percent / 100
    }
}
