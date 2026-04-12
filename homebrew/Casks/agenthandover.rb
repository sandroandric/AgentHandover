cask "agenthandover" do
  version "0.2.5"
  sha256 "e8b3cbfdbf10e433c5ca57b3fa6f0eab9d7f60807f6dc00c4df314d1cfbb2d15"

  url "https://github.com/sandroandric/AgentHandover/releases/download/v#{version}/AgentHandover-#{version}.pkg",
      verified: "github.com/sandroandric/AgentHandover/"
  name "AgentHandover"
  desc "Local, privacy-first workflow apprentice that generates AI-executable SOPs"
  homepage "https://github.com/sandroandric/AgentHandover"

  # AgentHandover uses the signed + notarized .pkg we ship on GitHub releases.
  # The pkg installs into /Applications, /usr/local/bin, and
  # /usr/local/lib/agenthandover — the Swift menu bar app hardcodes that
  # daemon path, so we cannot relocate into Homebrew's prefix.
  #
  # Why a cask and not a formula: the daemon must be spawned directly by
  # the menu bar app (via Process()) so macOS TCC attributes Screen
  # Recording + Accessibility to the app bundle.  Running the daemon via
  # launchd (as the old formula did) creates a second TCC principal and
  # Screen Recording silently fails.  See issue #1.  The cask path bypasses
  # all of that by running the exact same signed pkg that release users
  # install, so brew users get bit-identical behavior.

  depends_on macos: ">= :ventura"

  pkg "AgentHandover-#{version}.pkg"

  uninstall launchctl: "com.agenthandover.worker",
            pkgutil:   "com.agenthandover.pkg",
            delete:    [
              "/Applications/AgentHandover.app",
              "/usr/local/bin/agenthandover",
              "/usr/local/bin/agenthandover-connect",
              "/usr/local/bin/agenthandover-mcp",
              "/usr/local/bin/agenthandover-worker",
              "/usr/local/lib/agenthandover",
            ]

  zap trash: [
    "~/.agenthandover",
    "~/Library/Application Support/agenthandover",
    "~/Library/Caches/com.agenthandover.app",
    "~/Library/LaunchAgents/com.agenthandover.worker.plist",
    "~/Library/Preferences/com.agenthandover.app.plist",
  ]

  caveats <<~EOS
    AgentHandover runs entirely locally.  On first launch the menu bar
    app will walk you through:

      1. Accessibility + Screen Recording permissions (System Settings)
      2. Installing Ollama (opens https://ollama.com/download/mac) or
         `brew install ollama` — required for local AI models
      3. Pulling the Gemma 4 model tier that matches your Mac's RAM
      4. Loading the Chrome extension (optional, for in-browser capture)

    Gemma 4 models require Ollama 0.20.0 or later.

    To connect an agent after onboarding:
      agenthandover connect claude-code
      agenthandover connect codex
      agenthandover connect hermes
  EOS
end
