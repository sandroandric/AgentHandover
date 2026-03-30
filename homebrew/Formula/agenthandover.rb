class Agenthandover < Formula
  desc "Local, privacy-first workflow apprentice that generates AI-executable SOPs"
  homepage "https://github.com/sandroandric/AgentHandover"
  license "MIT"

  # HEAD-only formula until the first tagged release.
  #
  # Setup:
  #   1. Create tap repo: gh repo create sandroandric/homebrew-agenthandover --public
  #   2. Copy this file:  cp homebrew/Formula/agenthandover.rb <tap-repo>/Formula/
  #   3. Push & install:  brew tap sandroandric/agenthandover
  #                       brew install --HEAD agenthandover
  #
  # To cut a stable release later:
  #   1. Tag:    git tag v0.1.0 && git push --tags
  #   2. SHA:    curl -sL https://github.com/sandroandric/AgentHandover/archive/refs/tags/v0.1.0.tar.gz | shasum -a 256
  #   3. Add:    url "https://github.com/.../v0.1.0.tar.gz"
  #              sha256 "<computed hash>"
  head "https://github.com/sandroandric/AgentHandover.git", branch: "main"

  depends_on "rust" => :build
  depends_on "python@3.12"
  depends_on "node" => :build
  depends_on xcode: ["14.0", :build]

  def install
    # Build Rust binaries
    system "cargo", "build", "--release", "-p", "agenthandover-daemon"
    system "cargo", "build", "--release", "-p", "agenthandover-cli"
    bin.install "target/release/agenthandover-daemon"
    bin.install "target/release/agenthandover"

    # Install worker Python package into a venv
    venv = libexec/"venv"
    system Formula["python@3.12"].opt_bin/"python3.12", "-m", "venv", venv
    system venv/"bin/pip", "install", "--quiet", "worker/"
    # Symlink the venv python so launchd plist can reference a stable path
    (libexec/"bin").mkpath
    ln_sf venv/"bin/python", libexec/"bin/python"

    # Build Chrome extension
    cd "extension" do
      system "npm", "install", "--production=false"
      system "npx", "webpack", "--mode", "production"
    end
    (libexec/"extension").install Dir["extension/dist/*"]
    (libexec/"extension").install "extension/manifest.json"

    # Build SwiftUI menu bar app (requires Xcode CLT)
    cd "app/AgentHandoverApp" do
      system "swift", "build", "-c", "release"
      # Wrap the binary in a minimal .app bundle for ~/Applications
      app_binary = ".build/release/AgentHandoverApp"
      if File.exist?(app_binary)
        (libexec/"AgentHandover.app/Contents/MacOS").mkpath
        cp app_binary, libexec/"AgentHandover.app/Contents/MacOS/AgentHandover"
        if File.exist?("Sources/AgentHandoverApp/Info.plist")
          cp "Sources/AgentHandoverApp/Info.plist", libexec/"AgentHandover.app/Contents/Info.plist"
        end
      end
    end

    # Install launchd plists
    (libexec/"launchd").install Dir["resources/launchd/*.plist"]

    # Install uninstaller
    (libexec/"scripts").install "scripts/uninstall.sh"
  end

  def post_install
    # Create data directories under user's Application Support.
    # LaunchAgents log to ~/Library/Application Support/agenthandover/logs/
    # so we create that path (not var/) for consistency.
    app_support = Pathname.new(Dir.home)/"Library/Application Support/agenthandover"
    (app_support/"logs").mkpath
    (app_support/"artifacts").mkpath

    # Install LaunchAgent plists to ~/Library/LaunchAgents/
    # (CLI `agenthandover start` uses launchctl load on this directory)
    la_dir = Pathname.new(Dir.home)/"Library/LaunchAgents"
    la_dir.mkpath

    (la_dir/"com.agenthandover.daemon.plist").write <<~XML
      <?xml version="1.0" encoding="UTF-8"?>
      <!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
      <plist version="1.0">
      <dict>
          <key>Label</key>
          <string>com.agenthandover.daemon</string>
          <key>ProgramArguments</key>
          <array>
              <string>#{opt_bin}/agenthandover-daemon</string>
          </array>
          <key>RunAtLoad</key>
          <true/>
          <key>KeepAlive</key>
          <dict>
              <key>SuccessfulExit</key>
              <false/>
          </dict>
          <key>ThrottleInterval</key>
          <integer>10</integer>
          <key>ProcessType</key>
          <string>Background</string>
          <key>LowPriorityBackgroundIO</key>
          <true/>
          <key>StandardOutPath</key>
          <string>#{Dir.home}/Library/Application Support/agenthandover/logs/daemon.stdout.log</string>
          <key>StandardErrorPath</key>
          <string>#{Dir.home}/Library/Application Support/agenthandover/logs/daemon.stderr.log</string>
          <key>EnvironmentVariables</key>
          <dict>
              <key>RUST_LOG</key>
              <string>info</string>
          </dict>
      </dict>
      </plist>
    XML

    (la_dir/"com.agenthandover.worker.plist").write <<~XML
      <?xml version="1.0" encoding="UTF-8"?>
      <!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
      <plist version="1.0">
      <dict>
          <key>Label</key>
          <string>com.agenthandover.worker</string>
          <key>ProgramArguments</key>
          <array>
              <string>#{libexec}/venv/bin/python</string>
              <string>-m</string>
              <string>agenthandover_worker</string>
          </array>
          <key>RunAtLoad</key>
          <true/>
          <key>KeepAlive</key>
          <dict>
              <key>SuccessfulExit</key>
              <false/>
          </dict>
          <key>ThrottleInterval</key>
          <integer>10</integer>
          <key>ProcessType</key>
          <string>Background</string>
          <key>LowPriorityBackgroundIO</key>
          <true/>
          <key>StandardOutPath</key>
          <string>#{Dir.home}/Library/Application Support/agenthandover/logs/worker.stdout.log</string>
          <key>StandardErrorPath</key>
          <string>#{Dir.home}/Library/Application Support/agenthandover/logs/worker.stderr.log</string>
          <key>WorkingDirectory</key>
          <string>#{libexec}</string>
      </dict>
      </plist>
    XML

    # Symlink menu bar app into ~/Applications if it was built
    app_src = libexec/"AgentHandover.app"
    if app_src.exist?
      apps_dir = Pathname.new(Dir.home)/"Applications"
      apps_dir.mkpath
      # Symlink rather than copy so updates propagate
      ln_sf app_src, apps_dir/"AgentHandover.app"
    end

    # Install native messaging host manifest
    nm_dir = Pathname.new(Dir.home)/"Library/Application Support/Google/Chrome/NativeMessagingHosts"
    nm_dir.mkpath
    (nm_dir/"com.agenthandover.host.json").write <<~JSON
      {
          "name": "com.agenthandover.host",
          "description": "AgentHandover native messaging host",
          "path": "#{opt_bin}/agenthandover-daemon",
          "type": "stdio",
          "args": ["--native-messaging"],
          "allowed_origins": [
              "chrome-extension://knldjmfmopnpolahpmmgbagdohdnhkik/"
          ]
      }
    JSON
  end

  def caveats
    <<~EOS
      After installation, run:
        agenthandover doctor

      To start both daemon and worker:
        agenthandover start

      Note: `brew services start agenthandover` only manages the daemon.
      Use `agenthandover start` to launch both daemon and worker together.

      You will need to grant:
        1. Accessibility permission in System Settings > Privacy & Security
        2. Screen Recording permission in System Settings > Privacy & Security

      To load the Chrome extension:
        1. Open chrome://extensions
        2. Enable Developer Mode
        3. Click "Load unpacked" and select: #{libexec}/extension/

      The AgentHandover menu bar app is installed at:
        #{libexec}/AgentHandover.app
      A symlink is created in ~/Applications.

      The Chrome extension is pre-built during installation.
      To rebuild from source, clone the repo and run:
        cd AgentHandover/extension && npm install && npm run build
    EOS
  end

  service do
    run [opt_bin/"agenthandover-daemon"]
    keep_alive crashed: true
    # Log to ~/Library/Application Support/agenthandover/logs/ — same
    # location used by the LaunchAgent plists for consistency.
    log_path Pathname.new(Dir.home)/"Library/Application Support/agenthandover/logs/daemon.log"
    error_log_path Pathname.new(Dir.home)/"Library/Application Support/agenthandover/logs/daemon.error.log"
  end
end
