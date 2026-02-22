class Openmimic < Formula
  desc "Local, privacy-first workflow apprentice that generates AI-executable SOPs"
  homepage "https://github.com/sandroandric/OpenMimic"
  license "MIT"

  # HEAD-only formula until the first tagged release.
  #
  # Setup:
  #   1. Create tap repo: gh repo create sandroandric/homebrew-openmimic --public
  #   2. Copy this file:  cp homebrew/Formula/openmimic.rb <tap-repo>/Formula/
  #   3. Push & install:  brew tap sandroandric/openmimic
  #                       brew install --HEAD openmimic
  #
  # To cut a stable release later:
  #   1. Tag:    git tag v0.1.0 && git push --tags
  #   2. SHA:    curl -sL https://github.com/sandroandric/OpenMimic/archive/refs/tags/v0.1.0.tar.gz | shasum -a 256
  #   3. Add:    url "https://github.com/.../v0.1.0.tar.gz"
  #              sha256 "<computed hash>"
  head "https://github.com/sandroandric/OpenMimic.git", branch: "main"

  depends_on "rust" => :build
  depends_on "python@3.12"
  depends_on "node" => :build

  def install
    # Build Rust binaries
    system "cargo", "build", "--release", "-p", "oc-apprentice-daemon"
    system "cargo", "build", "--release", "-p", "openmimic-cli"
    bin.install "target/release/oc-apprentice-daemon"
    bin.install "target/release/openmimic"

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

    # Install launchd plists
    (libexec/"launchd").install Dir["resources/launchd/*.plist"]

    # Install uninstaller
    (libexec/"scripts").install "scripts/uninstall.sh"
  end

  def post_install
    # Create data directories under user's Application Support.
    # LaunchAgents log to ~/Library/Application Support/oc-apprentice/logs/
    # so we create that path (not var/) for consistency.
    app_support = Pathname.new(Dir.home)/"Library/Application Support/oc-apprentice"
    (app_support/"logs").mkpath
    (app_support/"artifacts").mkpath

    # Install LaunchAgent plists to ~/Library/LaunchAgents/
    # (CLI `openmimic start` uses launchctl load on this directory)
    la_dir = Pathname.new(Dir.home)/"Library/LaunchAgents"
    la_dir.mkpath

    (la_dir/"com.openmimic.daemon.plist").write <<~XML
      <?xml version="1.0" encoding="UTF-8"?>
      <!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
      <plist version="1.0">
      <dict>
          <key>Label</key>
          <string>com.openmimic.daemon</string>
          <key>ProgramArguments</key>
          <array>
              <string>#{opt_bin}/oc-apprentice-daemon</string>
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
          <string>#{Dir.home}/Library/Application Support/oc-apprentice/logs/daemon.stdout.log</string>
          <key>StandardErrorPath</key>
          <string>#{Dir.home}/Library/Application Support/oc-apprentice/logs/daemon.stderr.log</string>
          <key>EnvironmentVariables</key>
          <dict>
              <key>RUST_LOG</key>
              <string>info</string>
          </dict>
      </dict>
      </plist>
    XML

    (la_dir/"com.openmimic.worker.plist").write <<~XML
      <?xml version="1.0" encoding="UTF-8"?>
      <!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
      <plist version="1.0">
      <dict>
          <key>Label</key>
          <string>com.openmimic.worker</string>
          <key>ProgramArguments</key>
          <array>
              <string>#{libexec}/venv/bin/python</string>
              <string>-m</string>
              <string>oc_apprentice_worker</string>
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
          <string>#{Dir.home}/Library/Application Support/oc-apprentice/logs/worker.stdout.log</string>
          <key>StandardErrorPath</key>
          <string>#{Dir.home}/Library/Application Support/oc-apprentice/logs/worker.stderr.log</string>
          <key>WorkingDirectory</key>
          <string>#{libexec}</string>
      </dict>
      </plist>
    XML

    # Install native messaging host manifest
    nm_dir = Pathname.new(Dir.home)/"Library/Application Support/Google/Chrome/NativeMessagingHosts"
    nm_dir.mkpath
    (nm_dir/"com.openclaw.apprentice.json").write <<~JSON
      {
          "name": "com.openclaw.apprentice",
          "description": "OpenMimic native messaging host",
          "path": "#{opt_bin}/oc-apprentice-daemon",
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
        openmimic doctor

      To start both daemon and worker:
        openmimic start

      Note: `brew services start openmimic` only manages the daemon.
      Use `openmimic start` to launch both daemon and worker together.

      You will need to grant:
        1. Accessibility permission in System Settings > Privacy & Security
        2. Screen Recording permission in System Settings > Privacy & Security

      To load the Chrome extension:
        1. Open chrome://extensions
        2. Enable Developer Mode
        3. Click "Load unpacked" and select: #{libexec}/extension/

      The Chrome extension is pre-built during installation.
      To rebuild from source, clone the repo and run:
        cd OpenMimic/extension && npm install && npm run build
    EOS
  end

  service do
    run [opt_bin/"oc-apprentice-daemon"]
    keep_alive crashed: true
    # Log to ~/Library/Application Support/oc-apprentice/logs/ — same
    # location used by the LaunchAgent plists for consistency.
    log_path Pathname.new(Dir.home)/"Library/Application Support/oc-apprentice/logs/daemon.log"
    error_log_path Pathname.new(Dir.home)/"Library/Application Support/oc-apprentice/logs/daemon.error.log"
  end
end
