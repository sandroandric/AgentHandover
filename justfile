# AgentHandover — build and packaging commands

# Default recipe
default: build-all

# Build everything
build-all: build-daemon build-cli build-extension build-worker build-app
    @echo "All builds complete."

# Build daemon (native target, or universal if both targets available)
build-daemon:
    #!/usr/bin/env bash
    set -euo pipefail
    source ~/.cargo/env 2>/dev/null || true
    TARGETS=$(rustup target list --installed | grep apple-darwin || true)
    HAS_ARM=$(echo "$TARGETS" | grep -c aarch64 || true)
    HAS_X86=$(echo "$TARGETS" | grep -c x86_64 || true)
    if [ "$HAS_ARM" -gt 0 ] && [ "$HAS_X86" -gt 0 ]; then
        echo "Building universal daemon..."
        cargo build --release -p agenthandover-daemon --target aarch64-apple-darwin
        cargo build --release -p agenthandover-daemon --target x86_64-apple-darwin
        mkdir -p target/universal-release
        lipo -create \
            target/aarch64-apple-darwin/release/agenthandover-daemon \
            target/x86_64-apple-darwin/release/agenthandover-daemon \
            -output target/universal-release/agenthandover-daemon
        echo "Universal daemon binary: target/universal-release/agenthandover-daemon"
    else
        echo "Building daemon for native target..."
        cargo build --release -p agenthandover-daemon
        mkdir -p target/universal-release
        cp target/release/agenthandover-daemon target/universal-release/
        echo "Native daemon binary: target/universal-release/agenthandover-daemon"
    fi

# Build CLI (native target, or universal if both targets available)
build-cli:
    #!/usr/bin/env bash
    set -euo pipefail
    source ~/.cargo/env 2>/dev/null || true
    TARGETS=$(rustup target list --installed | grep apple-darwin || true)
    HAS_ARM=$(echo "$TARGETS" | grep -c aarch64 || true)
    HAS_X86=$(echo "$TARGETS" | grep -c x86_64 || true)
    if [ "$HAS_ARM" -gt 0 ] && [ "$HAS_X86" -gt 0 ]; then
        echo "Building universal CLI..."
        cargo build --release -p agenthandover-cli --target aarch64-apple-darwin
        cargo build --release -p agenthandover-cli --target x86_64-apple-darwin
        mkdir -p target/universal-release
        lipo -create \
            target/aarch64-apple-darwin/release/agenthandover \
            target/x86_64-apple-darwin/release/agenthandover \
            -output target/universal-release/agenthandover
        echo "Universal CLI binary: target/universal-release/agenthandover"
    else
        echo "Building CLI for native target..."
        cargo build --release -p agenthandover-cli
        mkdir -p target/universal-release
        cp target/release/agenthandover target/universal-release/
        echo "Native CLI binary: target/universal-release/agenthandover"
    fi

# Build worker (Python venv + install)
build-worker:
    #!/usr/bin/env bash
    set -euo pipefail
    cd worker
    if [ ! -d .venv ]; then
        python3 -m venv .venv
        echo "Created worker venv at worker/.venv"
    fi
    source .venv/bin/activate
    pip install -e ".[dev]" --quiet
    echo "Worker installed in worker/.venv"

# Build Chrome extension
build-extension:
    #!/usr/bin/env bash
    set -euo pipefail
    cd extension
    npm install
    npm run build 2>/dev/null || npx webpack --mode production
    echo "Extension built: extension/dist/"

# Build SwiftUI menu bar app
build-app:
    #!/usr/bin/env bash
    set -euo pipefail
    cd app/AgentHandoverApp
    swift build -c release
    echo "App built: app/AgentHandoverApp/.build/release/AgentHandoverApp"

# Run all tests
test-all: test-rust test-python test-extension
    @echo "All tests passed!"

# Run Rust tests
test-rust:
    #!/usr/bin/env bash
    source ~/.cargo/env 2>/dev/null || true
    cargo test --workspace

# Run Python tests (separate roots to avoid namespace collision)
test-python:
    #!/usr/bin/env bash
    set -euo pipefail
    python -m pytest worker/tests/ -v
    cd tests && python -m pytest e2e/ load/ -v

# Run extension tests
test-extension:
    #!/usr/bin/env bash
    cd extension
    npm test 2>/dev/null || npx vitest run

# Package .pkg installer
package-pkg: build-all build-app
    #!/usr/bin/env bash
    set -euo pipefail
    bash scripts/build-pkg.sh

# Clean all build artifacts
clean:
    #!/usr/bin/env bash
    source ~/.cargo/env 2>/dev/null || true
    cargo clean
    rm -rf target/universal-release
    rm -rf worker/dist worker/build worker/src/*.egg-info
    rm -rf extension/dist
    rm -rf app/AgentHandoverApp/.build
    echo "Cleaned."
