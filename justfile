# OpenMimic — build and packaging commands

# Default recipe
default: build-all

# Build everything
build-all: build-daemon build-cli build-extension
    @echo "All builds complete."

# Build daemon (universal binary for macOS)
build-daemon:
    #!/usr/bin/env bash
    set -euo pipefail
    source ~/.cargo/env 2>/dev/null || true
    echo "Building daemon for aarch64..."
    cargo build --release -p oc-apprentice-daemon --target aarch64-apple-darwin
    echo "Building daemon for x86_64..."
    cargo build --release -p oc-apprentice-daemon --target x86_64-apple-darwin
    echo "Creating universal binary..."
    mkdir -p target/universal-release
    lipo -create \
        target/aarch64-apple-darwin/release/oc-apprentice-daemon \
        target/x86_64-apple-darwin/release/oc-apprentice-daemon \
        -output target/universal-release/oc-apprentice-daemon
    echo "Universal daemon binary: target/universal-release/oc-apprentice-daemon"

# Build CLI (universal binary for macOS)
build-cli:
    #!/usr/bin/env bash
    set -euo pipefail
    source ~/.cargo/env 2>/dev/null || true
    echo "Building CLI for aarch64..."
    cargo build --release -p openmimic-cli --target aarch64-apple-darwin
    echo "Building CLI for x86_64..."
    cargo build --release -p openmimic-cli --target x86_64-apple-darwin
    echo "Creating universal binary..."
    mkdir -p target/universal-release
    lipo -create \
        target/aarch64-apple-darwin/release/openmimic \
        target/x86_64-apple-darwin/release/openmimic \
        -output target/universal-release/openmimic
    echo "Universal CLI binary: target/universal-release/openmimic"

# Build worker (Python wheel)
build-worker:
    #!/usr/bin/env bash
    set -euo pipefail
    cd worker
    pip install build 2>/dev/null || true
    python -m build --wheel
    echo "Worker wheel: worker/dist/"

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
    cd app/OpenMimicApp
    swift build -c release
    echo "App built: app/OpenMimicApp/.build/release/OpenMimicApp"

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
    rm -rf app/OpenMimicApp/.build
    echo "Cleaned."
