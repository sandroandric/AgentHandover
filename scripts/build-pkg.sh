#!/usr/bin/env bash
set -euo pipefail

VERSION="${VERSION:-0.1.0}"
PKG_ROOT="$(mktemp -d)"
SCRIPTS_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPTS_DIR/.." && pwd)"
OUTPUT="${REPO_ROOT}/target/AgentHandover-${VERSION}.pkg"

echo "=== Building AgentHandover ${VERSION} ==="

# Stage directories
mkdir -p "${PKG_ROOT}/usr/local/bin"
mkdir -p "${PKG_ROOT}/usr/local/lib/agenthandover/extension"
mkdir -p "${PKG_ROOT}/usr/local/lib/agenthandover/launchd"
mkdir -p "${PKG_ROOT}/Applications"

# Daemon lives OUTSIDE the app bundle so codesign --deep doesn't register
# it as a second TCC principal. The app owns Screen Recording; the daemon
# must be invisible to TCC's bundle scan.
DAEMON_EXEC="${PKG_ROOT}/usr/local/lib/agenthandover/ah-observer"

# Copy binaries
echo "Staging binaries..."
cp "${REPO_ROOT}/target/universal-release/agenthandover" "${PKG_ROOT}/usr/local/bin/"
# Cargo produces agenthandover-daemon; install as ah-observer to avoid
# macOS TCC caching the old "agenthandover-daemon" identity.
cp "${REPO_ROOT}/target/universal-release/agenthandover-daemon" "${DAEMON_EXEC}"

# Copy extension
echo "Staging extension..."
EXT_SRC="${REPO_ROOT}/extension"
EXT_DST="${PKG_ROOT}/usr/local/lib/agenthandover/extension"

if [ -d "${EXT_SRC}/dist" ]; then
    # Pre-built dist exists — copy contents flat so the extension dir
    # is directly loadable in Chrome (manifest.json + JS at root level).
    # webpack's CopyWebpackPlugin already copies manifest.json into dist/.
    cp -R "${EXT_SRC}/dist/." "${EXT_DST}/"
    echo "  Extension dist included (pre-built)."
elif command -v npm &>/dev/null && [ -f "${EXT_SRC}/package.json" ]; then
    # npm available — build the dist at package time
    echo "  Building extension with npm..."
    (cd "${EXT_SRC}" && npm install --ignore-scripts && npm run build)
    if [ -d "${EXT_SRC}/dist" ]; then
        # Copy contents flat (manifest.json + JS at root level)
        cp -R "${EXT_SRC}/dist/." "${EXT_DST}/"
        echo "  Extension dist built and included."
    else
        echo "  Warning: npm build did not produce dist/. Including source."
        cp -R "${EXT_SRC}/src" "${EXT_DST}/src"
        cp "${EXT_SRC}/manifest.json" "${EXT_DST}/"
        cp "${EXT_SRC}/package.json" "${EXT_DST}/"
        [ -f "${EXT_SRC}/tsconfig.json" ] && cp "${EXT_SRC}/tsconfig.json" "${EXT_DST}/"
        [ -f "${EXT_SRC}/webpack.config.js" ] && cp "${EXT_SRC}/webpack.config.js" "${EXT_DST}/"
    fi
elif [ -f "${EXT_SRC}/package.json" ]; then
    # No npm, no dist — include source for user to build
    echo "  npm not available, including extension source for manual build."
    cp -R "${EXT_SRC}/src" "${EXT_DST}/src"
    cp "${EXT_SRC}/manifest.json" "${EXT_DST}/"
    cp "${EXT_SRC}/package.json" "${EXT_DST}/"
    [ -f "${EXT_SRC}/tsconfig.json" ] && cp "${EXT_SRC}/tsconfig.json" "${EXT_DST}/"
    [ -f "${EXT_SRC}/webpack.config.js" ] && cp "${EXT_SRC}/webpack.config.js" "${EXT_DST}/"
fi

# Copy launchd plists (templates)
cp "${REPO_ROOT}/resources/launchd/"*.plist "${PKG_ROOT}/usr/local/lib/agenthandover/launchd/"

# Copy worker Python package (source only, no tests/build artifacts)
echo "Staging worker..."
mkdir -p "${PKG_ROOT}/usr/local/lib/agenthandover/worker"
cp -R "${REPO_ROOT}/worker/src" "${PKG_ROOT}/usr/local/lib/agenthandover/worker/src"
cp "${REPO_ROOT}/worker/pyproject.toml" "${PKG_ROOT}/usr/local/lib/agenthandover/worker/"

# Copy SwiftUI app if built — SPM produces a binary, not a .app bundle.
# Wrap it in a minimal .app structure for /Applications.
APP_BINARY="${REPO_ROOT}/app/AgentHandoverApp/.build/release/AgentHandoverApp"
if [ -f "${APP_BINARY}" ]; then
    APP_BUNDLE="${PKG_ROOT}/Applications/AgentHandover.app/Contents/MacOS"
    mkdir -p "${APP_BUNDLE}"
    cp "${APP_BINARY}" "${APP_BUNDLE}/AgentHandover"
    cp "${REPO_ROOT}/app/AgentHandoverApp/Sources/AgentHandoverApp/Info.plist" \
       "${PKG_ROOT}/Applications/AgentHandover.app/Contents/Info.plist"

    # Copy SPM resource bundle + icon to Contents/Resources
    RESOURCES_DIR="${PKG_ROOT}/Applications/AgentHandover.app/Contents/Resources"
    mkdir -p "${RESOURCES_DIR}"
    found_resource_bundle=false
    for RESOURCE_BUNDLE in "${REPO_ROOT}"/app/AgentHandoverApp/.build/release/*.bundle; do
        [ -d "${RESOURCE_BUNDLE}" ] || continue
        cp -R "${RESOURCE_BUNDLE}" "${RESOURCES_DIR}/"
        found_resource_bundle=true
        echo "  App resource bundle included: $(basename "${RESOURCE_BUNDLE}")"
        # Copy icon to top-level Resources for Finder/Dock
        if [ -f "${RESOURCE_BUNDLE}/AppIcon.icns" ]; then
            cp "${RESOURCE_BUNDLE}/AppIcon.icns" "${RESOURCES_DIR}/AppIcon.icns"
            echo "  App icon included."
        fi
    done
    if [ "${found_resource_bundle}" = false ]; then
        echo "  Warning: no SPM resource bundle found for AgentHandoverApp."
    fi
fi

# Codesign all binaries with Developer ID Application + hardened runtime + timestamp
CODESIGN_IDENTITY="${CODESIGN_IDENTITY:-}"
if [ -z "${CODESIGN_IDENTITY}" ]; then
    CODESIGN_IDENTITY=$(security find-identity -v -p codesigning 2>/dev/null \
        | grep "Developer ID Application" \
        | head -1 \
        | sed 's/.*"\(Developer ID Application:.*\)"/\1/')
fi

if [ -n "${CODESIGN_IDENTITY}" ]; then
    echo "Codesigning binaries with: ${CODESIGN_IDENTITY}"
    # Sign individual binaries first
    # Sign app binary and CLI with Developer ID
    for binary in \
        "${PKG_ROOT}/usr/local/bin/agenthandover" \
        "${PKG_ROOT}/Applications/AgentHandover.app/Contents/MacOS/AgentHandover"; do
        if [ -f "${binary}" ]; then
            codesign --force --options runtime --timestamp \
                --sign "${CODESIGN_IDENTITY}" "${binary}"
            echo "  Signed: $(basename "${binary}")"
        fi
    done

    # Sign daemon with Developer ID but use a DIFFERENT identifier so
    # TCC doesn't associate it with the old "ah-observer" principal.
    # The app now owns all TCC permissions; the daemon is just a helper.
    if [ -f "${DAEMON_EXEC}" ]; then
        codesign --force --options runtime --timestamp \
            --identifier "com.agenthandover.observer" \
            --sign "${CODESIGN_IDENTITY}" "${DAEMON_EXEC}"
        echo "  Signed: ah-observer (as com.agenthandover.observer)"
    fi

    # Sign the entire .app bundle (binds Info.plist to the signature).
    # This is REQUIRED for macOS Tahoe TCC — without bundle-level signing,
    # the app won't appear in Screen Recording settings even if the inner
    # binary is signed. --deep signs all nested bundles too.
    if [ -d "${PKG_ROOT}/Applications/AgentHandover.app" ]; then
        codesign --force --deep --options runtime --timestamp \
            --sign "${CODESIGN_IDENTITY}" \
            "${PKG_ROOT}/Applications/AgentHandover.app"
        echo "  Signed bundle: AgentHandover.app (deep)"
    fi
else
    echo "Warning: No Developer ID Application certificate found. Binaries unsigned."
    echo "  Notarization will fail without codesigned binaries."
fi

# Copy install scripts
SCRIPTS_STAGING="$(mktemp -d)"
cp "${REPO_ROOT}/resources/pkg/scripts/preinstall" "${SCRIPTS_STAGING}/"
cp "${REPO_ROOT}/resources/pkg/scripts/postinstall" "${SCRIPTS_STAGING}/"
chmod +x "${SCRIPTS_STAGING}/preinstall" "${SCRIPTS_STAGING}/postinstall"

# Generate component plist to disable app relocation.
# Without this, macOS Installer "relocates" the app back to wherever
# a previous copy was found (e.g. the build directory), instead of
# installing to /Applications.
echo "Generating component plist..."
COMPONENT_PLIST="$(mktemp -d)/component.plist"
pkgbuild --analyze --root "${PKG_ROOT}" "${COMPONENT_PLIST}" 2>/dev/null
# Set BundleIsRelocatable to false for all bundles
/usr/libexec/PlistBuddy -c "Set :0:BundleIsRelocatable false" "${COMPONENT_PLIST}" 2>/dev/null || true

# Build component .pkg
echo "Building component package..."
COMPONENT_PKG="$(mktemp -d)/agenthandover-component.pkg"
pkgbuild \
    --root "${PKG_ROOT}" \
    --scripts "${SCRIPTS_STAGING}" \
    --component-plist "${COMPONENT_PLIST}" \
    --identifier "com.agenthandover.pkg" \
    --version "${VERSION}" \
    --install-location "/" \
    "${COMPONENT_PKG}"

# Build product .pkg with distribution
echo "Building product package..."
mkdir -p "$(dirname "${OUTPUT}")"
productbuild \
    --distribution "${REPO_ROOT}/resources/pkg/distribution.xml" \
    --package-path "$(dirname "${COMPONENT_PKG}")" \
    --resources "${REPO_ROOT}/resources/pkg" \
    "${OUTPUT}"

# Sign the package if a Developer ID Installer identity is available
SIGN_IDENTITY="${SIGN_IDENTITY:-}"
if [ -z "${SIGN_IDENTITY}" ]; then
    # Auto-detect Developer ID Installer certificate
    SIGN_IDENTITY=$(security find-identity -v -p basic 2>/dev/null \
        | grep "Developer ID Installer" \
        | head -1 \
        | sed 's/.*"\(Developer ID Installer:.*\)"/\1/')
fi

if [ -n "${SIGN_IDENTITY}" ]; then
    echo "Signing with: ${SIGN_IDENTITY}"
    SIGNED_OUTPUT="${OUTPUT%.pkg}-signed.pkg"
    productsign --sign "${SIGN_IDENTITY}" "${OUTPUT}" "${SIGNED_OUTPUT}"
    mv "${SIGNED_OUTPUT}" "${OUTPUT}"
    echo "Package signed successfully."
else
    echo "Warning: No Developer ID Installer certificate found. Package is unsigned."
    echo "  Users will need to right-click → Open to bypass Gatekeeper."
fi

# Cleanup
rm -rf "${PKG_ROOT}" "${SCRIPTS_STAGING}" "${COMPONENT_PKG}"

echo ""
echo "=== Package built: ${OUTPUT} ==="
echo "Size: $(du -h "${OUTPUT}" | cut -f1)"
