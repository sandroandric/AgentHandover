#!/usr/bin/env bash
set -euo pipefail

VERSION="${VERSION:-0.1.0}"
PKG_ROOT="$(mktemp -d)"
SCRIPTS_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPTS_DIR/.." && pwd)"
OUTPUT="${REPO_ROOT}/target/OpenMimic-${VERSION}.pkg"

echo "=== Building OpenMimic ${VERSION} ==="

# Stage directories
mkdir -p "${PKG_ROOT}/usr/local/bin"
mkdir -p "${PKG_ROOT}/usr/local/lib/openmimic/extension"
mkdir -p "${PKG_ROOT}/usr/local/lib/openmimic/launchd"
mkdir -p "${PKG_ROOT}/Applications"

# Copy binaries
echo "Staging binaries..."
cp "${REPO_ROOT}/target/universal-release/oc-apprentice-daemon" "${PKG_ROOT}/usr/local/bin/"
cp "${REPO_ROOT}/target/universal-release/openmimic" "${PKG_ROOT}/usr/local/bin/"

# Copy extension
echo "Staging extension..."
EXT_SRC="${REPO_ROOT}/extension"
EXT_DST="${PKG_ROOT}/usr/local/lib/openmimic/extension"

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
cp "${REPO_ROOT}/resources/launchd/"*.plist "${PKG_ROOT}/usr/local/lib/openmimic/launchd/"

# Copy worker Python package (source only, no tests/build artifacts)
echo "Staging worker..."
mkdir -p "${PKG_ROOT}/usr/local/lib/openmimic/worker"
cp -R "${REPO_ROOT}/worker/src" "${PKG_ROOT}/usr/local/lib/openmimic/worker/src"
cp "${REPO_ROOT}/worker/pyproject.toml" "${PKG_ROOT}/usr/local/lib/openmimic/worker/"

# Copy SwiftUI app if built — SPM produces a binary, not a .app bundle.
# Wrap it in a minimal .app structure for /Applications.
APP_BINARY="${REPO_ROOT}/app/OpenMimicApp/.build/release/OpenMimicApp"
if [ -f "${APP_BINARY}" ]; then
    APP_BUNDLE="${PKG_ROOT}/Applications/OpenMimic.app/Contents/MacOS"
    mkdir -p "${APP_BUNDLE}"
    cp "${APP_BINARY}" "${APP_BUNDLE}/OpenMimic"
    cp "${REPO_ROOT}/app/OpenMimicApp/Sources/OpenMimicApp/Info.plist" \
       "${PKG_ROOT}/Applications/OpenMimic.app/Contents/Info.plist"
fi

# Copy install scripts
SCRIPTS_STAGING="$(mktemp -d)"
cp "${REPO_ROOT}/resources/pkg/scripts/preinstall" "${SCRIPTS_STAGING}/"
cp "${REPO_ROOT}/resources/pkg/scripts/postinstall" "${SCRIPTS_STAGING}/"
chmod +x "${SCRIPTS_STAGING}/preinstall" "${SCRIPTS_STAGING}/postinstall"

# Build component .pkg
echo "Building component package..."
COMPONENT_PKG="$(mktemp -d)/openmimic-component.pkg"
pkgbuild \
    --root "${PKG_ROOT}" \
    --scripts "${SCRIPTS_STAGING}" \
    --identifier "com.openmimic.pkg" \
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

# Cleanup
rm -rf "${PKG_ROOT}" "${SCRIPTS_STAGING}" "${COMPONENT_PKG}"

echo ""
echo "=== Package built: ${OUTPUT} ==="
echo "Size: $(du -h "${OUTPUT}" | cut -f1)"
