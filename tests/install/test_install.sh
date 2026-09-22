#!/usr/bin/env bash
# Install journey: runs the real preinstall/postinstall into temporary
# prefixes (AH_LIB_DIR, AH_BIN_DIR, AH_LAUNCH_AGENTS, AH_USER_HOME) and
# checks that the result is a working install, then runs the real CLI's
# `doctor` and `logs` against it.
#
# Scenarios:
#   fresh      first install; the candidate list starts with an old Python,
#              so the probe must skip it and pick a 3.11+ one
#   upgrade    second install over it, with a stub "app" process running
#              that preinstall must stop; then a broken venv is rebuilt
#   oldpython  only a Python older than 3.11 available: install must fail
#              with an actionable message
#   build-pkg  scripts/build-pkg.sh must refuse to package without the app
#              binary or with a missing/old Sparkle.framework
#
# Nothing here touches the live login session: AH_NO_SERVICES=1 skips
# launchctl/tccutil/defaults/open, and the "app" preinstall quits is a stub
# with its own name. Run unprivileged (ownership checks that need root are
# reported as skipped) or with sudo on a test Mac to also check that the lib
# dir is root-owned. Needs network access (or a warm pip cache) for the
# worker's dependencies.
#
# Usage: tests/install/test_install.sh
#   AH_TEST_PYTHON  Python 3.11+ to install with (default: newest Homebrew one)
#   AH_TEST_CLI     agenthandover CLI binary (default: target/debug, built if missing)
#   AH_TEST_DAEMON  daemon binary staged as ah-observer (default: target/debug)
#   AH_TEST_KEEP=1  keep the work directory for inspection

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
SCRIPTS="${REPO_ROOT}/resources/pkg/scripts"
# Short path: Unix socket paths are limited to 104 bytes on macOS.
WORK="$(mktemp -d /tmp/ahi.XXXXXX)"
STUB_NAME="AhInstallStub"
PIDS_TO_KILL=()

cleanup() {
    for pid in "${PIDS_TO_KILL[@]:-}"; do
        [ -n "${pid}" ] && kill "${pid}" 2>/dev/null
    done
    if [ "${AH_TEST_KEEP:-0}" = "1" ]; then
        echo "Work directory kept: ${WORK}"
    else
        rm -rf "${WORK}"
    fi
}
trap cleanup EXIT

PASSED=0
FAILED=0
SKIPPED=0
pass() { PASSED=$((PASSED + 1)); echo "  pass  $1"; }
fail() { FAILED=$((FAILED + 1)); echo "  FAIL  $1"; }
skip() { SKIPPED=$((SKIPPED + 1)); echo "  skip  $1"; }
check() { # check "description" command...
    local desc="$1"; shift
    if "$@" >/dev/null 2>&1; then pass "${desc}"; else fail "${desc}"; fi
}
contains() { grep -qF -- "$2" "$1"; }

IS_ROOT=0
if [ "$(id -u)" -eq 0 ]; then IS_ROOT=1; fi
INSTALL_USER="$(id -un)"
if [ "${IS_ROOT}" -eq 1 ]; then INSTALL_USER="${SUDO_USER:?run with sudo from a user account}"; fi

# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------

GOOD_PY="${AH_TEST_PYTHON:-}"
if [ -z "${GOOD_PY}" ]; then
    for v in 14 13 12 11; do
        for d in /opt/homebrew/bin /usr/local/bin; do
            if [ -x "${d}/python3.${v}" ]; then GOOD_PY="${d}/python3.${v}"; break 2; fi
        done
    done
fi
if [ -z "${GOOD_PY}" ]; then
    echo "No Python 3.11+ found; set AH_TEST_PYTHON." >&2
    exit 2
fi

# A Python older than 3.11: the system one when it is old (stock macOS
# ships 3.9 with the Command Line Tools), else a stub reporting 3.9.
OLD_PY="${WORK}/old-python/python3"
mkdir -p "$(dirname "${OLD_PY}")"
if xcode-select -p >/dev/null 2>&1 && [ -x /usr/bin/python3 ] \
    && ! /usr/bin/python3 -c 'import sys; sys.exit(0 if sys.version_info >= (3, 11) else 1)' 2>/dev/null; then
    OLD_PY=/usr/bin/python3
else
    printf '#!/bin/sh\n# stand-in for Python 3.9\nexit 1\n' > "${OLD_PY}"
    chmod +x "${OLD_PY}"
fi

CLI="${AH_TEST_CLI:-${REPO_ROOT}/target/debug/agenthandover}"
DAEMON="${AH_TEST_DAEMON:-${REPO_ROOT}/target/debug/agenthandover-daemon}"
if [ ! -x "${CLI}" ] || [ ! -x "${DAEMON}" ]; then
    echo "Building CLI and daemon..."
    (cd "${REPO_ROOT}" && cargo build -q -p agenthandover-cli -p agenthandover-daemon) || exit 2
fi
if [ "${IS_ROOT}" -eq 1 ]; then chmod 755 "${WORK}"; fi

echo "Work dir: ${WORK}"
echo "Python:   ${GOOD_PY} (old: ${OLD_PY})"
echo "Mode:     $([ "${IS_ROOT}" -eq 1 ] && echo "root, installing for ${INSTALL_USER}" || echo "unprivileged")"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Lay out what the .pkg payload puts on disk (see scripts/build-pkg.sh).
make_prefix() {
    local p="${WORK}/$1"
    mkdir -p "${p}/lib/launchd" "${p}/lib/worker" "${p}/lib/extension" "${p}/bin" \
        "${p}/LaunchAgents" "${p}/home/Library/Application Support/Google/Chrome"
    cp -R "${REPO_ROOT}/worker/src" "${p}/lib/worker/src"
    cp "${REPO_ROOT}/worker/pyproject.toml" "${p}/lib/worker/"
    cp "${REPO_ROOT}/resources/launchd/com.agenthandover.worker.plist" "${p}/lib/launchd/"
    cp "${REPO_ROOT}/extension/manifest.json" "${p}/lib/extension/"
    cp "${DAEMON}" "${p}/lib/ah-observer"
    cp "${CLI}" "${p}/bin/agenthandover"
    # A payload file left group-writable, as an old install might have.
    chmod 775 "${p}/lib/worker"
    if [ "${IS_ROOT}" -eq 1 ]; then chown -R "${INSTALL_USER}" "${p}/home"; fi
    echo "${p}"
}

# run_script <prefix> <preinstall|postinstall> <python candidates> <log>
run_script() {
    local p="$1" script="$2" candidates="$3" log="$4"
    env AH_LIB_DIR="${p}/lib" AH_BIN_DIR="${p}/bin" \
        AH_LAUNCH_AGENTS="${p}/LaunchAgents" AH_USER_HOME="${p}/home" \
        AH_APP_PATH="${p}/Applications/AgentHandover.app" \
        AH_NO_SERVICES=1 AH_APP_BUNDLE_ID="" AH_APP_PROCESS="${STUB_NAME}" \
        AH_PYTHON_CANDIDATES="${candidates}" \
        bash "${SCRIPTS}/${script}" >"${log}" 2>&1
}

run_doctor() { # run_doctor <prefix> <log>; returns doctor's exit code
    local p="$1"
    env NO_COLOR=1 HOME="${p}/home" AH_LIB_DIR="${p}/lib" \
        AH_LAUNCH_AGENTS="${p}/LaunchAgents" "${p}/bin/agenthandover" doctor >"$2" 2>&1
}

data_dir() { echo "$1/home/Library/Application Support/agenthandover"; }
mode_of() { stat -f '%Lp' "$1"; }
owner_of() { stat -f '%Su' "$1"; }
not_group_or_other_writable() { [ $(( 0$(mode_of "$1") & 022 )) -eq 0 ]; }
plist_value() { plutil -extract "$2" raw -o - "$1" 2>/dev/null; }

check_installed_prefix() {
    local p="$1" lib="$1/lib" dd
    dd="$(data_dir "$1")"
    local py="${lib}/venv/bin/python"
    check "venv python runs" "${py}" -c "import sys"
    check "worker, mcp and ollama import in the venv" \
        "${py}" -c "import agenthandover_worker, mcp, ollama"
    check "mcp pinned to >=1.30,<2" "${py}" -c \
        "import importlib.metadata as m; v=tuple(int(x) for x in m.version('mcp').split('.')[:2]); assert (1,30) <= v < (2,0), v"
    check "mlx-embeddings not installed by the installer" \
        "${py}" -c "import importlib.util, sys; sys.exit(importlib.util.find_spec('mlx_embeddings') is not None)"
    check "pip output logged to logs/pip-install.log" test -s "${dd}/logs/pip-install.log"
    check "default config disables image embeddings" contains "${dd}/config.toml" "image_embeddings = false"
    check "install marker written" test -f "${dd}/config.toml.bak"

    local plist="${p}/LaunchAgents/com.agenthandover.worker.plist"
    check "worker plist: RunAtLoad true" test "$(plist_value "${plist}" RunAtLoad)" = "true"
    check "worker plist: KeepAlive.SuccessfulExit false" \
        test "$(plist_value "${plist}" KeepAlive.SuccessfulExit)" = "false"
    check "worker plist: PATH has /opt/homebrew/bin and /usr/local/bin" \
        contains <(plist_value "${plist}" EnvironmentVariables.PATH) "/opt/homebrew/bin:/usr/local/bin"
    check "worker plist: runs this prefix's venv" \
        test "$(plist_value "${plist}" ProgramArguments.0)" = "${lib}/venv/bin/python"
    check "worker plist: logs under the user's data dir" \
        test "$(plist_value "${plist}" StandardErrorPath)" = "${dd}/logs/worker.stderr.log"
    check "worker plist: stdout log under the user's data dir" \
        test "$(plist_value "${plist}" StandardOutPath)" = "${dd}/logs/worker.stdout.log"
    check "worker plist: no placeholder or /tmp path left" \
        bash -c "! grep -qE '__AH_LOG_DIR__|/tmp/' \"\$1\"" _ "${plist}"

    local nm="${p}/home/Library/Application Support/Google/Chrome/NativeMessagingHosts/com.agenthandover.host.json"
    check "native messaging manifest points at this prefix's daemon" \
        contains "${nm}" "\"path\": \"${lib}/ah-observer\""
    check "agenthandover-mcp linked into the bin dir" \
        test "$(readlink "${p}/bin/agenthandover-mcp")" = "${lib}/venv/bin/agenthandover-mcp"

    check "lib dir not group/other-writable" not_group_or_other_writable "${lib}"
    check "worker payload not group/other-writable" not_group_or_other_writable "${lib}/worker"
    check "daemon binary not group/other-writable" not_group_or_other_writable "${lib}/ah-observer"
    check "venv owned by ${INSTALL_USER}" test "$(owner_of "${lib}/venv")" = "${INSTALL_USER}"
    if [ "${IS_ROOT}" -eq 1 ]; then
        check "lib dir owned by root" test "$(owner_of "${lib}")" = "root"
        check "daemon binary owned by root" test "$(owner_of "${lib}/ah-observer")" = "root"
        check "worker payload owned by root" test "$(owner_of "${lib}/worker/pyproject.toml")" = "root"
    else
        skip "root ownership of lib dir and daemon binary (run with sudo to check)"
    fi
}

# A stand-in for AgentHandover.app's observation and capture servers.
start_stub_servers() {
    local dd="$1" ready="${WORK}/stub-ready"
    rm -f "${ready}"
    "${GOOD_PY}" - "${dd}" "${ready}" <<'PY' &
import json, os, socket, sys, threading
dd, ready = sys.argv[1], sys.argv[2]
reply = json.dumps({
    "accessibility_granted": True, "screen_recording_granted": True,
    "secure_field_focused": False, "focused_window": None,
    "display_topology": [], "cursor_global_px": None,
}).encode() + b"\n"
socks = []
for name in ("observation.sock", "capture.sock"):
    path = os.path.join(dd, name)
    if os.path.exists(path):
        os.unlink(path)
    s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    s.bind(path)
    s.listen(4)
    socks.append(s)
def serve(s):
    while True:
        conn, _ = s.accept()
        with conn:
            conn.makefile("rb").readline()
            conn.sendall(reply)
for s in socks:
    threading.Thread(target=serve, args=(s,), daemon=True).start()
open(ready, "w").close()
threading.Event().wait()
PY
    PIDS_TO_KILL+=("$!")
    disown "$!"
    for _ in $(seq 1 50); do [ -f "${ready}" ] && return 0; sleep 0.1; done
    return 1
}

# ---------------------------------------------------------------------------
# fresh
# ---------------------------------------------------------------------------
echo ""
echo "== fresh install (old Python first in the candidate list) =="
P="$(make_prefix fresh)"
run_script "${P}" preinstall "" "${WORK}/fresh-pre.log"
check "preinstall exits 0" test $? -eq 0
run_script "${P}" postinstall "${OLD_PY} ${GOOD_PY}" "${WORK}/fresh-post.log"
rc=$?
check "postinstall exits 0" test "${rc}" -eq 0
[ "${rc}" -eq 0 ] || sed 's/^/    | /' "${WORK}/fresh-post.log"
check "probe skipped the old Python and used ${GOOD_PY}" contains "${WORK}/fresh-post.log" "Using Python: ${GOOD_PY}"
check_installed_prefix "${P}"

echo ""
echo "== doctor against the fresh prefix, app not running =="
run_doctor "${P}" "${WORK}/doctor-noapp.log"
check "doctor exits non-zero when the app is not running" test $? -ne 0
DD="$(data_dir "${P}")"
check "doctor names the observation socket" contains "${WORK}/doctor-noapp.log" "FAIL Observation server (${DD}/observation.sock)"
check "doctor explains the app is not running" contains "${WORK}/doctor-noapp.log" "AgentHandover.app is not running or its capture server failed to bind"
check "doctor: daemon binary in the prefix" contains "${WORK}/doctor-noapp.log" "pass Daemon binary (${P}/lib/ah-observer)"
check "doctor: venv healthy and worker version matches CLI" contains "${WORK}/doctor-noapp.log" "pass Python virtual environment (worker "
check "doctor: worker plist" contains "${WORK}/doctor-noapp.log" "pass Worker launchd plist"
check "doctor: native messaging host" contains "${WORK}/doctor-noapp.log" "pass Native messaging host"

echo ""
echo "== doctor with stub capture servers =="
if start_stub_servers "${DD}"; then
    run_doctor "${P}" "${WORK}/doctor-app.log"
    check "doctor: observation server answers" contains "${WORK}/doctor-app.log" "pass Observation server"
    check "doctor: accessibility granted" contains "${WORK}/doctor-app.log" "pass Accessibility permission"
    check "doctor: screen recording granted" contains "${WORK}/doctor-app.log" "pass Screen Recording permission"
    check "doctor: capture socket" contains "${WORK}/doctor-app.log" "pass Capture socket"
    if grep -q "^  FAIL" "${WORK}/doctor-app.log"; then
        echo "    remaining doctor failures (Ollama and models are machine state):"
        grep "^  FAIL" "${WORK}/doctor-app.log" | sed 's/^/    | /'
    fi
else
    fail "stub capture servers did not start"
fi

echo ""
echo "== logs daemon =="
mkdir -p "${DD}/logs"
echo "stub daemon line" > "${DD}/logs/daemon.2026-09-14.log"
env HOME="${P}/home" "${P}/bin/agenthandover" logs daemon > "${WORK}/logs.out" 2>&1
check "'agenthandover logs daemon' prints daemon.<date>.log" contains "${WORK}/logs.out" "stub daemon line"

# ---------------------------------------------------------------------------
# upgrade
# ---------------------------------------------------------------------------
echo ""
echo "== upgrade over the fresh prefix, stub app running =="
# A tiny compiled binary named like the app. (Copying /bin/sleep and
# re-signing it ad hoc no longer launches on macOS 26: platform binaries
# lose their entitlements when re-signed.)
printf '#include <unistd.h>\n#include <stdlib.h>\nint main(int c, char **v) { sleep(c > 1 ? atoi(v[1]) : 600); return 0; }\n' \
    | cc -x c -o "${WORK}/${STUB_NAME}" -
if [ "${IS_ROOT}" -eq 1 ]; then
    sudo -u "${INSTALL_USER}" "${WORK}/${STUB_NAME}" 600 &
else
    "${WORK}/${STUB_NAME}" 600 &
fi
STUB_PID=$!
disown "${STUB_PID}"
PIDS_TO_KILL+=("${STUB_PID}")
sleep 0.5
check "stub app process is running" pgrep -x "${STUB_NAME}"
PIP_LOG_BEFORE=$(wc -c < "${DD}/logs/pip-install.log")
run_script "${P}" preinstall "" "${WORK}/upgrade-pre.log"
check "preinstall exits 0" test $? -eq 0
check "preinstall stopped the running app" bash -c "! pgrep -x ${STUB_NAME}"
chmod 777 "${P}/lib/worker"   # simulate an old install's loose permissions
run_script "${P}" postinstall "${GOOD_PY}" "${WORK}/upgrade-post.log"
rc=$?
check "upgrade postinstall exits 0" test "${rc}" -eq 0
[ "${rc}" -eq 0 ] || sed 's/^/    | /' "${WORK}/upgrade-post.log"
check "upgrade updated the worker in place" contains "${WORK}/upgrade-post.log" "Updating worker package"
check "upgrade appended pip output to the log" test "$(wc -c < "${DD}/logs/pip-install.log")" -gt "${PIP_LOG_BEFORE}"
check_installed_prefix "${P}"

echo ""
echo "== upgrade over a broken venv =="
rm -f "${P}/lib/venv/bin/python"
ln -s /nonexistent/python3 "${P}/lib/venv/bin/python"
run_doctor "${P}" "${WORK}/doctor-broken.log"
check "doctor FAILs on the broken venv" contains "${WORK}/doctor-broken.log" "FAIL Python virtual environment"
run_script "${P}" postinstall "${GOOD_PY}" "${WORK}/rebuild-post.log"
rc=$?
check "postinstall exits 0" test "${rc}" -eq 0
[ "${rc}" -eq 0 ] || sed 's/^/    | /' "${WORK}/rebuild-post.log"
check "postinstall rebuilt the venv" contains "${WORK}/rebuild-post.log" "Existing Python environment is broken or too old; rebuilding it."
check "rebuilt venv imports the worker" "${P}/lib/venv/bin/python" -c "import agenthandover_worker, mcp"

# ---------------------------------------------------------------------------
# oldpython
# ---------------------------------------------------------------------------
echo ""
echo "== fresh install with only an old Python =="
P="$(make_prefix oldpython)"
run_script "${P}" postinstall "${OLD_PY}" "${WORK}/old-post.log"
check "postinstall exits non-zero" test $? -ne 0
check "error names the requirement" contains "${WORK}/old-post.log" "AgentHandover needs Python 3.11 or later"
check "error lists what was checked" contains "${WORK}/old-post.log" "${OLD_PY}"
check "error gives the fix" contains "${WORK}/old-post.log" "brew install python@3.13"
check "no venv left behind" test ! -e "${P}/lib/venv"

# ---------------------------------------------------------------------------
# build-pkg
# ---------------------------------------------------------------------------
echo ""
echo "== build-pkg.sh refuses to package a broken app bundle =="
# A throwaway copy of the inputs build-pkg.sh reads, and a fake `swift`
# whose behaviour each case sets. Every case fails before pkgbuild runs.
make_build_tree() {
    local t="${WORK}/build-$1"
    mkdir -p "${t}/scripts" "${t}/target/universal-release" "${t}/extension/dist" \
        "${t}/app/AgentHandoverApp/Sources/AgentHandoverApp" "${t}/resources/launchd" \
        "${t}/resources/pkg/scripts" "${t}/worker" "${t}/fakebin"
    cp "${REPO_ROOT}/scripts/build-pkg.sh" "${t}/scripts/"
    cp "${REPO_ROOT}/app/AgentHandoverApp/Sources/AgentHandoverApp/Info.plist" \
        "${t}/app/AgentHandoverApp/Sources/AgentHandoverApp/"
    cp "${REPO_ROOT}/resources/launchd/com.agenthandover.worker.plist" "${t}/resources/launchd/"
    cp "${SCRIPTS}/preinstall" "${SCRIPTS}/postinstall" "${SCRIPTS}/resolve-user.sh" "${t}/resources/pkg/scripts/"
    cp -R "${REPO_ROOT}/worker/src" "${t}/worker/src"
    cp "${REPO_ROOT}/worker/pyproject.toml" "${t}/worker/"
    touch "${t}/target/universal-release/agenthandover" "${t}/target/universal-release/agenthandover-daemon"
    echo '{}' > "${t}/extension/dist/manifest.json"
    # $2: shell body for the fake swift, run in app/AgentHandoverApp
    printf '#!/bin/bash\n%s\n' "$2" > "${t}/fakebin/swift"
    chmod +x "${t}/fakebin/swift"
    echo "${t}"
}
run_build() { # run_build <tree> <log>
    env PATH="$1/fakebin:${PATH}" CODESIGN_IDENTITY="none-for-test" VERSION=0.0.0-test \
        bash "$1/scripts/build-pkg.sh" >"$2" 2>&1
}
FAKE_FRAMEWORK='mkdir -p .build/release/Sparkle.framework/Versions/B/Resources
ln -sfn Versions/B/Resources .build/release/Sparkle.framework/Resources
cat > .build/release/Sparkle.framework/Versions/B/Resources/Info.plist <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<plist version="1.0"><dict><key>CFBundleShortVersionString</key><string>SPARKLE_VERSION</string></dict></plist>
EOF'
FAKE_APP='mkdir -p .build/release && cp /bin/echo .build/release/AgentHandoverApp'

T="$(make_build_tree nobinary 'exit 0')"
run_build "${T}" "${WORK}/build-nobinary.log"
check "no app binary: build-pkg exits non-zero" test $? -ne 0
check "no app binary: says why" contains "${WORK}/build-nobinary.log" "FATAL: ${T}/app/AgentHandoverApp/.build/release/AgentHandoverApp missing after swift build"

T="$(make_build_tree nosparkle "${FAKE_APP}")"
run_build "${T}" "${WORK}/build-nosparkle.log"
check "no Sparkle: build-pkg exits non-zero" test $? -ne 0
check "no Sparkle: says why" contains "${WORK}/build-nosparkle.log" "Sparkle.framework is missing"

T="$(make_build_tree oldsparkle "${FAKE_APP}
${FAKE_FRAMEWORK//SPARKLE_VERSION/2.9.1}")"
run_build "${T}" "${WORK}/build-oldsparkle.log"
check "Sparkle 2.9.1: build-pkg exits non-zero" test $? -ne 0
check "Sparkle 2.9.1: says why" contains "${WORK}/build-oldsparkle.log" "staged Sparkle is 2.9.1; need >= 2.9.6"

T="$(make_build_tree goodsparkle "${FAKE_APP}
${FAKE_FRAMEWORK//SPARKLE_VERSION/2.9.6}")"
run_build "${T}" "${WORK}/build-goodsparkle.log"
check "Sparkle 2.9.6 with an app binary passes the staging checks" \
    contains "${WORK}/build-goodsparkle.log" "Staged app and Sparkle 2.9.6 verified."

# ---------------------------------------------------------------------------
# minimum macOS
# ---------------------------------------------------------------------------
echo ""
echo "== minimum macOS agrees everywhere (14, Sonoma) =="
check "app Info.plist LSMinimumSystemVersion is 14.0" test "$(plist_value \
    "${REPO_ROOT}/app/AgentHandoverApp/Sources/AgentHandoverApp/Info.plist" LSMinimumSystemVersion)" = "14.0"
check "Package.swift targets .macOS(.v14)" contains "${REPO_ROOT}/app/AgentHandoverApp/Package.swift" ".macOS(.v14)"
check "cask depends_on Sonoma" contains "${REPO_ROOT}/homebrew/Casks/agenthandover.rb" 'depends_on macos: ">= :sonoma"'
check "installer distribution requires 14.0" contains "${REPO_ROOT}/resources/pkg/distribution.xml" '<os-version min="14.0"/>'

echo ""
echo "Install journey: ${PASSED} passed, ${FAILED} failed, ${SKIPPED} skipped"
[ "${FAILED}" -eq 0 ]
