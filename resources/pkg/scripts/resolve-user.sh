# shellcheck shell=bash
# AgentHandover installer: resolve the user to install for.
#
# Sourced by preinstall and postinstall (pkgbuild ships every file in the
# scripts directory next to them). Sets CONSOLE_USER, CONSOLE_UID,
# USER_HOME and DETECT_METHOD, or prints an error and returns 1.
#
# The installer runs as root, so $USER is "root". We need the human behind
# the keyboard: the person whose ~/Library/LaunchAgents and
# ~/Library/Application Support we write to.
#
# v0.2.3 used `stat -f '%Su' /dev/console` with a fallback to $USER. That
# broke for a user whose /dev/console resolved to root at install time: the
# whole install targeted /var/root/Library/LaunchAgents, `agenthandover
# start` failed with `launchctl: Load failed: 5`, and the failure was silent
# because /var/root exists. Since v0.2.4 we cascade through four methods,
# each validated against UID >= 500, and hard-fail if none finds a user:
#   1. scutil State:/Users/ConsoleUser  (Apple's source for the GUI user)
#   2. SUDO_USER                        (`sudo installer -pkg` over SSH)
#   3. stat -f '%Su' /dev/console       (legacy, works on most machines)
#   4. dscl first user with UID >= 500  (last resort)
#
# Unprivileged runs (a developer or the install test harness installing
# into a temporary prefix, see tests/install/) cannot install for anyone
# else, so they install for the invoking user. AH_USER_HOME overrides the
# home directory in both modes.

_ah_uid_of() {
    id -u "$1" 2>/dev/null || true
}

_ah_valid_user() {
    local _u="$1"
    if [ -z "$_u" ] || [ "$_u" = "root" ] || [ "$_u" = "loginwindow" ]; then
        return 1
    fi
    local _uid
    _uid=$(_ah_uid_of "$_u")
    [ -n "$_uid" ] && [ "$_uid" -ge 500 ] 2>/dev/null
}

ah_resolve_user() {
    CONSOLE_USER=""
    DETECT_METHOD=""

    if [ "$(id -u)" -ne 0 ]; then
        CONSOLE_USER=$(id -un)
        DETECT_METHOD="unprivileged run"
    fi

    if [ -z "${CONSOLE_USER}" ]; then
        local _m1
        _m1=$(scutil <<< "show State:/Users/ConsoleUser" 2>/dev/null | awk '/Name :/ && $3 != "loginwindow" { print $3; exit }')
        if _ah_valid_user "${_m1}"; then
            CONSOLE_USER="${_m1}"
            DETECT_METHOD="scutil"
        fi
    fi
    if [ -z "${CONSOLE_USER}" ] && _ah_valid_user "${SUDO_USER:-}"; then
        CONSOLE_USER="${SUDO_USER}"
        DETECT_METHOD="SUDO_USER"
    fi
    if [ -z "${CONSOLE_USER}" ]; then
        local _m3
        _m3=$(stat -f '%Su' /dev/console 2>/dev/null)
        if _ah_valid_user "${_m3}"; then
            CONSOLE_USER="${_m3}"
            DETECT_METHOD="stat-console"
        fi
    fi
    if [ -z "${CONSOLE_USER}" ]; then
        local _m4
        _m4=$(dscl . list /Users UniqueID 2>/dev/null | awk '$2 >= 500 && $2 < 60000 { print $1; exit }')
        if _ah_valid_user "${_m4}"; then
            CONSOLE_USER="${_m4}"
            DETECT_METHOD="dscl-first-real-user"
        fi
    fi

    if [ -z "${CONSOLE_USER}" ]; then
        cat >&2 <<'ERR'
ERROR: AgentHandover could not determine which user to install for.

All four detection methods returned empty or invalid:
  1. scutil 'State:/Users/ConsoleUser'  (the GUI console user)
  2. SUDO_USER env var                    (the user who ran sudo)
  3. stat -f '%Su' /dev/console           (the console device owner)
  4. dscl first-real-user scan            (any user with UID >= 500)

This usually happens when no one is logged in at the GUI console, or the
install ran before the user's login session fully initialized.

To fix: log in to the Mac via the GUI, then re-run:
    sudo installer -pkg AgentHandover-*.pkg -target /
ERR
        return 1
    fi

    CONSOLE_UID=$(_ah_uid_of "${CONSOLE_USER}")
    if [ -n "${AH_USER_HOME:-}" ]; then
        USER_HOME="${AH_USER_HOME}"
    else
        USER_HOME=$(dscl . -read /Users/"${CONSOLE_USER}" NFSHomeDirectory 2>/dev/null | awk '{print $2}')
        if [ -z "$USER_HOME" ] || [ ! -d "$USER_HOME" ]; then
            USER_HOME=$(eval echo ~"$CONSOLE_USER")
        fi
    fi
    if [ -z "$USER_HOME" ] || [ ! -d "$USER_HOME" ] || [ "$USER_HOME" = "/var/root" ]; then
        echo "ERROR: resolved CONSOLE_USER='${CONSOLE_USER}' to USER_HOME='${USER_HOME}', which is invalid." >&2
        echo "Refusing to install into a system or missing home directory." >&2
        return 1
    fi
    return 0
}

# Run a command as the install user: directly when we already are that user
# (unprivileged run), through sudo when the installer runs as root. The
# script's PATH is passed on explicitly (sudo substitutes secure_path when
# sudoers sets one), so a
# command resolves the same way in both modes; the install test harness
# relies on that to put stand-ins first on the PATH.
as_user() {
    if [ "$(id -u)" -eq 0 ]; then
        sudo -H -u "${CONSOLE_USER}" /usr/bin/env "PATH=${PATH}" "$@"
    else
        "$@"
    fi
}
