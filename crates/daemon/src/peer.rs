//! Who may talk to the daemon's Unix sockets.
//!
//! The sockets live in a 0700 directory (`status::socket_dir`), so only the
//! user's own processes reach them.  Each accepted connection is then
//! checked, in order:
//!
//! 1. the peer runs as this user (`getpeereid`);
//! 2. its executable (`LOCAL_PEERPID` + `proc_pidpath`) passes
//!    `PeerPolicy::path_allowed` — a cheap pre-check;
//! 3. when the daemon itself is signed with a Team ID, the peer's code
//!    signature, looked up by its audit token (`LOCAL_PEERTOKEN`, so a
//!    recycled PID cannot pass), satisfies `anchor apple generic and
//!    certificate leaf[subject.OU] = <team> and identifier <id>`.  An
//!    unsigned development build has no team to pin and relies on 1 and 2.

use std::ffi::{CStr, CString};
use std::os::fd::AsRawFd;
use std::path::{Path, PathBuf};

/// The menu bar app's executable name inside its bundle.
pub const APP_EXECUTABLE_SUFFIX: &str = "/AgentHandover.app/Contents/MacOS/AgentHandover";
/// The menu bar app's signing identifier (its bundle id).
pub const APP_IDENTIFIER: &str = "com.agenthandover.app";

extern "C" {
    fn ah_audit_token_satisfies(token: *const u8, len: usize, requirement: *const libc::c_char) -> i32;
    fn ah_pid_satisfies(pid: i32, requirement: *const libc::c_char) -> i32;
    fn ah_self_team_id(buf: *mut libc::c_char, len: usize) -> i32;
}

pub struct PeerPolicy {
    /// Pre-check on the peer's resolved executable path.
    pub path_allowed: Box<dyn Fn(&Path) -> bool + Send + Sync>,
    /// Signing identifiers accepted when a Team ID is pinned.
    pub identifiers: Vec<String>,
    /// The Team ID peers must be signed with; `None` skips the check.
    pub team_id: Option<String>,
}

impl PeerPolicy {
    /// The menu bar app, wherever its bundle is installed; pinned to this
    /// daemon's own Team ID when the daemon is signed.
    pub fn menu_bar_app() -> Self {
        PeerPolicy {
            path_allowed: Box::new(|p| p.to_string_lossy().ends_with(APP_EXECUTABLE_SUFFIX)),
            identifiers: vec![APP_IDENTIFIER.to_string()],
            team_id: own_team_id(),
        }
    }

    /// The signing requirement a peer must satisfy, or `None` when unsigned.
    pub fn requirement(&self) -> Option<String> {
        let team = self.team_id.as_ref()?;
        if self.identifiers.is_empty() {
            return None;
        }
        let ids = self
            .identifiers
            .iter()
            .map(|id| format!("identifier \"{id}\""))
            .collect::<Vec<_>>()
            .join(" or ");
        Some(format!(
            "anchor apple generic and certificate leaf[subject.OU] = \"{team}\" and ({ids})"
        ))
    }

    /// `Ok(())` when the peer on `stream` is allowed, else why it is not.
    pub fn check<S: AsRawFd>(&self, stream: &S) -> Result<(), String> {
        let fd = stream.as_raw_fd();
        let (mut uid, mut gid) = (0 as libc::uid_t, 0 as libc::gid_t);
        // SAFETY: fd is a connected socket owned by `stream`; uid/gid are valid out-pointers.
        if unsafe { libc::getpeereid(fd, &mut uid, &mut gid) } != 0 {
            return Err("no peer credentials".into());
        }
        // SAFETY: getuid has no preconditions.
        if uid != unsafe { libc::getuid() } {
            return Err(format!("peer runs as uid {uid}"));
        }
        let pid: libc::pid_t = getsockopt_local(fd, libc::LOCAL_PEERPID)
            .ok_or_else(|| "no peer pid".to_string())?;
        let path = executable_path(pid).ok_or_else(|| format!("peer {pid} has no path"))?;
        if !(self.path_allowed)(&path) {
            return Err(format!("peer executable {} not allowed", path.display()));
        }
        let Some(requirement) = self.requirement() else {
            return Ok(());
        };
        let token: [u8; 32] = getsockopt_local(fd, libc::LOCAL_PEERTOKEN)
            .ok_or_else(|| "no peer audit token".to_string())?;
        match audit_token_satisfies(&token, &requirement) {
            0 => Ok(()),
            status => Err(format!(
                "peer {} fails the code requirement ({status})",
                path.display()
            )),
        }
    }
}

/// `getsockopt(fd, SOL_LOCAL, option)` into a `T`, or `None` on failure.
fn getsockopt_local<T: Copy + Default>(fd: i32, option: libc::c_int) -> Option<T> {
    let mut value = T::default();
    let mut len = std::mem::size_of::<T>() as libc::socklen_t;
    // SAFETY: `value` is a writable T of `len` bytes.
    let rc = unsafe {
        libc::getsockopt(fd, libc::SOL_LOCAL, option, &mut value as *mut T as *mut libc::c_void, &mut len)
    };
    (rc == 0 && len as usize == std::mem::size_of::<T>()).then_some(value)
}

/// The resolved executable path of `pid`.
pub fn executable_path(pid: libc::pid_t) -> Option<PathBuf> {
    let mut buf = vec![0u8; libc::PROC_PIDPATHINFO_MAXSIZE as usize];
    // SAFETY: buf is writable for its full length.
    let n = unsafe { libc::proc_pidpath(pid, buf.as_mut_ptr() as *mut libc::c_void, buf.len() as u32) };
    if n <= 0 {
        return None;
    }
    buf.truncate(n as usize);
    Some(PathBuf::from(String::from_utf8_lossy(&buf).into_owned()))
}

fn audit_token_satisfies(token: &[u8; 32], requirement: &str) -> i32 {
    let Ok(req) = CString::new(requirement) else { return -1 };
    // SAFETY: token is 32 readable bytes; req is a valid C string.
    unsafe { ah_audit_token_satisfies(token.as_ptr(), token.len(), req.as_ptr()) }
}

/// OSStatus of checking process `pid` against `requirement` (0 = satisfied).
pub fn pid_satisfies(pid: libc::pid_t, requirement: &str) -> i32 {
    let Ok(req) = CString::new(requirement) else { return -1 };
    // SAFETY: req is a valid C string.
    unsafe { ah_pid_satisfies(pid, req.as_ptr()) }
}

/// The Team ID this process is signed with, or `None` (unsigned / ad hoc).
pub fn own_team_id() -> Option<String> {
    let mut buf = [0 as libc::c_char; 64];
    // SAFETY: buf is writable for its full length; the C side NUL-terminates.
    if unsafe { ah_self_team_id(buf.as_mut_ptr(), buf.len()) } != 1 {
        return None;
    }
    // SAFETY: ah_self_team_id wrote a NUL-terminated string into buf.
    let team = unsafe { CStr::from_ptr(buf.as_ptr()) }.to_string_lossy().into_owned();
    (!team.is_empty()).then_some(team)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn requirement_pins_team_and_identifier() {
        let policy = PeerPolicy {
            path_allowed: Box::new(|_| true),
            identifiers: vec![APP_IDENTIFIER.to_string()],
            team_id: Some("ABCDE12345".to_string()),
        };
        assert_eq!(
            policy.requirement().unwrap(),
            "anchor apple generic and certificate leaf[subject.OU] = \"ABCDE12345\" \
             and (identifier \"com.agenthandover.app\")"
        );
        let unsigned = PeerPolicy { team_id: None, ..policy };
        assert_eq!(unsigned.requirement(), None);
    }

    #[test]
    fn app_path_rule_matches_only_the_bundle_executable() {
        let policy = PeerPolicy::menu_bar_app();
        let allowed = |p: &str| (policy.path_allowed)(Path::new(p));
        assert!(allowed("/Applications/AgentHandover.app/Contents/MacOS/AgentHandover"));
        assert!(allowed("/Users/x/Applications/AgentHandover.app/Contents/MacOS/AgentHandover"));
        assert!(!allowed("/tmp/AgentHandover"));
        assert!(!allowed("/Applications/Evil.app/Contents/MacOS/AgentHandover"));
        assert!(!allowed("/Applications/AgentHandover.app/Contents/MacOS/AgentHandover-helper"));
    }

    #[test]
    fn signature_check_accepts_signed_code_and_rejects_the_unsigned_test_binary() {
        // This test binary is unsigned (linker ad hoc signature): no team.
        assert_eq!(own_team_id(), None);
        let pinned = "anchor apple generic and certificate leaf[subject.OU] = \"ABCDE12345\" \
                      and identifier \"com.agenthandover.app\"";
        assert_ne!(pid_satisfies(std::process::id() as i32, pinned), 0);

        let mut sleeper = std::process::Command::new("/bin/sleep").arg("5").spawn().unwrap();
        let pid = sleeper.id() as i32;
        assert_eq!(pid_satisfies(pid, "anchor apple"), 0, "an Apple-signed binary is valid");
        assert_ne!(pid_satisfies(pid, pinned), 0);
        sleeper.kill().unwrap();
        let _ = sleeper.wait();
    }

    #[test]
    fn executable_path_of_self_is_the_test_binary() {
        let me = executable_path(std::process::id() as i32).unwrap();
        let exe = std::env::current_exe().unwrap().canonicalize().unwrap();
        assert_eq!(me, exe);
    }
}
