//! `[privacy.zones]` evaluated before the shutter.
//!
//! The observer asks [`PrivacyZones::zone`] every tick, before any capture
//! request reaches the app: a [`Zone::Blocked`] window gets no screenshot,
//! no OCR and no title; a [`Zone::MetadataOnly`] window gets its app-switch
//! and title events but no screenshot or OCR.
//!
//! The rules match the worker's `privacy_zones.PrivacyZoneChecker` (which
//! stays as the second line, on stored events):
//! - app patterns (`full_observation`, `blocked`, `metadata_only`) are
//!   matched against the focused window's app (its app id with the
//!   `pid:<pid>:` prefix stripped) and its title; `full_observation` wins
//!   over `blocked`, `blocked` over `metadata_only`;
//! - `blocked_urls` are matched against the browser's current page URL;
//! - `auto_pause` windows (`HH:MM-HH:MM`, local time, overnight allowed)
//!   block everything;
//! - the built-in defaults ([`DEFAULT_BLOCKED_APPS`],
//!   [`DEFAULT_BLOCKED_URLS`]) are always added to the user's lists;
//! - every match is an fnmatch-style glob (`*`, `?`, `[seq]`, `[!seq]`),
//!   case-insensitive.

use agenthandover_common::config::PrivacyZonesConfig;
use chrono::NaiveTime;

/// Apps always blocked (the worker's `_DEFAULT_BLOCKED`; a Python test
/// keeps the two lists equal).
pub const DEFAULT_BLOCKED_APPS: &[&str] = &[
    "1Password",
    "1Password 7",
    "1Password 8",
    "Bitwarden",
    "LastPass",
    "Dashlane",
    "KeePassXC",
    "Keychain Access",
    // macOS authentication and system UIs: password prompts, Touch ID,
    // the login window, permission dialogs, Apple's Passwords app, the
    // screen saver's unlock prompt and system alerts.
    "SecurityAgent",
    "coreautha",
    "loginwindow",
    "universalAccessAuthWarn",
    "Passwords",
    "ScreenSaverEngine",
    "UserNotificationCenter",
];

/// URL patterns always blocked (the worker's `_DEFAULT_BLOCKED_URLS`).
pub const DEFAULT_BLOCKED_URLS: &[&str] = &[
    "*bank*",
    "*banking*",
    "*.gov/*",
    "*payroll*",
    "*paycheck*",
    "*salary*",
    "*benefits*",
    "*1password.com*",
    "*bitwarden.com*",
    "*lastpass.com*",
    "*dashlane.com*",
    "*accounts.google.com/signin*",
    "*login.*",
    "*signin.*",
];

/// Apps whose page URL the extension reports (Chromium browsers).
const BROWSER_APPS: &[&str] = &[
    "google chrome",
    "google chrome canary",
    "chromium",
    "brave browser",
    "microsoft edge",
    "arc",
    "vivaldi",
    "comet",
];

/// What the observer may record for the focused window. Ordered from least
/// to most restrictive.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Zone {
    Full,
    MetadataOnly,
    Blocked,
}

impl Zone {
    pub fn as_str(self) -> &'static str {
        match self {
            Zone::Full => "full",
            Zone::MetadataOnly => "metadata_only",
            Zone::Blocked => "blocked",
        }
    }
}

/// The compiled zone rules. Patterns are case-folded once, here.
#[derive(Debug, Clone, Default)]
pub struct PrivacyZones {
    full_observation: Vec<String>,
    metadata_only: Vec<String>,
    blocked: Vec<String>,
    blocked_urls: Vec<String>,
    auto_pause: Vec<(NaiveTime, NaiveTime)>,
}

fn folded(patterns: &[String]) -> Vec<String> {
    patterns.iter().map(|p| p.to_lowercase()).collect()
}

fn with_defaults(defaults: &[&str], user: &[String]) -> Vec<String> {
    let mut all: Vec<String> = defaults.iter().map(|p| p.to_lowercase()).collect();
    for p in folded(user) {
        if !all.contains(&p) {
            all.push(p);
        }
    }
    all
}

impl PrivacyZones {
    pub fn new(config: &PrivacyZonesConfig) -> Self {
        let auto_pause = config
            .auto_pause
            .iter()
            .filter_map(|w| {
                let parsed = parse_time_window(w);
                if parsed.is_none() {
                    tracing::warn!(window = %w, "Ignoring invalid [privacy.zones] auto_pause window (want HH:MM-HH:MM)");
                }
                parsed
            })
            .collect();
        Self {
            full_observation: folded(&config.full_observation),
            metadata_only: folded(&config.metadata_only),
            blocked: with_defaults(DEFAULT_BLOCKED_APPS, &config.blocked),
            blocked_urls: with_defaults(DEFAULT_BLOCKED_URLS, &config.blocked_urls),
            auto_pause,
        }
    }

    /// The zone for the focused window.
    ///
    /// `app_id` is the observation socket's `app_id` (`pid:<pid>:<name>`),
    /// `title` its window title, `page_url` the browser's current page URL
    /// (used only when the focused app is a browser), `now` local time.
    pub fn zone(&self, app_id: &str, title: &str, page_url: Option<&str>, now: NaiveTime) -> Zone {
        if self.auto_paused(now) {
            return Zone::Blocked;
        }
        let app = strip_pid_prefix(app_id).to_lowercase();
        let title = title.to_lowercase();
        let candidates = [app.as_str(), title.as_str()];
        let any_match = |patterns: &[String]| {
            patterns.iter().any(|p| {
                candidates
                    .iter()
                    .any(|c| !c.is_empty() && glob_match(p, c))
            })
        };

        let app_zone = if any_match(&self.full_observation) {
            Zone::Full
        } else if any_match(&self.blocked) {
            Zone::Blocked
        } else if any_match(&self.metadata_only) {
            Zone::MetadataOnly
        } else {
            Zone::Full
        };

        let url_zone = match page_url {
            Some(url) if is_browser(&app) && self.url_blocked(url) => Zone::Blocked,
            _ => Zone::Full,
        };
        app_zone.max(url_zone)
    }

    /// Whether `url` matches a blocked URL pattern.
    pub fn url_blocked(&self, url: &str) -> bool {
        let url = url.to_lowercase();
        !url.is_empty() && self.blocked_urls.iter().any(|p| glob_match(p, &url))
    }

    fn auto_paused(&self, now: NaiveTime) -> bool {
        self.auto_pause.iter().any(|&(start, end)| {
            if start <= end {
                start <= now && now <= end
            } else {
                now >= start || now <= end
            }
        })
    }
}

/// Whether the (case-folded, pid-stripped) app name is a Chromium browser
/// whose page URL the extension reports.
pub fn is_browser(app: &str) -> bool {
    BROWSER_APPS.contains(&app)
}

/// `pid:1234:Google Chrome` -> `Google Chrome`; anything else unchanged.
pub fn strip_pid_prefix(app_id: &str) -> &str {
    if let Some(rest) = app_id.strip_prefix("pid:") {
        if let Some((pid, name)) = rest.split_once(':') {
            if !pid.is_empty() && pid.bytes().all(|b| b.is_ascii_digit()) {
                return name;
            }
        }
    }
    app_id
}

fn parse_time_window(window: &str) -> Option<(NaiveTime, NaiveTime)> {
    let (start, end) = window.trim().split_once('-')?;
    let parse = |s: &str| NaiveTime::parse_from_str(s.trim(), "%H:%M").ok();
    Some((parse(start)?, parse(end)?))
}

/// fnmatch-style glob: `*` any run, `?` one char, `[seq]` / `[!seq]` a set
/// (ranges `a-z` allowed). A `[` without a closing `]` is a literal. The
/// whole of `text` must match.
pub fn glob_match(pattern: &str, text: &str) -> bool {
    let p: Vec<char> = pattern.chars().collect();
    let t: Vec<char> = text.chars().collect();
    let (mut pi, mut ti) = (0usize, 0usize);
    // Backtrack point: pattern index after the last `*`, and the text
    // index that `*` currently extends to.
    let mut star: Option<(usize, usize)> = None;

    while ti < t.len() {
        let step = if pi < p.len() {
            match p[pi] {
                '*' => {
                    star = Some((pi + 1, ti));
                    pi += 1;
                    continue;
                }
                '?' => Some(pi + 1),
                '[' => match match_set(&p, pi, t[ti]) {
                    Some((true, next)) => Some(next),
                    Some((false, _)) => None,
                    None => (t[ti] == '[').then_some(pi + 1),
                },
                c => (c == t[ti]).then_some(pi + 1),
            }
        } else {
            None
        };
        match step {
            Some(next) => {
                pi = next;
                ti += 1;
            }
            None => match star {
                Some((star_pi, star_ti)) => {
                    pi = star_pi;
                    ti = star_ti + 1;
                    star = Some((star_pi, star_ti + 1));
                }
                None => return false,
            },
        }
    }
    p[pi..].iter().all(|&c| c == '*')
}

/// Match `c` against the set starting at `p[start] == '['`. Returns
/// `(matched, index after ']')`, or `None` when the set is unterminated.
fn match_set(p: &[char], start: usize, c: char) -> Option<(bool, usize)> {
    let mut i = start + 1;
    let negate = p.get(i) == Some(&'!');
    if negate {
        i += 1;
    }
    let mut matched = false;
    let mut first = true;
    loop {
        let &ch = p.get(i)?;
        if ch == ']' && !first {
            return Some((matched != negate, i + 1));
        }
        first = false;
        if p.get(i + 1) == Some(&'-') && p.get(i + 2).is_some_and(|&e| e != ']') {
            let end = p[i + 2];
            if ch <= c && c <= end {
                matched = true;
            }
            i += 3;
        } else {
            if ch == c {
                matched = true;
            }
            i += 1;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn zones(blocked: &[&str], metadata_only: &[&str], blocked_urls: &[&str]) -> PrivacyZones {
        let s = |v: &[&str]| v.iter().map(|x| x.to_string()).collect();
        PrivacyZones::new(&PrivacyZonesConfig {
            full_observation: Vec::new(),
            metadata_only: s(metadata_only),
            blocked: s(blocked),
            blocked_urls: s(blocked_urls),
            auto_pause: Vec::new(),
        })
    }

    fn noon() -> NaiveTime {
        NaiveTime::from_hms_opt(12, 0, 0).unwrap()
    }

    #[test]
    fn glob_matches_like_fnmatch() {
        assert!(glob_match("1password*", "1password 8"));
        assert!(glob_match("*bank*", "https://mybank.com/"));
        assert!(glob_match("*.gov/*", "https://irs.gov/refund"));
        assert!(!glob_match("*.gov/*", "https://irs.gov"));
        assert!(glob_match("a?c", "abc"));
        assert!(!glob_match("a?c", "ac"));
        assert!(glob_match("[a-c]x", "bx"));
        assert!(!glob_match("[!a-c]x", "bx"));
        assert!(glob_match("[!a-c]x", "dx"));
        assert!(glob_match("a[b", "a[b"), "unterminated set is literal");
        assert!(glob_match("*", ""));
        assert!(!glob_match("keychain access", "keychain access 2"));
        assert!(glob_match("*a*b*c", "xxaxxbxxbxc"));
    }

    #[test]
    fn pid_prefix_is_stripped() {
        assert_eq!(strip_pid_prefix("pid:412:1Password"), "1Password");
        assert_eq!(strip_pid_prefix("pid:412:Name: with colon"), "Name: with colon");
        assert_eq!(strip_pid_prefix("com.google.Chrome"), "com.google.Chrome");
        assert_eq!(strip_pid_prefix("pid:x:Foo"), "pid:x:Foo");
    }

    #[test]
    fn blocked_app_matches_app_or_title_case_insensitively() {
        let z = zones(&["Signal*"], &[], &[]);
        assert_eq!(z.zone("pid:9:signal", "Chats", None, noon()), Zone::Blocked);
        assert_eq!(z.zone("pid:9:TextEdit", "SIGNAL notes", None, noon()), Zone::Blocked);
        assert_eq!(z.zone("pid:9:TextEdit", "notes", None, noon()), Zone::Full);
    }

    #[test]
    fn defaults_block_password_managers_without_config() {
        let z = zones(&[], &[], &[]);
        assert_eq!(z.zone("pid:77:1Password", "", None, noon()), Zone::Blocked);
        assert_eq!(z.zone("pid:77:Keychain Access", "Login", None, noon()), Zone::Blocked);
    }

    #[test]
    fn defaults_block_system_authentication_uis() {
        let z = zones(&[], &[], &[]);
        for app in [
            "SecurityAgent",
            "coreautha",
            "loginwindow",
            "universalAccessAuthWarn",
            "Passwords",
            "ScreenSaverEngine",
            "UserNotificationCenter",
        ] {
            let app_id = format!("pid:9:{app}");
            assert_eq!(z.zone(&app_id, "", None, noon()), Zone::Blocked, "{app}");
        }
        // Matching is on the whole owner name: an app merely containing one
        // of these names is not blocked.
        assert_eq!(z.zone("pid:9:Passwords Helper Notes", "", None, noon()), Zone::Full);
    }

    #[test]
    fn metadata_only_and_precedence() {
        let z = zones(&["Messages"], &["Messages", "Slack"], &[]);
        assert_eq!(z.zone("pid:1:Slack", "general", None, noon()), Zone::MetadataOnly);
        assert_eq!(z.zone("pid:1:Messages", "", None, noon()), Zone::Blocked, "blocked beats metadata_only");

        let z = PrivacyZones::new(&PrivacyZonesConfig {
            full_observation: vec!["1password".into()],
            ..Default::default()
        });
        assert_eq!(z.zone("pid:1:1Password", "", None, noon()), Zone::Full, "full_observation unblocks a default");
    }

    #[test]
    fn blocked_url_applies_only_to_a_browser() {
        let z = zones(&[], &[], &["*vault.example.com*"]);
        let url = Some("https://vault.example.com/items");
        assert_eq!(z.zone("pid:5:Google Chrome", "Vault", url, noon()), Zone::Blocked);
        assert_eq!(z.zone("pid:5:Arc", "Vault", url, noon()), Zone::Blocked);
        assert_eq!(z.zone("pid:5:TextEdit", "Vault", url, noon()), Zone::Full);
        assert_eq!(z.zone("pid:5:Google Chrome", "Docs", Some("https://docs.example.com/"), noon()), Zone::Full);
        // Defaults apply too.
        assert_eq!(z.zone("pid:5:Google Chrome", "x", Some("https://www.mybank.com/"), noon()), Zone::Blocked);
    }

    #[test]
    fn auto_pause_blocks_everything_including_overnight() {
        let z = PrivacyZones::new(&PrivacyZonesConfig {
            auto_pause: vec!["22:00-06:00".into(), "garbage".into()],
            ..Default::default()
        });
        let at = |h, m| NaiveTime::from_hms_opt(h, m, 0).unwrap();
        assert_eq!(z.zone("pid:1:TextEdit", "", None, at(23, 0)), Zone::Blocked);
        assert_eq!(z.zone("pid:1:TextEdit", "", None, at(5, 59)), Zone::Blocked);
        assert_eq!(z.zone("pid:1:TextEdit", "", None, at(12, 0)), Zone::Full);
    }
}
