use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Event {
    pub id: Uuid,
    pub timestamp: DateTime<Utc>,
    pub kind: EventKind,
    pub window: Option<WindowInfo>,
    pub display_topology: Vec<DisplayInfo>,
    pub primary_display_id: String,
    pub cursor_global_px: Option<CursorPosition>,
    pub ui_scale: Option<f64>,
    pub artifact_ids: Vec<String>,
    pub metadata: serde_json::Value,
    /// IDs of displays that the window spans across, if it spans multiple.
    /// None if there is no window or it fits within a single display.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub display_ids_spanned: Option<Vec<u32>>,
    /// The focused UI element as Accessibility reported it when the event
    /// was taken (the observer's dwell, scroll-read and focus-recording
    /// snapshots).  Stored in the `focused_element_json` column.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub focused_element: Option<FocusedElement>,
}

/// The focused UI element, read by AgentHandover.app through Accessibility
/// (`ObservationService.focusedElement`) and served over the observation
/// socket's `focused_element` command.
///
/// `value` is present only when `[privacy] capture_input_values` is on; a
/// secure (password) field is never read at all.  `ancestors` runs from the
/// parent up towards the window (at most six levels), so a cell or an
/// unlabelled field can be named by its row, group or toolbar.
/// `column_header` is the title of a table cell's column header when the
/// app exposes one.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
pub struct FocusedElement {
    pub role: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub subrole: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub title: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub value: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub column_header: Option<String>,
    /// The window's `AXDocument` (a file URL) when the app exposes one.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub document: Option<String>,
    /// The `AXURL` of the enclosing web area, when there is one.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub url: Option<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub ancestors: Vec<AxAncestor>,
}

/// One ancestor of the focused element.  `index` is the position, among
/// this ancestor's children, of the child on the path to the focused
/// element.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
pub struct AxAncestor {
    pub role: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub title: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub index: Option<u32>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "type")]
pub enum EventKind {
    FocusChange,
    WindowTitleChange,
    ClickIntent { target_description: String },
    DwellSnapshot,
    ScrollReadSnapshot,
    ClipboardChange {
        content_types: Vec<String>,
        byte_size: u64,
        high_entropy: bool,
        content_hash: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        content_preview: Option<String>,
    },
    /// The user pasted (a `paste` event in the browser extension).
    /// `matched_copy_hash` is the SHA-256 of the pasteboard content at that
    /// moment when a `ClipboardChange` with the same hash was recorded in
    /// the last 30 minutes, else None.
    PasteDetected {
        matched_copy_hash: Option<String>,
    },
    SecureFieldFocus,
    /// A command run in an interactive shell that has the opt-in hook
    /// (`agenthandover setup --shell`) installed.
    ShellCommand {
        cwd: String,
        command: String,
    },
    /// The Mac slept between `slept_at` and `woke_at` (NSWorkspace
    /// notifications seen by the app).  A hard boundary: no episode or
    /// task segment spans it.
    SystemSleep {
        slept_at: DateTime<Utc>,
        woke_at: DateTime<Utc>,
    },
    AppSwitch {
        from_app: String,
        to_app: String,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WindowInfo {
    pub window_id: String,
    pub app_id: String,
    pub title: String,
    pub bounds_global_px: [i32; 4],
    pub z_order: u32,
    pub is_fullscreen: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DisplayInfo {
    pub display_id: String,
    pub bounds_global_px: [i32; 4],
    pub scale_factor: f64,
    pub orientation: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CursorPosition {
    pub x: i32,
    pub y: i32,
}
