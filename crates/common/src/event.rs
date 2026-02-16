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
    pub artifact_ids: Vec<Uuid>,
    pub metadata: serde_json::Value,
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
    },
    PasteDetected {
        matched_copy_hash: Option<String>,
    },
    SecureFieldFocus,
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
