//! The RAM tier table: which models a Mac runs.
//!
//! The table's one source is `TIERS` in the worker's `model_profiles.py`.
//! `resources/tiers.json` is its committed JSON form (a worker test pins it
//! to the Python table), compiled in here; the installer writes the same
//! bytes to the data directory for the app's onboarding.  So the CLI, the
//! daemon's config defaults, onboarding and the worker name the same models.

use std::collections::BTreeMap;
use std::sync::OnceLock;

use serde::Deserialize;

/// `resources/tiers.json`, as the worker writes it.
pub const TIERS_JSON: &str = include_str!("../../../resources/tiers.json");

/// One row of the tier table.
#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
pub struct Tier {
    pub id: String,
    pub label: String,
    pub subtitle: String,
    pub min_ram_gb: u64,
    pub annotation_model: String,
    pub sop_model: String,
    /// The SOP model's A/B arm (`[vlm] sop_ab = true`); empty when none.
    #[serde(default)]
    pub ab_sop_model: String,
    /// The tier's earlier default models (`PREVIOUS_TIER_DEFAULTS` in the
    /// worker), which a shipped writer may have pinned in a config.
    #[serde(default)]
    pub previous_defaults: Vec<ModelPair>,
}

/// An `(annotation_model, sop_model)` pair.
#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
pub struct ModelPair {
    pub annotation_model: String,
    pub sop_model: String,
}

impl Tier {
    /// True when `annotation_model` + `sop_model` are one of this tier's
    /// earlier defaults (not its current one): a config still pinning them
    /// missed the worker's tier migration.
    pub fn is_previous_default(&self, annotation_model: &str, sop_model: &str) -> bool {
        (annotation_model, sop_model) != (self.annotation_model.as_str(), self.sop_model.as_str())
            && self
                .previous_defaults
                .iter()
                .any(|p| p.annotation_model == annotation_model && p.sop_model == sop_model)
    }
}

/// The whole of `tiers.json`.
#[derive(Debug, Clone, Deserialize)]
pub struct TierTable {
    pub schema_version: u32,
    pub basis: String,
    pub minimum_ollama_version_for_gemma4: String,
    pub embedding_model: String,
    pub model_sizes_gb: BTreeMap<String, f64>,
    /// Ascending by `min_ram_gb`; the first row takes any Mac.
    pub tiers: Vec<Tier>,
}

impl TierTable {
    /// The tier named `id`.
    pub fn get(&self, id: &str) -> Option<&Tier> {
        self.tiers.iter().find(|t| t.id == id)
    }

    /// The largest tier whose `min_ram_gb` fits `ram_gb`.
    pub fn recommend(&self, ram_gb: u64) -> &Tier {
        self.tiers
            .iter()
            .filter(|t| t.min_ram_gb <= ram_gb)
            .last()
            .unwrap_or(&self.tiers[0])
    }

    /// Every model `tier` needs pulled, annotation model first, no repeats.
    pub fn models<'a>(&'a self, tier: &'a Tier) -> Vec<&'a str> {
        let mut out: Vec<&str> = Vec::new();
        for m in [
            tier.annotation_model.as_str(),
            tier.sop_model.as_str(),
            self.embedding_model.as_str(),
        ] {
            if !out.contains(&m) {
                out.push(m);
            }
        }
        out
    }
}

/// The compiled-in tier table.
pub fn table() -> &'static TierTable {
    static TABLE: OnceLock<TierTable> = OnceLock::new();
    TABLE.get_or_init(|| {
        serde_json::from_str(TIERS_JSON).expect("resources/tiers.json is valid")
    })
}

/// Physical memory in whole GiB (what "a 16 GB Mac" reports); 0 if unknown.
pub fn machine_ram_gb() -> u64 {
    // SAFETY: sysconf only reads system configuration values.
    let (pages, page_size) =
        unsafe { (libc::sysconf(libc::_SC_PHYS_PAGES), libc::sysconf(libc::_SC_PAGESIZE)) };
    if pages <= 0 || page_size <= 0 {
        return 0;
    }
    (pages as u64 * page_size as u64) >> 30
}

/// The tier this Mac's RAM recommends.
pub fn recommended_tier() -> &'static Tier {
    table().recommend(machine_ram_gb())
}

/// The `[vlm]` models a config names: its `tier` (else the RAM tier)
/// supplies the keys it leaves out, as the worker's config loader does.
/// Returns `(tier, annotation_model, sop_model)`.
pub fn resolve_models(
    tier: Option<&str>,
    annotation_model: Option<&str>,
    sop_model: Option<&str>,
) -> (&'static Tier, String, String) {
    let tier = tier
        .and_then(|id| table().get(id))
        .unwrap_or_else(recommended_tier);
    let pick = |v: Option<&str>, fallback: &str| {
        v.filter(|s| !s.is_empty()).unwrap_or(fallback).to_string()
    };
    (
        tier,
        pick(annotation_model, &tier.annotation_model),
        pick(sop_model, &tier.sop_model),
    )
}

/// The keys of `[vlm]` a tier choice writes, in order.
const MANAGED_KEYS: [&str; 4] = ["mode", "tier", "annotation_model", "sop_model"];

/// The managed `[vlm]` lines for `tier`; byte-identical to the worker's
/// `model_profiles.vlm_block` and the app's `ModelTiers.vlmBlock`.
pub fn vlm_block(tier: &Tier) -> String {
    format!(
        "[vlm]\nmode = \"local\"\ntier = \"{}\"\nannotation_model = \"{}\"\nsop_model = \"{}\"\n",
        tier.id, tier.annotation_model, tier.sop_model
    )
}

/// `config` with `tier` chosen: the managed keys of `[vlm]` are replaced by
/// the tier's (right after the header, other `[vlm]` keys kept), or the
/// block is appended when there is no `[vlm]` section.  The app's
/// onboarding applies the same edit (`ModelTiers.applyVLMBlock`).
pub fn apply_vlm_block(config: &str, tier: &Tier) -> String {
    let block = vlm_block(tier);
    let lines: Vec<&str> = config.split_inclusive('\n').collect();
    let Some(header) = lines.iter().position(|l| l.trim() == "[vlm]") else {
        let mut out = config.to_string();
        if !out.is_empty() && !out.ends_with('\n') {
            out.push('\n');
        }
        out.push('\n');
        out.push_str(&block);
        return out;
    };
    let mut out: String = lines[..header].concat();
    out.push_str(&block);
    let mut in_section = true;
    for line in &lines[header + 1..] {
        let trimmed = line.trim();
        if trimmed.starts_with('[') {
            in_section = false;
        }
        let managed = in_section
            && trimmed
                .split_once('=')
                .map(|(k, _)| MANAGED_KEYS.contains(&k.trim()))
                .unwrap_or(false);
        if !managed {
            out.push_str(line);
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture(name: &str) -> String {
        let path = format!("{}/../../tests/fixtures/vlm_block/{}", env!("CARGO_MANIFEST_DIR"), name);
        std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{path}: {e}"))
    }

    #[test]
    fn table_parses_and_is_ordered() {
        let t = table();
        assert_eq!(t.schema_version, 1);
        assert_eq!(t.tiers[0].min_ram_gb, 0);
        assert!(t.tiers.windows(2).all(|w| w[0].min_ram_gb <= w[1].min_ram_gb));
        assert_eq!(t.minimum_ollama_version_for_gemma4, "0.34.0");
        assert_eq!(t.embedding_model, "nomic-embed-text");
    }

    #[test]
    fn recommend_by_ram() {
        let t = table();
        let ids: Vec<&str> = [8, 15, 16, 24, 36, 48, 128]
            .iter()
            .map(|&r| t.recommend(r).id.as_str())
            .collect();
        assert_eq!(
            ids,
            ["standard", "standard", "recommended", "performance", "performance",
             "max_quality", "max_quality"]
        );
    }

    #[test]
    fn annotator_is_the_small_model_on_every_tier() {
        assert!(table().tiers.iter().all(|t| t.annotation_model == "qwen3.5:4b"));
        let standard = table().get("standard").unwrap();
        assert_eq!(table().models(standard), ["qwen3.5:4b", "nomic-embed-text"]);
    }

    #[test]
    fn the_measured_split() {
        // WP-F6: 16 GB runs qwen3.5:4b for both roles; Gemma 4 writes Skills from 24 GB.
        let sop: Vec<(&str, &str)> = table()
            .tiers
            .iter()
            .map(|t| (t.id.as_str(), t.sop_model.as_str()))
            .collect();
        assert_eq!(
            sop,
            [("standard", "qwen3.5:4b"), ("recommended", "qwen3.5:4b"),
             ("performance", "gemma4:12b-it-qat"), ("max_quality", "gemma4:31b-it-qat")]
        );
        let recommended = table().get("recommended").unwrap();
        assert_eq!(table().models(recommended), ["qwen3.5:4b", "nomic-embed-text"]);
    }

    #[test]
    fn resolve_fills_missing_keys_from_the_tier() {
        let (tier, ann, sop) = resolve_models(Some("performance"), None, Some(""));
        assert_eq!(tier.id, "performance");
        assert_eq!((ann.as_str(), sop.as_str()), ("qwen3.5:4b", "gemma4:12b-it-qat"));
        let (_, ann, _) = resolve_models(Some("nope"), Some("custom:1"), None);
        assert_eq!(ann, "custom:1");
    }

    #[test]
    fn apply_matches_the_shared_fixture() {
        let tier = table().get("performance").unwrap();
        assert_eq!(apply_vlm_block(&fixture("input.toml"), tier), fixture("performance.toml"));
        // Idempotent.
        let once = apply_vlm_block(&fixture("input.toml"), tier);
        assert_eq!(apply_vlm_block(&once, tier), once);
    }

    /// The worker's tier migration (`model_profiles.migrate_vlm_block`)
    /// makes the same edit as `setup --tier`: a config on the 16 GB tier's
    /// pre-WP-F6 models becomes recommended-migrated.toml.
    #[test]
    fn apply_matches_the_migration_fixture() {
        let tier = table().get("recommended").unwrap();
        assert_eq!(
            apply_vlm_block(&fixture("recommended-previous.toml"), tier),
            fixture("recommended-migrated.toml")
        );
    }

    #[test]
    fn previous_defaults_come_from_tiers_json() {
        let recommended = table().get("recommended").unwrap();
        assert_eq!(
            recommended.previous_defaults,
            [ModelPair {
                annotation_model: "qwen3.5:4b".into(),
                sop_model: "gemma4:12b-it-qat".into()
            }]
        );
        assert!(recommended.is_previous_default("qwen3.5:4b", "gemma4:12b-it-qat"));
        assert!(!recommended.is_previous_default("qwen3.5:4b", "qwen3.5:4b"));
        assert!(!recommended.is_previous_default("qwen3.5:4b", "gemma4:31b-it-qat"));
        for id in ["standard", "performance", "max_quality"] {
            assert!(table().get(id).unwrap().previous_defaults.is_empty(), "{id}");
        }
    }

    #[test]
    fn apply_appends_to_a_config_without_vlm() {
        let tier = table().get("standard").unwrap();
        let out = apply_vlm_block("[embedding]\nimage_embeddings = false", tier);
        assert_eq!(out, format!("[embedding]\nimage_embeddings = false\n\n{}", vlm_block(tier)));
    }
}
