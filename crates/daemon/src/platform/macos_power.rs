use anyhow::Result;
use tracing::warn;

/// Power state information from IOKit.
#[derive(Debug, Clone)]
pub struct PowerState {
    pub is_on_ac: bool,
    pub battery_percent: Option<u32>,
    pub is_charging: bool,
}

/// Get current power state using IOKit.
pub fn get_power_state() -> Result<PowerState> {
    // Use IOKit via system command as a portable approach
    // Full IOKit FFI can be added later for better performance
    let output = std::process::Command::new("pmset")
        .arg("-g")
        .arg("batt")
        .output()?;

    let stdout = String::from_utf8_lossy(&output.stdout);

    let is_on_ac = stdout.contains("AC Power");
    let is_charging = stdout.contains("charging") && !stdout.contains("not charging");

    // Parse battery percentage
    let battery_percent = stdout
        .lines()
        .find(|l| l.contains('%'))
        .and_then(|l| {
            l.split('\t')
                .find(|s| s.contains('%'))
                .and_then(|s| s.trim().split('%').next().and_then(|n| n.trim().parse::<u32>().ok()))
        });

    Ok(PowerState {
        is_on_ac,
        battery_percent,
        is_charging,
    })
}

/// Get CPU temperature (approximate, via thermal pressure).
/// Returns None if not available.
pub fn get_cpu_temperature_celsius() -> Option<f64> {
    // macOS doesn't expose exact CPU temp via public API easily
    // We use thermal pressure as a proxy
    let output = std::process::Command::new("pmset")
        .arg("-g")
        .arg("therm")
        .output()
        .ok()?;

    let stdout = String::from_utf8_lossy(&output.stdout);

    // Parse CPU_Speed_Limit (100 = cool, lower = throttled)
    if stdout.contains("CPU_Speed_Limit") {
        // If speed limit is < 100, system is thermally constrained
        // Map this to an approximate temperature
        if let Some(limit) = stdout
            .lines()
            .find(|l| l.contains("CPU_Speed_Limit"))
            .and_then(|l| l.split_whitespace().last())
            .and_then(|n| n.parse::<u32>().ok())
        {
            // Rough mapping: 100 = ~50C, 80 = ~70C, 50 = ~90C
            let approx_temp = 50.0 + (100.0 - limit as f64) * 0.5;
            return Some(approx_temp);
        }
    }

    None
}

/// Check if system meets power/thermal requirements for heavy work.
pub fn check_power_gates(
    require_ac: bool,
    min_battery_percent: u32,
    max_temp_c: u32,
    max_cpu_percent: u32,
) -> Result<PowerGateResult> {
    let power = get_power_state()?;
    let temp = get_cpu_temperature_celsius();

    let mut gates_passed = true;
    let mut reasons = Vec::new();

    if require_ac && !power.is_on_ac {
        gates_passed = false;
        reasons.push("Not on AC power".to_string());
    }

    if let Some(batt) = power.battery_percent {
        if batt < min_battery_percent {
            gates_passed = false;
            reasons.push(format!(
                "Battery {}% < {}% minimum",
                batt, min_battery_percent
            ));
        }
    }

    if let Some(t) = temp {
        if t > max_temp_c as f64 {
            gates_passed = false;
            reasons.push(format!("Temperature {:.0}\u{00B0}C > {}\u{00B0}C max", t, max_temp_c));
        }
    }

    // CPU usage check via sysctl
    let cpu_percent = get_cpu_usage_percent().unwrap_or(0);
    if cpu_percent > max_cpu_percent {
        gates_passed = false;
        reasons.push(format!("CPU {}% > {}% max", cpu_percent, max_cpu_percent));
    }

    if !gates_passed {
        warn!(reasons = ?reasons, "Power/thermal gates not met");
    }

    Ok(PowerGateResult {
        passed: gates_passed,
        power_state: power,
        cpu_temp_c: temp,
        cpu_percent,
        rejection_reasons: reasons,
    })
}

#[derive(Debug)]
pub struct PowerGateResult {
    pub passed: bool,
    pub power_state: PowerState,
    pub cpu_temp_c: Option<f64>,
    pub cpu_percent: u32,
    pub rejection_reasons: Vec<String>,
}

fn get_cpu_usage_percent() -> Option<u32> {
    let output = std::process::Command::new("ps")
        .args(["-A", "-o", "%cpu"])
        .output()
        .ok()?;

    let stdout = String::from_utf8_lossy(&output.stdout);
    let total: f64 = stdout
        .lines()
        .skip(1) // header
        .filter_map(|l| l.trim().parse::<f64>().ok())
        .sum();

    // Divide by number of cores for per-core average
    let cores = num_cpus().unwrap_or(1);
    Some((total / cores as f64) as u32)
}

fn num_cpus() -> Option<u32> {
    let output = std::process::Command::new("sysctl")
        .args(["-n", "hw.ncpu"])
        .output()
        .ok()?;
    String::from_utf8_lossy(&output.stdout)
        .trim()
        .parse()
        .ok()
}
