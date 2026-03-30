use anyhow::Result;
use colored::Colorize;

pub fn run(purge_data: bool) -> Result<()> {
    println!("{}", "AgentHandover Uninstaller".bold());
    println!();

    // Step 1: Stop services
    println!("Stopping services...");
    super::service::stop("all").ok();

    // Step 2: Remove launchd plists
    let home = std::env::var("HOME").unwrap_or_else(|_| "/tmp".to_string());
    let launch_agents = std::path::PathBuf::from(&home).join("Library/LaunchAgents");

    for label in &[
        "com.agenthandover.daemon.plist",
        "com.agenthandover.worker.plist",
    ] {
        let path = launch_agents.join(label);
        if path.exists() {
            std::fs::remove_file(&path)?;
            println!("  Removed {}", path.display());
        }
    }

    // Step 3: Remove native messaging host manifest from all supported browsers
    let manifest_name = "com.agenthandover.host.json";
    let mut nm_removed = false;
    for dir in crate::paths::native_messaging_hosts_dirs() {
        let nm_path = dir.join(manifest_name);
        if nm_path.exists() {
            std::fs::remove_file(&nm_path)?;
            println!("  Removed {}", nm_path.display());
            nm_removed = true;
        }
    }
    if !nm_removed {
        println!("  No native messaging host manifests found");
    }

    // Step 4: Remove PID files
    let data_dir = agenthandover_common::status::status_dir();
    for pid_file in &["daemon.pid", "worker.pid"] {
        let path = data_dir.join(pid_file);
        if path.exists() {
            std::fs::remove_file(&path)?;
        }
    }

    // Step 5: Remove status files
    for status_file in &["daemon-status.json", "worker-status.json"] {
        let path = data_dir.join(status_file);
        if path.exists() {
            std::fs::remove_file(&path)?;
        }
    }

    if purge_data {
        println!();
        println!("{}", "Purging user data...".yellow());
        if data_dir.exists() {
            std::fs::remove_dir_all(&data_dir)?;
            println!("  Removed {}", data_dir.display());
        }
    } else {
        println!();
        println!("User data preserved at: {}", data_dir.display());
        println!("To also remove data, run: agenthandover uninstall --purge-data");
    }

    println!();
    println!("{}", "Uninstall complete.".green().bold());
    println!(
        "You may also want to remove /usr/local/bin/agenthandover and /usr/local/bin/agenthandover-daemon"
    );
    Ok(())
}
