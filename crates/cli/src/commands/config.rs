use anyhow::Result;

fn config_path() -> std::path::PathBuf {
    let home = std::env::var("HOME").unwrap_or_else(|_| "/tmp".to_string());
    if cfg!(target_os = "macos") {
        std::path::PathBuf::from(home)
            .join("Library/Application Support/oc-apprentice/config.toml")
    } else {
        std::path::PathBuf::from(home).join(".config/oc-apprentice/config.toml")
    }
}

pub fn show() -> Result<()> {
    let path = config_path();
    if path.exists() {
        let content = std::fs::read_to_string(&path)?;
        println!("{}", content);
    } else {
        println!("No config file found at: {}", path.display());
        println!();
        println!("To create one, copy the example config:");
        println!("  cp config.example.toml \"{}\"", path.display());
    }
    Ok(())
}

pub fn edit() -> Result<()> {
    let path = config_path();
    let editor = std::env::var("EDITOR").unwrap_or_else(|_| "nano".to_string());

    if !path.exists() {
        println!("Config file doesn't exist yet. Creating from defaults...");
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        // Write the embedded example config as a starting point
        std::fs::write(&path, include_str!("../../../../config.example.toml"))?;
    }

    let status = std::process::Command::new(&editor)
        .arg(path.display().to_string())
        .status()?;
    if !status.success() {
        eprintln!("Editor exited with error");
    }
    Ok(())
}

pub fn path() -> Result<()> {
    println!("{}", config_path().display());
    Ok(())
}
