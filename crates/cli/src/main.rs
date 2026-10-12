use anyhow::Result;
use clap::{Parser, Subcommand};

mod browser_extension;
mod commands;
mod display;
#[cfg(test)]
mod docs_tests;
mod machine;
mod ollama;
mod paths;
mod trigger;

#[derive(Parser)]
#[command(name = "agenthandover", version, about = "AgentHandover CLI — manage the apprentice system")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Show daemon and worker status
    Status {
        /// Show what capture and the models cost in the last hour
        #[arg(long)]
        compute: bool,
    },
    /// Start services (daemon launched directly, worker via launchd)
    Start {
        /// Which service to start: daemon, worker, or all (default)
        #[arg(default_value = "all")]
        service: String,
    },
    /// Stop services (daemon via SIGTERM, worker via launchd)
    Stop {
        /// Which service to stop: daemon, worker, or all (default)
        #[arg(default_value = "all")]
        service: String,
    },
    /// Restart services
    Restart {
        /// Which service to restart: daemon, worker, or all (default)
        #[arg(default_value = "all")]
        service: String,
    },
    /// View service logs
    Logs {
        /// Which service: daemon or worker
        #[arg(default_value = "daemon")]
        service: String,
        /// Follow the log (like tail -f)
        #[arg(short, long)]
        follow: bool,
        /// Number of lines to show
        #[arg(short = 'n', long, default_value = "50")]
        lines: usize,
    },
    /// Manage configuration
    Config {
        #[command(subcommand)]
        action: ConfigAction,
    },
    /// List and view generated Skills
    #[command(alias = "sops")]
    Skills {
        #[command(subcommand)]
        action: SopsAction,
    },
    /// Live-updating status dashboard (refreshes every 2s)
    Watch,
    /// Interactive setup wizard
    Setup {
        /// Check status only, don't modify anything
        #[arg(long)]
        check: bool,
        /// Set up Chrome extension only
        #[arg(long)]
        extension: bool,
        /// Set up VLM only
        #[arg(long)]
        vlm: bool,
        /// Model tier to set up (standard, recommended, performance,
        /// max_quality); default: the config's tier, else the RAM's
        #[arg(long)]
        tier: Option<String>,
        /// Install the opt-in zsh hook that records the commands you run
        #[arg(long, conflicts_with = "shell_remove")]
        shell: bool,
        /// Remove the zsh hook
        #[arg(long)]
        shell_remove: bool,
    },
    /// Run pre-flight checks
    Doctor {
        /// Print only the Ollama and memory checks, as JSON (what the
        /// app's onboarding shows)
        #[arg(long, conflicts_with = "extension_json")]
        ollama_json: bool,
        /// Print only whether the browser extension is built and loaded in
        /// a browser, as JSON (what the app's setup shows)
        #[arg(long)]
        extension_json: bool,
    },
    /// Uninstall AgentHandover
    Uninstall {
        /// Also remove user data (database, SOPs, config)
        #[arg(long)]
        purge_data: bool,
    },
    /// Focus recording mode — record a single workflow demonstration
    Focus {
        #[command(subcommand)]
        action: FocusAction,
    },
    /// Export SOPs in a specific format
    Export {
        /// Output format
        #[arg(long, default_value = "skill-md")]
        format: String,
        /// Export a specific SOP by slug (default: all)
        #[arg(long)]
        sop: Option<String>,
        /// Output directory (default: workspace/skills)
        #[arg(long)]
        output: Option<String>,
    },
    /// Search activity annotations (full-text search)
    Search {
        /// Search query (e.g. "domain research", "Stripe dashboard")
        query: String,
        /// Filter by date (YYYY-MM-DD)
        #[arg(long)]
        date: Option<String>,
        /// Filter by app name
        #[arg(long)]
        app: Option<String>,
        /// Maximum results to return
        #[arg(short = 'n', long, default_value = "20")]
        limit: usize,
    },
    /// Connect an AI agent (claude-code, codex, openclaw, hermes, mcp)
    Connect {
        /// Which agent to connect: claude-code, codex, openclaw, hermes, mcp
        agent: String,
        /// Options passed to the connector (codex: --path, --append, --force)
        #[arg(trailing_var_arg = true, allow_hyphen_values = true)]
        options: Vec<String>,
    },
    /// Model bake-off: replay a journey cassette's prompts against a tier's
    /// live models and report tok/s and a golden-diff score
    Bench {
        /// Tier to measure (standard, recommended, performance, max_quality)
        #[arg(long)]
        tier: String,
        /// Options passed to the bench (--cassette PATH, --host URL,
        /// --annotators a,b,c, --limit N, --out FILE)
        #[arg(trailing_var_arg = true, allow_hyphen_values = true)]
        options: Vec<String>,
    },
    /// Memory: search, forget or pause what was on your screen
    Memory {
        #[command(subcommand)]
        action: MemoryAction,
    },
    /// Recall what you were doing at a given time
    Recall {
        /// Date to recall (YYYY-MM-DD, default: today)
        #[arg(long)]
        date: Option<String>,
        /// Filter by app name
        #[arg(long)]
        app: Option<String>,
        /// Start time filter (HH:MM)
        #[arg(long)]
        start: Option<String>,
        /// End time filter (HH:MM)
        #[arg(long)]
        end: Option<String>,
    },
}

#[derive(Subcommand)]
enum MemoryAction {
    /// Search Memory by meaning ("the Acme invoice", "the error in the terminal")
    Search {
        /// What you are looking for
        query: String,
        /// Only moments after this time (2026-10-12 14:30, 14:30, 2026-10-12)
        #[arg(long)]
        since: Option<String>,
        /// Only moments before this time
        #[arg(long)]
        until: Option<String>,
        /// Only this app
        #[arg(long)]
        app: Option<String>,
        /// Maximum moments to show (1-100)
        #[arg(short = 'n', long, default_value = "20",
              value_parser = clap::value_parser!(u32).range(1..=100))]
        limit: u32,
        /// Print the API's JSON
        #[arg(long)]
        json: bool,
    },
    /// Forget a stretch of time, an app, a moment or everything (asks first)
    Forget {
        /// The last while (30m, 1h, 2d)
        #[arg(long)]
        last: Option<String>,
        /// Today so far
        #[arg(long)]
        today: bool,
        /// From this time (with --until, or up to now)
        #[arg(long)]
        since: Option<String>,
        /// Up to this time (needs --since)
        #[arg(long)]
        until: Option<String>,
        /// This app (on its own: on every day)
        #[arg(long)]
        app: Option<String>,
        /// One moment, by the id `memory search` shows
        #[arg(long)]
        moment: Option<String>,
        /// Everything Memory and the activity history hold
        #[arg(long)]
        all: bool,
        /// Do not ask for confirmation
        #[arg(short = 'y', long)]
        yes: bool,
    },
    /// Pause Memory (Skills still learn; pause observing in the app for that)
    Pause {
        /// For how long (30m, 1h, 2d); default: until you resume
        #[arg(long = "for", value_name = "DURATION")]
        duration: Option<String>,
    },
    /// Resume Memory
    Resume,
    /// What Memory holds, and whether it is on or paused
    Status {
        /// Print the API's JSON
        #[arg(long)]
        json: bool,
    },
    /// Let agents search Memory over MCP (descriptions, never images), or stop them
    Agents {
        /// on or off
        #[arg(value_parser = ["on", "off"])]
        state: String,
    },
}

#[derive(Subcommand)]
enum FocusAction {
    /// Start recording a workflow demonstration
    Start {
        /// A descriptive title for the workflow (e.g. "Expense report filing")
        title: String,
    },
    /// Stop the active focus recording session
    Stop,
    /// Answer the focus recording's pending questions in the terminal
    Finalize,
    /// Skip the pending questions (the Skill keeps them as not yet known)
    SkipQuestions,
}

#[derive(Subcommand)]
enum ConfigAction {
    /// Show current configuration
    Show,
    /// Open config file in $EDITOR
    Edit,
    /// Print the config file path
    Path,
}

#[derive(Subcommand)]
enum SopsAction {
    /// List all generated SOPs (exported + drafts)
    List,
    /// Show a specific SOP by slug (works for exported and draft SOPs)
    Show { slug: String },
    /// Print the SOPs directory path
    Dir,
    /// List draft SOPs awaiting review
    Drafts,
    /// Approve a draft SOP (agents see it only once promoted to agent_ready)
    Approve {
        /// SOP slug or ID
        slug_or_id: String,
    },
    /// Reject a draft SOP
    Reject {
        /// SOP slug or ID
        slug_or_id: String,
    },
    /// Promote a procedure's lifecycle state (e.g., draft → reviewed → verified → agent_ready)
    Promote {
        /// Procedure slug
        slug: String,
        /// Target lifecycle state: draft, reviewed, verified, agent_ready
        to_state: String,
    },
    /// List failed SOP generations
    Failed,
    /// Retry a failed SOP generation
    Retry {
        /// Failure ID to retry
        failure_id: String,
    },
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    match cli.command {
        Commands::Status { compute: false } => commands::status::run(),
        Commands::Status { compute: true } => commands::status::run_compute(),
        Commands::Start { service } => commands::service::start(&service),
        Commands::Stop { service } => commands::service::stop(&service),
        Commands::Restart { service } => commands::service::restart(&service),
        Commands::Logs {
            service,
            follow,
            lines,
        } => commands::logs::run(&service, follow, lines),
        Commands::Config { action } => match action {
            ConfigAction::Show => commands::config::show(),
            ConfigAction::Edit => commands::config::edit(),
            ConfigAction::Path => commands::config::path(),
        },
        Commands::Skills { action } => match action {
            SopsAction::List => commands::sops::list(),
            SopsAction::Show { slug } => commands::sops::show(&slug),
            SopsAction::Dir => commands::sops::dir(),
            SopsAction::Drafts => commands::sops::drafts(),
            SopsAction::Approve { slug_or_id } => commands::sops::approve(&slug_or_id),
            SopsAction::Reject { slug_or_id } => commands::sops::reject(&slug_or_id),
            SopsAction::Promote { slug, to_state } => commands::sops::promote(&slug, &to_state),
            SopsAction::Failed => commands::sops::failed(),
            SopsAction::Retry { failure_id } => commands::sops::retry(&failure_id),
        },
        Commands::Watch => commands::watch::run(),
        Commands::Setup { shell: true, .. } => commands::shell_hook::run(false),
        Commands::Setup { shell_remove: true, .. } => commands::shell_hook::run(true),
        Commands::Setup { check, extension, vlm, tier, .. } => {
            commands::setup::run(check, extension, vlm, tier.as_deref())
        }
        Commands::Doctor { ollama_json: true, .. } => commands::doctor::print_ollama_json(),
        Commands::Doctor { extension_json: true, .. } => commands::doctor::print_extension_json(),
        Commands::Doctor { .. } => commands::doctor::run(),
        Commands::Uninstall { purge_data } => commands::uninstall::run(purge_data),
        Commands::Focus { action } => match action {
            FocusAction::Start { title } => commands::focus::start(&title),
            FocusAction::Stop => commands::focus::stop(),
            FocusAction::Finalize => commands::focus::finalize(),
            FocusAction::SkipQuestions => commands::focus::skip_questions(),
        },
        Commands::Connect { agent, options } => {
            // Delegate to the Python connector: the entry point on PATH, else
            // the pkg venv. Fall back only when the first cannot be started;
            // a connector that ran and refused (exit 1) is not run twice.
            let status = std::process::Command::new("agenthandover-connect")
                .arg(&agent)
                .args(&options)
                .status()
                .or_else(|_| {
                    std::process::Command::new("/usr/local/lib/agenthandover/venv/bin/python")
                        .args(["-m", "agenthandover_worker.agent_connect", &agent])
                        .args(&options)
                        .status()
                });
            match status {
                Ok(s) if s.success() => Ok(()),
                Ok(s) => std::process::exit(s.code().unwrap_or(1)),
                Err(_) => anyhow::bail!("Could not run agent connect. Is the worker installed?"),
            }
        }
        Commands::Bench { tier, options } => {
            // The bench lives in the worker (it replays cassettes through
            // the worker's Ollama client); run it with the worker's python.
            let Some(python) = paths::find_venv_python() else {
                anyhow::bail!("Could not find the worker's Python environment. Is the worker installed?");
            };
            let status = std::process::Command::new(python)
                .args(["-m", "agenthandover_worker.bench", "--tier", &tier])
                .args(&options)
                .status()?;
            if !status.success() {
                std::process::exit(status.code().unwrap_or(1));
            }
            Ok(())
        }
        Commands::Export { format, sop, output } => {
            commands::export::run(&format, sop.as_deref(), output.as_deref())
        }
        Commands::Search {
            query,
            date,
            app,
            limit,
        } => commands::search::search(&query, date.as_deref(), app.as_deref(), limit),
        Commands::Memory { action } => run_memory(action),
        Commands::Recall {
            date,
            app,
            start,
            end,
        } => commands::search::recall(
            date.as_deref(),
            app.as_deref(),
            start.as_deref(),
            end.as_deref(),
        ),
    }
}

fn run_memory(action: MemoryAction) -> Result<()> {
    use commands::memory;
    let api = memory::worker_api();
    let (now, tz) = memory::now();
    let out = &mut std::io::stdout();
    match action {
        MemoryAction::Search { query, since, until, app, limit, json } => {
            let args = memory::SearchArgs {
                query: &query,
                since: since.as_deref(),
                until: until.as_deref(),
                app: app.as_deref(),
                limit,
                json,
            };
            memory::search(&api, out, &args, now, &tz)
        }
        MemoryAction::Forget { last, today, since, until, app, moment, all, yes } => {
            let args = memory::ForgetArgs {
                last: last.as_deref(),
                today,
                since: since.as_deref(),
                until: until.as_deref(),
                app: app.as_deref(),
                moment: moment.as_deref(),
                all,
                yes,
            };
            memory::forget(&api, out, &mut std::io::stdin().lock(), &args, now, &tz)
        }
        MemoryAction::Pause { duration } => memory::pause(&api, out, duration.as_deref(), now, &tz),
        MemoryAction::Resume => memory::resume(&api, out),
        MemoryAction::Status { json } => memory::status(&api, out, json, now, &tz),
        MemoryAction::Agents { state } => memory::agents_switch(state == "on"),
    }
}
