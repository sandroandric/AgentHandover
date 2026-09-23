---
name: daily-news-digest
description: 'Posts a daily digest of 3 tech articles to the #daily-news Slack channel to keep the team updated on database internals, on-device AI, and testing tools.'
argument-hint: '[article_title] [article_url] [summary_text] [note_content]'
allowed-tools: Read, WebFetch
---

# Daily news digest

Collects articles from Hacker News, research guides, and GitHub projects, summarizes them in a Notes app, and posts the compiled digest to a Slack channel.

Outcome: A comprehensive daily news digest containing summaries and links to selected articles is posted to the #daily-news Slack channel.

## Goal

Posts a daily digest of 3 tech articles to the #daily-news Slack channel to keep the team updated on database internals, on-device AI, and testing tools.

## Prerequisites

- Access to Google Chrome browser
- Access to the Notes app
- Access to the Slack application
- Account on Hacker News
- Account on Slack (#daily-news channel)

## When to use

- When a user wants to curate and share their daily reading of technology news with a team or group.
- Strategy: The user collects articles from Hacker News, research guides, and GitHub projects, copying the title and URL for each item into a specific note titled 'Daily news - 15 Sep' before summarizing key observations. They then copy the full content of this note and paste it directly into the Slack message box to share the curated list with the team.

## Inputs

- `article_title` (text, required): The title of a news article selected from the feed. (example `SQLite's WAL mode, explained with diagrams`)
- `article_url` (url, required): The URL of the article being processed. (example `https://blog.example.org/sqlite-wal-explained`)
- `summary_text` (text, required): The specific observation or summary text added to the note. (example `Readers keep going while one writer appends; checkpoints fold the log back.`)
- `note_content` (text, required): The complete text of the daily news note. (example `Daily news - 15 Sep | 1. SQLite's WAL mode, explained with diagrams | https://blog.example.org/sqlite-wal-explained | Readers keep going while one writer appends; checkpoints fold the log back.`)

Steps refer to inputs as `{{name}}`.

## Steps

1. [step_1] Navigate to the Hacker News homepage
   Detail: Open the browser and go to the main feed to identify relevant articles for the daily digest.
   App: Google Chrome
   Target: https://news.ycombinator.com/
   Verify: The Hacker News homepage displays the list of top stories.
   Anchor: link "Show HN: A tiny HTTP cassette recorder for tests" (`role=link[name="Show HN: A tiny HTTP cassette recorder for tests"]`)
2. [step_2] Click on the article titled 'SQLite's WAL mode, explained with diagrams'
   Detail: Select the specific article to read and gather information for the first news item.
   App: Google Chrome
   Target: https://news.ycombinator.com/
   Verify: The browser navigates to the article page at {{article_url}}.
3. [step_3] Copy the article title and URL
   Detail: Capture the metadata for the first news item to include in the Notes app.
   App: Google Chrome
   Target: {{article_url}}
   Verify: The text is copied to the clipboard (indicated by the system action log).
4. [step_4] Open the Notes app
   Detail: Access the application where the daily news will be compiled.
   App: Notes
   Target: Notes
   Verify: The Notes app interface is visible.
5. [step_5] Open the 'Daily news - 15 Sep' note
   Detail: Locate the existing note for the current date to add the first article.
   App: Notes
   Target: Notes
   Verify: The note titled 'Daily news - 15 Sep' is open and editable.
6. [step_6] Paste the title and URL for the SQLite article
   Detail: Insert the previously copied information as the first entry in the note.
   App: Notes
   Target: Daily news - 15 Sep
   Input: {{article_title}} | {{article_url}}
   Verify: The text appears in the note body.
7. [step_7] Type the summary sentence about SQLite WAL mode
   Detail: Add the specific observation or takeaway from the article to the note.
   App: Notes
   Target: Daily news - 15 Sep
   Input: {{summary_text}}
   Verify: The sentence is appended to the note content.
8. [step_8] Navigate to the research article about on-device vision models
   Detail: Find the second article to include in the digest.
   App: Google Chrome
   Target: https://news.ycombinator.com/
   Verify: The browser navigates to https://research.example.com/on-device-vision-guide.
9. [step_9] Copy the title and URL of the on-device vision guide
   Detail: Capture the metadata for the second news item.
   App: Google Chrome
   Target: https://research.example.com/on-device-vision-guide
   Verify: The text is copied to the clipboard.
10. [step_10] Paste the title and URL for the on-device vision guide
   Detail: Insert the second article's metadata into the Notes app.
   App: Notes
   Target: Daily news - 15 Sep
   Input: A field guide to on-device vision models | https://research.example.com/on-device-vision-guide
   Verify: The text appears in the note body.
11. [step_11] Type the observation about quantised models
   Detail: Add the specific observation from the research article to the note.
   App: Notes
   Target: Daily news - 15 Sep
   Input: Small quantised models now read screenshots well enough for UI tasks.
   Verify: The sentence is appended to the note content.
12. [step_12] Navigate to the GitHub repository page
   Detail: Find the third article to include in the digest.
   App: Google Chrome
   Target: https://news.ycombinator.com/
   Verify: The browser navigates to https://github.com/example/cassette.
13. [step_13] Copy the title and URL of the HTTP cassette recorder project
   Detail: Capture the metadata for the third news item.
   App: Google Chrome
   Target: https://github.com/example/cassette
   Verify: The text is copied to the clipboard.
14. [step_14] Paste the title and URL for the HTTP cassette recorder
   Detail: Insert the third article's metadata into the Notes app.
   App: Notes
   Target: Daily news - 15 Sep
   Input: Show HN: A tiny HTTP cassette recorder for tests | https://github.com/example/cassette
   Verify: The text appears in the note body.
15. [step_15] Type the observation about the cassette recorder
   Detail: Add the specific observation from the GitHub project page to the note.
   App: Notes
   Target: Daily news - 15 Sep
   Input: Records real API answers once and replays them byte for byte.
   Verify: The sentence is appended to the note content.
16. [step_16] Copy the full content of the Daily news note
   Detail: Capture the compiled digest to share it externally.
   App: Notes
   Target: Daily news - 15 Sep
   Verify: The system action log indicates content was copied to the clipboard.
17. [step_17] Navigate to the #daily-news Slack channel
   Detail: Open the destination channel to post the digest.
   App: Google Chrome
   Target: https://app.slack.com/client/T0EXAMPLE/C0DAILYNEWS
   Verify: The Slack channel #daily-news is active and the message composition box is visible.
   Anchor: textbox "Message #daily-news" (`[aria-label="Message #daily-news"]`)
18. [step_18] Paste the copied news digest into the Slack message box
   Detail: Insert the compiled note content into the Slack channel.
   App: Google Chrome
   Target: #daily-news
   Input: {{note_content}} | 2. A field guide to on-device vision models | https://research.example.com/on-device-vision-guide | Small quantised models now read screenshots well enough for UI tasks. | 3. Show HN: A tiny HTTP cassette recorder for tests | https://github.com/example/cassette | Records real API answers once and replays them byte for byte
   Verify: The full text of the digest appears in the Slack message composition box.
19. [step_19] Click the Send button to post the message
   Detail: Submit the message to the channel to complete the sharing process.
   App: Google Chrome
   Target: #daily-news
   Verify: The message appears in the channel history and the composition box is cleared.

## Success criteria

- The message containing the full daily news digest is successfully posted to the #daily-news Slack channel.
- The Notes app contains the three articles with their summaries.

## Common errors

- If the user accidentally clicks the wrong article, they should verify the title matches the intended article before proceeding.
- If the note is empty when trying to paste, the user must ensure they have opened the correct date-specific note.
- If the Slack message does not send, check if the user has permission to the #daily-news channel.

## Decisions

- If User has identified 3 distinct technical topics of interest.: User copies the full content of the 'Daily news - 15 Sep' note and pastes it into the Slack message box.
- Choose by: Selects specific high-interest technical topics from the available feed, skipping unrelated items.
  - Example: Selects 'SQLite's WAL mode, explained with diagrams' from the Hacker News feed (Frame 1) but skips 'The quiet death of the RSS reader, and its return' (Frame 1).
  - Example: Selects the GitHub project 'HTTP cassette recorder' (Frame 15) instead of other potential links.
- Open question the user did not answer: "Is the 'Daily news - 15 Sep' note already open in the Notes app, or do I need to create it if it doesn't exist?" Ask the user; do not guess.
- Open question the user did not answer: "What specific text should I type for the summaries in Steps 7, 9, and 11 since only generic descriptions ('SQLite WAL mode', 'quantised models', 'cassette recorder') are provided?" Ask the user; do not guess.
- Open question the user did not answer: "Are there any other articles currently visible on the Hacker News homepage that should be collected before clicking the specific SQLite link?" Ask the user; do not guess.

## Guardrails

- User never posts individual articles directly to Slack; they always compile them into a single note first.
- User never includes the full article text, only the title, URL, and a single summary sentence per item.
- Trust level: observe

## Voice

- Output template: 1. {{Title}} | {{URL}}
{{Summary Sentence}}
2. {{Title}} | {{URL}}
{{Summary Sentence}} (variables: Title, URL, Summary Sentence)

## Timing

- Collection (~0 min): Browsing Hacker News, research sites, and GitHub to identify and copy 3 items.
- Compilation (~0 min): Opening the 'Daily news - 15 Sep' note and pasting titles/URLs with typed summaries.
- Distribution (~0 min): Copying the note content and pasting it into the #daily-news Slack channel.

## Evidence

- Observed 1 demonstration(s); confidence 0.93
- Learned from: focus

## Execution protocol

If you have access to the AgentHandover MCP server, report your progress:
1. Call `report_execution_start` with slug `daily-news-digest` before starting.
2. Call `report_step_result` after each step with the id shown in brackets (for example `step_1`). If you did something other than the step says, set `status` to `deviated` and describe what you did in `actual_action`.
3. Call `report_execution_complete` when finished.

This feedback improves the Skill for future runs.

---
*Generated by AgentHandover v0.4.0 (procedure schema 3.2.0)*
