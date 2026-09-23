---
name: daily-news-digest
description: Creates a daily news digest note in the Notes app for personal use to organize technical news from Hacker News and other sources on a daily basis.
argument-hint: '[sqlite_url] [vision_guide_url] [cassette_recorder_url]'
allowed-tools: Read, WebFetch
---

# Daily news digest

Collect relevant technical news from Hacker News and other sources, then organize them into a structured daily news note in the Notes app.

Outcome: A formatted daily news note containing titles, URLs, and summaries of gathered technical articles.

## Goal

Creates a daily news digest note in the Notes app for personal use to organize technical news from Hacker News and other sources on a daily basis.

## Prerequisites

- Access to Hacker News
- Access to the Notes app
- Access to research and blog websites

## When to use

- When you want to curate and save a list of interesting technical articles and links for later reference.
- Strategy: The user identifies high-value technical content on Hacker News, such as "SQLite's WAL mode, explained with diagrams," and performs deep reading in Google Chrome to evaluate the content before curation. They then systematically synthesize these stories into a structured "Daily news" note, using numbered lists, labels, and brief summaries to create a searchable archive of links.

## Inputs

- `sqlite_url` (url, required): The URL of the SQLite WAL article. (example `https://blog.example.org/sqlite-wal-explained`)
- `vision_guide_url` (url, required): The URL of the on-device vision guide. (example `https://research.example.com/on-device-vision-guide`)
- `cassette_recorder_url` (url, required): The URL of the HTTP cassette recorder repository. (example `https://github.com/example/cassette`)

Steps refer to inputs as `{{name}}`.

## Steps

1. [step_1] Navigate to Hacker News
   Detail: Open the Hacker News homepage to browse the latest top stories.
   App: Google Chrome
   Target: https://news.ycombinator.com/
   Verify: The Hacker News homepage is visible with a list of stories.
2. [step_2] Click on the SQLite WAL mode story
   Detail: Navigate to the specific article about SQLite's Write-Ahead Logging to read the content.
   App: Google Chrome
   Target: https://news.ycombinator.com/
   Input: SQLite's WAL mode, explained with diagrams
   Verify: The browser navigates to {{sqlite_url}}.
3. [step_3] Copy the SQLite article title and URL
   Detail: Copy the specific title and URL from the blog page to include in the notes.
   App: Google Chrome
   Target: {{sqlite_url}}
   Input: SQLite's WAL mode, explained with diagrams | {{sqlite_url}}
   Verify: The content is copied to the clipboard.
4. [step_4] Create a new note in the Notes app
   Detail: Switch to the Notes app and create a new entry for the daily news.
   App: Notes
   Target: Notes app
   Verify: A new note screen is open with a title field.
5. [step_5] Paste the SQLite URL into the labels field
   Detail: Paste the copied URL into the labels section of the new note.
   App: Notes
   Target: Notes app - New Note
   Input: {{sqlite_url}}
   Verify: The URL appears in the labels field of the note.
6. [step_6] Type the SQLite article title into the note body
   Detail: Enter the descriptive title of the article into the main text area of the note.
   App: Notes
   Target: Notes app - New Note
   Input: SQLite's WAL mode, explained with diagrams
   Verify: The text 'SQLite's WAL mode, explained with diagrams' is visible in the note body.
7. [step_7] Navigate to the Daily news - 15 Sep note
   Detail: Switch to the existing daily news note for the current date.
   App: Notes
   Target: Daily news - 15 Sep
   Verify: The 'Daily news - 15 Sep' note is displayed.
8. [step_8] Paste the URL into the title field
   Detail: Add the article URL to the title of the note for easy access.
   App: Notes
   Target: Daily news - 15 Sep
   Input: {{sqlite_url}}
   Verify: The URL is appended to the title field.
9. [step_9] Format the note title with a numbered list
   Detail: Add '1. ' prefix to the note title to organize it as a list.
   App: Notes
   Target: Daily news - 15 Sep
   Input: 1.
   Verify: The title now begins with '1. '.
10. [step_10] Paste the URL into the note text field
   Detail: Paste the article URL into the body of the note.
   App: Notes
   Target: Daily news - 15 Sep
   Input: {{sqlite_url}}
   Verify: The URL appears in the text field of the note.
11. [step_11] Type the full numbered title
   Detail: Complete the title entry with the full description.
   App: Notes
   Target: Daily news - 15 Sep
   Input: SQLite's WAL mode, explained with diagrams
   Verify: The title field displays '1. SQLite's WAL mode, explained with diagrams'.
12. [step_12] Add a descriptive label to the note
   Detail: Add a summary label to the note to provide context on the content.
   App: Notes
   Target: Daily news - 15 Sep
   Input: Readers keep going while one writer appends; checkpoints fold the log back.
   Verify: The summary text appears in the labels section of the note.
13. [step_13] Copy the on-device vision guide URL
   Detail: Navigate to and copy the URL for the second news item.
   App: Google Chrome
   Target: {{vision_guide_url}}
   Input: A field guide to on-device vision models | {{vision_guide_url}}
   Verify: The URL and title are copied to the clipboard.
14. [step_14] Copy the HTTP cassette recorder link
   Detail: Navigate to Hacker News and copy the third news item link.
   App: Google Chrome
   Target: https://news.ycombinator.com
   Input: Show HN: A tiny HTTP cassette recorder for tests | {{cassette_recorder_url}}
   Verify: The GitHub link is copied to the clipboard.
15. [step_15] Verify the final news list in Notes
   Detail: Review the compiled list of news items in the Notes app.
   App: Notes
   Target: Daily news - 15 Sep
   Verify: The note contains three numbered items with their respective links and summaries.
16. [step_16] Review news in Slack
   Detail: Check the daily-news channel in Slack to see if the gathered news matches the team's feed.
   App: Google Chrome
   Target: https://example.com/slack
   Verify: The Slack channel #daily-news is visible with the news items.

## Success criteria

- A note exists in the Notes app titled 'Daily news - 15 Sep' containing at least three numbered entries with URLs and summaries.
- The final Slack page is visible showing the news items.

## Common errors

- Incorrectly pasting the URL into the title field instead of the body
- fix by deleting the text and re-pasting in the correct field.
- Forgetting to add the '1. ' prefix
- fix by manually editing the note title to include numbering.

## Decisions

- If The user finds a story on Hacker News that requires further context or validation.: Navigates to the external URL in Chrome and reads the full article before adding it to the Notes app.
- Choose by: Technical depth and educational value (confidence 90%)
  - Example: SQLite's WAL mode, explained with diagrams
  - Example: Designing a scheduler that respects laptop batteries
- Choose by: High engagement/interest on Hacker News (confidence 80%)
  - Example: The quiet death of the RSS reader, and its return
- Open question the user did not answer: "Where are the 'on-device vision guide' and 'HTTP cassette recorder' links located? They are not on Hacker News, so I need the specific URL or page to find them." Ask the user; do not guess.
- Open question the user did not answer: "For Step 7, should I dynamically generate the note title based on the current date (e.g., 'Daily news - [Today's Date]'), or is there a specific note I should always target?" Ask the user; do not guess.
- Open question the user did not answer: "How should I determine the number for the 'numbered list' in the note title? Should it be the sequential number of items added to the note today?" Ask the user; do not guess.

## Guardrails

- Never adds a link to the digest without first reading the content in the browser.
- Always uses a specific 'Daily news - [Date]' note structure to maintain chronological organization.
- Trust level: observe

## Voice

- Output template: {{Number}}. {{Title}} (variables: Number, Title)
- Output template: Label: {{URL}} (variables: URL)

## Timing

- Typical duration: 30 minutes
- Research and Curation (~20 min): Browsing Hacker News and reading selected articles in Google Chrome to determine relevance.
- Note Organization (~7 min): Creating/updating the daily note in the Notes app, including titles, URLs, and summaries.
- Verification (~3 min): Reviewing the final list in Notes and checking Slack for additional context.

## Evidence

- Observed 1 demonstration(s); confidence 0.91
- Learned from: focus

## Execution protocol

If you have access to the AgentHandover MCP server, report your progress:
1. Call `report_execution_start` with slug `daily-news-digest` before starting.
2. Call `report_step_result` after each step with the id shown in brackets (for example `step_1`). If you did something other than the step says, set `status` to `deviated` and describe what you did in `actual_action`.
3. Call `report_execution_complete` when finished.

This feedback improves the Skill for future runs.

---
*Generated by AgentHandover v0.4.0 (procedure schema 3.2.0)*
