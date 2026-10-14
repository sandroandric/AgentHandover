---
name: daily-news-digest
description: 'Posts a daily digest of three tech news items (one from Hacker News, one research article, one GitHub project) to the #daily-news Slack channel to the team on a daily cadence.'
argument-hint: '[date] [article_title] [article_url] [summary_text]'
allowed-tools: Read, WebFetch
---

# Daily news digest

The user collects tech news articles from Hacker News, research guides, and GitHub repositories, summarizes them in a Notes app, and posts the compiled digest to a Slack channel.

Outcome: The user successfully compiles a daily news update with links and summaries from multiple sources and posts it to the #daily-news Slack channel.

## Goal

Posts a daily digest of three tech news items (one from Hacker News, one research article, one GitHub project) to the #daily-news Slack channel to the team on a daily cadence.

## Prerequisites

- Access to Google Chrome browser
- Access to the Notes app
- Access to the Slack app (via web or desktop)
- An account on Hacker News
- A Slack workspace with a #daily-news channel

## When to use

- When a user wants to curate and share a summary of technology news and research findings for a specific day.
- Strategy: The user iteratively collects three specific articles from Hacker News, research guides, and GitHub repositories, copying their titles and URLs into a Notes app to draft summaries before pasting the final content into the Slack message box. They explicitly navigate back to Hacker News between research items to find the next relevant story, ensuring the digest covers database internals, on-device vision models, and HTTP testing tools.

## Inputs

- `date` (date, required): The date of the daily news digest. (example `15 Sep`)
- `article_title` (text, required): The title of a news article being summarized. (example `SQLite's WAL mode, explained with diagrams`)
- `article_url` (url, required): The URL of the article being summarized. (example `https://blog.example.org/sqlite-wal-explained`)
- `summary_text` (text, required): A short sentence summarizing the article content. (example `Readers keep going while one writer appends; checkpoints fold the log back.`)

Steps refer to inputs as `{{name}}`.

## Steps

1. [step_1] Navigate to the Hacker News homepage
   Detail: Open the browser and go to the main feed to identify relevant articles for the daily digest.
   App: Google Chrome
   Target: https://news.ycombinator.com/
   Verify: The Hacker News feed page loads, displaying the list of top stories.
2. [step_2] Click on the article titled 'SQLite's WAL mode, explained with diagrams'
   Detail: Select the first relevant article to read its content and gather information for the summary.
   App: Google Chrome
   Target: https://news.ycombinator.com/
   Verify: The browser navigates to the article page at {{article_url}}.
   Anchor: link "SQLite's WAL mode, explained with diagrams" (`role=link[name="SQLite's WAL mode, explained with diagrams"]`)
3. [step_3] Copy the article title and URL
   Detail: Select the text '{{article_title}}' and the URL to prepare it for pasting into the Notes app.
   App: Google Chrome
   Target: {{article_url}}
   Verify: The text is copied to the clipboard (indicated by the Copy Action log).
4. [step_4] Open the Notes app and create a new note
   Detail: Launch the Notes application to begin drafting the daily news entry.
   App: Notes
   Target: Notes
   Verify: The Notes app interface is open and ready for input.
5. [step_5] Type the note header and paste the first article details
   Detail: Enter the date, title, and the copied URL for the SQLite article.
   App: Notes
   Target: Notes
   Input: Daily news - {{date}} | 1. {{article_title}} | {{article_url}}
   Verify: The text appears in the note body.
6. [step_6] Type a summary sentence about SQLite's WAL mode
   Detail: Add a specific observation or key takeaway from the article to the note.
   App: Notes
   Target: Notes
   Input: {{summary_text}}
   Verify: The sentence is appended to the existing note content.
7. [step_7] Navigate to the research article about on-device vision models
   Detail: Return to the browser to find the second article for the digest.
   App: Google Chrome
   Target: https://news.ycombinator.com/
   Verify: The browser navigates to https://research.example.com/on-device-vision-guide.
   Anchor: link "A field guide to on-device vision models" (`role=link[name="A field guide to on-device vision models"]`)
8. [step_8] Copy the title and URL of the on-device vision article
   Detail: Select the text 'A field guide to on-device vision models' and the URL to save for the next step.
   App: Google Chrome
   Target: https://research.example.com/on-device-vision-guide
   Verify: The text is copied to the clipboard.
9. [step_9] Paste the second article details into the Notes app
   Detail: Insert the copied title and URL for the on-device vision article into the note.
   App: Notes
   Target: Notes - Daily news - {{date}}
   Input: 2. A field guide to on-device vision models | https://research.example.com/on-device-vision-guide
   Verify: The second entry appears in the note body.
10. [step_10] Type a summary sentence about quantised models
   Detail: Add a specific observation about quantised models to the second entry.
   App: Notes
   Target: Notes
   Input: Small quantised models now read screenshots well enough for UI tasks.
   Verify: The sentence is appended to the second entry.
11. [step_11] Navigate to the GitHub repository page for the HTTP cassette recorder
   Detail: Find the third article to include in the daily news summary.
   App: Google Chrome
   Target: https://news.ycombinator.com/
   Verify: The browser navigates to https://github.com/example/cassette.
   Anchor: link "Show HN: A tiny HTTP cassette recorder for tests" (`role=link[name="Show HN: A tiny HTTP cassette recorder for tests"]`)
12. [step_12] Copy the title and URL of the HTTP cassette recorder project
   Detail: Select the text 'Show HN: A tiny HTTP cassette recorder for tests' and the URL to save.
   App: Google Chrome
   Target: https://github.com/example/cassette
   Verify: The text is copied to the clipboard.
13. [step_13] Paste the third article details into the Notes app
   Detail: Insert the copied title and URL for the HTTP cassette recorder project into the note.
   App: Notes
   Target: Notes - Daily news - {{date}}
   Input: 3. Show HN: A tiny HTTP cassette recorder for tests | https://github.com/example/cassette
   Verify: The third entry appears in the note body.
14. [step_14] Type a summary sentence about the HTTP cassette recorder
   Detail: Add a specific observation about the HTTP cassette recorder to the third entry.
   App: Notes
   Target: Notes
   Input: Records real API answers once and replays them byte for byte.
   Verify: The sentence is appended to the third entry.
15. [step_15] Copy the entire note content
   Detail: Select and copy the full text of the daily news note to the clipboard.
   App: Notes
   Target: Notes
   Verify: The content is copied to the clipboard (indicated by the Copy Action log).
16. [step_16] Navigate to the Slack #daily-news channel
   Detail: Open the Slack application and switch to the designated channel for posting.
   App: Google Chrome
   Target: https://app.slack.com/client/T0EXAMPLE/C0DAILYNEWS
   Verify: The Slack interface shows the #daily-news channel with an active message textbox.
   Anchor: textbox "Message #daily-news" (`[aria-label="Message #daily-news"]`)
17. [step_17] Paste the copied note content into the Slack message box
   Detail: Insert the previously copied daily news text into the Slack message field.
   App: Google Chrome
   Target: https://app.slack.com/client/T0EXAMPLE/C0DAILYNEWS
   Input: Daily news - {{date}} | 1. {{article_title}} | {{article_url}} | {{summary_text}} | 2. A field guide to on-device vision models | https://research.example.com/on-device-vision-guide | Small quantised models now read screenshots well enough for UI tasks. | 3. Show HN: A tiny HTTP cassette recorder for tests | https://github.com/example/cassette | Records real API answers once and replays them byte for byte
   Verify: The full text appears in the Slack message box.
18. [step_18] Post the message in the #daily-news channel
   Detail: Submit the message to complete the daily news digest workflow.
   App: Google Chrome
   Target: https://app.slack.com/client/T0EXAMPLE/C0DAILYNEWS
   Verify: The message is sent and appears in the channel feed.
   Anchor: button "Send now" (`[aria-label="Send now"]`)

## Success criteria

- Done when "Send now" is clicked on app.slack.com/client/T0EXAMPLE/C0DAILYNEWS (observed in 1 of 1 runs)
- The note in the Notes app contains all three articles with their summaries.
- The message in the Slack channel contains the same content as the note.
- The user successfully navigated from Hacker News to Notes to Slack without errors.

## Common errors

- If the user forgets to copy the note content before pasting to Slack, the message box will remain empty.
- If the user navigates to the wrong Slack channel, the message will be posted to the incorrect channel.
- If the user does not save the note in Notes before copying, the content will be lost.

## Decisions

- Open question the user did not answer: "What starts this task?" Ask the user; do not guess.
- Open question the user did not answer: "How do you choose between the options in this task?" Ask the user; do not guess.
- Open question the user did not answer: "How do you check that this task worked?" Ask the user; do not guess.

## Guardrails

- User never posts articles from Hacker News directly without first copying the title and URL into the Notes app.
- User never includes more than three articles in a single digest.
- User never posts to Slack without first drafting the content in the Notes app.
- Trust level: observe

## Voice

- Output template: {{Article Title}}: {{Summary Sentence}} | {{URL}} (variables: Article Title, Summary Sentence, URL)

## Timing

- collection (~0 min): User alternates between browsing Hacker News and reading external articles, copying details to Notes.
- composition (~0 min): User types summary sentences and pastes collected links into the Notes app.
- distribution (~0 min): User copies the note and pastes it into the Slack #daily-news channel.

## Not yet known

- Trigger (what starts the task: a time, an event or a message): not learned from the recordings. Ask the user; do not assume.
- Decisions (how the user chooses between options): not learned from the recordings. Ask the user; do not assume.
- Verification (how the user checks the result): not learned from the recordings. Ask the user; do not assume.

## Evidence

- Observed 1 demonstration(s); confidence 0.93
- Learned from: focus
- 2026-09-15 focus session `1a85097c-b382-4864-93f3-3b1ba42dd34f`

## Execution protocol

If you have access to the AgentHandover MCP server, report your progress:
1. Call `report_execution_start` with slug `daily-news-digest` before starting.
2. Call `report_step_result` after each step with the id shown in brackets (for example `step_1`). If you did something other than the step says, set `status` to `deviated` and describe what you did in `actual_action`.
3. Call `report_execution_complete` when finished.

This feedback improves the Skill for future runs.

---
*Generated by AgentHandover v2.0.2 (procedure schema 3.2.0)*
