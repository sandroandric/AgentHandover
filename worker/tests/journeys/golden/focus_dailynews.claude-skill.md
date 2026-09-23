---
name: daily-news-digest
description: 'Posts a daily digest of tech news and project updates to the #daily-news channel on Slack to the user''s team.'
argument-hint: '[article_title] [article_url] [article_summary]'
allowed-tools: Read, WebFetch
---

# Daily news digest

The user collects articles from Hacker News and external sources, saves them to a Notes app, and posts a summarized digest to a Slack channel.

Outcome: A formatted daily news update containing three articles with links is posted to the #daily-news Slack channel.

## Goal

Posts a daily digest of tech news and project updates to the #daily-news channel on Slack to the user's team.

## Prerequisites

- Access to Google Chrome browser
- Access to the Notes app
- Access to the Slack web interface
- Account on Hacker News
- Account on the target blog/research sites
- Account on the target GitHub repository
- Account on the Slack workspace

## When to use

- When a user wants to curate and share their daily reading list with a team.
- Strategy: The user manually curates three specific topics daily: a Hacker News article on SQLite's WAL mode, a research article on on-device vision models, and a GitHub project on HTTP cassettes, copying each title, URL, and summary into a single note before pasting it into Slack.

## Inputs

- `article_title` (text, required): The title of an article or project being added to the digest. (example `SQLite's WAL mode, explained with diagrams`)
- `article_url` (url, required): The URL of the article or project. (example `https://blog.example.org/sqlite-wal-explained`)
- `article_summary` (text, required): A short sentence summarizing the article or project. (example `Readers keep going while one writer appends; checkpoints fold the log back.`)

Steps refer to inputs as `{{name}}`.

## Steps

1. [step_1] Navigate to the Hacker News website
   Detail: Open the browser and go to the main feed to browse available articles for the digest.
   App: Google Chrome
   Target: https://news.ycombinator.com/
   Verify: The Hacker News homepage loads displaying the list of articles.
   Anchor: link "SQLite's WAL mode, explained with diagrams" (`role=link[name="SQLite's WAL mode, explained with diagrams"]`)
2. [step_2] Click on the article titled 'SQLite's WAL mode, explained with diagrams'
   Detail: Select the first relevant article to read its content and gather details for the note.
   App: Google Chrome
   Target: https://news.ycombinator.com/
   Verify: The browser navigates to the article page at {{article_url}}.
3. [step_3] Open the Notes app
   Detail: Switch to the Notes application to begin creating the daily news entry.
   App: Notes
   Target: Notes
   Verify: The Notes app interface is visible.
4. [step_4] Type the title and URL for the SQLite article into the note
   Detail: Enter the first entry of the daily news list with the article title and link.
   App: Notes
   Target: Notes - Daily news - 15 Sep
   Input: 1. {{article_title}} | {{article_url}}
   Verify: The text appears in the note editor.
5. [step_5] Type the summary sentence for the SQLite article
   Detail: Add the specific detail about WAL mode behavior to the note.
   App: Notes
   Target: Notes - Daily news - 15 Sep
   Input: {{article_summary}}
   Verify: The sentence appears after the URL in the note.
6. [step_6] Navigate back to Google Chrome
   Detail: Return to the browser to find the next article to include in the digest.
   App: Google Chrome
   Target: https://news.ycombinator.com/
   Verify: The Hacker News feed is visible again.
   Anchor: link "A field guide to on-device vision models" (`role=link[name="A field guide to on-device vision models"]`)
7. [step_7] Click on the article titled 'A field guide to on-device vision models'
   Detail: Select the second relevant article to read its content.
   App: Google Chrome
   Target: https://news.ycombinator.com/
   Verify: The browser navigates to the article page at https://research.example.com/on-device-vision-guide.
8. [step_8] Navigate back to the Notes app
   Detail: Switch to the Notes application to add the second entry.
   App: Notes
   Target: Notes - Daily news - 15 Sep
   Verify: The Notes app interface is visible.
9. [step_9] Type the title and URL for the on-device vision article into the note
   Detail: Enter the second entry of the daily news list with the article title and link.
   App: Notes
   Target: Notes - Daily news - 15 Sep
   Input: 2. A field guide to on-device vision models | https://research.example.com/on-device-vision-guide
   Verify: The text appears in the note editor.
10. [step_10] Type the summary sentence for the on-device vision article
   Detail: Add the specific detail about quantised models reading screenshots to the note.
   App: Notes
   Target: Notes - Daily news - 15 Sep
   Input: Small quantised models now read screenshots well enough for UI tasks.
   Verify: The sentence appears after the URL in the note.
11. [step_11] Navigate back to Google Chrome
   Detail: Return to the browser to find the third article to include in the digest.
   App: Google Chrome
   Target: https://news.ycombinator.com/
   Verify: The Hacker News feed is visible again.
   Anchor: link "Show HN: A tiny HTTP cassette recorder for tests" (`role=link[name="Show HN: A tiny HTTP cassette recorder for tests"]`)
12. [step_12] Navigate to the GitHub repository page for the HTTP cassette recorder
   Detail: Search for and open the third relevant article about the HTTP cassette recorder project.
   App: Google Chrome
   Target: https://github.com/example/cassette
   Verify: The GitHub repository page loads showing the project description.
13. [step_13] Navigate back to the Notes app
   Detail: Switch to the Notes application to add the third entry.
   App: Notes
   Target: Notes - Daily news - 15 Sep
   Verify: The Notes app interface is visible.
14. [step_14] Type the title and URL for the HTTP cassette recorder article into the note
   Detail: Enter the third entry of the daily news list with the project title and link.
   App: Notes
   Target: Notes - Daily news - 15 Sep
   Input: 3. Show HN: A tiny HTTP cassette recorder for tests | https://github.com/example/cassette
   Verify: The text appears in the note editor.
15. [step_15] Type the summary sentence for the HTTP cassette recorder article
   Detail: Add the specific detail about the recorder's functionality to the note.
   App: Notes
   Target: Notes - Daily news - 15 Sep
   Input: Records real API answers once and replays them byte for byte.
   Verify: The sentence appears after the URL in the note.
16. [step_16] Copy the entire note content to the clipboard
   Detail: Select and copy the full text of the daily news note to use as the message body.
   App: Notes
   Target: Notes - Daily news - 15 Sep
   Verify: The system indicates the content has been copied (e.g., visual cue or clipboard confirmation).
17. [step_17] Navigate to the Slack web interface
   Detail: Open the browser and go to the Slack workspace to post the update.
   App: Google Chrome
   Target: https://app.slack.com/client/T0EXAMPLE/C0DAILYNEWS
   Verify: The Slack interface loads showing the #daily-news channel.
   Anchor: textbox "Message #daily-news" (`[aria-label="Message #daily-news"]`)
18. [step_18] Paste the copied note content into the message textbox
   Detail: Insert the previously copied daily news note into the Slack message field.
   App: Google Chrome
   Target: https://app.slack.com/client/T0EXAMPLE/C0DAILYNEWS
   Verify: The full text of the note appears in the message input box.
   Anchor: textbox "Message #daily-news" (`[aria-label="Message #daily-news"]`)
19. [step_19] Type the message prefix 'Message #daily-news'
   Detail: Add the required prefix to the message before posting.
   App: Google Chrome
   Target: https://app.slack.com/client/T0EXAMPLE/C0DAILYNEWS
   Input: Message #daily-news
   Verify: The text 'Message #daily-news' appears at the top of the message input box.
20. [step_20] Click the Send button to post the message
   Detail: Submit the message to the #daily-news channel.
   App: Google Chrome
   Target: https://app.slack.com/client/T0EXAMPLE/C0DAILYNEWS
   Verify: The message appears in the channel history and the input box is cleared.

## Success criteria

- The message containing the three articles and their summaries is successfully posted to the #daily-news Slack channel.
- The note in the Notes app contains all three articles with titles, URLs, and summaries.

## Common errors

- Forgetting to include the URL in the note entry.
- Pasting the wrong content into the Slack message.
- Navigating to the wrong Slack channel.
- Not copying the full note content before pasting into Slack.

## Decisions

- If User has finished reading the three selected articles and needs to transfer the content to Slack.: Copies the entire note content to the clipboard and pastes it into the Slack message textbox.
- Choose by: Selects articles based on specific technical keywords found in the Hacker News feed, specifically 'SQLite's WAL mode' and 'on-device vision models', and manually navigates to GitHub for the HTTP cassette recorder. (confidence 100%)
  - Example: Frame 1 shows the user browsing Hacker News with items including 'SQLite's WAL mode, explained with diagrams' and 'A field guide to on-device vision models'.
  - Example: Frame 15 shows the user navigating to 'https://github.com/example/cassette' to read 'Show HN: A tiny HTTP cassette recorder for tests'.

## Guardrails

- The user never uses browser extensions or automated tools to scrape articles; they manually click each link and type summaries.
- The user never posts to any channel other than #daily-news on Slack.
- Trust level: observe

## Voice

- Output template: Daily news - {{date}} | 1. {{article_title}} | {{article_url}} | {{summary_sentence}} | 2. {{article_title}} | {{article_url}} | {{summary_sentence}} | 3. {{article_title}} | {{article_url}} | {{summary_sentence}} (variables: date, article_title, article_url, summary_sentence)

## Timing

- curation (~0 min): Reading articles on Hacker News, GitHub, and external blogs to extract titles, URLs, and summaries.
- documentation (~0 min): Typing the collected information into a Notes app entry.
- distribution (~0 min): Navigating to Slack, pasting the note, and sending the message.

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
