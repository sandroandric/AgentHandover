# AgentHandover -- Launch Day Playbook

**Goal:** Concentrated single-day virality. Every action fires on the same day to create overlapping waves.

**Key narrative angle:** "OpenClaw can DO things. But it doesn't know YOUR workflows. AgentHandover watches you work and teaches it -- self-improving Skills that get better every execution. Work once, hand over forever."

---

## The OpenClaw Leverage Play

OpenClaw went viral through a combination of a well-timed Hacker News post, prominent devs tweeting about it, and the inherent shareability of the concept. AgentHandover can ride this wave because:

1. OpenClaw users immediately need a way to teach it their workflows (that's you)
2. `agenthandover connect openclaw` is a one-liner -- zero friction
3. Every OpenClaw community (GitHub Discussions, Discord, Reddit threads, Twitter conversations) is a warm audience
4. The tagline writes itself: "OpenClaw does the work. AgentHandover teaches it yours."

---

## Launch Day Timeline (All times Pacific)

### Pre-dawn: 12:00-5:00 AM PT
- **ProductHunt goes live at midnight PT.** Have the page ready and scheduled.
- Post the X thread immediately at 12:03 AM PT (catches the global audience -- Europe is waking up).
- DM 5-10 people you know who are active on PH to leave early comments (NOT upvote requests -- ask them to try it and comment honestly).

### Morning wave: 6:00-8:00 AM PT
- **Show HN post** at 7:00 AM PT (Tuesday-Thursday optimal; engineers check HN before standup). Your Show HN title should be factual and technical, e.g.:
  > Show HN: AgentHandover -- watch you work on Mac, produce self-improving Skills for AI agents (all local)
- First comment on your own HN post: 60-word TL;DR + one question to seed discussion. Something like: "I built this because I kept hand-writing Claude Code skills that went stale. AgentHandover watches me work, extracts the strategy/guardrails/voice, and produces Skills that improve from execution feedback. Runs an 11-stage local pipeline (Qwen + nomic-embed). Curious -- how are others handling the gap between what agents can do and teaching them your specific workflows?"
- Post to Reddit communities (see list below) -- stagger by 15-20 minutes.

### Midday wave: 10:00 AM - 12:00 PM PT
- **Engage with every HN and PH comment** within 10 minutes. This is the single highest-leverage activity of the day.
- Cross-post to dev.to and Hashnode (technical blog post format).
- Post in relevant Discord servers.
- Reply to OpenClaw-related tweets with a natural mention.

### Afternoon wave: 2:00-4:00 PM PT
- Second X post (different angle -- maybe the architecture diagram or a GIF of the pipeline).
- Post in Indie Hackers, Lobsters.
- Reply to anyone who shared or commented -- keep engagement momentum alive.

### Evening wave: 5:00-7:00 PM PT
- Share daily stats if ProductHunt is going well ("wow, #3 product of the day, thank you").
- Post a "lessons learned" or behind-the-scenes angle on X.
- Engage with late-arriving HN comments.

---

## Platform-by-Platform Breakdown

### 1. ProductHunt

**Day:** Tuesday, Wednesday, or Thursday (lower competition than people think; avoid Monday/Friday only if you want high total volume). Given you're launching "this week," aim for Tuesday March 31 or Wednesday April 1.

**What to prepare:**
- Tagline (60 chars max): "Watch you work. Teach your agents. Self-improving Skills."
- Description: Lead with the problem (agents are powerful but don't know your workflows), then the solution, then the OpenClaw connection.
- First comment (maker comment): Your personal story. Why you built this. Be vulnerable and specific. "I was writing the same Claude Code skills over and over, and they kept going stale..."
- Gallery: 3-5 images. Screenshot of the menu bar app, the Skill format example from your README, the architecture diagram, a before/after (hand-written skill vs. AgentHandover-generated skill).
- PH-only perk: Consider offering early adopter status, a "founding user" badge in the app, or priority feature requests.

**Algorithm note:** PH in 2026 rewards steady engagement throughout the day, not upvote spikes. Upvotes from active PH users weigh more than from new accounts. Answer every comment in under 10 minutes for the first 4 hours.

### 2. Hacker News (Show HN)

**This is probably your highest-leverage channel.** OpenClaw's breakout started on HN.

**Title format:** Factual, technical, no hype. Include "local" and "Mac" -- HN loves local-first and privacy.
> Show HN: AgentHandover -- learns your Mac workflows and produces self-improving agent Skills (all local, no cloud)

**What HN cares about:**
- Privacy angle (everything local, no telemetry, encryption at rest)
- Technical depth (11-stage pipeline, Rust daemon, Qwen VLM, vector embeddings)
- Open source / source available (BSL 1.1 -- be upfront about this, HN will ask)
- The "self-improving" feedback loop (this is genuinely novel)

**What will get you killed on HN:**
- Marketing language
- Buzzwords without substance
- Not being transparent about the license (BSL 1.1 is not OSI-approved -- own it, explain why)

**Critical:** You need 8-10 genuine upvotes and 2-3 thoughtful comments in the first 30 minutes. Have a small group (5-10 real HN users) ready to check it out and engage organically. Do NOT use public "upvote my post" calls -- HN shadow-bans those IPs.

### 3. X / Twitter

**Thread structure (post at 12:03 AM PT, catches Europe morning + US night owls):**

Tweet 1 (hook):
> I've been watching AI agents try to do my work for months. They're powerful but clueless about MY workflows. So I built something: AgentHandover watches you work on your Mac, understands what you're doing and why, and produces self-improving Skills that agents like OpenClaw and Claude Code can execute. Here's what it does:

Tweet 2: The problem (agents are generic, hand-written skills go stale)

Tweet 3: The solution in one sentence + the Skill example from your README

Tweet 4: How it works (11-stage pipeline, mention local/private, Qwen VLM)

Tweet 5: The self-improving loop (execution feedback -- this is the novel bit)

Tweet 6: OpenClaw integration ("agenthandover connect openclaw" -- one command)

Tweet 7: CTA -- link to GitHub repo + ProductHunt

**Algorithm note:** On X, text-only posts outperform video by 30%. Conversation depth matters most -- a reply that gets a reply from you is weighted 150x more than a like. So reply to everyone.

**Hashtags / tags:** Tag @OpenClaw (or Peter Steinberger's account), @AnthropicAI, @ClaudeCode if they exist. Use #buildinpublic, #opensource, #aiagents.

**Second post (afternoon):** Different angle. Maybe the architecture ASCII diagram, or a specific workflow example ("I recorded myself doing code review for 3 sessions. AgentHandover produced a Skill with my review checklist, tone, and guardrails. Now Claude Code runs it.").

### 4. Reddit (stagger posts by 15-20 min starting 7:30 AM PT)

**Tier 1 -- highest signal, post first:**
- **r/LocalLLaMA** -- THE subreddit for this. Lead with "runs Qwen locally via Ollama, no cloud APIs." This community will love the local-first privacy angle.
- **r/MachineLearning** -- More research-oriented. Lead with the technical pipeline (VLM annotation, semantic clustering, behavioral synthesis).
- **r/ClaudeAI** -- Direct audience. Lead with Claude Code integration and the Skills format.

**Tier 2 -- strong fit:**
- **r/artificial** -- General AI news. Frame it as "teaching agents your workflows."
- **r/selfhosted** -- Privacy and local-first crowd. They'll appreciate the "no telemetry, everything on your machine" angle.
- **r/macapps** -- It's a Mac menu bar app. This is their thing.
- **r/programming** -- Technical angle. The Rust daemon + Python worker architecture.

**Tier 3 -- worth posting if time allows:**
- **r/singularity** -- "What if your AI agent could learn by watching you work?"
- **r/ChatGPT** -- Large audience, lower signal. Keep it practical.
- **r/opensource** -- Source available, BSL 1.1 discussion.
- **r/SideProject** -- Indie dev audience, they love launch stories.
- **r/startups** -- If you frame it as a product story.

**Reddit rules of engagement:**
- Each subreddit gets a DIFFERENT post, tailored to what that community cares about.
- Never cross-post the same text.
- Be in the comments replying to questions within minutes.
- Reddit AI communities punish hype -- be factual, technical, honest.

### 5. Hacker News Adjacent

- **Lobsters** (lobste.rs) -- Invite-only but technically rigorous. If you have an account or know someone, post it. HN-quality audience, less noise.
- **dev.to** -- Write a technical post: "How I built an 11-stage pipeline to teach AI agents my workflows." Publish on launch day morning.
- **Hashnode** -- Same post, cross-published.

### 6. Discord Servers

- **OpenClaw Discord** (if one exists) -- This is the #1 warm audience
- **MCP Community Discord** (11,800+ members, discord.com/invite/model-context-protocol) -- AgentHandover IS an MCP server, this is a direct-fit audience
- **MCP Contributor Discord** -- if you're an active contributor, post here too
- **Claude Code / Anthropic community** Discord
- **Ollama** Discord -- you use their models, this is relevant
- **Indie Hackers** Discord
- **Mac Power Users** type communities
- **AI agent builders** -- search for "AI agents" Discord servers

### 7. Newsletters and Aggregators (submit on launch day morning)

These are free submissions that can generate traffic if they pick you up:

- **Hacker Newsletter** (hackernewsletter.com) -- curates top HN posts weekly
- **TLDR Newsletter** (tldr.tech) -- submit via their form, huge dev audience
- **Ben's Bites** (bensbites.com) -- AI-focused newsletter, large subscriber base
- **The Rundown AI** -- AI newsletter, submit your launch
- **PulseMCP** (pulsemcp.com) -- MCP-specific newsletter (they already covered OpenClaw going viral)
- **Console.dev** -- curates developer tools weekly
- **Changelog** (changelog.com) -- open source news
- **Import AI** -- AI newsletter
- **AlphaSignal** -- AI/ML newsletter

### 8. Indie Hacker / Builder Communities

- **Indie Hackers** (indiehackers.com) -- post a "Show IH" with your story
- **WIP.co** -- maker community, post a launch milestone
- **Makerlog** -- log your launch publicly
- **Hacker Noon** -- submit a technical article

---

## The Viral Hooks (pick 2-3, use different ones on different platforms)

1. **"Work once, hand over forever"** -- The tagline. Simple, memorable, aspirational.

2. **"Your agents are powerful but clueless"** -- Frames the problem everyone with AI agents faces.

3. **"Not static playbooks. Self-improving ones."** -- Differentiator from every other SOP/workflow tool.

4. **"I recorded myself doing code review for 3 sessions. AgentHandover now runs it for me with my exact checklist, tone, and guardrails."** -- Concrete, relatable example.

5. **"11 stages between your screen and an agent-ready Skill. Zero cloud APIs."** -- Technical credibility + privacy.

6. **"OpenClaw does the work. AgentHandover teaches it yours."** -- Leverages the OpenClaw name recognition.

7. **"It watched me reply to Reddit comments for a week. Then it wrote a Skill that matches my voice so well I can't tell the difference."** -- The voice analysis angle is genuinely creepy/cool.

---

## Content to Prepare Before Launch Day

Since you're going with repo + PH page only, here's the minimum you should prepare:

1. **ProductHunt page** -- tagline, description, 3-5 images, maker comment draft
2. **Show HN post** -- title + first comment (the 60-word TL;DR) pre-written
3. **X thread** -- 7 tweets pre-written, ready to post
4. **Reddit posts** -- at least Tier 1 (3 posts), ideally Tier 2 (4 more), each tailored
5. **One-paragraph pitch** -- for Discord servers and newsletter submissions
6. **FAQ ready in your head** for common questions:
   - "How is this different from screen recording + ChatGPT?" (Answer: 11-stage pipeline, self-improving feedback loop, voice analysis, semantic dedup)
   - "Is this open source?" (Answer: BSL 1.1 -- source available, non-commercial. Converts to Apache 2.0 in 2030. Be transparent.)
   - "Does it work on Linux/Windows?" (Answer: Mac only for now. Be honest about the roadmap.)
   - "Privacy?" (Answer: Everything local, screenshots deleted after annotation, auto-redaction, encryption at rest, no telemetry.)

---

## Maximizing Viral Potential

### The compounding effect
The reason everything fires on the same day: someone sees it on HN, checks X, sees it trending there too, checks PH, it's climbing there. This creates the perception of "this is everywhere" which is itself the trigger for sharing. Three platforms trending simultaneously > one platform at 3x the engagement.

### Reply to EVERYTHING
On launch day, your only job is engaging. Every HN comment, every Reddit reply, every X mention, every PH comment. The algorithms on every platform reward active conversation. On X specifically, your reply to someone's reply is weighted 150x more than a like. On HN, active threads stay on the front page longer.

### Seed the OpenClaw connection
If you can get even one tweet or comment from someone in the OpenClaw ecosystem acknowledging AgentHandover, that's rocket fuel. Consider reaching out to Peter Steinberger directly (a respectful, short DM about the integration). Even a retweet from him would expose you to the entire OpenClaw audience.

### The "Show, Don't Tell" factor
OpenClaw went viral partly because "talk to an AI agent through WhatsApp" is immediately shareable. For AgentHandover, the equivalent shareable moment is: "It watched me work and produced THIS" -- showing a generated Skill. If you can record even a 30-second GIF of the menu bar app showing a Skill being generated from a recording session, that's worth 1000 words of explanation.

---

## Paid Options (small budget, high leverage)

If you have any budget at all:

1. **X Promoted post** ($50-100) -- Boost your launch thread to followers of AI/developer accounts. Target followers of @AnthropicAI, @OpenAI, @petersteinberger, @maboroshi, etc.

2. **Reddit promoted post** ($50-100) -- Target r/LocalLLaMA, r/MachineLearning. Reddit ads in niche subreddits are surprisingly cheap and well-targeted.

3. **Sponsoring a newsletter issue** ($100-500) -- TLDR, Ben's Bites, or PulseMCP. One mention in a newsletter with 100K+ subscribers can drive thousands of visits.

4. **ProductHunt launch service** ($200-500) -- Services like LaunchPedia or OpenHunts help coordinate upvotes and engagement from active PH users. Mixed reputation but some people swear by them.

---

## Post-Launch (same day, evening)

- Screenshot your PH ranking, GitHub stars, HN points -- post as a "thank you" thread on X.
- If you hit any milestone (top 5 PH, HN front page, 100+ stars), post it immediately. Milestones are shareable.
- Update your GitHub README with any social proof badges (PH badge, star count).
- Reply to every remaining comment you missed.

---

## Risk Mitigation

- **HN license debate:** BSL 1.1 will get scrutinized. Have your talking points ready. Be honest: "I chose BSL because I want the code to be inspectable and the project sustainable. It converts to Apache 2.0 in 2030."
- **"Just use screen recording + Claude":** Your answer is the 11-stage pipeline, the self-improving feedback loop, and voice analysis. No prompt engineering can replicate behavioral synthesis across multiple sessions.
- **Mac only:** Acknowledge it. "Mac first because screen capture + menu bar UX is best here. Linux is on the roadmap."
- **Privacy concerns about screen recording:** Lead with the privacy section. Screenshots are temporary, auto-redaction, encryption, no telemetry. This is actually a strength because everything is local.
