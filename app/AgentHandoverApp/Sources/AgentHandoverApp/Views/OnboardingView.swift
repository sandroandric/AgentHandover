import SwiftUI

/// Premium onboarding experience — 8 screens with progressive disclosure.
///
/// Screens: Welcome → Two Ways to Teach → What You Get → Review Cycle →
///          Permissions → AI Model → Browser Extension → Ready (First Recording)
struct OnboardingView: View {
    @EnvironmentObject var appState: AppState
    @State private var currentStep = 0
    @State private var extensionPath: String = ""
    @State private var chromeOpenError: String? = nil
    @State private var vlmPullInProgress = false
    @State private var vlmPullOutput = ""
    @State private var serviceStartFailed = false

    // Cloud VLM state
    enum VLMMode: String, CaseIterable {
        case local = "Local"
        case cloud = "Cloud"
    }

    enum RemoteProvider: String, CaseIterable, Identifiable {
        case openai = "openai"
        case anthropic = "anthropic"
        case google = "google"

        var id: String { rawValue }

        var displayName: String {
            switch self {
            case .openai: return "OpenAI"
            case .anthropic: return "Anthropic (Claude)"
            case .google: return "Google (Gemini)"
            }
        }

        var defaultModel: String {
            switch self {
            case .openai: return "gpt-4.1-mini"
            case .anthropic: return "claude-sonnet-4-6-20260320"
            case .google: return "gemini-2.5-flash"
            }
        }

        var envVar: String {
            switch self {
            case .openai: return "OPENAI_API_KEY"
            case .anthropic: return "ANTHROPIC_API_KEY"
            case .google: return "GOOGLE_API_KEY"
            }
        }
    }

    @State private var vlmMode: VLMMode = .local
    @State private var selectedProvider: RemoteProvider = .openai
    @State private var apiKeyInput: String = ""
    @State private var customModelName: String = ""
    @State private var apiKeyValidating = false
    @State private var apiKeyValid: Bool? = nil
    @State private var remoteConsentGiven = false

    // Focus recording from onboarding
    @State private var firstRecordingTitle: String = ""

    // Clipboard copy feedback
    @State private var pathCopied = false

    // Record button pulse
    @State private var recordPulse = false

    /// Called when onboarding completes (sets hasCompletedOnboarding).
    var onComplete: (() -> Void)?

    private let totalSteps = 8

    // MARK: - Design Tokens (Warm Amber Palette)

    private let warmOrange = Color(red: 0.92, green: 0.57, blue: 0.20)       // #EA9134
    private let warmAmber = Color(red: 0.95, green: 0.68, blue: 0.30)        // #F2AD4D
    private let warmBrown = Color(red: 0.42, green: 0.28, blue: 0.15)        // #6B4726
    private let warmCream = Color(red: 0.98, green: 0.96, blue: 0.92)        // #FAF5EB
    private let warmWhite = Color(red: 0.995, green: 0.985, blue: 0.975)     // warm-shifted white

    private var bgWash: LinearGradient {
        LinearGradient(
            colors: [warmOrange.opacity(0.05), Color.clear],
            startPoint: .topLeading,
            endPoint: .bottomTrailing
        )
    }

    private var brandGradient: LinearGradient {
        LinearGradient(
            colors: [warmOrange, Color(red: 0.88, green: 0.48, blue: 0.16)],
            startPoint: .topLeading,
            endPoint: .bottomTrailing
        )
    }

    // Card styling
    private let cardRadius: CGFloat = 16
    private let cardPadding: CGFloat = 18
    private let sectionSpacing: CGFloat = 24

    private var cardBg: Color { warmWhite }
    private var cardBorder: Color { warmBrown.opacity(0.08) }
    private var cardShadowColor: Color { warmBrown.opacity(0.06) }

    // Typography
    private let heroFont = Font.system(size: 32, weight: .bold, design: .rounded)
    private let sectionFont = Font.system(size: 16, weight: .semibold, design: .rounded)
    private let bodyFont = Font.system(size: 14)
    private let captionFont = Font.system(size: 11)
    private let monoFont = Font.system(size: 12, design: .monospaced)

    private let captionColor = Color.secondary.opacity(0.7)

    var body: some View {
        ZStack {
            // Warm background wash
            bgWash.ignoresSafeArea()

            VStack(spacing: 0) {
                // Progress indicator
                progressBar
                    .padding(.top, 20)
                    .padding(.horizontal, 44)

                Spacer()

                // Current step content
                stepContent(for: currentStep)
                    .padding(.horizontal, 44)

                Spacer()

                // Navigation
                navigationBar
                    .padding(.horizontal, 44)
                    .padding(.bottom, 28)
            }
        }
        .onAppear {
            resolveExtensionPath()
        }
    }

    // MARK: - Progress Bar

    private var progressBar: some View {
        VStack(spacing: 8) {
            GeometryReader { geometry in
                ZStack(alignment: .leading) {
                    // Track
                    Capsule()
                        .fill(warmBrown.opacity(0.06))
                        .frame(height: 3)

                    // Fill
                    Capsule()
                        .fill(brandGradient)
                        .frame(
                            width: geometry.size.width * CGFloat(currentStep + 1) / CGFloat(totalSteps),
                            height: 3
                        )
                        .animation(.easeOut(duration: 0.45), value: currentStep)
                }
            }
            .frame(height: 3)

            Text("\(currentStep + 1) of \(totalSteps)")
                .font(.system(size: 10, weight: .medium, design: .rounded))
                .foregroundColor(captionColor)
                .tracking(0.3)
        }
    }

    // MARK: - Navigation Bar

    private var navigationBar: some View {
        HStack {
            if currentStep > 0 {
                Button("Back") {
                    withAnimation(.spring(response: 0.35, dampingFraction: 0.8)) { currentStep -= 1 }
                }
                .buttonStyle(.plain)
                .foregroundColor(warmBrown.opacity(0.5))
                .font(.system(size: 13, weight: .medium, design: .rounded))
            }

            Spacer()

            switch currentStep {
            case 0:
                Button {
                    withAnimation(.spring(response: 0.35, dampingFraction: 0.8)) { currentStep += 1 }
                } label: {
                    HStack(spacing: 7) {
                        Text("Get Started")
                            .font(.system(size: 15, weight: .semibold, design: .rounded))
                        Image(systemName: "arrow.right")
                            .font(.system(size: 12, weight: .bold))
                    }
                    .padding(.horizontal, 24)
                    .padding(.vertical, 11)
                    .background(brandGradient)
                    .foregroundColor(.white)
                    .clipShape(Capsule())
                    .shadow(color: warmOrange.opacity(0.25), radius: 8, y: 3)
                }
                .buttonStyle(.plain)

            case 1, 2, 3:
                nextButton {
                    withAnimation(.spring(response: 0.35, dampingFraction: 0.8)) { currentStep += 1 }
                }

            case 4:
                // Permissions — blocked until both granted, with skip option
                VStack(spacing: 4) {
                    nextButton(disabled: !appState.accessibilityGranted || !appState.screenRecordingGranted) {
                        withAnimation(.spring(response: 0.35, dampingFraction: 0.8)) { currentStep += 1 }
                    }

                    if !appState.accessibilityGranted || !appState.screenRecordingGranted {
                        Button("Skip for now") {
                            withAnimation(.spring(response: 0.35, dampingFraction: 0.8)) { currentStep += 1 }
                        }
                        .font(.system(size: 11, design: .rounded))
                        .foregroundColor(captionColor)
                        .buttonStyle(.plain)
                    }
                }

            case 5:
                // VLM Setup — blocked until model ready
                VStack(spacing: 2) {
                    nextButton(disabled: !appState.vlmAvailable) {
                        withAnimation(.spring(response: 0.35, dampingFraction: 0.8)) { currentStep += 1 }
                    }

                    if !appState.vlmAvailable {
                        Text("Set up an AI model above to continue")
                            .font(.system(size: 11, design: .rounded))
                            .foregroundColor(warmOrange)
                    }
                }

            case 6:
                // Browser extension — optional
                HStack(spacing: 12) {
                    if !appState.extensionConnected {
                        Button("Skip") {
                            withAnimation(.spring(response: 0.35, dampingFraction: 0.8)) { currentStep += 1 }
                        }
                        .foregroundColor(captionColor)
                        .buttonStyle(.plain)
                        .font(.system(size: 13, design: .rounded))
                    }

                    nextButton {
                        withAnimation(.spring(response: 0.35, dampingFraction: 0.8)) { currentStep += 1 }
                    }
                }

            case 7:
                // Ready — final step, no Next button
                EmptyView()

            default:
                EmptyView()
            }
        }
    }

    /// Consistent warm "Next" button used across nav bar.
    private func nextButton(disabled: Bool = false, action: @escaping () -> Void) -> some View {
        Button {
            action()
        } label: {
            HStack(spacing: 5) {
                Text("Next")
                    .font(.system(size: 14, weight: .semibold, design: .rounded))
                Image(systemName: "arrow.right")
                    .font(.system(size: 10, weight: .bold))
            }
            .padding(.horizontal, 20)
            .padding(.vertical, 9)
            .background(
                ZStack {
                    if disabled {
                        Capsule().fill(warmBrown.opacity(0.08))
                    } else {
                        Capsule().fill(brandGradient)
                    }
                }
            )
            .foregroundColor(disabled ? .secondary : .white)
            .clipShape(Capsule())
            .shadow(color: disabled ? .clear : warmOrange.opacity(0.15), radius: 6, y: 2)
        }
        .buttonStyle(.plain)
        .disabled(disabled)
    }

    // MARK: - Step Content

    @ViewBuilder
    private func stepContent(for step: Int) -> some View {
        switch step {
        case 0: welcomeStep
        case 1: teachByDoingStep
        case 2: whatYoullGetStep
        case 3: reviewCycleStep
        case 4: permissionsStep
        case 5: vlmSetupStep
        case 6: chromeExtensionStep
        case 7: readyStep
        default: EmptyView()
        }
    }

    // MARK: - Mascot Image Helper

    /// Loads the mascot image with multiple fallback strategies.
    @ViewBuilder
    private func mascotImage(height: CGFloat) -> some View {
        if let nsImg = Bundle.module.image(forResource: "mascot") {
            Image(nsImage: nsImg)
                .resizable()
                .aspectRatio(contentMode: .fit)
                .frame(height: height)
        } else if let nsImg = NSImage(named: "mascot") {
            Image(nsImage: nsImg)
                .resizable()
                .aspectRatio(contentMode: .fit)
                .frame(height: height)
        } else {
            // Last resort: styled SF Symbol
            Image(systemName: "binoculars.fill")
                .font(.system(size: height * 0.6, weight: .medium))
                .foregroundStyle(brandGradient)
                .frame(height: height)
        }
    }

    // MARK: - Screen 1: Welcome

    private var welcomeStep: some View {
        VStack(spacing: 28) {
            // Hero: mascot, large and proud
            mascotImage(height: 120)
                .shadow(color: warmOrange.opacity(0.12), radius: 20, y: 6)

            // Title
            VStack(spacing: 8) {
                Text("AgentHandover")
                    .font(heroFont)
                    .foregroundColor(warmBrown)

                Text("Your work, turned into agent instructions")
                    .font(.system(size: 15, weight: .medium, design: .rounded))
                    .foregroundColor(.secondary)
            }

            // Three value props as clean text lines (no cards, no icons)
            VStack(spacing: 10) {
                valuePropLine("Watches your screen silently")
                valuePropLine("Learns repeatable workflows")
                valuePropLine("Writes procedures agents can follow")
            }
            .padding(.top, 4)

            // Privacy badge
            HStack(spacing: 7) {
                Image(systemName: "lock.fill")
                    .font(.system(size: 10, weight: .semibold))
                    .foregroundColor(warmOrange)
                Text("Everything runs locally on your Mac")
                    .font(.system(size: 12, weight: .medium, design: .rounded))
                    .foregroundColor(warmBrown.opacity(0.6))
            }
            .padding(.horizontal, 16)
            .padding(.vertical, 9)
            .background(
                Capsule()
                    .fill(warmOrange.opacity(0.06))
            )
        }
    }

    private func valuePropLine(_ text: String) -> some View {
        Text(text)
            .font(.system(size: 14, weight: .regular))
            .foregroundColor(.secondary)
            .lineSpacing(5)
    }

    // MARK: - Screen 2: Two Ways to Teach

    private var teachByDoingStep: some View {
        VStack(spacing: sectionSpacing) {
            Text("Two ways to teach")
                .font(heroFont)
                .foregroundColor(warmBrown)

            HStack(spacing: 16) {
                // Focus Recording — warm, recommended
                VStack(alignment: .leading, spacing: 14) {
                    // Recommended ribbon
                    HStack {
                        Spacer()
                        Text("Recommended")
                            .font(.system(size: 10, weight: .bold, design: .rounded))
                            .tracking(0.5)
                            .textCase(.uppercase)
                            .foregroundColor(.white)
                            .padding(.horizontal, 10)
                            .padding(.vertical, 4)
                            .background(
                                Capsule().fill(brandGradient)
                            )
                    }

                    Image(systemName: "record.circle")
                        .font(.system(size: 26))
                        .foregroundColor(warmOrange)

                    Text("Focus Recording")
                        .font(.system(size: 15, weight: .semibold, design: .rounded))
                        .foregroundColor(warmBrown)

                    VStack(alignment: .leading, spacing: 7) {
                        numberedStep(1, "Click Record")
                        numberedStep(2, "Do the task as usual")
                        numberedStep(3, "Stop \u{2014} AI analyzes in 2\u{2013}5 min")
                    }
                }
                .padding(cardPadding)
                .frame(maxWidth: .infinity, alignment: .leading)
                .background(
                    RoundedRectangle(cornerRadius: cardRadius)
                        .fill(
                            LinearGradient(
                                colors: [warmOrange.opacity(0.06), warmAmber.opacity(0.03)],
                                startPoint: .topLeading,
                                endPoint: .bottomTrailing
                            )
                        )
                )
                .overlay(
                    RoundedRectangle(cornerRadius: cardRadius)
                        .stroke(warmOrange.opacity(0.25), lineWidth: 1.5)
                )
                .shadow(color: warmOrange.opacity(0.08), radius: 14, y: 4)

                // Passive Learning — cool/neutral
                VStack(alignment: .leading, spacing: 14) {
                    Spacer().frame(height: 24) // align with ribbon space

                    Image(systemName: "eye")
                        .font(.system(size: 26))
                        .foregroundColor(.secondary.opacity(0.6))

                    Text("Passive Learning")
                        .font(.system(size: 15, weight: .semibold, design: .rounded))
                        .foregroundColor(.primary.opacity(0.7))

                    VStack(alignment: .leading, spacing: 7) {
                        Text("Watches for repeated patterns")
                            .font(bodyFont)
                            .foregroundColor(.secondary)
                        Text("Gets smarter over days")
                            .font(bodyFont)
                            .foregroundColor(.secondary)
                        Text("No effort required")
                            .font(bodyFont)
                            .foregroundColor(.secondary)
                    }
                }
                .padding(cardPadding)
                .frame(maxWidth: .infinity, alignment: .leading)
                .background(
                    RoundedRectangle(cornerRadius: cardRadius)
                        .fill(Color.primary.opacity(0.02))
                )
                .overlay(
                    RoundedRectangle(cornerRadius: cardRadius)
                        .stroke(Color.primary.opacity(0.06), lineWidth: 1)
                )
            }

            Text("Start with a Focus Recording. After you stop, the AI analyzes your screenshots in 2\u{2013}5 minutes and asks a few questions to finalize the procedure.")
                .font(captionFont)
                .foregroundColor(captionColor)
                .multilineTextAlignment(.center)
                .lineSpacing(3)
                .frame(maxWidth: 460)
        }
    }

    private func numberedStep(_ n: Int, _ text: String) -> some View {
        HStack(spacing: 8) {
            Text("\(n)")
                .font(.system(size: 11, weight: .bold, design: .rounded))
                .foregroundColor(warmOrange)
                .frame(width: 20, height: 20)
                .background(
                    Circle().fill(warmOrange.opacity(0.1))
                )
            Text(text)
                .font(.system(size: 13))
                .foregroundColor(.secondary)
        }
    }

    // MARK: - Screen 3: What You Get

    private var whatYoullGetStep: some View {
        VStack(spacing: sectionSpacing) {
            Text("What your agent receives")
                .font(heroFont)
                .foregroundColor(warmBrown)

            // Mock procedure — styled as a real paper document
            HStack(spacing: 0) {
                // Orange left border stripe
                warmOrange
                    .frame(width: 3)
                    .clipShape(RoundedRectangle(cornerRadius: 1.5))

                VStack(alignment: .leading, spacing: 0) {
                    // Document title (serif-like rounded font for contrast)
                    Text("File Expense Report")
                        .font(.system(size: 18, weight: .semibold, design: .rounded))
                        .foregroundColor(warmBrown)
                        .padding(.bottom, 3)

                    Text("Expensify workflow \u{00B7} 5 steps")
                        .font(.system(size: 11, weight: .medium))
                        .foregroundColor(warmBrown.opacity(0.5))
                        .padding(.bottom, 16)

                    // Strategy
                    docSectionLabel("Strategy")
                    Text("Open Expensify, upload receipt, categorize, submit for approval")
                        .font(.system(size: 12))
                        .foregroundColor(.secondary)
                        .lineSpacing(3)
                        .padding(.bottom, 16)

                    // Steps
                    docSectionLabel("Steps")
                    VStack(alignment: .leading, spacing: 5) {
                        docStep(1, "Open Expensify in Chrome")
                        docStep(2, "Click \"New Expense\"")
                        docStep(3, "Upload receipt photo")
                        docStep(4, "Select category: Travel")
                        docStep(5, "Submit for manager approval")
                    }
                    .padding(.bottom, 16)

                    // Verification & Guardrails
                    HStack(alignment: .top, spacing: 20) {
                        VStack(alignment: .leading, spacing: 5) {
                            docSectionLabel("Verification")
                            Text("\"Expense submitted\" confirmation")
                                .font(.system(size: 11))
                                .foregroundColor(.secondary)
                        }
                        .frame(maxWidth: .infinity, alignment: .leading)

                        VStack(alignment: .leading, spacing: 5) {
                            docSectionLabel("Guardrails")
                            Text("Never submit without receipt")
                                .font(.system(size: 11))
                                .foregroundColor(.secondary)
                            Text("Max $500 without pre-approval")
                                .font(.system(size: 11))
                                .foregroundColor(.secondary)
                        }
                        .frame(maxWidth: .infinity, alignment: .leading)
                    }
                    .padding(.bottom, 14)

                    // Thin divider
                    Rectangle()
                        .fill(warmBrown.opacity(0.06))
                        .frame(height: 1)
                        .padding(.bottom, 10)

                    // Footer
                    HStack(spacing: 14) {
                        HStack(spacing: 4) {
                            Image(systemName: "clock")
                                .font(.system(size: 9))
                                .foregroundColor(warmBrown.opacity(0.4))
                            Text("~5 min")
                                .font(.system(size: 11))
                                .foregroundColor(warmBrown.opacity(0.4))
                        }
                        HStack(spacing: 4) {
                            Image(systemName: "chart.bar.fill")
                                .font(.system(size: 9))
                                .foregroundColor(Color.green.opacity(0.7))
                            Text("Confidence: 92%")
                                .font(.system(size: 11))
                                .foregroundColor(warmBrown.opacity(0.4))
                        }
                    }
                }
                .padding(20)
            }
            .background(
                RoundedRectangle(cornerRadius: cardRadius)
                    .fill(warmCream.opacity(0.6))
            )
            .overlay(
                RoundedRectangle(cornerRadius: cardRadius)
                    .stroke(warmBrown.opacity(0.06), lineWidth: 1)
            )
            .shadow(color: warmBrown.opacity(0.08), radius: 16, y: 5)

            Text("Exported as a SKILL.md that Claude Code, OpenClaw, and other agents can execute.")
                .font(captionFont)
                .foregroundColor(captionColor)
                .multilineTextAlignment(.center)
                .frame(maxWidth: 460)
        }
    }

    private func docSectionLabel(_ text: String) -> some View {
        Text(text.uppercased())
            .font(.system(size: 10, weight: .bold, design: .rounded))
            .foregroundColor(warmBrown.opacity(0.4))
            .tracking(0.8)
            .padding(.bottom, 5)
    }

    private func docStep(_ number: Int, _ text: String) -> some View {
        HStack(alignment: .center, spacing: 9) {
            Text("\(number)")
                .font(.system(size: 10, weight: .bold, design: .rounded))
                .foregroundColor(warmOrange)
                .frame(width: 18, height: 18)
                .background(
                    Circle().fill(warmOrange.opacity(0.1))
                )
            Text(text)
                .font(.system(size: 12))
                .foregroundColor(.secondary)
        }
    }

    // MARK: - Screen 4: Review Cycle (Vertical Timeline)

    private var reviewCycleStep: some View {
        VStack(spacing: sectionSpacing) {
            Text("You stay in control")
                .font(heroFont)
                .foregroundColor(warmBrown)

            // Vertical timeline
            VStack(alignment: .leading, spacing: 0) {
                timelineNode(
                    icon: "camera.fill",
                    title: "Record or Observe",
                    subtitle: "You do your work normally",
                    isHighlighted: false,
                    isLast: false
                )
                timelineNode(
                    icon: "brain.head.profile",
                    title: "AI Analyzes",
                    subtitle: "Extracts steps from your screen recordings",
                    isHighlighted: false,
                    isLast: false
                )
                timelineNode(
                    icon: "person.fill",
                    title: "You Review",
                    subtitle: "Approve, edit, or reject each procedure",
                    isHighlighted: true,
                    isLast: false
                )
                timelineNode(
                    icon: "cpu",
                    title: "Agent Ready",
                    subtitle: "Only approved procedures reach agents",
                    isHighlighted: false,
                    isLast: true
                )
            }
            .padding(.leading, 4)

            // Menu bar callout
            HStack(spacing: 12) {
                Image(systemName: "hand.tap.fill")
                    .font(.system(size: 18))
                    .foregroundColor(warmOrange)

                VStack(alignment: .leading, spacing: 3) {
                    Text("Review from your menu bar")
                        .font(.system(size: 13, weight: .semibold, design: .rounded))
                        .foregroundColor(warmBrown)
                    Text("Approve with one tap, or edit to refine.")
                        .font(bodyFont)
                        .foregroundColor(.secondary)
                }
            }
            .padding(cardPadding)
            .frame(maxWidth: .infinity, alignment: .leading)
            .background(
                RoundedRectangle(cornerRadius: cardRadius)
                    .fill(warmOrange.opacity(0.04))
            )
            .overlay(
                RoundedRectangle(cornerRadius: cardRadius)
                    .stroke(warmOrange.opacity(0.15), lineWidth: 1)
            )
            .shadow(color: warmOrange.opacity(0.05), radius: 10, y: 2)
        }
    }

    private func timelineNode(
        icon: String,
        title: String,
        subtitle: String,
        isHighlighted: Bool,
        isLast: Bool
    ) -> some View {
        HStack(alignment: .top, spacing: 14) {
            // Dot + connecting line
            VStack(spacing: 0) {
                ZStack {
                    Circle()
                        .fill(isHighlighted ? warmOrange : warmBrown.opacity(0.12))
                        .frame(width: 28, height: 28)

                    if isHighlighted {
                        Circle()
                            .stroke(warmOrange.opacity(0.3), lineWidth: 2)
                            .frame(width: 36, height: 36)
                    }

                    Image(systemName: icon)
                        .font(.system(size: 12, weight: .semibold))
                        .foregroundColor(isHighlighted ? .white : warmBrown.opacity(0.5))
                }

                if !isLast {
                    Rectangle()
                        .fill(warmBrown.opacity(0.08))
                        .frame(width: 1.5, height: 28)
                }
            }

            VStack(alignment: .leading, spacing: 2) {
                Text(title)
                    .font(.system(size: 14, weight: isHighlighted ? .bold : .semibold, design: .rounded))
                    .foregroundColor(isHighlighted ? warmOrange : warmBrown.opacity(0.8))
                Text(subtitle)
                    .font(.system(size: 12))
                    .foregroundColor(.secondary)
            }
            .padding(.top, 3)
        }
    }

    // MARK: - Screen 5: Permissions

    private var permissionsStep: some View {
        VStack(spacing: sectionSpacing) {
            Text("Two permissions to enable")
                .font(heroFont)
                .foregroundColor(warmBrown)

            VStack(spacing: 12) {
                // Accessibility card
                permissionCard(
                    icon: "hand.raised.circle.fill",
                    title: "Accessibility",
                    description: "Read window titles and UI elements",
                    granted: appState.accessibilityGranted,
                    action: {
                        PermissionChecker.requestAccessibility()
                    },
                    actionLabel: "Grant Access"
                )

                // Screen Recording card
                permissionCard(
                    icon: "rectangle.dashed.badge.record",
                    title: "Screen Recording",
                    description: "Capture screenshots for AI analysis",
                    granted: appState.screenRecordingGranted,
                    action: {
                        PermissionChecker.openScreenRecordingSettings()
                    },
                    actionLabel: "Open Settings"
                )
            }

            HStack(spacing: 7) {
                Image(systemName: "eye.fill")
                    .font(.system(size: 10))
                    .foregroundColor(warmOrange.opacity(0.7))
                Text("AgentHandover reads your screen. It never types, clicks, or takes actions.")
                    .font(captionFont)
                    .foregroundColor(captionColor)
            }
            .multilineTextAlignment(.center)
            .frame(maxWidth: 440)
        }
    }

    private func permissionCard(
        icon: String,
        title: String,
        description: String,
        granted: Bool,
        action: @escaping () -> Void,
        actionLabel: String
    ) -> some View {
        HStack(spacing: 14) {
            Image(systemName: icon)
                .font(.system(size: 22))
                .foregroundColor(granted ? .green : warmOrange)
                .frame(width: 36)

            VStack(alignment: .leading, spacing: 3) {
                Text(title)
                    .font(.system(size: 14, weight: .semibold, design: .rounded))
                    .foregroundColor(warmBrown)
                Text(description)
                    .font(.system(size: 12))
                    .foregroundColor(.secondary)
            }

            Spacer()

            if granted {
                HStack(spacing: 5) {
                    Image(systemName: "checkmark.circle.fill")
                        .foregroundColor(.green)
                    Text("Granted")
                        .foregroundColor(.green)
                }
                .font(.system(size: 12, weight: .medium, design: .rounded))
                .padding(.horizontal, 12)
                .padding(.vertical, 6)
                .background(
                    Capsule().fill(Color.green.opacity(0.08))
                )
            } else {
                Button(actionLabel) {
                    action()
                }
                .font(.system(size: 12, weight: .medium, design: .rounded))
                .buttonStyle(.bordered)
                .controlSize(.small)
            }
        }
        .padding(cardPadding)
        .background(
            RoundedRectangle(cornerRadius: cardRadius)
                .fill(granted ? Color.green.opacity(0.02) : cardBg)
        )
        .overlay(
            RoundedRectangle(cornerRadius: cardRadius)
                .stroke(
                    granted ? Color.green.opacity(0.2) : cardBorder,
                    lineWidth: 1
                )
        )
        .shadow(color: cardShadowColor, radius: 12, y: 3)
    }

    // MARK: - Screen 6: VLM Setup (Required)

    private var vlmSetupStep: some View {
        VStack(spacing: 16) {
            Image(systemName: "brain.head.profile")
                .font(.system(size: 36))
                .foregroundColor(warmOrange)
                .shadow(color: warmOrange.opacity(0.15), radius: 12, y: 3)

            Text("Set up your AI")
                .font(heroFont)
                .foregroundColor(warmBrown)

            Text("A small AI model runs on your Mac to understand what\u{2019}s on your screen.")
                .font(bodyFont)
                .foregroundColor(.secondary)
                .multilineTextAlignment(.center)
                .lineSpacing(5)
                .frame(maxWidth: 440)

            if appState.vlmAvailable {
                PermissionStatusBadge(
                    granted: true,
                    grantedLabel: "AI Model Ready",
                    deniedLabel: ""
                )
            } else {
                // Local / Cloud toggle
                Picker("Mode", selection: $vlmMode) {
                    ForEach(VLMMode.allCases, id: \.self) { mode in
                        Text(mode.rawValue).tag(mode)
                    }
                }
                .pickerStyle(.segmented)
                .frame(maxWidth: 240)

                if vlmMode == .cloud {
                    cloudVLMContent
                } else {
                    localVLMContent
                }
            }
        }
    }

    // MARK: - Local VLM Content

    private var localVLMContent: some View {
        VStack(spacing: 10) {
            let ollamaInstalled = isOllamaInstalled()

            if ollamaInstalled {
                PermissionStatusBadge(
                    granted: true,
                    grantedLabel: "Ollama Installed",
                    deniedLabel: ""
                )

                if vlmPullInProgress {
                    VStack(spacing: 6) {
                        ProgressView()
                            .progressViewStyle(.circular)
                            .controlSize(.small)
                        Text("Pulling models...")
                            .font(.system(size: 12, design: .rounded))
                            .foregroundColor(.secondary)
                        if !vlmPullOutput.isEmpty {
                            Text(vlmPullOutput)
                                .font(.system(size: 11, design: .monospaced))
                                .foregroundColor(.secondary)
                                .lineLimit(2)
                        }
                    }
                } else {
                    VStack(alignment: .leading, spacing: 8) {
                        HStack(spacing: 6) {
                            Image(systemName: "arrow.down.circle.fill")
                                .foregroundColor(warmOrange)
                                .font(.system(size: 14))
                            Text("~6 GB download \u{00B7} Runs on Apple Silicon")
                                .font(.system(size: 12))
                                .foregroundColor(.secondary)
                        }

                        VStack(alignment: .leading, spacing: 4) {
                            modelRow("qwen3.5:2b", "2.7 GB", "Screen annotation \u{2014} reads your screen and describes what you\u{2019}re doing")
                            modelRow("qwen3.5:4b", "3.4 GB", "SOP generation \u{2014} writes step-by-step procedures from observations")
                            modelRow("all-minilm:l6-v2", "45 MB", "Task matching \u{2014} groups similar work together")
                        }

                        Button("Pull All Recommended Models") {
                            pullOllamaModel()
                        }
                        .font(.system(size: 13, weight: .semibold, design: .rounded))
                        .buttonStyle(.borderedProminent)
                        .tint(warmOrange)

                        Text("Or use any Ollama-compatible model \u{2014} edit annotation_model and sop_model in config.toml after setup.")
                            .font(.system(size: 11))
                            .foregroundColor(.secondary)
                            .frame(maxWidth: 380)
                    }
                    .frame(maxWidth: 440)
                }
            } else {
                VStack(spacing: 8) {
                    Text("Ollama not installed")
                        .font(.system(size: 12, design: .rounded))
                        .foregroundColor(.secondary)

                    Button("Download Ollama") {
                        if let url = URL(string: "https://ollama.com/download/mac") {
                            NSWorkspace.shared.open(url)
                        }
                    }
                    .buttonStyle(.bordered)

                    Text("Or install via: brew install ollama")
                        .font(.system(size: 11, design: .monospaced))
                        .foregroundColor(.secondary)
                }
            }
        }
    }

    // MARK: - Cloud VLM Content

    private var cloudVLMContent: some View {
        VStack(spacing: 12) {
            // Privacy consent
            if !remoteConsentGiven {
                VStack(spacing: 8) {
                    HStack(spacing: 6) {
                        Image(systemName: "exclamationmark.triangle.fill")
                            .foregroundColor(warmOrange)
                        Text("Privacy Notice")
                            .font(.system(size: 12, weight: .semibold, design: .rounded))
                    }
                    Text("Cloud VLM sends screenshots of your desktop to a third-party API for analysis. Only enable this if you accept this trade-off.")
                        .font(.system(size: 11))
                        .foregroundColor(.secondary)
                        .multilineTextAlignment(.center)
                        .frame(maxWidth: 350)

                    Button("I Understand & Consent") {
                        remoteConsentGiven = true
                    }
                    .buttonStyle(.bordered)
                }
            } else {
                // Provider picker
                Picker("Provider", selection: $selectedProvider) {
                    ForEach(RemoteProvider.allCases) { provider in
                        Text(provider.displayName).tag(provider)
                    }
                }
                .pickerStyle(.menu)
                .frame(maxWidth: 280)
                .onChange(of: selectedProvider) { _ in
                    customModelName = ""
                }

                // Model selection
                HStack(spacing: 8) {
                    Text("Model:")
                        .font(.system(size: 12))
                        .foregroundColor(.secondary)
                    TextField("Model name", text: $customModelName)
                        .textFieldStyle(.roundedBorder)
                        .frame(maxWidth: 220)
                }

                Text("Default: \(selectedProvider.defaultModel)")
                    .font(.system(size: 11))
                    .foregroundColor(.secondary)

                // API Key input
                SecureField("API Key", text: $apiKeyInput)
                    .textFieldStyle(.roundedBorder)
                    .frame(maxWidth: 300)

                Text("Stored securely in macOS Keychain")
                    .font(.system(size: 11))
                    .foregroundColor(.secondary)

                // Save & Test button
                HStack(spacing: 8) {
                    Button("Save Configuration") {
                        saveCloudVLMConfig()
                    }
                    .font(.system(size: 13, weight: .semibold, design: .rounded))
                    .buttonStyle(.borderedProminent)
                    .tint(warmOrange)
                    .disabled(apiKeyInput.count < 10)

                    if apiKeyValidating {
                        ProgressView()
                            .controlSize(.small)
                    }
                }

                if let valid = apiKeyValid {
                    PermissionStatusBadge(
                        granted: valid,
                        grantedLabel: "Configuration Saved",
                        deniedLabel: "Failed to save"
                    )
                }
            }
        }
    }

    // MARK: - Cloud VLM Config Save

    private func saveCloudVLMConfig() {
        apiKeyValidating = true

        DispatchQueue.global(qos: .userInitiated).async {
            // Store key in Keychain
            let stored = KeychainHelper.store(
                key: "agenthandover-\(selectedProvider.rawValue)-key",
                value: apiKeyInput
            )

            // Write config.toml update
            if stored {
                writeRemoteVLMConfig(
                    provider: selectedProvider.rawValue,
                    model: customModelName.isEmpty ? selectedProvider.defaultModel : customModelName,
                    apiKeyEnv: selectedProvider.envVar
                )
            }

            DispatchQueue.main.async {
                apiKeyValidating = false
                apiKeyValid = stored
                if stored && !apiKeyInput.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
                    appState.vlmAvailable = true
                }
            }
        }
    }

    private func writeRemoteVLMConfig(provider: String, model: String, apiKeyEnv: String) {
        let home = FileManager.default.homeDirectoryForCurrentUser
        let configDir = home
            .appendingPathComponent("Library/Application Support/agenthandover")
        let configPath = configDir.appendingPathComponent("config.toml")

        try? FileManager.default.createDirectory(at: configDir, withIntermediateDirectories: true)

        // Read existing config or start fresh
        var content = (try? String(contentsOf: configPath, encoding: .utf8)) ?? ""

        // Strip any existing remote-mode keys from [vlm] section to avoid
        // duplicates on repeated saves.  Then insert the new values.
        let remoteKeys = ["mode", "provider", "model", "api_key_env"]
        for key in remoteKeys {
            // Match lines like: mode = "remote"  or  provider = "openai"
            // (with optional leading whitespace and any quoted value)
            let pattern = "(?m)^[ \\t]*\(key)[ \\t]*=[ \\t]*\"[^\"]*\"[ \\t]*\\n?"
            if let regex = try? NSRegularExpression(pattern: pattern) {
                let range = NSRange(content.startIndex..., in: content)
                content = regex.stringByReplacingMatches(
                    in: content, range: range, withTemplate: ""
                )
            }
        }

        let newFields = "mode = \"remote\"\nprovider = \"\(provider)\"\nmodel = \"\(model)\"\napi_key_env = \"\(apiKeyEnv)\"\n"

        if content.contains("[vlm]") {
            if let vlmRange = content.range(of: "[vlm]") {
                let afterVlm = content[vlmRange.upperBound...]
                if let nextSection = afterVlm.range(of: "\n[") {
                    content.insert(contentsOf: "\n" + newFields, at: nextSection.lowerBound)
                } else {
                    content += "\n" + newFields
                }
            }
        } else {
            content += "\n[vlm]\n" + newFields
        }

        try? content.write(to: configPath, atomically: true, encoding: .utf8)
    }

    // MARK: - Screen 7: Browser Extension (Optional, Load Unpacked)

    private var chromeExtensionStep: some View {
        VStack(spacing: sectionSpacing) {
            HStack(spacing: 10) {
                Text("Browser workflows")
                    .font(heroFont)
                    .foregroundColor(warmBrown)

                Text("Optional")
                    .font(.system(size: 10, weight: .bold, design: .rounded))
                    .tracking(0.3)
                    .padding(.horizontal, 10)
                    .padding(.vertical, 4)
                    .background(
                        Capsule().fill(warmBrown.opacity(0.06))
                    )
                    .foregroundColor(warmBrown.opacity(0.5))
            }

            // What the extension does
            HStack(spacing: 12) {
                Image(systemName: "globe.badge.chevron.backward")
                    .font(.system(size: 20))
                    .foregroundColor(warmOrange.opacity(0.8))
                    .frame(width: 36)

                Text("Adds CSS selectors, form field names, and page structure to your procedures \u{2014} making browser automation more precise.")
                    .font(bodyFont)
                    .foregroundColor(.secondary)
                    .lineSpacing(5)
            }
            .padding(cardPadding)
            .frame(maxWidth: .infinity, alignment: .leading)
            .background(
                RoundedRectangle(cornerRadius: cardRadius)
                    .fill(cardBg)
            )
            .overlay(
                RoundedRectangle(cornerRadius: cardRadius)
                    .stroke(cardBorder, lineWidth: 1)
            )
            .shadow(color: cardShadowColor, radius: 12, y: 3)

            // Connection status and install instructions
            if appState.extensionConnected {
                extensionConnectedView
            } else if !extensionPath.isEmpty {
                extensionReadyView
            } else {
                extensionNotFoundView
            }

            // Supported browsers note
            HStack(spacing: 6) {
                Image(systemName: "info.circle")
                    .font(.system(size: 10))
                    .foregroundColor(warmBrown.opacity(0.3))
                Text("Works with Chrome, Brave, and Edge")
                    .font(captionFont)
                    .foregroundColor(captionColor)
            }
        }
    }

    // Extension already connected
    private var extensionConnectedView: some View {
        VStack(spacing: 10) {
            Image(systemName: "checkmark.circle.fill")
                .font(.system(size: 28))
                .foregroundColor(.green)

            Text("Browser extension connected!")
                .font(.system(size: 14, weight: .semibold, design: .rounded))
                .foregroundColor(warmBrown)

            Text("You\u{2019}re getting enhanced browser context in your procedures.")
                .font(bodyFont)
                .foregroundColor(.secondary)
                .multilineTextAlignment(.center)
        }
        .padding(20)
        .frame(maxWidth: .infinity)
        .background(
            RoundedRectangle(cornerRadius: cardRadius)
                .fill(Color.green.opacity(0.03))
        )
        .overlay(
            RoundedRectangle(cornerRadius: cardRadius)
                .stroke(Color.green.opacity(0.15), lineWidth: 1)
        )
    }

    // Extension files found — Load Unpacked flow
    private var extensionReadyView: some View {
        VStack(alignment: .leading, spacing: 14) {
            // Status header
            HStack(spacing: 8) {
                Image(systemName: "checkmark.circle.fill")
                    .font(.system(size: 16))
                    .foregroundColor(.green)
                Text("Extension ready to install")
                    .font(.system(size: 14, weight: .semibold, design: .rounded))
                    .foregroundColor(warmBrown)
            }

            // Three numbered steps
            VStack(alignment: .leading, spacing: 10) {
                // Step 1: Open extensions page
                HStack(alignment: .top, spacing: 12) {
                    stepCircle(number: 1)
                    VStack(alignment: .leading, spacing: 6) {
                        Text("Open your browser\u{2019}s extension page")
                            .font(.system(size: 13, weight: .medium, design: .rounded))
                        Button {
                            openBrowserExtensionsPage()
                        } label: {
                            HStack(spacing: 5) {
                                Image(systemName: "arrow.up.right.square")
                                    .font(.system(size: 11))
                                Text("Open Extensions Page")
                                    .font(.system(size: 12, weight: .medium, design: .rounded))
                            }
                        }
                        .buttonStyle(.bordered)
                        .controlSize(.small)
                    }
                }

                // Step 2: Developer Mode
                HStack(alignment: .top, spacing: 12) {
                    stepCircle(number: 2)
                    VStack(alignment: .leading, spacing: 3) {
                        Text("Enable Developer Mode")
                            .font(.system(size: 13, weight: .medium, design: .rounded))
                        Text("Toggle in the top-right corner of the extensions page")
                            .font(captionFont)
                            .foregroundColor(.secondary)
                    }
                }

                // Step 3: Load unpacked
                HStack(alignment: .top, spacing: 12) {
                    stepCircle(number: 3)
                    VStack(alignment: .leading, spacing: 6) {
                        Text("Click \"Load unpacked\" and select this folder:")
                            .font(.system(size: 13, weight: .medium, design: .rounded))

                        // Path display with copy button
                        HStack(spacing: 0) {
                            Text(extensionPath)
                                .font(monoFont)
                                .foregroundColor(warmBrown.opacity(0.7))
                                .textSelection(.enabled)
                                .lineLimit(1)
                                .truncationMode(.middle)

                            Spacer(minLength: 8)

                            Button {
                                let pasteboard = NSPasteboard.general
                                pasteboard.clearContents()
                                pasteboard.setString(extensionPath, forType: .string)
                                withAnimation { pathCopied = true }
                                DispatchQueue.main.asyncAfter(deadline: .now() + 2.0) {
                                    withAnimation { pathCopied = false }
                                }
                            } label: {
                                HStack(spacing: 4) {
                                    Image(systemName: pathCopied ? "checkmark" : "doc.on.doc")
                                        .font(.system(size: 10))
                                    Text(pathCopied ? "Copied" : "Copy")
                                        .font(.system(size: 11, weight: .medium, design: .rounded))
                                }
                                .foregroundColor(pathCopied ? .green : warmOrange)
                            }
                            .buttonStyle(.plain)
                        }
                        .padding(.horizontal, 10)
                        .padding(.vertical, 7)
                        .background(
                            RoundedRectangle(cornerRadius: 10)
                                .fill(warmCream.opacity(0.5))
                        )
                        .overlay(
                            RoundedRectangle(cornerRadius: 10)
                                .stroke(warmBrown.opacity(0.06), lineWidth: 1)
                        )
                    }
                }
            }

            if let error = chromeOpenError {
                Text(error)
                    .font(captionFont)
                    .foregroundColor(.red)
            }
        }
        .padding(cardPadding)
        .background(
            RoundedRectangle(cornerRadius: cardRadius)
                .fill(cardBg)
        )
        .overlay(
            RoundedRectangle(cornerRadius: cardRadius)
                .stroke(Color.green.opacity(0.15), lineWidth: 1)
        )
        .shadow(color: cardShadowColor, radius: 12, y: 3)
    }

    // Extension not found — coming soon
    private var extensionNotFoundView: some View {
        VStack(spacing: 12) {
            Image(systemName: "clock.badge.checkmark")
                .font(.system(size: 20))
                .foregroundColor(warmBrown.opacity(0.3))

            Text("Extension will be available on the Chrome Web Store soon.")
                .font(bodyFont)
                .foregroundColor(.secondary)
                .multilineTextAlignment(.center)

            Text("For now, you can skip this step \u{2014} AgentHandover works great without it.")
                .font(captionFont)
                .foregroundColor(captionColor)
                .multilineTextAlignment(.center)
                .frame(maxWidth: 400)
        }
        .padding(20)
        .frame(maxWidth: .infinity)
        .background(
            RoundedRectangle(cornerRadius: cardRadius)
                .fill(Color.primary.opacity(0.02))
        )
        .overlay(
            RoundedRectangle(cornerRadius: cardRadius)
                .stroke(cardBorder, lineWidth: 1)
        )
    }

    private func stepCircle(number: Int) -> some View {
        Text("\(number)")
            .font(.system(size: 12, weight: .bold, design: .rounded))
            .foregroundColor(.white)
            .frame(width: 24, height: 24)
            .background(
                Circle().fill(brandGradient)
            )
    }

    // MARK: - Screen 8: Ready — First Recording

    private var readyStep: some View {
        VStack(spacing: 20) {
            // Small mascot at top
            mascotImage(height: 64)

            Text("You\u{2019}re ready!")
                .font(heroFont)
                .foregroundColor(warmBrown)

            // Summary checks
            HStack(spacing: 14) {
                readinessChip(
                    icon: "checkmark.shield.fill",
                    label: "Permissions",
                    ok: appState.accessibilityGranted && appState.screenRecordingGranted
                )
                readinessChip(
                    icon: "brain.head.profile",
                    label: "AI Model",
                    ok: appState.vlmAvailable
                )
                readinessChip(
                    icon: "globe",
                    label: "Extension",
                    ok: appState.extensionConnected,
                    optional: true
                )
            }

            // Main recording card
            VStack(spacing: 16) {
                Text("Record your first workflow")
                    .font(.system(size: 17, weight: .semibold, design: .rounded))
                    .foregroundColor(warmBrown)

                Text("What\u{2019}s something you do regularly?")
                    .font(bodyFont)
                    .foregroundColor(.secondary)

                TextField("e.g. File expense report, Check inbox, Deploy code...", text: $firstRecordingTitle)
                    .textFieldStyle(.roundedBorder)
                    .font(bodyFont)
                    .frame(maxWidth: 340)

                let isDisabled = firstRecordingTitle.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
                    || !appState.accessibilityGranted
                    || !appState.vlmAvailable

                Button {
                    startServicesAndRecord()
                } label: {
                    HStack(spacing: 10) {
                        ZStack {
                            // Pulse ring
                            Circle()
                                .fill(Color.white.opacity(0.3))
                                .frame(width: 22, height: 22)
                                .scaleEffect(recordPulse ? 1.4 : 1.0)
                                .opacity(recordPulse ? 0.0 : 0.5)

                            // Solid dot
                            Circle()
                                .fill(Color.white)
                                .frame(width: 11, height: 11)
                        }
                        Text("Start Recording")
                            .font(.system(size: 15, weight: .bold, design: .rounded))
                    }
                    .padding(.horizontal, 28)
                    .padding(.vertical, 13)
                    .background(
                        Capsule()
                            .fill(isDisabled ? warmBrown.opacity(0.15) : Color.red)
                            .shadow(
                                color: isDisabled ? .clear : Color.red.opacity(0.3),
                                radius: 12, y: 4
                            )
                    )
                    .foregroundColor(isDisabled ? .secondary : .white)
                }
                .buttonStyle(.plain)
                .disabled(isDisabled)
                .onAppear {
                    withAnimation(.easeInOut(duration: 1.5).repeatForever(autoreverses: false)) {
                        recordPulse = true
                    }
                }
            }
            .padding(24)
            .frame(maxWidth: .infinity)
            .background(
                RoundedRectangle(cornerRadius: 18)
                    .fill(cardBg)
            )
            .overlay(
                RoundedRectangle(cornerRadius: 18)
                    .stroke(warmOrange.opacity(0.12), lineWidth: 1)
            )
            .shadow(color: warmOrange.opacity(0.08), radius: 18, y: 5)

            // Secondary: Just start observing
            Button("Or start observing \u{2192}") {
                startServicesOnly()
            }
            .foregroundColor(warmOrange)
            .buttonStyle(.plain)
            .font(.system(size: 13, weight: .medium, design: .rounded))

            if serviceStartFailed {
                Text("Services may not have started. Check agenthandover status in Terminal.")
                    .font(captionFont)
                    .foregroundColor(.red)
            } else if !appState.accessibilityGranted {
                Text("Accessibility permission is required (go back to step 5)")
                    .font(captionFont)
                    .foregroundColor(warmOrange)
            } else if !appState.vlmAvailable {
                Text("An AI model must be configured (go back to step 6)")
                    .font(captionFont)
                    .foregroundColor(warmOrange)
            }

            HStack(spacing: 4) {
                Text("AgentHandover lives in your menu bar")
                    .font(captionFont)
                    .foregroundColor(captionColor)
                Image(systemName: "arrow.up.right")
                    .font(.system(size: 9))
                    .foregroundColor(captionColor)
                Text("\u{2014} that\u{2019}s your control center")
                    .font(captionFont)
                    .foregroundColor(captionColor)
            }
        }
    }

    private func readinessChip(icon: String, label: String, ok: Bool, optional: Bool = false) -> some View {
        HStack(spacing: 6) {
            Image(systemName: ok ? "checkmark.circle.fill" : (optional ? "minus.circle" : "xmark.circle.fill"))
                .font(.system(size: 13))
                .foregroundColor(ok ? .green : (optional ? warmBrown.opacity(0.3) : warmOrange))
            Text(label)
                .font(.system(size: 12, weight: .medium, design: .rounded))
                .foregroundColor(ok ? warmBrown : .secondary)
        }
        .padding(.horizontal, 12)
        .padding(.vertical, 7)
        .background(
            Capsule()
                .fill(ok ? Color.green.opacity(0.05) : Color.primary.opacity(0.02))
        )
        .overlay(
            Capsule()
                .stroke(ok ? Color.green.opacity(0.12) : cardBorder, lineWidth: 1)
        )
    }

    // MARK: - Actions

    private func startServicesOnly() {
        let ok = ServiceController.startAll()
        if ok {
            onComplete?()
            NSApplication.shared.keyWindow?.close()
        } else {
            serviceStartFailed = true
        }
    }

    private func startServicesAndRecord() {
        let ok = ServiceController.startAll()
        if ok {
            // Write focus-session.json to trigger a recording
            let sessionId = UUID().uuidString
            let signal: [String: Any] = [
                "session_id": sessionId,
                "title": firstRecordingTitle.trimmingCharacters(in: .whitespacesAndNewlines),
                "started_at": ISO8601DateFormatter().string(from: Date()),
                "status": "recording",
            ]
            writeFocusSignalFile(signal)
            onComplete?()
            NSApplication.shared.keyWindow?.close()
        } else {
            serviceStartFailed = true
        }
    }

    private func writeFocusSignalFile(_ signal: [String: Any]) {
        let home = FileManager.default.homeDirectoryForCurrentUser
        let dir = home.appendingPathComponent("Library/Application Support/agenthandover")
        let target = dir.appendingPathComponent("focus-session.json")
        let tmp = dir.appendingPathComponent(".focus-session.json.tmp")

        do {
            try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
            let data = try JSONSerialization.data(withJSONObject: signal, options: .prettyPrinted)
            try data.write(to: tmp, options: .atomic)
            if FileManager.default.fileExists(atPath: target.path) {
                try FileManager.default.removeItem(at: target)
            }
            try FileManager.default.moveItem(at: tmp, to: target)
        } catch {
            // Silently fail — the recording won't start but services are already running.
            // The user can start a recording from the menu bar.
        }
    }

    // MARK: - Helpers

    private func resolveExtensionPath() {
        // 1. Check installed paths (pkg + Homebrew opt symlinks)
        let installedPaths: [String] = [
            "/usr/local/lib/agenthandover/extension",
            "/usr/local/opt/agenthandover/libexec/extension",
            "/opt/homebrew/opt/agenthandover/libexec/extension",
        ]

        for candidate in installedPaths {
            let manifestPath = (candidate as NSString).appendingPathComponent("manifest.json")
            if FileManager.default.fileExists(atPath: manifestPath) {
                extensionPath = candidate
                return
            }
        }

        // 2. Resolve Homebrew Cellar path by following the CLI binary symlink.
        //    This covers cases where the opt symlink is broken or not yet created.
        //    Pattern: /usr/local/bin/agenthandover -> .../Cellar/agenthandover/HEAD-xxx/bin/agenthandover
        //             -> .../Cellar/agenthandover/HEAD-xxx/libexec/extension/
        let cliBinaryPaths = ["/usr/local/bin/agenthandover", "/opt/homebrew/bin/agenthandover"]
        for binaryPath in cliBinaryPaths {
            let url = URL(fileURLWithPath: binaryPath)
            let resolved = url.resolvingSymlinksInPath().path
                .components(separatedBy: "/")
            if !resolved.isEmpty {
                if let binIdx = resolved.lastIndex(of: "bin") {
                    let prefix = resolved[..<binIdx].joined(separator: "/")
                    let cellarExt = prefix + "/libexec/extension"
                    if FileManager.default.fileExists(atPath: cellarExt + "/manifest.json") {
                        extensionPath = cellarExt
                        return
                    }
                }
            }
        }

        // 3. Check for dev/source build by walking ancestors from the app binary
        //    AND from the CLI binary to find extension/dist relative to repo root.
        var searchRoots: [URL] = []

        // From the running app binary
        if let execPath = Bundle.main.executableURL {
            searchRoots.append(execPath.deletingLastPathComponent())
        }

        // From the CLI binary
        if let cliPath = findCLIBinary() {
            searchRoots.append(URL(fileURLWithPath: cliPath).deletingLastPathComponent())
        }

        // Also check common source build locations
        let home = FileManager.default.homeDirectoryForCurrentUser.path
        let commonPaths = [
            "\(home)/Desktop/openmimic/extension/dist",
            "\(home)/Projects/AgentHandover/extension/dist",
            "\(home)/Developer/AgentHandover/extension/dist",
        ]
        for path in commonPaths {
            if FileManager.default.fileExists(atPath: path + "/manifest.json") {
                extensionPath = path
                return
            }
        }

        for root in searchRoots {
            var dir = root
            for _ in 0..<8 {
                let candidate = dir.appendingPathComponent("extension/dist").path
                if FileManager.default.fileExists(atPath: candidate + "/manifest.json") {
                    extensionPath = candidate
                    return
                }
                dir = dir.deletingLastPathComponent()
            }
        }
    }

    /// Find the agenthandover CLI binary on the system.
    private func findCLIBinary() -> String? {
        let knownPaths = [
            "/usr/local/bin/agenthandover",
            "/opt/homebrew/bin/agenthandover",
        ]
        for path in knownPaths {
            if FileManager.default.fileExists(atPath: path) {
                // Resolve symlinks to find the real location
                return URL(fileURLWithPath: path).resolvingSymlinksInPath().path
            }
        }
        return nil
    }

    private func openBrowserExtensionsPage() {
        chromeOpenError = nil

        // Try Chrome first, then Brave, then Edge
        let browsers: [(name: String, url: String)] = [
            ("Google Chrome", "chrome://extensions"),
            ("Brave Browser", "brave://extensions"),
            ("Microsoft Edge", "edge://extensions"),
        ]

        for browser in browsers {
            let process = Process()
            process.executableURL = URL(fileURLWithPath: "/usr/bin/open")
            process.arguments = ["-a", browser.name, browser.url]
            do {
                try process.run()
                process.waitUntilExit()
                if process.terminationStatus == 0 {
                    return
                }
            } catch {
                continue
            }
        }

        chromeOpenError = "Could not open a supported browser. Open Chrome, Brave, or Edge extensions page manually."
    }

    private func copyPathAndOpenChrome() {
        chromeOpenError = nil

        // Copy extension path to clipboard
        let pasteboard = NSPasteboard.general
        pasteboard.clearContents()
        let copied = pasteboard.setString(extensionPath, forType: .string)
        if !copied {
            chromeOpenError = "Failed to copy path to clipboard."
            return
        }

        // Open chrome://extensions via /usr/bin/open -a "Google Chrome"
        let process = Process()
        process.executableURL = URL(fileURLWithPath: "/usr/bin/open")
        process.arguments = ["-a", "Google Chrome", "chrome://extensions"]
        do {
            try process.run()
            process.waitUntilExit()
            if process.terminationStatus != 0 {
                chromeOpenError = "Could not open Chrome. Is Google Chrome installed?"
            }
        } catch {
            chromeOpenError = "Could not open Chrome: \(error.localizedDescription)"
        }
    }

    private func isOllamaInstalled() -> Bool {
        let paths = [
            "/usr/local/bin/ollama",
            "/opt/homebrew/bin/ollama",
        ]
        return paths.contains { FileManager.default.fileExists(atPath: $0) }
    }

    private func findOllamaPath() -> String? {
        let paths = [
            "/usr/local/bin/ollama",
            "/opt/homebrew/bin/ollama",
        ]
        return paths.first { FileManager.default.fileExists(atPath: $0) }
    }

    private func modelRow(_ name: String, _ size: String, _ description: String) -> some View {
        HStack(alignment: .top, spacing: 8) {
            Text("\u{2022}")
                .foregroundColor(warmOrange)
                .font(.system(size: 12))
            VStack(alignment: .leading, spacing: 1) {
                HStack(spacing: 6) {
                    Text(name)
                        .font(.system(size: 12, weight: .medium, design: .monospaced))
                    Text(size)
                        .font(.system(size: 11))
                        .foregroundColor(.secondary)
                }
                Text(description)
                    .font(.system(size: 11))
                    .foregroundColor(.secondary)
            }
        }
    }

    private func pullOllamaModel() {
        guard let ollamaPath = findOllamaPath() else { return }

        vlmPullInProgress = true
        vlmPullOutput = "Starting download..."

        let models = [
            ("qwen3.5:2b", "scene annotation"),
            ("qwen3.5:4b", "SOP generation"),
            ("all-minilm:l6-v2", "embeddings"),
        ]

        DispatchQueue.global(qos: .userInitiated).async {
            for (index, (model, purpose)) in models.enumerated() {
                DispatchQueue.main.async {
                    vlmPullOutput = "[\(index + 1)/\(models.count)] Pulling \(model) (\(purpose))..."
                }

                let process = Process()
                process.executableURL = URL(fileURLWithPath: ollamaPath)
                process.arguments = ["pull", model]

                let pipe = Pipe()
                process.standardOutput = pipe
                process.standardError = pipe

                do {
                    try process.run()

                    // Read output asynchronously
                    pipe.fileHandleForReading.readabilityHandler = { handle in
                        let data = handle.availableData
                        if !data.isEmpty, let output = String(data: data, encoding: .utf8) {
                            let lastLine = output.components(separatedBy: "\n")
                                .filter { !$0.isEmpty }
                                .last ?? ""
                            DispatchQueue.main.async {
                                vlmPullOutput = "[\(index + 1)/\(models.count)] \(model): \(String(lastLine.prefix(60)))"
                            }
                        }
                    }

                    process.waitUntilExit()
                    pipe.fileHandleForReading.readabilityHandler = nil

                    if process.terminationStatus != 0 {
                        DispatchQueue.main.async {
                            vlmPullInProgress = false
                            vlmPullOutput = "Failed to pull \(model). Make sure Ollama is running."
                        }
                        return
                    }
                } catch {
                    DispatchQueue.main.async {
                        vlmPullInProgress = false
                        vlmPullOutput = "Failed to run ollama: \(error.localizedDescription)"
                    }
                    return
                }
            }

            DispatchQueue.main.async {
                vlmPullInProgress = false
                vlmPullOutput = "All models downloaded successfully!"
                appState.vlmAvailable = true
            }
        }
    }
}

// MARK: - Models

enum OnboardingAction {
    case none
    case accessibility
    case screenRecording
    case chromeExtension
    case vlmSetup
}

// MARK: - Subviews

/// Simple macOS Keychain wrapper for storing/retrieving API keys.
struct KeychainHelper {
    static func store(key: String, value: String) -> Bool {
        guard let data = value.data(using: .utf8) else { return false }

        // Delete existing item first
        let deleteQuery: [String: Any] = [
            kSecClass as String: kSecClassGenericPassword,
            kSecAttrService as String: "com.agenthandover.app",
            kSecAttrAccount as String: key,
        ]
        SecItemDelete(deleteQuery as CFDictionary)

        // Add new item
        let addQuery: [String: Any] = [
            kSecClass as String: kSecClassGenericPassword,
            kSecAttrService as String: "com.agenthandover.app",
            kSecAttrAccount as String: key,
            kSecValueData as String: data,
        ]
        let status = SecItemAdd(addQuery as CFDictionary, nil)
        return status == errSecSuccess
    }

    static func retrieve(key: String) -> String? {
        let query: [String: Any] = [
            kSecClass as String: kSecClassGenericPassword,
            kSecAttrService as String: "com.agenthandover.app",
            kSecAttrAccount as String: key,
            kSecReturnData as String: true,
            kSecMatchLimit as String: kSecMatchLimitOne,
        ]

        var result: AnyObject?
        let status = SecItemCopyMatching(query as CFDictionary, &result)
        guard status == errSecSuccess,
              let data = result as? Data,
              let string = String(data: data, encoding: .utf8) else {
            return nil
        }
        return string
    }
}

struct PermissionStatusBadge: View {
    let granted: Bool
    let grantedLabel: String
    let deniedLabel: String

    private let warmOrange = Color(red: 0.92, green: 0.57, blue: 0.20)

    var body: some View {
        HStack(spacing: 6) {
            Image(systemName: granted ? "checkmark.circle.fill" : "xmark.circle.fill")
                .foregroundColor(granted ? .green : warmOrange)
            Text(granted ? grantedLabel : deniedLabel)
                .font(.system(size: 12, weight: .medium, design: .rounded))
                .foregroundColor(granted ? .green : warmOrange)
        }
        .padding(.horizontal, 14)
        .padding(.vertical, 8)
        .background(
            Capsule()
                .fill((granted ? Color.green : warmOrange).opacity(0.08))
        )
        .overlay(
            Capsule()
                .stroke((granted ? Color.green : warmOrange).opacity(0.15), lineWidth: 1)
        )
    }
}
