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
    @State private var enableImageEmbeddings = false
    @State private var remoteConsentGiven = false
    @State private var useCustomModels = false
    @State private var customAnnotationModel = "qwen3.5:2b"
    @State private var customSOPModel = "qwen3.5:4b"
    @State private var ollamaInstalled = false
    @State private var ollamaRunning = false
    @State private var ollamaHasRequiredModels = false
    @State private var remoteConfigSaved = false
    @State private var vlmStateRefreshing = false
    @State private var vlmStatusMessage = ""
    @State private var hasSeededVLMFromConfig = false

    // Focus recording from onboarding
    @State private var firstRecordingTitle: String = ""

    // Clipboard copy feedback
    @State private var pathCopied = false

    // Record button pulse
    @State private var recordPulse = false

    /// Called when onboarding completes (sets hasCompletedOnboarding).
    var onComplete: (() -> Void)?

    private let totalSteps = 9

    // MARK: - Design Tokens (Contra-inspired)

    private let darkNavy = Color(red: 0.09, green: 0.10, blue: 0.12)        // #18191F
    private let warmOrange = Color(red: 0.92, green: 0.57, blue: 0.20)      // #EA9134
    private let goldenYellow = Color(red: 1.0, green: 0.74, blue: 0.07)     // #FFBD12
    private let warmCream = Color(red: 1.0, green: 0.96, blue: 0.88)        // #FFF5E0
    private let lightGray = Color(red: 0.96, green: 0.96, blue: 0.96)       // #F5F5F5
    private let brightGreen = Color(red: 0.18, green: 0.80, blue: 0.34)     // #2ECC57

    // Contra constants
    private let contraRadius: CGFloat = 16
    private let contraBorder: CGFloat = 2
    private let thickBorder: CGFloat = 3

    // Typography
    private let heroFont = Font.system(size: 36, weight: .black, design: .rounded)
    private let titleFont = Font.system(size: 28, weight: .bold, design: .rounded)
    private let bodyFont = Font.system(size: 14)
    private let captionFont = Font.system(size: 12)
    private let monoFont = Font.system(size: 12, design: .monospaced)

    /// Whether current screen uses a colored (non-white) background.
    private var isColoredScreen: Bool {
        currentStep == 0
    }

    /// Longer setup screens need the full vertical lane between progress and
    /// footer so their inner ScrollViews can actually scroll instead of being
    /// squeezed by surrounding spacers.
    private var usesTopAlignedStepLayout: Bool {
        currentStep >= 6
    }

    var body: some View {
        ZStack {
            // Background: solid orange for welcome, white for content screens, cream for ready
            if currentStep == 0 {
                warmOrange.ignoresSafeArea()
            } else if currentStep == 7 {
                warmCream.ignoresSafeArea()
            } else {
                Color.white.ignoresSafeArea()
            }

            VStack(spacing: 0) {
                // Progress indicator
                progressBar
                    .padding(.top, 20)
                    .padding(.horizontal, 44)
                    .padding(.bottom, 16)

                // Current step content
                stepContent(for: currentStep)
                    .padding(.horizontal, 44)
                    .frame(
                        maxWidth: .infinity,
                        maxHeight: .infinity,
                        alignment: usesTopAlignedStepLayout ? .top : .center
                    )

                // Navigation
                navigationBar
                    .padding(.horizontal, 44)
                    .padding(.bottom, 28)
            }
        }
        .onAppear {
            resolveExtensionPath()
            seedVLMStateFromConfigIfNeeded()
            refreshVLMSetupState(force: currentStep == 6)
        }
        .onChange(of: currentStep) { newStep in
            if newStep == 6 {
                seedVLMStateFromConfigIfNeeded()
                refreshVLMSetupState(force: true)
            }

            if newStep >= 7 {
                ServiceController.installNativeMessagingHostManifest()
            } else {
                ServiceController.removeNativeMessagingHostManifest()
            }
        }
        .onReceive(NotificationCenter.default.publisher(for: NSApplication.didBecomeActiveNotification)) { _ in
            if currentStep == 6 {
                refreshVLMSetupState(force: true)
            }
        }
    }

    // MARK: - Progress Bar

    private var progressBar: some View {
        VStack(spacing: 8) {
            GeometryReader { geometry in
                ZStack(alignment: .leading) {
                    // Track
                    Capsule()
                        .fill(isColoredScreen ? Color.white.opacity(0.25) : darkNavy.opacity(0.08))
                        .frame(height: 4)

                    // Fill
                    Capsule()
                        .fill(isColoredScreen ? Color.white : darkNavy)
                        .frame(
                            width: geometry.size.width * CGFloat(currentStep + 1) / CGFloat(totalSteps),
                            height: 4
                        )
                        .animation(.easeOut(duration: 0.45), value: currentStep)
                }
            }
            .frame(height: 4)

            Text("Step \(currentStep + 1) of \(totalSteps)")
                .font(.system(size: 10, weight: .bold, design: .rounded))
                .foregroundColor(isColoredScreen ? Color.white.opacity(0.6) : darkNavy.opacity(0.4))
                .tracking(0.5)
        }
    }

    // MARK: - Navigation Bar (Contra two-button footer)

    private var navigationBar: some View {
        HStack {
            // Left: Back/Skip as ghost button
            if currentStep > 0 {
                Button("Back") {
                    withAnimation(.spring(response: 0.35, dampingFraction: 0.8)) { currentStep -= 1 }
                }
                .buttonStyle(.plain)
                .font(.system(size: 14, weight: .semibold, design: .rounded))
                .foregroundColor(currentStep == 7 ? darkNavy.opacity(0.4) : darkNavy.opacity(0.4))
                .padding(.horizontal, 20)
                .padding(.vertical, 11)
                .background(
                    RoundedRectangle(cornerRadius: contraRadius)
                        .stroke(currentStep == 0 ? Color.white.opacity(0.4) : darkNavy.opacity(0.15), lineWidth: contraBorder)
                )
            }

            Spacer()

            switch currentStep {
            case 0:
                contraButton("Get Started", style: .whiteFilled) {
                    withAnimation(.spring(response: 0.35, dampingFraction: 0.8)) { currentStep += 1 }
                }

            case 1, 2, 3:
                contraButton("Next", icon: "arrow.right", style: .darkFilled) {
                    withAnimation(.spring(response: 0.35, dampingFraction: 0.8)) { currentStep += 1 }
                }

            case 4:
                // Accessibility
                VStack(spacing: 6) {
                    contraButton(
                        appState.accessibilityGranted ? "Next" : "Continue anyway",
                        icon: "arrow.right",
                        style: .darkFilled,
                        disabled: false
                    ) {
                        withAnimation(.spring(response: 0.35, dampingFraction: 0.8)) { currentStep += 1 }
                    }
                    if !appState.accessibilityGranted {
                        Text("You can grant this later via agenthandover doctor")
                            .font(.system(size: 11))
                            .foregroundColor(darkNavy.opacity(0.35))
                    }
                }

            case 5:
                // Screen Recording
                VStack(spacing: 6) {
                    contraButton(
                        appState.screenRecordingGranted ? "Next" : "Continue anyway",
                        icon: "arrow.right",
                        style: .darkFilled,
                        disabled: false
                    ) {
                        withAnimation(.spring(response: 0.35, dampingFraction: 0.8)) { currentStep += 1 }
                    }
                    if !appState.screenRecordingGranted {
                        Text("You can grant this later via agenthandover doctor")
                            .font(.system(size: 11))
                            .foregroundColor(darkNavy.opacity(0.35))
                    }
                }

            case 6:
                // VLM Setup -- blocked until model ready
                VStack(spacing: 4) {
                    if onboardingVLMReady {
                        HStack(spacing: 6) {
                            Image(systemName: "checkmark.circle.fill")
                                .foregroundColor(.green)
                            Text("AI models ready")
                                .font(.system(size: 13, weight: .bold, design: .rounded))
                                .foregroundColor(.green)
                        }
                        .padding(.bottom, 4)
                    }

                    contraButton(
                        "Next",
                        icon: "arrow.right",
                        style: .darkFilled,
                        disabled: !onboardingVLMReady
                    ) {
                        withAnimation(.spring(response: 0.35, dampingFraction: 0.8)) { currentStep += 1 }
                    }

                    if !onboardingVLMReady {
                        Text("Set up an AI model above to continue")
                            .font(.system(size: 12, weight: .medium, design: .rounded))
                            .foregroundColor(warmOrange)
                    }
                }

            case 7:
                // Browser extension -- optional
                HStack(spacing: 12) {
                    if !appState.extensionConnected {
                        contraButton("Skip", style: .outlined) {
                            withAnimation(.spring(response: 0.35, dampingFraction: 0.8)) { currentStep += 1 }
                        }
                    }

                    contraButton("Next", icon: "arrow.right", style: .darkFilled) {
                        withAnimation(.spring(response: 0.35, dampingFraction: 0.8)) { currentStep += 1 }
                    }
                }

            case 8:
                // Ready -- final step, no Next button
                EmptyView()

            default:
                EmptyView()
            }
        }
    }

    // MARK: - Contra Button Styles

    enum ContraButtonStyle {
        case darkFilled      // Dark navy background, white text
        case whiteFilled     // White background, dark text (for colored screens)
        case outlined        // Ghost/outlined button
    }

    private func contraButton(
        _ label: String,
        icon: String? = nil,
        style: ContraButtonStyle = .darkFilled,
        disabled: Bool = false,
        action: @escaping () -> Void
    ) -> some View {
        Button {
            action()
        } label: {
            HStack(spacing: 7) {
                Text(label)
                    .font(.system(size: 15, weight: .bold, design: .rounded))
                if let icon = icon {
                    Image(systemName: icon)
                        .font(.system(size: 11, weight: .bold))
                }
            }
            .padding(.horizontal, 24)
            .padding(.vertical, 12)
            .background(buttonBackground(style: style, disabled: disabled))
            .foregroundColor(buttonForeground(style: style, disabled: disabled))
            .clipShape(RoundedRectangle(cornerRadius: contraRadius))
            .overlay(
                RoundedRectangle(cornerRadius: contraRadius)
                    .stroke(buttonBorderColor(style: style, disabled: disabled), lineWidth: style == .outlined ? contraBorder : 0)
            )
        }
        .buttonStyle(.plain)
        .disabled(disabled)
    }

    private func buttonBackground(style: ContraButtonStyle, disabled: Bool) -> some ShapeStyle {
        switch style {
        case .darkFilled:
            return AnyShapeStyle(disabled ? darkNavy.opacity(0.12) : darkNavy)
        case .whiteFilled:
            return AnyShapeStyle(Color.white)
        case .outlined:
            return AnyShapeStyle(Color.clear)
        }
    }

    private func buttonForeground(style: ContraButtonStyle, disabled: Bool) -> Color {
        switch style {
        case .darkFilled:
            return disabled ? darkNavy.opacity(0.3) : .white
        case .whiteFilled:
            return darkNavy
        case .outlined:
            return darkNavy.opacity(0.5)
        }
    }

    private func buttonBorderColor(style: ContraButtonStyle, disabled: Bool) -> Color {
        switch style {
        case .outlined:
            return darkNavy.opacity(0.15)
        default:
            return .clear
        }
    }

    // MARK: - Step Content

    @ViewBuilder
    private func stepContent(for step: Int) -> some View {
        switch step {
        case 0: welcomeStep
        case 1: teachByDoingStep
        case 2: whatYoullGetStep
        case 3: reviewCycleStep
        case 4: accessibilityStep
        case 5: screenRecordingStep
        case 6: vlmSetupStep
        case 7: chromeExtensionStep
        case 8: readyStep
        default: EmptyView()
        }
    }

    // MARK: - Mascot Image Helper

    /// Loads the mascot image with multiple fallback strategies.
    @ViewBuilder
    private func mascotImage(height: CGFloat) -> some View {
        if let nsImg = onboardingMascotImage() {
            Image(nsImage: nsImg)
                .resizable()
                .aspectRatio(contentMode: .fit)
                .frame(height: height)
        } else {
            // Last resort: styled SF Symbol
            Image(systemName: "binoculars.fill")
                .font(.system(size: height * 0.6, weight: .medium))
                .foregroundColor(.white)
                .frame(height: height)
        }
    }

    private func onboardingMascotImage() -> NSImage? {
        if let image = NSImage(named: "mascot") {
            return image
        }

        let bundleNames = [
            "AgentHandoverApp_AgentHandoverApp.bundle",
            "AgentHandover_AgentHandoverApp.bundle",
        ]

        let candidateBundles: [Bundle] = [
            Bundle.main,
            Bundle(for: OnboardingViewHost.self),
        ] + bundleNames.compactMap { name in
            guard let resourceURL = Bundle.main.resourceURL else { return nil }
            let bundleURL = resourceURL.appendingPathComponent(name)
            return Bundle(url: bundleURL)
        }

        for bundle in candidateBundles {
            if let image = bundle.image(forResource: "mascot") {
                return image
            }
            if let url = bundle.url(forResource: "mascot", withExtension: "png"),
               let image = NSImage(contentsOf: url) {
                return image
            }
        }

        return nil
    }

    // MARK: - Screen 1: Welcome (SOLID ORANGE BACKGROUND)

    private var welcomeStep: some View {
        VStack(spacing: 24) {
            // Mascot inside golden circle with thick dark border
            ZStack {
                Circle()
                    .fill(goldenYellow)
                    .frame(width: 200, height: 200)
                    .overlay(
                        Circle()
                            .stroke(darkNavy, lineWidth: thickBorder)
                    )

                mascotImage(height: 160)
            }

            // Title + tagline
            VStack(spacing: 10) {
                Text("AgentHandover")
                    .font(.system(size: 36, weight: .black, design: .rounded))
                    .foregroundColor(.white)

                Text("Work once. Hand over forever.")
                    .font(.system(size: 20, weight: .semibold, design: .rounded))
                    .foregroundColor(.white)

                Text("AgentHandover studies how you work -every app, every step, every pattern -and teaches agents like OpenClaw, Claude Code, and Codex to do it exactly the way you would.")
                    .font(bodyFont)
                    .foregroundColor(.white.opacity(0.85))
                    .multilineTextAlignment(.center)
                    .lineSpacing(5)
                    .frame(maxWidth: 440)

                Text("No instructions to write. No workflows to document. It just watches and learns.")
                    .font(captionFont)
                    .foregroundColor(.white.opacity(0.7))
                    .multilineTextAlignment(.center)
            }

            // Three value props as simple white text lines
            VStack(spacing: 8) {
                valuePropLine("Watches silently as you work")
                valuePropLine("Learns your patterns and decisions")
                valuePropLine("Teaches your agents to do it for you")
            }
            .padding(.top, 2)

            // Privacy badge
            HStack(spacing: 7) {
                Image(systemName: "lock.fill")
                    .font(.system(size: 10, weight: .semibold))
                    .foregroundColor(.white.opacity(0.7))
                Text("Everything runs locally on your Mac")
                    .font(.system(size: 12, weight: .medium, design: .rounded))
                    .foregroundColor(.white.opacity(0.7))
            }
        }
    }

    private func valuePropLine(_ text: String) -> some View {
        Text(text)
            .font(.system(size: 14, weight: .medium, design: .rounded))
            .foregroundColor(.white.opacity(0.9))
    }

    // MARK: - Screen 2: Two Ways to Teach (WHITE BACKGROUND)

    private var teachByDoingStep: some View {
        VStack(spacing: sectionSpacing) {
            Text("Two ways to teach")
                .font(titleFont)
                .foregroundColor(darkNavy)

            HStack(spacing: 16) {
                // Focus Recording -- golden yellow background, recommended
                VStack(alignment: .leading, spacing: 14) {
                    // Recommended badge: dark navy pill
                    HStack {
                        Spacer()
                        Text("Recommended")
                            .font(.system(size: 10, weight: .bold, design: .rounded))
                            .tracking(0.5)
                            .textCase(.uppercase)
                            .foregroundColor(.white)
                            .padding(.horizontal, 12)
                            .padding(.vertical, 5)
                            .background(
                                Capsule().fill(darkNavy)
                            )
                    }

                    Image(systemName: "record.circle")
                        .font(.system(size: 28, weight: .medium))
                        .foregroundColor(darkNavy)

                    Text("Focus Recording")
                        .font(.system(size: 16, weight: .bold, design: .rounded))
                        .foregroundColor(darkNavy)

                    Text("Show your agent how it\u{2019}s done. Record yourself doing the task once -AgentHandover figures out the rest.")
                        .font(.system(size: 12))
                        .foregroundColor(darkNavy.opacity(0.7))
                        .lineSpacing(5)

                    VStack(alignment: .leading, spacing: 7) {
                        numberedStep(1, "Click Record")
                        numberedStep(2, "Do the task as usual")
                        numberedStep(3, "Stop -AI analyzes in 2-5 min")
                    }

                    Text("Your agent gets a complete handoff document with steps, strategy, guardrails, and verification criteria.")
                        .font(.system(size: 11))
                        .foregroundColor(darkNavy.opacity(0.5))
                        .lineSpacing(4)
                }
                .padding(20)
                .frame(maxWidth: .infinity, alignment: .leading)
                .background(
                    RoundedRectangle(cornerRadius: contraRadius)
                        .fill(goldenYellow)
                )
                .overlay(
                    RoundedRectangle(cornerRadius: contraRadius)
                        .stroke(darkNavy, lineWidth: contraBorder)
                )

                // Passive Learning -- light gray background
                VStack(alignment: .leading, spacing: 14) {
                    Spacer().frame(height: 28) // align with ribbon space

                    Image(systemName: "eye")
                        .font(.system(size: 28, weight: .medium))
                        .foregroundColor(darkNavy.opacity(0.5))

                    Text("Passive Learning")
                        .font(.system(size: 16, weight: .bold, design: .rounded))
                        .foregroundColor(darkNavy)

                    Text("Just work normally. AgentHandover spots patterns in your daily work and builds handoff documents automatically. The more you work, the smarter it gets.")
                        .font(.system(size: 12))
                        .foregroundColor(darkNavy.opacity(0.6))
                        .lineSpacing(5)
                }
                .padding(20)
                .frame(maxWidth: .infinity, alignment: .leading)
                .background(
                    RoundedRectangle(cornerRadius: contraRadius)
                        .fill(lightGray)
                )
                .overlay(
                    RoundedRectangle(cornerRadius: contraRadius)
                        .stroke(darkNavy, lineWidth: contraBorder)
                )
            }

            Text("We recommend starting with a Focus Recording -you\u{2019}ll have your first agent-ready handoff in minutes.")
                .font(captionFont)
                .foregroundColor(darkNavy.opacity(0.4))
                .multilineTextAlignment(.center)
                .lineSpacing(3)
                .frame(maxWidth: 460)
        }
    }

    private let sectionSpacing: CGFloat = 24

    private func numberedStep(_ n: Int, _ text: String) -> some View {
        HStack(spacing: 8) {
            Text("\(n)")
                .font(.system(size: 11, weight: .bold, design: .rounded))
                .foregroundColor(.white)
                .frame(width: 22, height: 22)
                .background(
                    Circle().fill(darkNavy)
                )
            Text(text)
                .font(.system(size: 13, weight: .medium))
                .foregroundColor(darkNavy.opacity(0.7))
        }
    }

    // MARK: - Screen 3: What You Get (WHITE BACKGROUND)

    private var whatYoullGetStep: some View {
        ScrollView(showsIndicators: false) {
        VStack(spacing: sectionSpacing) {
            Text("What your agent receives")
                .font(titleFont)
                .foregroundColor(darkNavy)

            // Illustrative example badge
            HStack(spacing: 6) {
                Image(systemName: "lightbulb.fill")
                    .font(.system(size: 10))
                Text("Illustrative example")
                    .font(.system(size: 11, weight: .semibold))
            }
            .foregroundColor(darkNavy.opacity(0.5))
            .padding(.horizontal, 12)
            .padding(.vertical, 5)
            .background(
                Capsule()
                    .fill(goldenYellow.opacity(0.25))
            )
            .overlay(
                Capsule()
                    .stroke(goldenYellow.opacity(0.4), lineWidth: 1)
            )

            // Mock procedure -- card with thick dark border
            HStack(spacing: 0) {
                // Orange left accent stripe
                warmOrange
                    .frame(width: 4)
                    .clipShape(RoundedRectangle(cornerRadius: 2))

                VStack(alignment: .leading, spacing: 0) {
                    // Document title
                    Text("ILLUSTRATIVE EXAMPLE")
                        .font(.system(size: 9, weight: .bold, design: .monospaced))
                        .foregroundColor(warmOrange)
                        .tracking(1)
                        .padding(.bottom, 4)
                    Text("Reddit Community Marketing")
                        .font(.system(size: 18, weight: .bold, design: .rounded))
                        .foregroundColor(darkNavy)
                        .padding(.bottom, 3)

                    Text("Daily engagement workflow \u{00B7} 6 steps \u{00B7} 4 sessions learned")
                        .font(.system(size: 11, weight: .medium))
                        .foregroundColor(darkNavy.opacity(0.4))
                        .padding(.bottom, 14)

                    // Strategy
                    docSectionLabel("Strategy")
                    Text("Browse target subreddits for posts about marketing tools or growth hacking. Engage with high-signal posts (10+ comments, posted within 48h, not promotional). Write authentic replies that acknowledge the problem, share personal experience, and softly mention the product.")
                        .font(.system(size: 12))
                        .foregroundColor(darkNavy.opacity(0.6))
                        .lineSpacing(4)
                        .padding(.bottom, 14)

                    // Steps
                    docSectionLabel("Steps")
                    VStack(alignment: .leading, spacing: 5) {
                        docStep(1, "Open Reddit and navigate to r/startups")
                        docStep(2, "Scan posts -skip promotional, skip < 10 comments")
                        docStep(3, "Open high-signal post and read top comments")
                        docStep(4, "Write reply: acknowledge \u{2192} experience \u{2192} mention product")
                        docStep(5, "Submit and verify not auto-removed")
                        docStep(6, "Repeat for r/marketing, r/growthacking (max 5/day)")
                    }
                    .padding(.bottom, 14)

                    // Selection Criteria & Guardrails
                    HStack(alignment: .top, spacing: 20) {
                        VStack(alignment: .leading, spacing: 5) {
                            docSectionLabel("Selection Criteria")
                            docBullet("Posts with 10+ comments")
                            docBullet("Not promotional or competitor")
                            docBullet("Posted within 48 hours")
                            docBullet("Relevant to [product category]")
                        }
                        .frame(maxWidth: .infinity, alignment: .leading)

                        VStack(alignment: .leading, spacing: 5) {
                            docSectionLabel("Guardrails")
                            docBullet("Max 5 replies per day")
                            docBullet("Never identical phrasing")
                            docBullet("Never reply to own posts")
                            docBullet("Empathy-first tone always")
                        }
                        .frame(maxWidth: .infinity, alignment: .leading)
                    }
                    .padding(.bottom, 12)

                    // Thin divider
                    Rectangle()
                        .fill(darkNavy.opacity(0.08))
                        .frame(height: 1)
                        .padding(.bottom, 10)

                    // Footer -- Timing
                    HStack(spacing: 14) {
                        HStack(spacing: 4) {
                            Image(systemName: "clock")
                                .font(.system(size: 9))
                                .foregroundColor(darkNavy.opacity(0.35))
                            Text("~15 min daily \u{00B7} 9-10am")
                                .font(.system(size: 11))
                                .foregroundColor(darkNavy.opacity(0.35))
                        }
                        HStack(spacing: 4) {
                            Image(systemName: "chart.bar.fill")
                                .font(.system(size: 9))
                                .foregroundColor(brightGreen)
                            Text("Confidence: 89%")
                                .font(.system(size: 11))
                                .foregroundColor(darkNavy.opacity(0.35))
                        }
                    }
                }
                .padding(20)
            }
            .background(
                RoundedRectangle(cornerRadius: contraRadius)
                    .fill(Color.white)
            )
            .overlay(
                RoundedRectangle(cornerRadius: contraRadius)
                    .stroke(darkNavy, lineWidth: contraBorder)
            )

            Text("This is what your agent receives -not just steps, but the strategy, decisions, and guardrails behind them.")
                .font(captionFont)
                .foregroundColor(darkNavy.opacity(0.4))
                .multilineTextAlignment(.center)
                .frame(maxWidth: 460)
        }
        } // ScrollView
    }

    private func docSectionLabel(_ text: String) -> some View {
        Text(text.uppercased())
            .font(.system(size: 10, weight: .bold, design: .rounded))
            .foregroundColor(darkNavy.opacity(0.35))
            .tracking(0.8)
            .padding(.bottom, 5)
    }

    private func docStep(_ number: Int, _ text: String) -> some View {
        HStack(alignment: .center, spacing: 9) {
            Text("\(number)")
                .font(.system(size: 10, weight: .bold, design: .rounded))
                .foregroundColor(.white)
                .frame(width: 18, height: 18)
                .background(
                    Circle().fill(warmOrange)
                )
            Text(text)
                .font(.system(size: 12))
                .foregroundColor(darkNavy.opacity(0.6))
        }
    }

    private func docBullet(_ text: String) -> some View {
        HStack(alignment: .top, spacing: 6) {
            Text("\u{2022}")
                .font(.system(size: 10))
                .foregroundColor(darkNavy.opacity(0.3))
            Text(text)
                .font(.system(size: 11))
                .foregroundColor(darkNavy.opacity(0.6))
        }
    }

    // MARK: - Screen 4: Review Cycle (WHITE BACKGROUND, THICK TIMELINE)

    private var reviewCycleStep: some View {
        VStack(spacing: sectionSpacing) {
            Text("You stay in control")
                .font(titleFont)
                .foregroundColor(darkNavy)

            // Vertical timeline with THICK connecting line
            VStack(alignment: .leading, spacing: 0) {
                timelineNode(
                    icon: "camera.fill",
                    title: "Record or Observe",
                    subtitle: "Work normally -AgentHandover captures everything",
                    color: warmOrange,
                    isHighlighted: false,
                    isLast: false
                )
                timelineNode(
                    icon: "brain.head.profile",
                    title: "AI Analyzes",
                    subtitle: "Extracts strategy, steps, decisions, and guardrails",
                    color: darkNavy,
                    isHighlighted: false,
                    isLast: false
                )
                timelineNode(
                    icon: "person.fill",
                    title: "You Review",
                    subtitle: "Approve, refine, or reject with one tap",
                    color: goldenYellow,
                    isHighlighted: true,
                    isLast: false
                )
                timelineNode(
                    icon: "cpu",
                    title: "Agent Ready",
                    subtitle: "Your agent executes exactly how you would",
                    color: brightGreen,
                    isHighlighted: false,
                    isLast: true
                )
            }
            .padding(.leading, 4)

            // Menu bar callout
            HStack(spacing: 14) {
                Image(systemName: "hand.tap.fill")
                    .font(.system(size: 20, weight: .medium))
                    .foregroundColor(darkNavy)

                VStack(alignment: .leading, spacing: 3) {
                    Text("Review from your menu bar")
                        .font(.system(size: 14, weight: .bold, design: .rounded))
                        .foregroundColor(darkNavy)
                    Text("All of this lives in your menu bar -review drafts, approve handoffs, and monitor your agents from one place.")
                        .font(bodyFont)
                        .foregroundColor(darkNavy.opacity(0.6))
                        .lineSpacing(5)
                }
            }
            .padding(20)
            .frame(maxWidth: .infinity, alignment: .leading)
            .background(
                RoundedRectangle(cornerRadius: contraRadius)
                    .fill(lightGray)
            )
            .overlay(
                RoundedRectangle(cornerRadius: contraRadius)
                    .stroke(darkNavy, lineWidth: contraBorder)
            )
        }
    }

    private func timelineNode(
        icon: String,
        title: String,
        subtitle: String,
        color: Color,
        isHighlighted: Bool,
        isLast: Bool
    ) -> some View {
        HStack(alignment: .top, spacing: 16) {
            // Dot + connecting line
            VStack(spacing: 0) {
                ZStack {
                    Circle()
                        .fill(isHighlighted ? goldenYellow : color)
                        .frame(width: 40, height: 40)
                        .overlay(
                            Circle()
                                .stroke(darkNavy, lineWidth: thickBorder)
                        )

                    Image(systemName: icon)
                        .font(.system(size: 15, weight: .bold))
                        .foregroundColor(isHighlighted ? darkNavy : .white)
                }

                if !isLast {
                    Rectangle()
                        .fill(darkNavy)
                        .frame(width: thickBorder, height: 28)
                }
            }

            VStack(alignment: .leading, spacing: 3) {
                Text(title)
                    .font(.system(size: 15, weight: .bold, design: .rounded))
                    .foregroundColor(isHighlighted ? darkNavy : darkNavy.opacity(0.8))
                Text(subtitle)
                    .font(.system(size: 13))
                    .foregroundColor(darkNavy.opacity(0.5))
            }
            .padding(.top, 8)
        }
    }

    // MARK: - Screen 5: Accessibility (WHITE BACKGROUND)

    private enum PermissionKind {
        case accessibility
        case screenRecording
    }

    @State private var probingPermission: PermissionKind? = nil
    @State private var awaitingSettingsReturn: PermissionKind? = nil
    @State private var screenRecordingNeedsRestart = false
    @State private var screenRecordingNeedsManualSettings = false

    private var accessibilityStep: some View {
        singlePermissionStep(
            icon: "hand.raised.circle.fill",
            title: "Enable Accessibility",
            description: "AgentHandover reads window titles and UI elements to understand your workflow.",
            instruction: "Toggle on AgentHandover in the list",
            granted: appState.accessibilityGranted,
            isScreenRecording: false
        )
    }

    // MARK: - Screen 6: Screen Recording (WHITE BACKGROUND)

    private var screenRecordingStep: some View {
        singlePermissionStep(
            icon: "rectangle.dashed.badge.record",
            title: "Enable Screen Recording",
            description: "AgentHandover captures screenshots to analyze what you see on screen.",
            instruction: "Toggle on AgentHandover in the list",
            granted: appState.screenRecordingGranted,
            isScreenRecording: true
        )
    }

    private var resolvedAnnotationModel: String {
        let custom = customAnnotationModel.trimmingCharacters(in: .whitespacesAndNewlines)
        return useCustomModels && !custom.isEmpty ? custom : "qwen3.5:2b"
    }

    private var resolvedSOPModel: String {
        let custom = customSOPModel.trimmingCharacters(in: .whitespacesAndNewlines)
        return useCustomModels && !custom.isEmpty ? custom : "qwen3.5:4b"
    }

    private var requiredLocalModels: [String] {
        Array(Set([resolvedAnnotationModel, resolvedSOPModel, "nomic-embed-text"])).sorted()
    }

    private var localVLMReady: Bool {
        ollamaInstalled && ollamaRunning && ollamaHasRequiredModels
    }

    private var cloudVLMReady: Bool {
        remoteConfigSaved || apiKeyValid == true
    }

    private var onboardingVLMReady: Bool {
        vlmMode == .cloud ? cloudVLMReady : localVLMReady
    }

    private var hasOllamaAppBundle: Bool {
        FileManager.default.fileExists(atPath: "/Applications/Ollama.app")
    }

    private var vlmModeSwitcher: some View {
        HStack(spacing: 10) {
            vlmModeButton(.local)
            vlmModeButton(.cloud)
        }
        .frame(maxWidth: 280)
    }

    private func vlmModeButton(_ mode: VLMMode) -> some View {
        let selected = vlmMode == mode
        return Button {
            vlmMode = mode
        } label: {
            Text(mode.rawValue)
                .font(.system(size: 13, weight: .bold, design: .rounded))
                .foregroundColor(selected ? .white : darkNavy)
                .frame(maxWidth: .infinity)
                .padding(.vertical, 10)
                .background(
                    RoundedRectangle(cornerRadius: 12)
                        .fill(selected ? darkNavy : Color.white)
                )
                .overlay(
                    RoundedRectangle(cornerRadius: 12)
                        .stroke(selected ? darkNavy : darkNavy.opacity(0.2), lineWidth: contraBorder)
                )
        }
        .buttonStyle(.plain)
    }

    /// Shared view for a single permission step.
    private func singlePermissionStep(
        icon: String,
        title: String,
        description: String,
        instruction: String,
        granted: Bool,
        isScreenRecording: Bool
    ) -> some View {
        let permission: PermissionKind = isScreenRecording ? .screenRecording : .accessibility
        let isProbing = probingPermission == permission

        return VStack(spacing: sectionSpacing) {
            Text(title)
                .font(titleFont)
                .foregroundColor(darkNavy)

            Text(description)
                .font(bodyFont)
                .foregroundColor(darkNavy.opacity(0.6))
                .multilineTextAlignment(.center)
                .frame(maxWidth: 400)

            permissionCard(
                icon: icon,
                title: title.replacingOccurrences(of: "Enable ", with: ""),
                description: instruction,
                granted: granted,
                action: {
                    openPermissionSettings(isScreenRecording: isScreenRecording)
                },
                actionLabel: isProbing ? "Starting..." : permissionActionLabel(
                    permission: permission,
                    granted: granted
                )
            )

            if screenRecordingNeedsRestart && isScreenRecording {
                Button(action: { restartApp() }) {
                    HStack(spacing: 7) {
                        Image(systemName: "arrow.triangle.2.circlepath")
                            .font(.system(size: 11))
                        Text("Restart AgentHandover to apply")
                            .font(.system(size: 12, weight: .medium))
                            .foregroundColor(warmOrange)
                    }
                }
                .buttonStyle(.plain)
                .padding(.top, 4)
            } else if !granted {
                Button(action: { recheckPermission(permission) }) {
                    HStack(spacing: 7) {
                        if isProbing {
                            ProgressView()
                                .controlSize(.small)
                        } else {
                            Image(systemName: "arrow.clockwise")
                                .font(.system(size: 11))
                        }
                        Text(isProbing ? "Checking..." : "I've granted it - recheck")
                            .font(.system(size: 12, weight: .medium))
                            .foregroundColor(darkNavy.opacity(0.6))
                    }
                }
                .buttonStyle(.plain)
                .disabled(isProbing)
                .padding(.top, 4)
            }

            HStack(spacing: 7) {
                Image(systemName: "eye.fill")
                    .font(.system(size: 11))
                    .foregroundColor(darkNavy.opacity(0.3))
                Text("AgentHandover reads your screen. It never types, clicks, or takes actions.")
                    .font(captionFont)
                    .foregroundColor(darkNavy.opacity(0.4))
            }
            .multilineTextAlignment(.center)
            .frame(maxWidth: 440)
        }
        // Auto-recheck ONLY when returning from System Settings (not on step entry)
        .onReceive(NotificationCenter.default.publisher(for: NSApplication.didBecomeActiveNotification)) { _ in
            if awaitingSettingsReturn == permission && !granted {
                awaitingSettingsReturn = nil
                recheckPermission(permission, returnedFromSettings: true)
            }
        }
    }

    /// Open Settings for a specific permission.
    /// For Screen Recording: first trigger the app-owned grant flow. Only on a
    /// second explicit click do we deep-link to Settings as a manual fallback.
    /// For Accessibility: request from the app principal and let macOS drive
    /// the Settings experience for AgentHandover itself.
    private func openPermissionSettings(isScreenRecording: Bool = false) {
        let permission: PermissionKind = isScreenRecording ? .screenRecording : .accessibility
        guard probingPermission != permission else { return }
        probingPermission = permission

        Task { @MainActor in
            if isScreenRecording {
                if screenRecordingNeedsManualSettings {
                    ServiceController.prepareForAppOwnedPermissionRequest()
                    PermissionChecker.openScreenRecordingSettings()
                    awaitingSettingsReturn = .screenRecording
                    probingPermission = nil
                    return
                }

                let granted = await PermissionChecker.requestScreenRecordingAndOpenSettingsIfNeeded()
                if granted {
                    let status = await PermissionChecker.resolveScreenRecordingStatus(
                        timeoutNanoseconds: 1_500_000_000
                    )
                    appState.screenRecordingGranted = status.granted
                    screenRecordingNeedsRestart = status.granted && !status.captureReady
                    screenRecordingNeedsManualSettings = false
                    awaitingSettingsReturn = status.granted ? nil : .screenRecording
                } else {
                    appState.screenRecordingGranted = false
                    screenRecordingNeedsRestart = false
                    screenRecordingNeedsManualSettings = true
                    awaitingSettingsReturn = .screenRecording
                }
                probingPermission = nil
            } else {
                let granted = await PermissionChecker.requestAccessibilityAndOpenSettingsIfNeeded()
                appState.accessibilityGranted = granted
                probingPermission = nil
                awaitingSettingsReturn = granted ? nil : .accessibility
            }
        }
    }

    /// Restart the app so macOS applies Screen Recording permission.
    private func restartApp() {
        let url = URL(fileURLWithPath: Bundle.main.bundlePath)
        let config = NSWorkspace.OpenConfiguration()
        config.createsNewApplicationInstance = true
        NSWorkspace.shared.openApplication(at: url, configuration: config) { _, _ in
            DispatchQueue.main.async {
                NSApp.terminate(nil)
            }
        }
    }

    /// Recheck a single permission without cross-contaminating the other screen.
    private func recheckPermission(
        _ permission: PermissionKind,
        returnedFromSettings: Bool = false
    ) {
        guard probingPermission != permission else { return }
        probingPermission = permission

        Task { @MainActor in
            switch permission {
            case .screenRecording:
                let status = await PermissionChecker.resolveScreenRecordingStatus(
                    timeoutNanoseconds: returnedFromSettings ? 5_000_000_000 : 1_500_000_000
                )
                appState.screenRecordingGranted = status.granted
                screenRecordingNeedsRestart = status.granted && !status.captureReady
                screenRecordingNeedsManualSettings = !status.granted

            case .accessibility:
                appState.accessibilityGranted = PermissionChecker.isAccessibilityGranted()
            }

            if probingPermission == permission {
                probingPermission = nil
            }
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
                .font(.system(size: 24, weight: .medium))
                .foregroundColor(granted ? brightGreen : darkNavy)
                .frame(width: 40)

            VStack(alignment: .leading, spacing: 3) {
                Text(title)
                    .font(.system(size: 15, weight: .bold, design: .rounded))
                    .foregroundColor(darkNavy)
                Text(description)
                    .font(.system(size: 12))
                    .foregroundColor(darkNavy.opacity(0.5))
            }

            Spacer()

            if granted {
                HStack(spacing: 5) {
                    Image(systemName: "checkmark.circle.fill")
                        .foregroundColor(brightGreen)
                    Text("Granted")
                        .foregroundColor(brightGreen)
                }
                .font(.system(size: 13, weight: .bold, design: .rounded))
                .padding(.horizontal, 14)
                .padding(.vertical, 8)
                .background(
                    RoundedRectangle(cornerRadius: contraRadius)
                        .fill(brightGreen.opacity(0.1))
                )
                .overlay(
                    RoundedRectangle(cornerRadius: contraRadius)
                        .stroke(brightGreen, lineWidth: contraBorder)
                )
            } else {
                Button(actionLabel) {
                    action()
                }
                .font(.system(size: 13, weight: .bold, design: .rounded))
                .foregroundColor(.white)
                .padding(.horizontal, 16)
                .padding(.vertical, 8)
                .background(
                    RoundedRectangle(cornerRadius: contraRadius)
                        .fill(darkNavy)
                )
                .buttonStyle(.plain)
            }
        }
        .padding(20)
        .background(
            RoundedRectangle(cornerRadius: contraRadius)
                .fill(granted ? brightGreen.opacity(0.04) : Color.white)
        )
        .overlay(
            RoundedRectangle(cornerRadius: contraRadius)
                .stroke(granted ? brightGreen : darkNavy, lineWidth: contraBorder)
        )
    }

    private func permissionActionLabel(permission: PermissionKind, granted: Bool) -> String {
        guard !granted else { return "Granted" }
        switch permission {
        case .accessibility:
            return "Open Settings"
        case .screenRecording:
            return screenRecordingNeedsManualSettings ? "Open Settings" : "Grant Access"
        }
    }

    // MARK: - Screen 7: VLM Setup (Required, WHITE BACKGROUND)

    private var vlmSetupStep: some View {
        ScrollView(showsIndicators: false) {
            VStack(spacing: 18) {
                Image(systemName: "brain.head.profile")
                    .font(.system(size: 40, weight: .medium))
                    .foregroundColor(darkNavy)

                Text("Set up your AI")
                    .font(titleFont)
                    .foregroundColor(darkNavy)

                Text("Choose a local or cloud AI model to understand what\u{2019}s on your screen.")
                    .font(bodyFont)
                    .foregroundColor(darkNavy.opacity(0.5))
                    .multilineTextAlignment(.center)
                    .lineSpacing(5)
                    .frame(maxWidth: 440)

                if onboardingVLMReady {
                    PermissionStatusBadge(
                        granted: true,
                        grantedLabel: vlmMode == .cloud ? "Cloud AI Ready" : "Local AI Ready",
                        deniedLabel: ""
                    )
                }

                vlmModeSwitcher

                if vlmStateRefreshing {
                    ProgressView("Checking AI setup...")
                        .font(.system(size: 12, weight: .medium, design: .rounded))
                        .tint(warmOrange)
                }

                if vlmMode == .cloud {
                    cloudVLMContent
                } else {
                    localVLMContent
                }
            }
            .frame(maxWidth: 520)
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .top)
    }

    // MARK: - Local VLM Content

    private var localVLMContent: some View {
        VStack(spacing: 12) {
            if ollamaInstalled {
                PermissionStatusBadge(
                    granted: true,
                    grantedLabel: "Ollama Installed",
                    deniedLabel: ""
                )

                if !ollamaRunning {
                    VStack(spacing: 10) {
                        Text("Open Ollama once to start the local AI service")
                            .font(.system(size: 13, weight: .semibold, design: .rounded))
                            .foregroundColor(darkNavy)

                        Text(hasOllamaAppBundle
                            ? "After a fresh install, Ollama usually needs one launch before AgentHandover can pull models."
                            : "If you installed Ollama through Homebrew, run `ollama serve` once before pulling models.")
                            .font(.system(size: 12))
                            .foregroundColor(darkNavy.opacity(0.5))
                            .multilineTextAlignment(.center)
                            .frame(maxWidth: 360)

                        HStack(spacing: 10) {
                            if hasOllamaAppBundle {
                                Button("Open Ollama") {
                                    launchOllamaApp()
                                }
                                .font(.system(size: 14, weight: .bold, design: .rounded))
                                .foregroundColor(.white)
                                .padding(.horizontal, 18)
                                .padding(.vertical, 10)
                                .background(
                                    RoundedRectangle(cornerRadius: contraRadius)
                                        .fill(darkNavy)
                                )
                                .buttonStyle(.plain)
                            }

                            Button("Recheck") {
                                refreshVLMSetupState(force: true)
                            }
                            .font(.system(size: 13, weight: .bold, design: .rounded))
                            .foregroundColor(darkNavy)
                            .padding(.horizontal, 16)
                            .padding(.vertical, 10)
                            .background(
                                RoundedRectangle(cornerRadius: contraRadius)
                                    .fill(Color.white)
                            )
                            .overlay(
                                RoundedRectangle(cornerRadius: contraRadius)
                                    .stroke(darkNavy, lineWidth: contraBorder)
                            )
                            .buttonStyle(.plain)
                        }

                        if !vlmStatusMessage.isEmpty {
                            Text(vlmStatusMessage)
                                .font(.system(size: 11))
                                .foregroundColor(darkNavy.opacity(0.45))
                                .multilineTextAlignment(.center)
                        }
                    }
                    .padding(20)
                    .background(
                        RoundedRectangle(cornerRadius: contraRadius)
                            .fill(Color.white)
                    )
                    .overlay(
                        RoundedRectangle(cornerRadius: contraRadius)
                            .stroke(darkNavy, lineWidth: contraBorder)
                    )
                    .frame(maxWidth: 440)
                } else if vlmPullInProgress {
                    VStack(spacing: 8) {
                        ProgressView()
                            .progressViewStyle(.circular)
                            .controlSize(.small)
                        Text("Pulling models...")
                            .font(.system(size: 13, weight: .medium, design: .rounded))
                            .foregroundColor(darkNavy.opacity(0.5))
                        if !vlmPullOutput.isEmpty {
                            Text(vlmPullOutput)
                                .font(.system(size: 11, design: .monospaced))
                                .foregroundColor(darkNavy.opacity(0.4))
                                .lineLimit(2)
                        }
                    }
                } else {
                    VStack(alignment: .leading, spacing: 10) {
                        HStack(spacing: 8) {
                            Image(systemName: ollamaHasRequiredModels ? "checkmark.circle.fill" : "arrow.down.circle.fill")
                                .foregroundColor(ollamaHasRequiredModels ? brightGreen : warmOrange)
                                .font(.system(size: 16))
                            Text(ollamaHasRequiredModels ? "Required local models are already available." : "~6 GB download \u{00B7} Runs on Apple Silicon")
                                .font(.system(size: 13, weight: .medium))
                                .foregroundColor(darkNavy.opacity(0.5))
                        }

                        HStack(spacing: 8) {
                            Image(systemName: "bolt.circle.fill")
                                .foregroundColor(warmOrange)
                                .font(.system(size: 16))
                            Text("Pulling models only works while the Ollama app is open.")
                                .font(.system(size: 13, weight: .medium))
                                .foregroundColor(darkNavy.opacity(0.5))
                        }

                        VStack(alignment: .leading, spacing: 6) {
                            modelRow("qwen3.5:2b", "2.7 GB", "Screen annotation - reads your screen and describes what you're doing")
                            modelRow("qwen3.5:4b", "3.4 GB", "SOP generation - writes step-by-step procedures from observations")
                            modelRow("nomic-embed-text", "274 MB", "Semantic search - finds similar workflows by meaning")
                        }

                        // Image embeddings toggle
                        HStack(spacing: 10) {
                            Toggle("", isOn: $enableImageEmbeddings)
                                .toggleStyle(.switch)
                                .labelsHidden()

                            VStack(alignment: .leading, spacing: 2) {
                                Text("Visual search (recommended)")
                                    .font(.system(size: 13, weight: .semibold, design: .rounded))
                                    .foregroundColor(darkNavy)
                                Text("Embeds screenshots so agents can find visually similar screens. Downloads ~1 GB SigLIP model on first use. Requires Apple Silicon.")
                                    .font(.system(size: 11))
                                    .foregroundColor(darkNavy.opacity(0.5))
                                    .lineLimit(3)
                            }
                        }
                        .padding(12)
                        .background(
                            RoundedRectangle(cornerRadius: 10)
                                .fill(enableImageEmbeddings ? goldenYellow.opacity(0.15) : lightGray)
                        )
                        .overlay(
                            RoundedRectangle(cornerRadius: 10)
                                .stroke(enableImageEmbeddings ? warmOrange.opacity(0.5) : darkNavy.opacity(0.1), lineWidth: 1)
                        )

                        Button(ollamaHasRequiredModels ? "Refresh Models" : "Pull All Models") {
                            pullOllamaModel()
                        }
                        .font(.system(size: 14, weight: .bold, design: .rounded))
                        .foregroundColor(.white)
                        .padding(.horizontal, 20)
                        .padding(.vertical, 10)
                        .background(
                            RoundedRectangle(cornerRadius: contraRadius)
                                .fill(darkNavy)
                        )
                        .buttonStyle(.plain)

                        Button {
                            useCustomModels.toggle()
                        } label: {
                            HStack(spacing: 8) {
                                Image(systemName: useCustomModels ? "chevron.down.circle.fill" : "chevron.right.circle")
                                    .foregroundColor(warmOrange)
                                Text(useCustomModels ? "Hide custom models" : "Use different models")
                                    .font(.system(size: 12, weight: .medium, design: .rounded))
                                    .foregroundColor(darkNavy.opacity(0.65))
                                Spacer()
                            }
                            .contentShape(Rectangle())
                        }
                        .buttonStyle(.plain)

                        if useCustomModels {
                            VStack(alignment: .leading, spacing: 8) {
                                HStack(spacing: 8) {
                                    Text("Annotation:")
                                        .font(.system(size: 12, weight: .medium))
                                        .foregroundColor(darkNavy.opacity(0.6))
                                        .frame(width: 80, alignment: .trailing)
                                    TextField("qwen3.5:2b", text: $customAnnotationModel)
                                        .textFieldStyle(.plain)
                                        .font(.system(size: 12, design: .monospaced))
                                        .padding(6)
                                        .background(
                                            RoundedRectangle(cornerRadius: 6)
                                                .fill(Color.white)
                                        )
                                        .overlay(
                                            RoundedRectangle(cornerRadius: 6)
                                                .stroke(darkNavy.opacity(0.15), lineWidth: 1)
                                        )
                                }
                                HStack(spacing: 8) {
                                    Text("Generation:")
                                        .font(.system(size: 12, weight: .medium))
                                        .foregroundColor(darkNavy.opacity(0.6))
                                        .frame(width: 80, alignment: .trailing)
                                    TextField("qwen3.5:4b", text: $customSOPModel)
                                        .textFieldStyle(.plain)
                                        .font(.system(size: 12, design: .monospaced))
                                        .padding(6)
                                        .background(
                                            RoundedRectangle(cornerRadius: 6)
                                                .fill(Color.white)
                                        )
                                        .overlay(
                                            RoundedRectangle(cornerRadius: 6)
                                                .stroke(darkNavy.opacity(0.15), lineWidth: 1)
                                        )
                                }
                                Text("Enter any Ollama model name (e.g. llama3.2-vision, gemma3)")
                                    .font(.system(size: 10))
                                    .foregroundColor(darkNavy.opacity(0.4))
                            }
                            .padding(.top, 6)
                        }

                        if !vlmStatusMessage.isEmpty {
                            Text(vlmStatusMessage)
                                .font(.system(size: 11))
                                .foregroundColor(darkNavy.opacity(0.45))
                        }
                    }
                    .padding(20)
                    .background(
                        RoundedRectangle(cornerRadius: contraRadius)
                            .fill(Color.white)
                    )
                    .overlay(
                        RoundedRectangle(cornerRadius: contraRadius)
                            .stroke(darkNavy, lineWidth: contraBorder)
                    )
                    .frame(maxWidth: 440)
                }
            } else {
                VStack(spacing: 10) {
                    Text("Ollama not installed")
                        .font(.system(size: 13, weight: .medium, design: .rounded))
                        .foregroundColor(darkNavy.opacity(0.5))

                    Button("Download Ollama") {
                        if let url = URL(string: "https://ollama.com/download/mac") {
                            NSWorkspace.shared.open(url)
                        }
                    }
                    .font(.system(size: 14, weight: .bold, design: .rounded))
                    .foregroundColor(.white)
                    .padding(.horizontal, 20)
                    .padding(.vertical, 10)
                    .background(
                        RoundedRectangle(cornerRadius: contraRadius)
                            .fill(darkNavy)
                    )
                    .buttonStyle(.plain)

                    Text("Or install via: brew install ollama")
                        .font(.system(size: 11, design: .monospaced))
                        .foregroundColor(darkNavy.opacity(0.4))

                    Button("I've installed Ollama — recheck") {
                        refreshVLMSetupState(force: true)
                    }
                    .font(.system(size: 12, weight: .semibold, design: .rounded))
                    .foregroundColor(darkNavy.opacity(0.7))
                    .buttonStyle(.plain)
                }
                .padding(20)
                .background(
                    RoundedRectangle(cornerRadius: contraRadius)
                        .fill(Color.white)
                )
                .overlay(
                    RoundedRectangle(cornerRadius: contraRadius)
                        .stroke(darkNavy, lineWidth: contraBorder)
                )
            }
        }
    }

    // MARK: - Cloud VLM Content

    private var cloudVLMContent: some View {
        VStack(spacing: 14) {
            // Privacy consent
            if !remoteConsentGiven {
                VStack(spacing: 10) {
                    HStack(spacing: 6) {
                        Image(systemName: "exclamationmark.triangle.fill")
                            .foregroundColor(warmOrange)
                        Text("Privacy Notice")
                            .font(.system(size: 13, weight: .bold, design: .rounded))
                            .foregroundColor(darkNavy)
                    }
                    Text("Cloud VLM sends screenshots of your desktop to a third-party API for analysis. Only enable this if you accept this trade-off.")
                        .font(.system(size: 12))
                        .foregroundColor(darkNavy.opacity(0.5))
                        .multilineTextAlignment(.center)
                        .frame(maxWidth: 350)

                    Button("I Understand & Consent") {
                        remoteConsentGiven = true
                    }
                    .font(.system(size: 14, weight: .bold, design: .rounded))
                    .foregroundColor(.white)
                    .padding(.horizontal, 20)
                    .padding(.vertical, 10)
                    .background(
                        RoundedRectangle(cornerRadius: contraRadius)
                            .fill(darkNavy)
                    )
                    .buttonStyle(.plain)
                }
                .padding(20)
                .background(
                    RoundedRectangle(cornerRadius: contraRadius)
                        .fill(Color.white)
                )
                .overlay(
                    RoundedRectangle(cornerRadius: contraRadius)
                        .stroke(warmOrange, lineWidth: contraBorder)
                )
            } else {
                VStack(spacing: 12) {
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
                            .font(.system(size: 13, weight: .medium))
                            .foregroundColor(darkNavy.opacity(0.5))
                        TextField("Model name", text: $customModelName)
                            .textFieldStyle(.roundedBorder)
                            .frame(maxWidth: 220)
                    }

                    Text("Default: \(selectedProvider.defaultModel)")
                        .font(.system(size: 11))
                        .foregroundColor(darkNavy.opacity(0.4))

                    // API Key input
                    SecureField("API Key", text: $apiKeyInput)
                        .textFieldStyle(.roundedBorder)
                        .frame(maxWidth: 300)

                    Text("Stored securely in macOS Keychain")
                        .font(.system(size: 11))
                        .foregroundColor(darkNavy.opacity(0.4))

                    // Save & Test button
                    HStack(spacing: 8) {
                        Button("Save Configuration") {
                            saveCloudVLMConfig()
                        }
                        .font(.system(size: 14, weight: .bold, design: .rounded))
                        .foregroundColor(.white)
                        .padding(.horizontal, 20)
                        .padding(.vertical, 10)
                        .background(
                            RoundedRectangle(cornerRadius: contraRadius)
                                .fill(apiKeyInput.count < 10 ? darkNavy.opacity(0.2) : darkNavy)
                        )
                        .buttonStyle(.plain)
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
                .padding(20)
                .background(
                    RoundedRectangle(cornerRadius: contraRadius)
                        .fill(Color.white)
                )
                .overlay(
                    RoundedRectangle(cornerRadius: contraRadius)
                        .stroke(darkNavy, lineWidth: contraBorder)
                )
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
                    remoteConfigSaved = true
                    appState.vlmAvailable = onboardingVLMReady
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

    // MARK: - Screen 8: Browser Extension (Optional, Load Unpacked, WHITE BACKGROUND)

    private var chromeExtensionStep: some View {
        ScrollView(showsIndicators: false) {
            VStack(spacing: sectionSpacing) {
                HStack(spacing: 10) {
                    Text("Browser workflows")
                        .font(titleFont)
                        .foregroundColor(darkNavy)

                    Text("Optional")
                        .font(.system(size: 10, weight: .bold, design: .rounded))
                        .tracking(0.5)
                        .textCase(.uppercase)
                        .padding(.horizontal, 12)
                        .padding(.vertical, 5)
                        .background(
                            Capsule().fill(darkNavy.opacity(0.08))
                        )
                        .foregroundColor(darkNavy.opacity(0.4))
                }

                HStack(spacing: 14) {
                    Image(systemName: "globe.badge.chevron.backward")
                        .font(.system(size: 22, weight: .medium))
                        .foregroundColor(darkNavy)
                        .frame(width: 40)

                    Text("Adds CSS selectors, form field names, and page structure to your procedures -making browser automation more precise.")
                        .font(bodyFont)
                        .foregroundColor(darkNavy.opacity(0.6))
                        .lineSpacing(5)
                }
                .padding(20)
                .frame(maxWidth: .infinity, alignment: .leading)
                .background(
                    RoundedRectangle(cornerRadius: contraRadius)
                        .fill(Color.white)
                )
                .overlay(
                    RoundedRectangle(cornerRadius: contraRadius)
                        .stroke(darkNavy, lineWidth: contraBorder)
                )

                if appState.extensionConnected {
                    extensionConnectedView
                } else if !extensionPath.isEmpty {
                    extensionReadyView
                } else {
                    extensionNotFoundView
                }

                Text("We recommend setting up the extension now. It won't be detected here -- just proceed with Next once installed, or Skip to set it up later.")
                    .font(.system(size: 11))
                    .foregroundColor(darkNavy.opacity(0.5))
                    .multilineTextAlignment(.center)
                    .frame(maxWidth: 440)

                HStack(spacing: 6) {
                    Image(systemName: "info.circle")
                        .font(.system(size: 11))
                        .foregroundColor(darkNavy.opacity(0.3))
                    Text("Works with Chrome, Brave, Edge, and any Chromium-based browser")
                        .font(captionFont)
                        .foregroundColor(darkNavy.opacity(0.4))
                }
            }
            .frame(maxWidth: 520)
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .top)
    }

    // Extension already connected
    private var extensionConnectedView: some View {
        VStack(spacing: 12) {
            Image(systemName: "checkmark.circle.fill")
                .font(.system(size: 32, weight: .medium))
                .foregroundColor(brightGreen)

            Text("Browser extension connected!")
                .font(.system(size: 15, weight: .bold, design: .rounded))
                .foregroundColor(darkNavy)

            Text("You\u{2019}re getting enhanced browser context in your procedures.")
                .font(bodyFont)
                .foregroundColor(darkNavy.opacity(0.5))
                .multilineTextAlignment(.center)
        }
        .padding(24)
        .frame(maxWidth: .infinity)
        .background(
            RoundedRectangle(cornerRadius: contraRadius)
                .fill(brightGreen.opacity(0.05))
        )
        .overlay(
            RoundedRectangle(cornerRadius: contraRadius)
                .stroke(brightGreen, lineWidth: contraBorder)
        )
    }

    // Extension files found -- Load Unpacked flow
    private var extensionReadyView: some View {
        VStack(alignment: .leading, spacing: 14) {
            // Status header
            HStack(spacing: 8) {
                Image(systemName: "puzzlepiece.extension.fill")
                    .font(.system(size: 18))
                    .foregroundColor(warmOrange)
                Text("Extension ready to install")
                    .font(.system(size: 15, weight: .bold, design: .rounded))
                    .foregroundColor(darkNavy)
            }

            // Three numbered steps
            VStack(alignment: .leading, spacing: 12) {
                // Step 1: Open extensions page
                HStack(alignment: .top, spacing: 12) {
                    stepCircle(number: 1)
                    VStack(alignment: .leading, spacing: 8) {
                        Text("Open your browser\u{2019}s extension page")
                            .font(.system(size: 13, weight: .bold, design: .rounded))
                            .foregroundColor(darkNavy)
                        Button {
                            openBrowserExtensionsPage()
                        } label: {
                            HStack(spacing: 5) {
                                Image(systemName: "arrow.up.right.square")
                                    .font(.system(size: 11))
                                Text("Open Extensions Page")
                                    .font(.system(size: 12, weight: .bold, design: .rounded))
                            }
                            .foregroundColor(darkNavy)
                            .padding(.horizontal, 14)
                            .padding(.vertical, 7)
                            .background(
                                RoundedRectangle(cornerRadius: contraRadius)
                                    .stroke(darkNavy, lineWidth: contraBorder)
                            )
                        }
                        .buttonStyle(.plain)
                    }
                }

                // Step 2: Developer Mode
                HStack(alignment: .top, spacing: 12) {
                    stepCircle(number: 2)
                    VStack(alignment: .leading, spacing: 3) {
                        Text("Enable Developer Mode")
                            .font(.system(size: 13, weight: .bold, design: .rounded))
                            .foregroundColor(darkNavy)
                        Text("Toggle in the top-right corner of the extensions page")
                            .font(captionFont)
                            .foregroundColor(darkNavy.opacity(0.4))
                    }
                }

                // Step 3: Load unpacked
                HStack(alignment: .top, spacing: 12) {
                    stepCircle(number: 3)
                    VStack(alignment: .leading, spacing: 8) {
                        Text("Click \"Load unpacked\" and select this folder:")
                            .font(.system(size: 13, weight: .bold, design: .rounded))
                            .foregroundColor(darkNavy)

                        // Path display with copy button
                        HStack(spacing: 0) {
                            Text(extensionPath)
                                .font(monoFont)
                                .foregroundColor(darkNavy.opacity(0.6))
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
                                        .font(.system(size: 11, weight: .bold, design: .rounded))
                                }
                                .foregroundColor(pathCopied ? brightGreen : darkNavy)
                            }
                            .buttonStyle(.plain)
                        }
                        .padding(.horizontal, 12)
                        .padding(.vertical, 8)
                        .background(
                            RoundedRectangle(cornerRadius: 10)
                                .fill(lightGray)
                        )
                        .overlay(
                            RoundedRectangle(cornerRadius: 10)
                                .stroke(darkNavy.opacity(0.12), lineWidth: 1)
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
        .padding(20)
        .background(
            RoundedRectangle(cornerRadius: contraRadius)
                .fill(warmOrange.opacity(0.05))
        )
        .overlay(
            RoundedRectangle(cornerRadius: contraRadius)
                .stroke(warmOrange, lineWidth: contraBorder)
        )
    }

    // Extension not found -- coming soon
    private var extensionNotFoundView: some View {
        VStack(spacing: 12) {
            Image(systemName: "clock.badge.checkmark")
                .font(.system(size: 22, weight: .medium))
                .foregroundColor(darkNavy.opacity(0.3))

            Text("Extension will be available on the Chrome Web Store soon.")
                .font(bodyFont)
                .foregroundColor(darkNavy.opacity(0.5))
                .multilineTextAlignment(.center)

            Text("For now, you can skip this step -AgentHandover works great without it.")
                .font(captionFont)
                .foregroundColor(darkNavy.opacity(0.35))
                .multilineTextAlignment(.center)
                .frame(maxWidth: 400)
        }
        .padding(24)
        .frame(maxWidth: .infinity)
        .background(
            RoundedRectangle(cornerRadius: contraRadius)
                .fill(lightGray)
        )
        .overlay(
            RoundedRectangle(cornerRadius: contraRadius)
                .stroke(darkNavy, lineWidth: contraBorder)
        )
    }

    private func stepCircle(number: Int) -> some View {
        Text("\(number)")
            .font(.system(size: 13, weight: .bold, design: .rounded))
            .foregroundColor(.white)
            .frame(width: 28, height: 28)
            .background(
                Circle().fill(darkNavy)
            )
    }

    // MARK: - Screen 9: Ready -- First Recording (WARM CREAM BACKGROUND)

    private var readyStep: some View {
        VStack(spacing: 24) {
            // Small mascot at top
            mascotImage(height: 64)

            Text("You\u{2019}re all set!")
                .font(.system(size: 32, weight: .black, design: .rounded))
                .foregroundColor(darkNavy)

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
                    ok: onboardingVLMReady
                )
            }

            // Instructions card
            VStack(spacing: 16) {
                Text("How to use AgentHandover")
                    .font(.system(size: 18, weight: .bold, design: .rounded))
                    .foregroundColor(darkNavy)

                VStack(alignment: .leading, spacing: 12) {
                    instructionRow(
                        icon: "record.circle",
                        iconColor: .red,
                        title: "Focus Session",
                        description: "Record one specific task. Click Stop when done."
                    )
                    instructionRow(
                        icon: "eye.fill",
                        iconColor: warmOrange,
                        title: "Observe Me",
                        description: "Learns patterns silently in the background over time."
                    )

                    Text("Always click Stop when you finish a focus session -- otherwise it keeps recording.")
                        .font(.system(size: 11))
                        .foregroundColor(warmOrange.opacity(0.8))
                        .padding(.top, 4)
                }
                .frame(maxWidth: 400, alignment: .leading)
            }
            .padding(24)
            .frame(maxWidth: .infinity)
            .background(
                RoundedRectangle(cornerRadius: contraRadius)
                    .fill(Color(nsColor: .controlBackgroundColor))
            )
            .overlay(
                RoundedRectangle(cornerRadius: contraRadius)
                    .stroke(Color.primary.opacity(0.2), lineWidth: contraBorder)
            )

            // Single CTA: close onboarding and show menu bar
            Button {
                onComplete?()
                DispatchQueue.main.asyncAfter(deadline: .now() + 0.3) {
                    NSApplication.shared.keyWindow?.close()
                }
            } label: {
                Text("Open Menu Bar")
                    .font(.system(size: 15, weight: .bold, design: .rounded))
                    .padding(.horizontal, 32)
                    .padding(.vertical, 13)
                    .background(
                        RoundedRectangle(cornerRadius: contraRadius)
                            .fill(darkNavy)
                    )
                    .foregroundColor(.white)
            }
            .buttonStyle(.plain)

            HStack(spacing: 4) {
                Image(systemName: "menubar.rectangle")
                    .font(.system(size: 10))
                    .foregroundColor(darkNavy.opacity(0.3))
                Text("You can start anytime from the menu bar icon")
                    .font(captionFont)
                    .foregroundColor(darkNavy.opacity(0.35))
            }
        }
    }

    private func readinessChip(icon: String, label: String, ok: Bool, optional: Bool = false) -> some View {
        HStack(spacing: 6) {
            Image(systemName: ok ? "checkmark.circle.fill" : (optional ? "minus.circle" : "xmark.circle.fill"))
                .font(.system(size: 14))
                .foregroundColor(ok ? brightGreen : (optional ? darkNavy.opacity(0.3) : warmOrange))
            Text(label)
                .font(.system(size: 13, weight: .bold, design: .rounded))
                .foregroundColor(ok ? darkNavy : darkNavy.opacity(0.4))
        }
        .padding(.horizontal, 14)
        .padding(.vertical, 8)
        .background(
            RoundedRectangle(cornerRadius: contraRadius)
                .fill(ok ? brightGreen.opacity(0.08) : lightGray)
        )
        .overlay(
            RoundedRectangle(cornerRadius: contraRadius)
                .stroke(ok ? brightGreen : darkNavy.opacity(0.12), lineWidth: contraBorder)
        )
    }

    private func instructionRow(icon: String, iconColor: Color, title: String, description: String) -> some View {
        HStack(alignment: .top, spacing: 12) {
            Image(systemName: icon)
                .font(.system(size: 16, weight: .semibold))
                .foregroundColor(iconColor)
                .frame(width: 24, height: 24)
            VStack(alignment: .leading, spacing: 2) {
                Text(title)
                    .font(.system(size: 14, weight: .bold, design: .rounded))
                    .foregroundColor(darkNavy)
                Text(description)
                    .font(.system(size: 12))
                    .foregroundColor(darkNavy.opacity(0.6))
                    .fixedSize(horizontal: false, vertical: true)
            }
        }
    }

    // MARK: - Actions

    private func startServicesOnly() {
        DispatchQueue.global(qos: .userInitiated).async {
            let ok = ServiceController.startAll()
            DispatchQueue.main.async {
                if ok {
                    onComplete?()
                    NSApplication.shared.keyWindow?.close()
                } else {
                    serviceStartFailed = true
                }
            }
        }
    }

    private func startServicesAndRecord() {
        let title = firstRecordingTitle.trimmingCharacters(in: .whitespacesAndNewlines)

        // Write focus signal BEFORE starting services (daemon picks it up on first loop)
        let sessionId = UUID().uuidString
        let signal: [String: Any] = [
            "session_id": sessionId,
            "title": title,
            "started_at": ISO8601DateFormatter().string(from: Date()),
            "status": "recording",
        ]
        writeFocusSignalFile(signal)

        // Start services on background thread to avoid freezing the UI
        DispatchQueue.global(qos: .userInitiated).async {
            let ok = ServiceController.startAll()
            DispatchQueue.main.async {
                if ok {
                    onComplete?()
                    NSApplication.shared.keyWindow?.close()
                } else {
                    serviceStartFailed = true
                }
            }
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
            // Silently fail -- the recording won't start but services are already running.
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
        findOllamaPath() != nil
    }

    private func findOllamaPath() -> String? {
        let paths = [
            "/usr/local/bin/ollama",
            "/opt/homebrew/bin/ollama",
            "/Applications/Ollama.app/Contents/Resources/ollama",
        ]
        if let knownPath = paths.first(where: { FileManager.default.fileExists(atPath: $0) }) {
            return knownPath
        }

        let whichProcess = Process()
        whichProcess.executableURL = URL(fileURLWithPath: "/usr/bin/which")
        whichProcess.arguments = ["ollama"]
        let outputPipe = Pipe()
        whichProcess.standardOutput = outputPipe

        do {
            try whichProcess.run()
            whichProcess.waitUntilExit()
            guard whichProcess.terminationStatus == 0 else { return nil }
            let output = String(data: outputPipe.fileHandleForReading.readDataToEndOfFile(), encoding: .utf8)?
                .trimmingCharacters(in: .whitespacesAndNewlines)
            return (output?.isEmpty == false) ? output : nil
        } catch {
            return nil
        }
    }

    private func modelRow(_ name: String, _ size: String, _ description: String) -> some View {
        HStack(alignment: .top, spacing: 8) {
            Text("\u{2022}")
                .foregroundColor(warmOrange)
                .font(.system(size: 12, weight: .bold))
            VStack(alignment: .leading, spacing: 2) {
                HStack(spacing: 6) {
                    Text(name)
                        .font(.system(size: 12, weight: .bold, design: .monospaced))
                        .foregroundColor(darkNavy)
                    Text(size)
                        .font(.system(size: 11, weight: .medium))
                        .foregroundColor(darkNavy.opacity(0.4))
                }
                Text(description)
                    .font(.system(size: 11))
                    .foregroundColor(darkNavy.opacity(0.5))
            }
        }
    }

    private func pullOllamaModel() {
        guard let ollamaPath = findOllamaPath() else { return }

        vlmPullInProgress = true
        vlmPullOutput = "Starting download..."

        let annModel = useCustomModels ? customAnnotationModel : "qwen3.5:2b"
        let sopModel = useCustomModels ? customSOPModel : "qwen3.5:4b"
        let models = [
            (annModel, "scene annotation"),
            (sopModel, "Skill generation"),
            ("nomic-embed-text", "semantic search"),
        ]

        DispatchQueue.global(qos: .userInitiated).async {
            // Check which models are already pulled
            let listProcess = Process()
            listProcess.executableURL = URL(fileURLWithPath: ollamaPath)
            listProcess.arguments = ["list"]
            let listPipe = Pipe()
            listProcess.standardOutput = listPipe
            try? listProcess.run()
            listProcess.waitUntilExit()
            let installedModels = String(data: listPipe.fileHandleForReading.readDataToEndOfFile(), encoding: .utf8) ?? ""

            for (index, (model, purpose)) in models.enumerated() {
                // Skip if already installed
                if installedModels.contains(model) {
                    DispatchQueue.main.async {
                        vlmPullOutput = "[\(index + 1)/\(models.count)] \(model) already installed"
                    }
                    continue
                }

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

            // Save embedding config + custom models
            let imageEmbEnabled = self.enableImageEmbeddings
            self.saveEmbeddingConfig(imageEmbeddings: imageEmbEnabled)
            if self.useCustomModels {
                self.saveCustomModelConfig(annotation: annModel, sop: sopModel)
            }

            DispatchQueue.main.async {
                vlmPullInProgress = false
                vlmPullOutput = "All models downloaded successfully!"
                refreshVLMSetupState(force: true)
            }
        }
    }

    private func seedVLMStateFromConfigIfNeeded() {
        guard !hasSeededVLMFromConfig else { return }
        hasSeededVLMFromConfig = true
        appState.refreshStatus()
        if appState.vlmMode == "remote" {
            vlmMode = .cloud
            remoteConsentGiven = true
            remoteConfigSaved = appState.vlmProvider != nil || appState.vlmAvailable
        } else {
            vlmMode = .local
        }
        if let providerRaw = appState.vlmProvider, let provider = RemoteProvider(rawValue: providerRaw) {
            selectedProvider = provider
        }
    }

    private func refreshVLMSetupState(force: Bool = false) {
        guard force || currentStep == 6 else { return }
        let requiredModels = requiredLocalModels.map(canonicalModelName)
        vlmStateRefreshing = true
        appState.refreshStatus()
        remoteConfigSaved = appState.vlmMode == "remote" && (appState.vlmProvider != nil || appState.vlmAvailable)

        DispatchQueue.global(qos: .userInitiated).async {
            let ollamaPath = findOllamaPath()
            let installed = ollamaPath != nil
            var running = false
            var hasRequiredModels = false
            var statusMessage = ""

            if let ollamaPath {
                switch queryOllamaModels(at: ollamaPath) {
                case .success(let installedModels):
                    running = true
                    let missingModels = requiredModels.filter { !installedModels.contains($0) }
                    hasRequiredModels = missingModels.isEmpty
                    if missingModels.isEmpty {
                        statusMessage = "Local setup looks good."
                    } else if installedModels.isEmpty {
                        statusMessage = "Ollama is running, but no models are downloaded yet."
                    } else {
                        statusMessage = "Missing: " + missingModels.joined(separator: ", ")
                    }
                case .failure:
                    statusMessage = hasOllamaAppBundle
                        ? "Open Ollama once to start the local AI service, then recheck."
                        : "Run `ollama serve` in Terminal, then recheck."
                }
            } else {
                statusMessage = "Download and install Ollama to run models locally."
            }

            Task { @MainActor in
                self.ollamaInstalled = installed
                self.ollamaRunning = running
                self.ollamaHasRequiredModels = hasRequiredModels
                self.vlmStatusMessage = statusMessage
                self.vlmStateRefreshing = false
                self.appState.vlmAvailable = self.onboardingVLMReady
            }
        }
    }

    private func queryOllamaModels(at ollamaPath: String) -> Result<Set<String>, Error> {
        let process = Process()
        process.executableURL = URL(fileURLWithPath: ollamaPath)
        process.arguments = ["list"]
        let outputPipe = Pipe()
        process.standardOutput = outputPipe
        process.standardError = outputPipe

        do {
            try process.run()
            process.waitUntilExit()
            guard process.terminationStatus == 0 else {
                return .failure(NSError(domain: "OllamaList", code: Int(process.terminationStatus)))
            }

            let output = String(data: outputPipe.fileHandleForReading.readDataToEndOfFile(), encoding: .utf8) ?? ""
            let modelNames = Set(
                output
                    .components(separatedBy: "\n")
                    .dropFirst()
                    .compactMap { line -> String? in
                        let trimmed = line.trimmingCharacters(in: .whitespacesAndNewlines)
                        guard !trimmed.isEmpty else { return nil }
                        guard let rawName = trimmed.components(separatedBy: .whitespaces).first else { return nil }
                        return canonicalModelName(rawName)
                    }
            )
            return .success(modelNames)
        } catch {
            return .failure(error)
        }
    }

    private func canonicalModelName(_ name: String) -> String {
        let trimmed = name.trimmingCharacters(in: .whitespacesAndNewlines)
        if trimmed.hasSuffix(":latest") {
            return String(trimmed.dropLast(":latest".count))
        }
        return trimmed
    }

    private func launchOllamaApp() {
        let appURL = URL(fileURLWithPath: "/Applications/Ollama.app")
        guard FileManager.default.fileExists(atPath: appURL.path) else { return }

        let config = NSWorkspace.OpenConfiguration()
        config.activates = true
        NSWorkspace.shared.openApplication(at: appURL, configuration: config) { _, _ in
            DispatchQueue.main.asyncAfter(deadline: .now() + 1.0) {
                refreshVLMSetupState(force: true)
            }
        }
    }

    private func saveEmbeddingConfig(imageEmbeddings: Bool) {
        let home = FileManager.default.homeDirectoryForCurrentUser
        let configDir = home
            .appendingPathComponent("Library/Application Support/agenthandover")
        let configPath = configDir.appendingPathComponent("config.toml")

        try? FileManager.default.createDirectory(at: configDir, withIntermediateDirectories: true)

        var content = (try? String(contentsOf: configPath, encoding: .utf8)) ?? ""

        let embeddingSection = """

        [embedding]
        model = "nomic-embed-text"
        image_embeddings = \(imageEmbeddings)

        """

        // Remove existing [embedding] section if present
        if let range = content.range(of: "[embedding]") {
            let afterSection = content[range.lowerBound...]
            if let nextSection = afterSection.range(of: "\n[", range: content.index(after: range.lowerBound)..<content.endIndex) {
                content.removeSubrange(range.lowerBound..<nextSection.lowerBound)
            } else {
                content.removeSubrange(range.lowerBound..<content.endIndex)
            }
        }

        content += embeddingSection
        try? content.write(to: configPath, atomically: true, encoding: .utf8)
    }

    private func saveCustomModelConfig(annotation: String, sop: String) {
        let home = FileManager.default.homeDirectoryForCurrentUser
        let configDir = home
            .appendingPathComponent("Library/Application Support/agenthandover")
        let configPath = configDir.appendingPathComponent("config.toml")

        try? FileManager.default.createDirectory(at: configDir, withIntermediateDirectories: true)

        var content = (try? String(contentsOf: configPath, encoding: .utf8)) ?? ""

        // Update or add annotation_model and sop_model in [vlm] section
        let vlmFields = "annotation_model = \"\(annotation)\"\nsop_model = \"\(sop)\"\n"

        if content.contains("[vlm]") {
            // Remove existing model keys
            for key in ["annotation_model", "sop_model"] {
                let pattern = "(?m)^[ \\t]*\(key)[ \\t]*=[ \\t]*\"[^\"]*\"[ \\t]*\\n?"
                if let regex = try? NSRegularExpression(pattern: pattern) {
                    let range = NSRange(content.startIndex..., in: content)
                    content = regex.stringByReplacingMatches(in: content, range: range, withTemplate: "")
                }
            }
            // Insert after [vlm]
            if let vlmRange = content.range(of: "[vlm]") {
                let insertIdx = content.index(vlmRange.upperBound, offsetBy: 1, limitedBy: content.endIndex) ?? vlmRange.upperBound
                content.insert(contentsOf: "\n" + vlmFields, at: insertIdx)
            }
        } else {
            content += "\n[vlm]\n" + vlmFields
        }

        try? content.write(to: configPath, atomically: true, encoding: .utf8)
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

private final class OnboardingViewHost {}

struct PermissionStatusBadge: View {
    let granted: Bool
    let grantedLabel: String
    let deniedLabel: String

    private let darkNavy = Color(red: 0.09, green: 0.10, blue: 0.12)
    private let warmOrange = Color(red: 0.92, green: 0.57, blue: 0.20)
    private let brightGreen = Color(red: 0.18, green: 0.80, blue: 0.34)
    private let contraRadius: CGFloat = 16
    private let contraBorder: CGFloat = 2

    var body: some View {
        HStack(spacing: 6) {
            Image(systemName: granted ? "checkmark.circle.fill" : "xmark.circle.fill")
                .foregroundColor(granted ? brightGreen : warmOrange)
            Text(granted ? grantedLabel : deniedLabel)
                .font(.system(size: 13, weight: .bold, design: .rounded))
                .foregroundColor(granted ? brightGreen : warmOrange)
        }
        .padding(.horizontal, 16)
        .padding(.vertical, 9)
        .background(
            RoundedRectangle(cornerRadius: contraRadius)
                .fill((granted ? brightGreen : warmOrange).opacity(0.1))
        )
        .overlay(
            RoundedRectangle(cornerRadius: contraRadius)
                .stroke((granted ? brightGreen : warmOrange), lineWidth: contraBorder)
        )
    }
}
