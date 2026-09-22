// swift-tools-version: 5.9
import PackageDescription

let package = Package(
    name: "AgentHandoverApp",
    platforms: [
        .macOS(.v14)
    ],
    dependencies: [
        // 2.9.x only: 2.10 is in beta.
        .package(url: "https://github.com/sparkle-project/Sparkle", .upToNextMinor(from: "2.9.6")),
    ],
    targets: [
        .executableTarget(
            name: "AgentHandoverApp",
            dependencies: [
                .product(name: "Sparkle", package: "Sparkle"),
            ],
            path: "Sources/AgentHandoverApp",
            exclude: ["Info.plist"],
            resources: [
                .process("Resources/Assets.xcassets"),
                .process("Resources/mascot.png"),
                .copy("Resources/AppIcon.icns"),
            ]
        ),
        .testTarget(
            name: "AgentHandoverAppTests",
            dependencies: ["AgentHandoverApp"],
            path: "Tests/AgentHandoverAppTests"
        ),
    ]
)
