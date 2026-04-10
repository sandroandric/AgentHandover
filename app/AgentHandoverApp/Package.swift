// swift-tools-version: 5.9
import PackageDescription

let package = Package(
    name: "AgentHandoverApp",
    platforms: [
        .macOS(.v13)
    ],
    dependencies: [
        .package(url: "https://github.com/sparkle-project/Sparkle", from: "2.6.0"),
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
