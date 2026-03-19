// swift-tools-version: 5.9
import PackageDescription

let package = Package(
    name: "AgentHandoverApp",
    platforms: [
        .macOS(.v13)
    ],
    targets: [
        .executableTarget(
            name: "AgentHandoverApp",
            path: "Sources/AgentHandoverApp",
            exclude: ["Info.plist"],
            resources: [
                .process("Resources/Assets.xcassets"),
                .process("Resources/mascot.png"),
            ]
        ),
        .testTarget(
            name: "AgentHandoverAppTests",
            dependencies: ["AgentHandoverApp"],
            path: "Tests/AgentHandoverAppTests"
        ),
    ]
)
