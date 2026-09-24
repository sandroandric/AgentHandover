import Foundation
import Security

/// The cloud API keys earlier versions stored in the Keychain.
///
/// Up to 0.4.x onboarding offered a "Cloud" model mode and kept the
/// provider's API key as a generic password (service
/// `com.agenthandover.app`, account `agenthandover-<provider>-key`).
/// Cloud mode is gone in 0.5.0, so nothing reads those keys any more; the
/// first launch after the upgrade deletes them, once, and logs which.
enum LegacyAPIKeys {
    static let service = "com.agenthandover.app"
    static let accounts = ["openai", "anthropic", "google"].map { "agenthandover-\($0)-key" }
    /// Set once the removal ran, so later launches do not touch the Keychain.
    static let doneKey = "legacyAPIKeysRemoved"

    /// Delete the stored keys (once per install).  Returns the accounts
    /// that were deleted; `delete` is `SecItemDelete` for one account
    /// except in tests.
    @discardableResult
    static func removeOnce(
        defaults: UserDefaults = .standard,
        delete: (String) -> OSStatus = deleteItem
    ) -> [String] {
        guard !defaults.bool(forKey: doneKey) else { return [] }
        var removed: [String] = []
        var failed: [String] = []
        for account in accounts {
            switch delete(account) {
            case errSecSuccess: removed.append(account)
            case errSecItemNotFound: break
            case let status: failed.append("\(account) (OSStatus \(status))")
            }
        }
        if !removed.isEmpty {
            NSLog(
                "AgentHandover: cloud mode was removed in 0.5.0; deleted its stored API key(s) from the Keychain: %@",
                removed.joined(separator: ", ")
            )
        }
        if failed.isEmpty {
            defaults.set(true, forKey: doneKey)
        } else {
            // Tried again at the next launch.
            NSLog(
                "AgentHandover: could not delete the old cloud API key(s) from the Keychain: %@",
                failed.joined(separator: ", ")
            )
        }
        return removed
    }

    static func deleteItem(account: String) -> OSStatus {
        let query: [String: Any] = [
            kSecClass as String: kSecClassGenericPassword,
            kSecAttrService as String: service,
            kSecAttrAccount as String: account,
        ]
        return SecItemDelete(query as CFDictionary)
    }
}
