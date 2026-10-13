import AppKit
import CryptoKit
import Foundation
import Security

/// The key that seals Memory thumbnails (2.0).
///
/// Created once, on first use, as a 256-bit random key and kept as a
/// Keychain generic password (service `com.agenthandover.app`, account
/// `memory-thumbnail-key`, this device only). The app hands it to the
/// daemon over the capture socket (`{"command":"thumbnail_key"}`, peers
/// `PeerPolicy` accepts only); the daemon seals each kept frame's thumbnail
/// with it and never stores it. The app opens the thumbnails for display.
enum MemoryThumbnailKey {
    static let service = "com.agenthandover.app"
    static let account = "memory-thumbnail-key"
    static let byteCount = 32

    /// Where the key lives: the Keychain in the app, memory in tests.
    struct Store {
        enum Read {
            case found(Data)
            case notFound
            case failed(OSStatus)
        }
        var read: () -> Read
        /// Add the item; `errSecDuplicateItem` when another caller won.
        var add: (Data) -> OSStatus
    }

    /// The stored key, creating and storing it the first time. nil (logged)
    /// when the Keychain cannot be read or written, or holds something that
    /// is not a 256-bit key (left alone: thumbnails sealed with it would be
    /// lost).
    static func loadOrCreate(store: Store = .keychain) -> SymmetricKey? {
        switch store.read() {
        case .failed(let status):
            NSLog("MemoryThumbnailKey: Keychain read failed (OSStatus %d)", status)
            return nil
        case .found(let data):
            return key(from: data)
        case .notFound:
            break
        }
        let fresh = SymmetricKey(size: .bits256)
        let data = fresh.withUnsafeBytes { Data($0) }
        switch store.add(data) {
        case errSecSuccess:
            NSLog("MemoryThumbnailKey: created the Memory thumbnail key")
            return fresh
        case errSecDuplicateItem:
            // Created concurrently: use the stored one.
            if case .found(let stored) = store.read() { return key(from: stored) }
            return nil
        case let status:
            NSLog("MemoryThumbnailKey: Keychain add failed (OSStatus %d)", status)
            return nil
        }
    }

    private static func key(from data: Data) -> SymmetricKey? {
        guard data.count == byteCount else {
            NSLog("MemoryThumbnailKey: stored item is %d bytes, not a 256-bit key", data.count)
            return nil
        }
        return SymmetricKey(data: data)
    }

    // MARK: - Keychain

    /// The attributes of the stored item.
    static func addQuery(_ data: Data) -> [String: Any] {
        [
            kSecClass as String: kSecClassGenericPassword,
            kSecAttrService as String: service,
            kSecAttrAccount as String: account,
            kSecAttrAccessible as String: kSecAttrAccessibleAfterFirstUnlockThisDeviceOnly,
            kSecAttrSynchronizable as String: false,
            kSecValueData as String: data,
        ]
    }

    static func readQuery() -> [String: Any] {
        [
            kSecClass as String: kSecClassGenericPassword,
            kSecAttrService as String: service,
            kSecAttrAccount as String: account,
            kSecReturnData as String: true,
            kSecMatchLimit as String: kSecMatchLimitOne,
        ]
    }

    /// The key for this process, read (or created) once and then held in
    /// memory; a failure is not remembered, so the next request tries again.
    static func appKey() -> SymmetricKey? { cache.get() }

    private static let cache = Cache(load: { loadOrCreate() })

    final class Cache: @unchecked Sendable {
        private let lock = NSLock()
        private let load: () -> SymmetricKey?
        private var key: SymmetricKey?

        init(load: @escaping () -> SymmetricKey?) { self.load = load }

        func get() -> SymmetricKey? {
            lock.lock(); defer { lock.unlock() }
            if key == nil { key = load() }
            return key
        }
    }
}

extension MemoryThumbnailKey.Store {
    static let keychain = MemoryThumbnailKey.Store(
        read: {
            var result: CFTypeRef?
            let status = SecItemCopyMatching(MemoryThumbnailKey.readQuery() as CFDictionary, &result)
            switch status {
            case errSecSuccess:
                guard let data = result as? Data else { return .failed(errSecDecode) }
                return .found(data)
            case errSecItemNotFound: return .notFound
            default: return .failed(status)
            }
        },
        add: { SecItemAdd(MemoryThumbnailKey.addQuery($0) as CFDictionary, nil) }
    )
}

/// Encrypted Memory thumbnails: `<data dir>/thumbs/<event_id>.ahthumb`,
/// written by the daemon as AES-256-GCM `nonce (12) || ciphertext || tag (16)`
/// of a ~480 px JPEG (CryptoKit's combined layout).
enum MemoryThumbnail {
    static let fileExtension = "ahthumb"

    static var directory: URL {
        FileManager.default.homeDirectoryForCurrentUser
            .appendingPathComponent("Library/Application Support/agenthandover/thumbs")
    }

    static func fileURL(eventID: String, in directory: URL = MemoryThumbnail.directory) -> URL {
        directory.appendingPathComponent("\(eventID).\(fileExtension)")
    }

    /// The thumbnail JPEG inside `sealed`, or nil when the key is wrong or
    /// a byte was altered.
    static func open(_ sealed: Data, key: SymmetricKey) -> Data? {
        guard let box = try? AES.GCM.SealedBox(combined: sealed) else { return nil }
        return try? AES.GCM.open(box, using: key)
    }

    /// The thumbnail at `fileURL` for display, or nil when it is missing
    /// (expired, forgotten), cannot be opened with the key, or is not an image.
    static func decrypt(_ fileURL: URL, key: SymmetricKey? = MemoryThumbnailKey.appKey()) -> NSImage? {
        guard let key, let sealed = try? Data(contentsOf: fileURL),
              let jpeg = open(sealed, key: key) else { return nil }
        return NSImage(data: jpeg)
    }
}
