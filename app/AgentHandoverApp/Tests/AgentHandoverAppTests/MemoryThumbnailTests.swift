import CryptoKit
import XCTest
@testable import AgentHandoverApp

/// 2.0 Memory thumbnails: the app's key (created once, Keychain attributes),
/// opening what the daemon sealed, and the capture-socket key handshake that
/// only the allowed daemon passes. No test touches the real Keychain.
final class MemoryThumbnailTests: XCTestCase {

    /// An in-memory stand-in for the Keychain item.
    private final class FakeStore {
        var item: Data?
        var adds = 0
        var readFailure: OSStatus?

        var store: MemoryThumbnailKey.Store {
            MemoryThumbnailKey.Store(
                read: { [unowned self] in
                    if let readFailure { return .failed(readFailure) }
                    return item.map { .found($0) } ?? .notFound
                },
                add: { [unowned self] data in
                    if item != nil { return errSecDuplicateItem }
                    adds += 1
                    item = data
                    return errSecSuccess
                }
            )
        }
    }

    private func bytes(_ key: SymmetricKey) -> Data { key.withUnsafeBytes { Data($0) } }

    // MARK: - Key

    func testKeyIsCreatedOnceAndThenReused() throws {
        let fake = FakeStore()
        let first = try XCTUnwrap(MemoryThumbnailKey.loadOrCreate(store: fake.store))
        let second = try XCTUnwrap(MemoryThumbnailKey.loadOrCreate(store: fake.store))
        XCTAssertEqual(fake.adds, 1, "one key, created on first use")
        XCTAssertEqual(first.bitCount, 256)
        XCTAssertEqual(bytes(first), bytes(second))
        XCTAssertEqual(fake.item, bytes(first), "the stored item is the key")
    }

    func testKeychainFailureOrForeignItemGivesNoKeyAndOverwritesNothing() {
        let locked = FakeStore()
        locked.readFailure = errSecInteractionNotAllowed
        XCTAssertNil(MemoryThumbnailKey.loadOrCreate(store: locked.store))
        XCTAssertEqual(locked.adds, 0, "a failed read never creates a second key")

        let foreign = FakeStore()
        foreign.item = Data(repeating: 1, count: 16)
        XCTAssertNil(MemoryThumbnailKey.loadOrCreate(store: foreign.store))
        XCTAssertEqual(foreign.item, Data(repeating: 1, count: 16), "left alone")
    }

    func testCacheRetriesAFailureButHoldsAKey() {
        var loads = 0
        var result: SymmetricKey?
        let cache = MemoryThumbnailKey.Cache(load: { loads += 1; return result })
        XCTAssertNil(cache.get())
        result = SymmetricKey(size: .bits256)
        XCTAssertNotNil(cache.get())
        XCTAssertNotNil(cache.get())
        XCTAssertEqual(loads, 2, "a failure is retried, a key is held")
    }

    func testKeychainItemIsThisDeviceOnlyUnderTheAppService() {
        let query = MemoryThumbnailKey.addQuery(Data(count: 32))
        XCTAssertEqual(query[kSecAttrService as String] as? String, "com.agenthandover.app")
        XCTAssertEqual(query[kSecAttrAccount as String] as? String, "memory-thumbnail-key")
        XCTAssertEqual(query[kSecAttrAccessible as String] as? String,
                       kSecAttrAccessibleAfterFirstUnlockThisDeviceOnly as String)
        XCTAssertEqual(query[kSecAttrSynchronizable as String] as? Bool, false)
        XCTAssertEqual(query[kSecClass as String] as? String, kSecClassGenericPassword as String)
    }

    // MARK: - Decrypt

    private func hex(_ string: String) -> Data {
        var data = Data()
        var index = string.startIndex
        while index < string.endIndex {
            let next = string.index(index, offsetBy: 2)
            data.append(UInt8(string[index..<next], radix: 16)!)
            index = next
        }
        return data
    }

    /// The vector the daemon's Rust test pins
    /// (`thumbnail::tests::known_vector_is_the_cryptokit_combined_layout`).
    func testOpensWhatTheDaemonSealed() throws {
        let sealed = hex("0102030405060708090a0b0c"
                         + "b5047c38fcde0aa4646a5dc99f77c00dd3b66d306cc023"
                         + "f4defdab4c7c39b4f2bfe1d324a2caf6")
        let key = SymmetricKey(data: Data(repeating: 42, count: 32))
        let plain = try XCTUnwrap(MemoryThumbnail.open(sealed, key: key))
        XCTAssertEqual(String(decoding: plain, as: UTF8.self), "AgentHandover thumbnail")
        XCTAssertNil(MemoryThumbnail.open(sealed, key: SymmetricKey(data: Data(repeating: 7, count: 32))))
        var altered = sealed
        altered[altered.count - 1] ^= 1
        XCTAssertNil(MemoryThumbnail.open(altered, key: key))
    }

    func testDecryptsAThumbnailFileForDisplay() throws {
        let context = CGContext(
            data: nil, width: 120, height: 80, bitsPerComponent: 8, bytesPerRow: 0,
            space: CGColorSpace(name: CGColorSpace.sRGB)!,
            bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)!
        context.setFillColor(CGColor(red: 0.2, green: 0.4, blue: 0.8, alpha: 1))
        context.fill(CGRect(x: 0, y: 0, width: 120, height: 80))
        let jpeg = try XCTUnwrap(
            ScreenCaptureService.encodeJPEG(context.makeImage()!, scale: 1, quality: 60)).jpeg
        let key = SymmetricKey(size: .bits256)
        let sealed = try XCTUnwrap(try AES.GCM.seal(jpeg, using: key).combined)

        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("ahthumbs-\(UUID().uuidString.prefix(8))")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: dir) }
        let url = MemoryThumbnail.fileURL(eventID: "e1", in: dir)
        XCTAssertEqual(url.lastPathComponent, "e1.ahthumb")
        try sealed.write(to: url)

        let image = try XCTUnwrap(MemoryThumbnail.decrypt(url, key: key))
        XCTAssertEqual(image.size, NSSize(width: 120, height: 80))
        XCTAssertNil(MemoryThumbnail.decrypt(url, key: SymmetricKey(size: .bits256)), "wrong key")
        XCTAssertNil(MemoryThumbnail.decrypt(MemoryThumbnail.fileURL(eventID: "gone", in: dir), key: key),
                     "an expired or forgotten thumbnail")
        XCTAssertNil(MemoryThumbnail.decrypt(url, key: nil), "no key")
    }

    // MARK: - Key handshake on capture.sock

    private func temporarySocketPath() -> String {
        let dir = (NSTemporaryDirectory() as NSString)
            .appendingPathComponent("ahkey-\(UUID().uuidString.prefix(8))")
        return (dir as NSString).appendingPathComponent("capture.sock")
    }

    /// Connect, send `request`, read until the server closes. Throws when
    /// the connection itself fails.
    private func roundTrip(_ path: String, request: String) throws -> Data {
        let fd = socket(AF_UNIX, SOCK_STREAM, 0)
        defer { close(fd) }
        var addr = sockaddr_un()
        addr.sun_family = sa_family_t(AF_UNIX)
        withUnsafeMutablePointer(to: &addr.sun_path) { ptr in
            ptr.withMemoryRebound(to: CChar.self, capacity: 104) { dest in
                for (i, byte) in path.utf8CString.enumerated() { dest[i] = byte }
            }
        }
        let connected = withUnsafePointer(to: &addr) { ptr in
            ptr.withMemoryRebound(to: sockaddr.self, capacity: 1) {
                connect(fd, $0, socklen_t(MemoryLayout<sockaddr_un>.size))
            }
        }
        guard connected == 0 else { throw NSError(domain: NSPOSIXErrorDomain, code: Int(errno)) }
        _ = request.withCString { send(fd, $0, strlen($0), 0) }
        var response = Data()
        var buffer = [UInt8](repeating: 0, count: 4096)
        while true {
            let n = recv(fd, &buffer, buffer.count, 0)
            if n <= 0 { break }
            response.append(contentsOf: buffer[0..<n])
        }
        return response
    }

    private let keyRequest = #"{"command":"thumbnail_key"}"# + "\n"

    private func server(policy: PeerPolicy, key: SymmetricKey?) -> (ScreenCaptureServer, String) {
        let path = temporarySocketPath()
        let server = ScreenCaptureServer(socketPath: path, policy: policy, thumbnailKey: { key }) { _ in
            .failure(.timeout)
        }
        server.start()
        return (server, path)
    }

    func testAllowedPeerReceivesTheKey() throws {
        let key = SymmetricKey(size: .bits256)
        let me = PeerPolicy(allowedPaths: [PeerPolicy.executablePath(of: getpid())!], identifiers: [], teamID: nil)
        let (server, path) = server(policy: me, key: key)
        defer { server.stop() }
        let reply = try roundTrip(path, request: keyRequest)
        XCTAssertEqual(reply.count, 4 + 32)
        XCTAssertEqual(Array(reply.prefix(4)), [32, 0, 0, 0], "length u32 LE")
        XCTAssertEqual(reply.suffix(32), bytes(key))
    }

    func testKeyUnavailableIsAnError() throws {
        let me = PeerPolicy(allowedPaths: [PeerPolicy.executablePath(of: getpid())!], identifiers: [], teamID: nil)
        let (server, path) = server(policy: me, key: nil)
        defer { server.stop() }
        let reply = try roundTrip(path, request: keyRequest)
        let json = try JSONSerialization.jsonObject(with: reply) as? [String: String]
        XCTAssertEqual(json?["error"], "thumbnail key unavailable")
    }

    func testRefusedPeersNeverReceiveTheKey() throws {
        let key = SymmetricKey(size: .bits256)
        // Another executable path (the installed daemon's): this process is refused.
        let (daemonOnly, path1) = server(
            policy: PeerPolicy(allowedPaths: [PeerPolicy.daemonPath],
                               identifiers: [PeerPolicy.daemonIdentifier], teamID: nil),
            key: key)
        defer { daemonOnly.stop() }
        XCTAssertTrue(try roundTrip(path1, request: keyRequest).isEmpty)

        // The right path, but an unsigned client against a pinned Team ID:
        // refused by the code-signature check.
        let (signedOnly, path2) = server(
            policy: PeerPolicy(allowedPaths: [PeerPolicy.executablePath(of: getpid())!],
                               identifiers: [PeerPolicy.daemonIdentifier], teamID: "ABCDE12345"),
            key: key)
        defer { signedOnly.stop() }
        XCTAssertTrue(try roundTrip(path2, request: keyRequest).isEmpty,
                      "an unsigned client gets no key")
    }
}
