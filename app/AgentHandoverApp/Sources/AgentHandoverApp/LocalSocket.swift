import Foundation
import Security

/// The directory that holds the Unix sockets shared by the app and the
/// daemon (`capture.sock`, `observation.sock`, `control.sock`).
///
/// It is created mode 0700 and must be owned by the current user, so no
/// other account can even reach the sockets. Same-user processes still can;
/// `PeerPolicy` narrows that down to AgentHandover's own binaries.
enum SocketDirectory {
    static var path: String {
        FileManager.default.homeDirectoryForCurrentUser
            .appendingPathComponent("Library/Application Support/agenthandover/sockets").path
    }

    static func socketPath(_ name: String) -> String {
        (path as NSString).appendingPathComponent(name)
    }

    /// Create `dir` (0700) or tighten an existing one to 0700. Refuses a
    /// symlink, a non-directory, or a directory owned by someone else.
    static func preparePrivate(_ dir: String) -> Bool {
        let parent = (dir as NSString).deletingLastPathComponent
        try? FileManager.default.createDirectory(
            atPath: parent, withIntermediateDirectories: true, attributes: nil)
        if mkdir(dir, 0o700) != 0 && errno != EEXIST {
            NSLog("SocketDirectory: mkdir %@ failed: errno=%d", dir, errno)
            return false
        }
        var info = stat()
        guard lstat(dir, &info) == 0,
              info.st_mode & S_IFMT == S_IFDIR,
              info.st_uid == getuid() else {
            NSLog("SocketDirectory: %@ is not a directory owned by this user; refusing it", dir)
            return false
        }
        if info.st_mode & 0o777 != 0o700 && chmod(dir, 0o700) != 0 {
            NSLog("SocketDirectory: chmod 0700 %@ failed: errno=%d", dir, errno)
            return false
        }
        return true
    }
}

/// Who may talk to one of the app's sockets.
///
/// Checked on every accepted connection, in order:
/// 1. the peer runs as this user (`getpeereid`);
/// 2. its executable (`LOCAL_PEERPID` + `proc_pidpath`) is one of
///    `allowedPaths` (a cheap pre-check);
/// 3. when this app is signed with a Team ID, the peer's code signature
///    (looked up by its audit token, so a recycled PID cannot pass) satisfies
///    `anchor apple generic and certificate leaf[subject.OU] = <team> and
///    identifier <one of identifiers>`. An unsigned development build has
///    no team to pin and relies on 1 and 2.
struct PeerPolicy {
    let allowedPaths: Set<String>
    let identifiers: [String]
    let teamID: String?

    /// The daemon (`ah-observer`, signed as `com.agenthandover.observer`).
    static let daemonPath = "/usr/local/lib/agenthandover/ah-observer"
    static let daemonIdentifier = "com.agenthandover.observer"
    /// The CLI (`agenthandover doctor` probes observation.sock).
    static let cliPath = "/usr/local/bin/agenthandover"
    static let cliIdentifier = "agenthandover"

    static func production(paths: Set<String>, identifiers: [String]) -> PeerPolicy {
        PeerPolicy(allowedPaths: paths, identifiers: identifiers, teamID: ownTeamID())
    }

    /// The signing requirement a peer must satisfy, or nil when unsigned.
    var requirement: String? {
        guard let teamID, !identifiers.isEmpty else { return nil }
        let ids = identifiers.map { "identifier \"\($0)\"" }.joined(separator: " or ")
        return "anchor apple generic and certificate leaf[subject.OU] = \"\(teamID)\" and (\(ids))"
    }

    /// nil when the peer on `fd` is allowed, else the reason it is not.
    func refusal(for fd: Int32) -> String? {
        var uid = uid_t(0), gid = gid_t(0)
        guard getpeereid(fd, &uid, &gid) == 0 else { return "no peer credentials" }
        guard uid == getuid() else { return "peer runs as uid \(uid)" }

        var pid = pid_t(0)
        var pidLen = socklen_t(MemoryLayout<pid_t>.size)
        guard getsockopt(fd, SOL_LOCAL, LOCAL_PEERPID, &pid, &pidLen) == 0 else {
            return "no peer pid"
        }
        guard let path = Self.executablePath(of: pid) else { return "peer \(pid) has no path" }
        guard allowedPaths.contains(path) else { return "peer executable \(path) not allowed" }

        guard let requirement else { return nil }
        var token = audit_token_t()
        var tokenLen = socklen_t(MemoryLayout<audit_token_t>.size)
        guard getsockopt(fd, SOL_LOCAL, LOCAL_PEERTOKEN, &token, &tokenLen) == 0 else {
            return "no peer audit token"
        }
        if Self.validated.contains(token, requirement: requirement) { return nil }
        let status = Self.check(auditToken: token, requirement: requirement)
        guard status == errSecSuccess else {
            return "peer \(path) fails the code requirement (\(status))"
        }
        Self.validated.insert(token, requirement: requirement)
        return nil
    }

    /// Peer processes whose signature already passed. The check hashes the
    /// peer's code, and the daemon connects many times a minute: re-checking
    /// each connection was most of the idle app's CPU on the test Mac. A
    /// process is keyed by its audit token's PID and PID version (`exec`
    /// and PID reuse both change the version), so a new or re-executed
    /// process is checked again; refusals are never remembered.
    static let validated = ValidatedPeers()

    final class ValidatedPeers {
        private let lock = NSLock()
        private var keys: [String] = []
        private static let capacity = 32

        private static func key(_ token: audit_token_t, _ requirement: String) -> String {
            // audit_token_t: val[5] is the PID, val[7] its version.
            "\(token.val.5):\(token.val.7):\(requirement)"
        }

        func contains(_ token: audit_token_t, requirement: String) -> Bool {
            let key = Self.key(token, requirement)
            lock.lock(); defer { lock.unlock() }
            return keys.contains(key)
        }

        func insert(_ token: audit_token_t, requirement: String) {
            let key = Self.key(token, requirement)
            lock.lock(); defer { lock.unlock() }
            guard !keys.contains(key) else { return }
            keys.append(key)
            if keys.count > Self.capacity { keys.removeFirst(keys.count - Self.capacity) }
        }
    }

    static func executablePath(of pid: pid_t) -> String? {
        var buffer = [CChar](repeating: 0, count: Int(MAXPATHLEN))
        guard proc_pidpath(pid, &buffer, UInt32(buffer.count)) > 0 else { return nil }
        return String(cString: buffer)
    }

    /// `errSecSuccess` when the process behind `auditToken` is validly
    /// signed and satisfies `requirement`.
    static func check(auditToken: audit_token_t, requirement: String) -> OSStatus {
        var token = auditToken
        let tokenData = Data(bytes: &token, count: MemoryLayout<audit_token_t>.size)
        return check(guestAttributes: [kSecGuestAttributeAudit: tokenData], requirement: requirement)
    }

    /// Same check by PID (tests; production uses the audit token).
    static func check(pid: pid_t, requirement: String) -> OSStatus {
        check(guestAttributes: [kSecGuestAttributePid: pid], requirement: requirement)
    }

    private static func check(guestAttributes: [CFString: Any], requirement: String) -> OSStatus {
        var code: SecCode?
        var status = SecCodeCopyGuestWithAttributes(nil, guestAttributes as CFDictionary, [], &code)
        guard status == errSecSuccess, let code else { return status }
        var req: SecRequirement?
        status = SecRequirementCreateWithString(requirement as CFString, [], &req)
        guard status == errSecSuccess, let req else { return status }
        return SecCodeCheckValidity(code, [], req)
    }

    /// The Team ID this process is signed with, or nil (unsigned / ad hoc).
    static func ownTeamID() -> String? {
        var me: SecCode?
        var staticCode: SecStaticCode?
        var info: CFDictionary?
        guard SecCodeCopySelf([], &me) == errSecSuccess, let me,
              SecCodeCopyStaticCode(me, [], &staticCode) == errSecSuccess, let staticCode,
              SecCodeCopySigningInformation(
                  staticCode, SecCSFlags(rawValue: kSecCSSigningInformation), &info) == errSecSuccess
        else { return nil }
        return (info as? [String: Any])?[kSecCodeInfoTeamIdentifier as String] as? String
    }
}

/// A Unix socket server: binds `path` inside a private directory, accepts
/// on its own thread, checks each peer against `policy`, and runs `handler`
/// for each allowed connection on `handlerQueue` so a slow handler never
/// blocks the acceptor. The listening fd is closed only by the accept
/// thread after it has left the loop, so `stop()` never closes a descriptor
/// that is still in use.
final class UnixSocketListener {
    let path: String
    let exited = DispatchSemaphore(value: 0)
    private let name: String
    private let policy: PeerPolicy
    private let handlerQueue: DispatchQueue
    private let handler: (Int32) -> Void
    private let lock = NSLock()
    private var stopRequested = false
    private var fd: Int32 = -1

    /// Poll interval of the accept loop; bounds how long `stop()` takes to land.
    static let acceptPollMilliseconds: Int32 = 200

    init(name: String, path: String, policy: PeerPolicy, handlerQueue: DispatchQueue,
         handler: @escaping (Int32) -> Void) {
        self.name = name
        self.path = path
        self.policy = policy
        self.handlerQueue = handlerQueue
        self.handler = handler
    }

    var isStopRequested: Bool {
        lock.lock(); defer { lock.unlock() }
        return stopRequested
    }

    /// Bind and start accepting. Returns false (and logs) on failure.
    func start() -> Bool {
        guard SocketDirectory.preparePrivate((path as NSString).deletingLastPathComponent) else {
            return false
        }
        unlink(path)

        let fd = socket(AF_UNIX, SOCK_STREAM, 0)
        guard fd >= 0 else {
            NSLog("%@: failed to create socket: errno=%d", name, errno)
            return false
        }
        var addr = sockaddr_un()
        addr.sun_family = sa_family_t(AF_UNIX)
        let pathBytes = path.utf8CString
        guard pathBytes.count <= MemoryLayout.size(ofValue: addr.sun_path) else {
            NSLog("%@: socket path too long (%d bytes)", name, pathBytes.count)
            Darwin.close(fd)
            return false
        }
        withUnsafeMutablePointer(to: &addr.sun_path) { ptr in
            ptr.withMemoryRebound(to: CChar.self, capacity: pathBytes.count) { dest in
                for (i, byte) in pathBytes.enumerated() { dest[i] = byte }
            }
        }
        let bound = withUnsafePointer(to: &addr) { ptr in
            ptr.withMemoryRebound(to: sockaddr.self, capacity: 1) {
                Darwin.bind(fd, $0, socklen_t(MemoryLayout<sockaddr_un>.size))
            }
        }
        guard bound == 0, chmod(path, 0o600) == 0, Darwin.listen(fd, 16) == 0 else {
            NSLog("%@: bind/listen on %@ failed: errno=%d", name, path, errno)
            Darwin.close(fd)
            unlink(path)
            return false
        }
        self.fd = fd
        NSLog("%@: listening on %@", name, path)
        Thread.detachNewThread { [self] in acceptLoop() }
        return true
    }

    /// Ask the accept loop to exit; it closes the fd within one poll interval.
    func stop() {
        lock.lock(); stopRequested = true; lock.unlock()
        unlink(path)
    }

    private func acceptLoop() {
        defer {
            Darwin.close(fd)
            exited.signal()
        }
        while !isStopRequested {
            var pfd = pollfd(fd: fd, events: Int16(POLLIN), revents: 0)
            let ready = poll(&pfd, 1, Self.acceptPollMilliseconds)
            if ready < 0 {
                if errno == EINTR { continue }
                NSLog("%@: poll failed: errno=%d", name, errno)
                return
            }
            if ready == 0 || isStopRequested { continue }

            let client = accept(fd, nil, nil)
            guard client >= 0 else {
                if errno == EINTR || errno == EAGAIN || errno == ECONNABORTED { continue }
                NSLog("%@: accept failed: errno=%d", name, errno)
                return
            }
            if let reason = policy.refusal(for: client) {
                NSLog("%@: refused connection: %@", name, reason)
                Darwin.close(client)
                continue
            }
            var timeout = timeval(tv_sec: 10, tv_usec: 0)
            setsockopt(client, SOL_SOCKET, SO_RCVTIMEO, &timeout, socklen_t(MemoryLayout<timeval>.size))
            setsockopt(client, SOL_SOCKET, SO_SNDTIMEO, &timeout, socklen_t(MemoryLayout<timeval>.size))
            // A client that hung up must not kill the app with SIGPIPE.
            var noSigPipe: Int32 = 1
            setsockopt(client, SOL_SOCKET, SO_NOSIGPIPE, &noSigPipe, socklen_t(MemoryLayout<Int32>.size))

            let handler = self.handler
            handlerQueue.async {
                handler(client)
                Darwin.close(client)
            }
        }
    }
}

/// Write every byte of `data` to `fd`, looping over partial sends.
@discardableResult
func sendAll(_ fd: Int32, _ data: Data) -> Bool {
    data.withUnsafeBytes { raw in
        guard let base = raw.baseAddress else { return true }
        var sent = 0
        while sent < data.count {
            let n = Darwin.send(fd, base.advanced(by: sent), data.count - sent, 0)
            if n < 0 && errno == EINTR { continue }
            if n <= 0 { return false }
            sent += n
        }
        return true
    }
}
