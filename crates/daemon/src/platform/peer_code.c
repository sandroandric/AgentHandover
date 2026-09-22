// Code-signature checks for Unix-socket peers (see src/peer.rs).
//
// Kept in C so the Security framework's CoreFoundation types never cross
// into Rust: Rust passes plain bytes and strings and gets an OSStatus back.

#include <Security/Security.h>
#include <string.h>

// Validate the process behind `attrs` against `requirement`.
static OSStatus check_guest(CFDictionaryRef attrs, const char *requirement) {
    SecCodeRef code = NULL;
    OSStatus status = SecCodeCopyGuestWithAttributes(NULL, attrs, kSecCSDefaultFlags, &code);
    if (status != errSecSuccess) return status;

    CFStringRef text = CFStringCreateWithCString(NULL, requirement, kCFStringEncodingUTF8);
    SecRequirementRef req = NULL;
    status = text ? SecRequirementCreateWithString(text, kSecCSDefaultFlags, &req) : errSecParam;
    if (status == errSecSuccess) {
        status = SecCodeCheckValidity(code, kSecCSDefaultFlags, req);
    }
    if (req) CFRelease(req);
    if (text) CFRelease(text);
    CFRelease(code);
    return status;
}

// errSecSuccess when the process with this audit token (32 bytes, from
// getsockopt LOCAL_PEERTOKEN) is validly signed and satisfies `requirement`.
int32_t ah_audit_token_satisfies(const uint8_t *token, size_t len, const char *requirement) {
    CFDataRef data = CFDataCreate(NULL, token, (CFIndex)len);
    if (!data) return errSecAllocate;
    const void *keys[] = {kSecGuestAttributeAudit};
    const void *values[] = {data};
    CFDictionaryRef attrs = CFDictionaryCreate(NULL, keys, values, 1,
                                               &kCFTypeDictionaryKeyCallBacks,
                                               &kCFTypeDictionaryValueCallBacks);
    OSStatus status = attrs ? check_guest(attrs, requirement) : errSecAllocate;
    if (attrs) CFRelease(attrs);
    CFRelease(data);
    return status;
}

// Same check by PID (tests; the server uses the audit token).
int32_t ah_pid_satisfies(int32_t pid, const char *requirement) {
    CFNumberRef number = CFNumberCreate(NULL, kCFNumberSInt32Type, &pid);
    if (!number) return errSecAllocate;
    const void *keys[] = {kSecGuestAttributePid};
    const void *values[] = {number};
    CFDictionaryRef attrs = CFDictionaryCreate(NULL, keys, values, 1,
                                               &kCFTypeDictionaryKeyCallBacks,
                                               &kCFTypeDictionaryValueCallBacks);
    OSStatus status = attrs ? check_guest(attrs, requirement) : errSecAllocate;
    if (attrs) CFRelease(attrs);
    CFRelease(number);
    return status;
}

// Copy this process's signing Team ID into `buf` (NUL-terminated).
// Returns 1 when there is one, 0 when unsigned or ad hoc signed.
int32_t ah_self_team_id(char *buf, size_t len) {
    SecCodeRef self = NULL;
    SecStaticCodeRef staticCode = NULL;
    CFDictionaryRef info = NULL;
    int32_t found = 0;
    if (SecCodeCopySelf(kSecCSDefaultFlags, &self) == errSecSuccess &&
        SecCodeCopyStaticCode(self, kSecCSDefaultFlags, &staticCode) == errSecSuccess &&
        SecCodeCopySigningInformation(staticCode, kSecCSSigningInformation, &info) == errSecSuccess) {
        CFStringRef team = CFDictionaryGetValue(info, kSecCodeInfoTeamIdentifier);
        if (team && CFGetTypeID(team) == CFStringGetTypeID() &&
            CFStringGetCString(team, buf, (CFIndex)len, kCFStringEncodingUTF8)) {
            found = 1;
        }
    }
    if (info) CFRelease(info);
    if (staticCode) CFRelease(staticCode);
    if (self) CFRelease(self);
    return found;
}
