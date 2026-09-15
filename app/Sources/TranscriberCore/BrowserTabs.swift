import Foundation

public struct BrowserTab: Equatable {
    public let title: String
    public let url: String
    public init(title: String, url: String) {
        self.title = title
        self.url = url
    }
}

/// A call in a browser tab looks from the outside exactly like any other tab holding the mic, so
/// the only way to tell a meeting from YouTube is to ask the browser what it has open. AppleScript
/// is the one route that does not need an extension; it costs the Automation permission (macOS
/// asks once) and returns nothing at all when it is denied — hence «no tabs» is never an error,
/// just «we do not know yet».
public enum BrowserTabs {
    /// Hosts we treat as meeting services. Matched by suffix, so `us02web.zoom.us` counts too.
    public static let meetingHosts: [String] = ["talk.magnit.ru", "ktalk.ru", "talk.kontur.ru",
                                                "meet.google.com", "telemost.yandex.ru", "zoom.us"]

    /// A meeting tab is a meeting host **with a room in the path**: `talk.magnit.ru/` is the lobby
    /// the browser keeps open all day, `talk.magnit.ru/<room>` is an actual call.
    public static func isMeeting(_ tab: BrowserTab) -> Bool {
        guard let host = URLComponents(string: tab.url)?.host?.lowercased(), !host.isEmpty else { return false }
        guard meetingHosts.contains(where: { host == $0 || host.hasSuffix("." + $0) }) else { return false }
        let path = URLComponents(string: tab.url)?.path ?? ""
        return !path.isEmpty && path != "/"
    }

    public static func pickMeeting(_ tabs: [BrowserTab]) -> BrowserTab? {
        tabs.first(where: isMeeting)
    }

    /// Browser tab titles carry the service name («Синк — Толк», «Meet – Standup»); the note only
    /// wants the meeting itself. A title that is nothing but the service name carries no meeting.
    public static func cleanTitle(_ title: String) -> String {
        let t = title.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !t.isEmpty, !matches(barePattern, t) else { return "" }
        var out = replacing(suffixPattern, in: t)
        out = replacing(prefixPattern, in: out)
        out = replacing(whitespacePattern, in: out, with: " ")
        return out.trimmingCharacters(in: .whitespacesAndNewlines)
    }

    /// Every open tab of a browser, via AppleScript. `[]` on any failure — a missing Automation
    /// permission, a browser that is not running, an unknown bundle ID.
    public static func tabs(ofBrowser bundleID: String) -> [BrowserTab] {
        guard let source = scriptSource(for: bundleID) else {
            Log.shared.warn("browser tabs: unknown browser \(bundleID)")
            return []
        }
        guard let script = NSAppleScript(source: source) else {
            Log.shared.warn("browser tabs: script did not compile for \(bundleID)")
            return []
        }
        var error: NSDictionary?
        let result = script.executeAndReturnError(&error)
        if let error {
            let number = error["NSAppleScriptErrorNumber"] ?? "?"
            let message = error["NSAppleScriptErrorMessage"] ?? ""
            Log.shared.warn("browser tabs: \(bundleID) script failed (\(number)): \(message)")
            return []
        }
        return parse(result.stringValue ?? "")
    }

    /// Title and participants for a call open in `bundleID`, or nil when no tab looks like a call.
    /// Participants stay empty: the browser tells us nothing about who is in the room.
    public static func meetingInfo(forBrowser bundleID: String) -> MeetingInfo? {
        guard let tab = pickMeeting(tabs(ofBrowser: bundleID)) else { return nil }
        return MeetingInfo(title: cleanTitle(tab.title), participants: [])
    }

    // MARK: - AppleScript

    /// One string, `title <TAB> url <LF>` per tab, so the Swift side only has to split. Both loops
    /// are wrapped in `try`: Safari counts non-browser windows (preferences, downloads) among its
    /// `windows`, and asking those for tabs raises.
    private static func scriptSource(for bundleID: String) -> String? {
        let appName: String
        let titleProperty: String
        switch bundleID {
        case "com.google.Chrome": appName = "Google Chrome"; titleProperty = "title of t"
        case "com.microsoft.edgemac": appName = "Microsoft Edge"; titleProperty = "title of t"
        case "com.apple.Safari": appName = "Safari"; titleProperty = "name of t"
        default: return nil
        }
        return """
        tell application "\(appName)"
            set out to ""
            repeat with w in windows
                try
                    repeat with t in tabs of w
                        try
                            set out to out & (\(titleProperty)) & "\\t" & (URL of t) & "\\n"
                        end try
                    end repeat
                end try
            end repeat
            return out
        end tell
        """
    }

    static func parse(_ raw: String) -> [BrowserTab] {
        raw.split(separator: "\n", omittingEmptySubsequences: true).compactMap { line in
            let parts = line.split(separator: "\t", maxSplits: 1, omittingEmptySubsequences: false)
            guard parts.count == 2 else { return nil }
            let url = String(parts[1]).trimmingCharacters(in: .whitespacesAndNewlines)
            guard !url.isEmpty else { return nil }
            return BrowserTab(title: String(parts[0]), url: url)
        }
    }

    // MARK: - Title patterns

    /// «Google Meet» stands before «Meet» (and «Яндекс Телемост» before «Телемост»): alternation is
    /// leftmost-first, so the longer name has to come first or only its tail would be stripped.
    private static let serviceNames = "Google Meet|Meet|Толк|Zoom|Яндекс Телемост|Телемост"
    private static let suffixPattern = regex(#"\s*[—–-]\s*(\#(serviceNames))\s*$"#)
    private static let prefixPattern = regex(#"^\s*(\#(serviceNames))\s*[—–\-:]\s*"#)
    private static let barePattern = regex(#"^\s*(\#(serviceNames))\s*$"#)
    private static let whitespacePattern = regex(#"\s+"#)

    private static func regex(_ pattern: String) -> NSRegularExpression {
        try! NSRegularExpression(pattern: pattern, options: [.caseInsensitive])
    }

    private static func matches(_ re: NSRegularExpression, _ s: String) -> Bool {
        re.firstMatch(in: s, range: NSRange(s.startIndex..., in: s)) != nil
    }

    private static func replacing(_ re: NSRegularExpression, in s: String, with template: String = "") -> String {
        re.stringByReplacingMatches(in: s, range: NSRange(s.startIndex..., in: s), withTemplate: template)
    }
}
