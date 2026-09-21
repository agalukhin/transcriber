import Foundation

/// A recurring slot in the week that is private by definition: therapy is every Monday and
/// Thursday, always at the same hour. `weekday` follows `Calendar`: 1 = Sunday … 7 = Saturday.
/// `from`/`to` are local wall-clock times, `"HH:mm"`.
public struct PrivateWindow: Codable, Equatable {
    public var weekday: Int
    public var from: String
    public var to: String

    public init(weekday: Int, from: String, to: String) {
        self.weekday = weekday
        self.from = from
        self.to = to
    }
}

/// What must never reach the work vault. Two independent triggers, either one is enough:
/// a bundle-ID prefix (work calls happen only in Толк, so anything in Zoom is personal) and a
/// weekly time window. A session marked private tells the worker to write the transcript into
/// `~/Documents/Приватные записи/` instead of the vault.
///
/// Both ends of a window are inclusive and the windows are padded around the real appointment:
/// the cost of a false positive is one note in the wrong folder, the cost of a false negative is
/// a therapy transcript in a work vault.
public struct PrivacyRules: Codable, Equatable {
    public var windows: [PrivateWindow]
    public var apps: [String]

    public init(windows: [PrivateWindow], apps: [String]) {
        self.windows = windows
        self.apps = apps
    }

    public static let defaults = PrivacyRules(
        windows: [PrivateWindow(weekday: 2, from: "16:00", to: "16:50"),
                  PrivateWindow(weekday: 5, from: "20:00", to: "20:50")],
        apps: ["us.zoom"])

    public func isPrivate(app: String, at date: Date, calendar: Calendar = .current) -> Bool {
        if apps.contains(where: { !$0.isEmpty && app.hasPrefix($0) }) { return true }
        let parts = calendar.dateComponents([.weekday, .hour, .minute], from: date)
        guard let weekday = parts.weekday, let hour = parts.hour, let minute = parts.minute else { return false }
        let now = hour * 60 + minute
        return windows.contains { window in
            guard window.weekday == weekday,
                  let from = Self.minutes(window.from), let to = Self.minutes(window.to) else { return false }
            // A window that wraps past midnight («23:30»–«00:30») is two ranges, not one.
            return from <= to ? (now >= from && now <= to) : (now >= from || now <= to)
        }
    }

    /// Minutes since midnight for `"HH:mm"`; nil for anything else, so a typo in the config
    /// disables that one window instead of throwing the whole file away.
    static func minutes(_ text: String) -> Int? {
        let parts = text.split(separator: ":")
        guard parts.count == 2, let h = Int(parts[0]), let m = Int(parts[1]),
              (0...23).contains(h), (0...59).contains(m) else { return nil }
        return h * 60 + m
    }

    // MARK: - Config file

    /// Владислав edits the schedule here; the app re-reads it at every session start.
    public static var defaultURL: URL {
        FileManager.default.homeDirectoryForCurrentUser
            .appendingPathComponent("Claude/transcriber/app-config.json")
    }

    /// Rules from the `privacy` key of the JSON at `url`. A missing file is written out with the
    /// defaults (so there is something to edit); a broken one is ignored — falling back to the
    /// defaults keeps therapy private, which is the whole point of the file.
    public static func load(from url: URL = PrivacyRules.defaultURL) -> PrivacyRules {
        guard let data = try? Data(contentsOf: url) else {
            writeDefaults(to: url)
            return defaults
        }
        guard let object = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            invalidWarned.fire { Log.shared.warn("privacy rules: \(url.path) is not valid JSON, using defaults") }
            return defaults
        }
        guard let privacy = object["privacy"] else { return defaults }
        guard let sub = try? JSONSerialization.data(withJSONObject: privacy),
              let rules = try? JSONDecoder().decode(PrivacyRules.self, from: sub) else {
            invalidWarned.fire { Log.shared.warn("privacy rules: 'privacy' in \(url.path) is malformed, using defaults") }
            return defaults
        }
        return rules
    }

    private static func writeDefaults(to url: URL) {
        guard let encoded = try? JSONEncoder().encode(defaults),
              let privacy = try? JSONSerialization.jsonObject(with: encoded) else { return }
        let payload: [String: Any] = [
            "_comment": "privacy.windows: weekday 1=вс … 7=сб, время локальное HH:mm, обе границы включительно. "
                + "privacy.apps — префиксы bundle id. Запись, попавшая в окно или в это приложение, уходит в "
                + "«Приватные записи», а не в рабочий вольт.",
            "privacy": privacy,
        ]
        guard let data = try? JSONSerialization.data(withJSONObject: payload,
                                                     options: [.prettyPrinted, .sortedKeys, .withoutEscapingSlashes])
        else { return }
        try? FileManager.default.createDirectory(at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
        try? data.write(to: url, options: .atomic)
    }

    /// The rules are re-read at every session start, so a broken file would otherwise fill the log
    /// with the same line all day.
    private static let invalidWarned = WarnOnce()

    private final class WarnOnce {
        private let lock = NSLock()
        private var fired = false
        func fire(_ body: () -> Void) {
            lock.lock()
            let first = !fired
            fired = true
            lock.unlock()
            if first { body() }
        }
    }
}
