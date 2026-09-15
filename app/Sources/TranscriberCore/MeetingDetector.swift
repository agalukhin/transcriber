import Foundation

public enum MeetingEvent: Equatable {
    case started(app: String)
    case ended
}

/// Polled state machine: a meeting starts when a watched app records from the mic, and ends
/// when no watched app has used the mic for `endGrace` seconds.
public final class MeetingDetector {
    public static let defaultApps = ["kontur.talk", "us.zoom"]

    private let apps: [String]
    private let lister: () -> [AudioProcess]
    private let endGrace: TimeInterval
    public private(set) var activeApp: String?
    private var micLostAt: Date?

    public init(apps: [String] = MeetingDetector.defaultApps, lister: @escaping () -> [AudioProcess],
                endGrace: TimeInterval = 60) {
        self.apps = apps
        self.lister = lister
        self.endGrace = endGrace
    }

    /// Returns the watched app (by prefix) that currently records from the mic, if any.
    private func appUsingMic() -> String? {
        for p in lister() where p.isRunningInput {
            if let app = apps.first(where: { p.bundleID.hasPrefix($0) }) { return app }
        }
        return nil
    }

    /// Two accepted limitations:
    /// (a) if the active app switches within a single poll window, `activeApp` keeps the app
    ///     that was first seen using the mic and does not update to the newer one;
    /// (b) any watched app touching the mic outside an actual call (e.g. a Zoom mic test)
    ///     triggers a `.started` event — the user simply declines the prompt.
    public func poll(now: Date) -> MeetingEvent? {
        let using = appUsingMic()
        if activeApp == nil {
            guard let app = using else { return nil }
            activeApp = app
            micLostAt = nil
            return .started(app: app)
        }
        if using != nil {
            micLostAt = nil
            return nil
        }
        if let lost = micLostAt {
            if now.timeIntervalSince(lost) >= endGrace {
                activeApp = nil
                micLostAt = nil
                return .ended
            }
            return nil
        }
        micLostAt = now
        return nil
    }

    public func reset() {
        activeApp = nil
        micLostAt = nil
    }
}
