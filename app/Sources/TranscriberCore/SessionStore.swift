import Foundation

public struct SessionDir: Equatable {
    public let url: URL
    public var id: String { url.lastPathComponent }
    public var metaURL: URL { url.appendingPathComponent("meta.json") }
    public var stateURL: URL { url.appendingPathComponent("state.json") }
    public var stopURL: URL { url.appendingPathComponent("STOP") }
    public var micURL: URL { url.appendingPathComponent("mic.pcm") }
    public var sysURL: URL { url.appendingPathComponent("sys.pcm") }

    public func requestStop() throws {
        if !FileManager.default.fileExists(atPath: stopURL.path) {
            try Data().write(to: stopURL)
        }
    }

    private func state() -> [String: Any]? {
        guard let data = try? Data(contentsOf: stateURL) else { return nil }
        return (try? JSONSerialization.jsonObject(with: data)) as? [String: Any]
    }

    public func notePath() -> String? { state()?["note_path"] as? String }
    public func status() -> String? { state()?["status"] as? String }
}

public final class SessionStore {
    public let root: URL

    public init(root: URL) {
        self.root = root
    }

    public static var defaultRoot: URL {
        FileManager.default.homeDirectoryForCurrentUser.appendingPathComponent("Claude/transcriber/sessions")
    }

    public func create(app: String, title: String, participants: [String], now: Date = Date()) throws -> SessionDir {
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        let base = Self.idFormatter.string(from: now)
        var id = base
        var n = 0
        while FileManager.default.fileExists(atPath: root.appendingPathComponent(id, isDirectory: false).path) {
            n += 1
            id = "\(base)-\(n)"
        }
        let dir = SessionDir(url: root.appendingPathComponent(id, isDirectory: false))
        try FileManager.default.createDirectory(at: dir.url, withIntermediateDirectories: true)
        let meta: [String: Any] = [
            "id": id,
            "started_at": Self.isoFormatter.string(from: now),
            "app": app,
            "title": title,
            "participants": participants,
        ]
        let data = try JSONSerialization.data(withJSONObject: meta, options: [.prettyPrinted, .withoutEscapingSlashes])
        try data.write(to: dir.metaURL, options: .atomic)
        return dir
    }

    public func open(id: String) -> SessionDir? {
        let dir = SessionDir(url: root.appendingPathComponent(id, isDirectory: false))
        return FileManager.default.fileExists(atPath: dir.metaURL.path) ? dir : nil
    }

    private static let idFormatter: DateFormatter = {
        let f = DateFormatter()
        f.locale = Locale(identifier: "en_US_POSIX")
        f.dateFormat = "yyyy-MM-dd-HHmm"
        return f
    }()

    /// ISO 8601 with the local offset (`+03:00`), which the worker uses for vault filenames.
    private static let isoFormatter: DateFormatter = {
        let f = DateFormatter()
        f.locale = Locale(identifier: "en_US_POSIX")
        f.dateFormat = "yyyy-MM-dd'T'HH:mm:ssXXX"
        return f
    }()
}
