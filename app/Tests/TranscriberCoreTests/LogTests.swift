import Foundation
import Testing
@testable import TranscriberCore

@Test func logAppendsTimestampedLines() throws {
    let dir = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
    let log = Log(file: dir.appendingPathComponent("t.log"))
    log.info("hello")
    log.warn("careful")
    let text = try String(contentsOf: dir.appendingPathComponent("t.log"), encoding: .utf8)
    let lines = text.split(separator: "\n")
    #expect(lines.count == 2)
    #expect(lines[0].hasSuffix(" INFO hello"))
    #expect(lines[1].hasSuffix(" WARN careful"))
    #expect(lines[0].hasPrefix("20"))
}
