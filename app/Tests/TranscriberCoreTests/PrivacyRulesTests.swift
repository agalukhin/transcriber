import Foundation
import Testing
@testable import TranscriberCore

private let calendar = Calendar.current

/// 2026-09-21 is a Monday and 2026-09-24 the Thursday of the same week — asserted below rather
/// than trusted, so a wrong date fails here instead of silently weakening every other case.
private func at(_ day: Int, _ hour: Int, _ minute: Int) -> Date {
    calendar.date(from: DateComponents(year: 2026, month: 9, day: day, hour: hour, minute: minute))!
}

private func tempConfig() -> URL {
    FileManager.default.temporaryDirectory
        .appendingPathComponent(UUID().uuidString)
        .appendingPathComponent("app-config.json")
}

@Test func weekdayNumbersMatchCalendar() {
    #expect(calendar.component(.weekday, from: at(21, 12, 0)) == 2)   // понедельник
    #expect(calendar.component(.weekday, from: at(24, 12, 0)) == 5)   // четверг
}

@Test func therapyHourIsPrivate() {
    let rules = PrivacyRules.defaults
    #expect(rules.isPrivate(app: "com.google.Chrome", at: at(21, 16, 1)))    // та самая встреча
    #expect(rules.isPrivate(app: "com.apple.Safari", at: at(24, 20, 10)))
}

@Test func workHourInTalkIsNotPrivate() {
    let rules = PrivacyRules.defaults
    #expect(!rules.isPrivate(app: "kontur.talk", at: at(21, 14, 0)))
    #expect(!rules.isPrivate(app: "kontur.talk", at: at(24, 11, 30)))
    #expect(!rules.isPrivate(app: "com.google.Chrome", at: at(22, 16, 1)))   // вторник, то же время
}

/// Терапии в Толке не бывает, а рабочие встречи бывают только там — поэтому Толк перебивает окно.
@Test func talkIsNeverPrivateEvenInsideAWindow() {
    let rules = PrivacyRules.defaults
    #expect(!rules.isPrivate(app: "kontur.talk", at: at(21, 16, 10)))
    #expect(!rules.isPrivate(app: "kontur.talk", at: at(24, 20, 10)))
    #expect(!rules.isPrivate(app: "kontur.talk.helper", at: at(21, 16, 10)))   // префикс, не точное совпадение
}

/// В браузере bundle id не говорит ничего: рабочий звонок узнаётся только по открытой вкладке,
/// и ответ на этот вопрос приходит снаружи, флагом `workContext`.
@Test func workContextInBrowserBeatsTheWindow() {
    let rules = PrivacyRules.defaults
    #expect(!rules.isPrivate(app: "com.google.Chrome", at: at(21, 16, 10), workContext: true))
    #expect(rules.isPrivate(app: "com.google.Chrome", at: at(21, 16, 10), workContext: false))
    #expect(!rules.isPrivate(app: "com.google.Chrome", at: at(21, 14, 0), workContext: false))
}

@Test func zoomIsPrivateAtAnyTime() {
    let rules = PrivacyRules.defaults
    #expect(rules.isPrivate(app: "us.zoom", at: at(22, 11, 0)))
    #expect(rules.isPrivate(app: "us.zoom.xos", at: at(23, 9, 15)))   // префикс, не точное совпадение
}

@Test func windowBoundariesAreInclusive() {
    let rules = PrivacyRules.defaults
    #expect(!rules.isPrivate(app: "com.google.Chrome", at: at(21, 15, 59)))
    #expect(rules.isPrivate(app: "com.google.Chrome", at: at(21, 16, 0)))
    #expect(rules.isPrivate(app: "com.google.Chrome", at: at(21, 16, 50)))
    #expect(!rules.isPrivate(app: "com.google.Chrome", at: at(21, 16, 51)))
}

@Test func customRulesFromJSON() throws {
    let url = tempConfig()
    try FileManager.default.createDirectory(at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
    try """
    {"privacy": {"windows": [{"weekday": 3, "from": "09:00", "to": "10:00"}], "apps": ["com.example.diary"]}}
    """.write(to: url, atomically: true, encoding: .utf8)
    let rules = PrivacyRules.load(from: url)
    #expect(rules.windows == [PrivateWindow(weekday: 3, from: "09:00", to: "10:00")])
    #expect(calendar.component(.weekday, from: at(22, 9, 30)) == 3)   // вторник
    #expect(rules.isPrivate(app: "com.google.Chrome", at: at(22, 9, 30)))
    #expect(rules.isPrivate(app: "com.example.diary", at: at(21, 12, 0)))
    #expect(!rules.isPrivate(app: "com.google.Chrome", at: at(21, 16, 1)))   // дефолтные окна больше не действуют
    // Ключей workApps/workHosts в файле нет — берутся дефолты, Толк остаётся рабочим.
    #expect(rules.workApps == PrivacyRules.defaultWorkApps)
    #expect(rules.workHosts == PrivacyRules.defaultWorkHosts)
    #expect(!rules.isPrivate(app: "kontur.talk", at: at(22, 9, 30)))
}

@Test func workKeysComeFromJSONWhenPresent() throws {
    let url = tempConfig()
    try FileManager.default.createDirectory(at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
    try """
    {"privacy": {"windows": [{"weekday": 2, "from": "16:00", "to": "16:50"}], "apps": [], \
    "workApps": ["com.example.work"], "workHosts": ["calls.example.com"]}}
    """.write(to: url, atomically: true, encoding: .utf8)
    let rules = PrivacyRules.load(from: url)
    #expect(rules.workApps == ["com.example.work"])
    #expect(rules.workHosts == ["calls.example.com"])
    #expect(!rules.isPrivate(app: "com.example.work", at: at(21, 16, 10)))
    #expect(rules.isPrivate(app: "kontur.talk", at: at(21, 16, 10)))   // список переопределён целиком
}

@Test func missingFileIsWrittenWithDefaults() throws {
    let url = tempConfig()
    #expect(PrivacyRules.load(from: url) == .defaults)
    #expect(FileManager.default.fileExists(atPath: url.path))
    let written = PrivacyRules.load(from: url)   // второй заход читает уже файл, не ветку «нет файла»
    #expect(written == .defaults)
    let json = try JSONSerialization.jsonObject(with: Data(contentsOf: url)) as! [String: Any]
    let privacy = json["privacy"] as? [String: Any]
    #expect(privacy != nil)
    #expect(privacy?["workApps"] != nil)    // новые ключи видны в файле, есть что править руками
    #expect(privacy?["workHosts"] != nil)
    #expect(json["_comment"] != nil)   // подсказка про нумерацию дней недели для ручной правки
}

@Test func brokenFileFallsBackToDefaults() throws {
    let url = tempConfig()
    try FileManager.default.createDirectory(at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
    try "{ не json".write(to: url, atomically: true, encoding: .utf8)
    #expect(PrivacyRules.load(from: url) == .defaults)

    let other = tempConfig()
    try FileManager.default.createDirectory(at: other.deletingLastPathComponent(), withIntermediateDirectories: true)
    try #"{"privacy": {"windows": "каждый понедельник"}}"#.write(to: other, atomically: true, encoding: .utf8)
    #expect(PrivacyRules.load(from: other) == .defaults)

    let unrelated = tempConfig()
    try FileManager.default.createDirectory(at: unrelated.deletingLastPathComponent(), withIntermediateDirectories: true)
    try #"{"worker": {"model": "large"}}"#.write(to: unrelated, atomically: true, encoding: .utf8)
    #expect(PrivacyRules.load(from: unrelated) == .defaults)
}

@Test func malformedTimeDisablesOnlyThatWindow() {
    let rules = PrivacyRules(windows: [PrivateWindow(weekday: 2, from: "четыре", to: "17:30"),
                                       PrivateWindow(weekday: 2, from: "19:00", to: "20:00")],
                             apps: [])
    #expect(!rules.isPrivate(app: "com.google.Chrome", at: at(21, 16, 1)))
    #expect(rules.isPrivate(app: "com.google.Chrome", at: at(21, 19, 30)))
}
