import Foundation
import Testing
@testable import TranscriberCore

private func tab(_ url: String, _ title: String = "") -> BrowserTab {
    BrowserTab(title: title, url: url)
}

@Test func meetingTabNeedsAMeetingHostAndARoomInThePath() {
    #expect(BrowserTabs.isMeeting(tab("https://talk.magnit.ru/5170c4177c32422893005c8dd4ed834e")))
    #expect(BrowserTabs.isMeeting(tab("https://meet.google.com/abc-defg-hij")))
    #expect(BrowserTabs.isMeeting(tab("https://us02web.zoom.us/j/123")))
    #expect(BrowserTabs.isMeeting(tab("https://zoom.us/wc/123/join")))
    // Lobby pages: the host is right, but nobody is in a room.
    #expect(!BrowserTabs.isMeeting(tab("https://talk.magnit.ru/")))
    #expect(!BrowserTabs.isMeeting(tab("https://zoom.us/")))
    // Neighbouring hosts must not match on a substring.
    #expect(!BrowserTabs.isMeeting(tab("https://mail.google.com/x")))
    #expect(!BrowserTabs.isMeeting(tab("https://notzoom.us/j/123")))
    #expect(!BrowserTabs.isMeeting(tab("не ссылка")))
}

@Test func pickMeetingTakesTheFirstCallTab() {
    let tabs = [tab("https://mail.google.com/x", "Почта"),
                tab("https://talk.magnit.ru/", "Толк"),
                tab("https://talk.magnit.ru/room1", "ОКРs Q3: общий синхрон — Толк"),
                tab("https://meet.google.com/abc-defg-hij", "Meet – Standup")]
    #expect(BrowserTabs.pickMeeting(tabs) == tabs[2])
    #expect(BrowserTabs.pickMeeting([tab("https://talk.magnit.ru/", "Толк")]) == nil)
    #expect(BrowserTabs.pickMeeting([]) == nil)
}

@Test func cleanTitleStripsServiceNamesOnly() {
    #expect(BrowserTabs.cleanTitle("ОКРs Q3: общий синхрон — Толк") == "ОКРs Q3: общий синхрон")
    #expect(BrowserTabs.cleanTitle("Standup - Google Meet") == "Standup")
    #expect(BrowserTabs.cleanTitle("Meet – Standup") == "Standup")
    #expect(BrowserTabs.cleanTitle("Google Meet: Разбор инцидента") == "Разбор инцидента")
    #expect(BrowserTabs.cleanTitle("  Синк   про   роадмап  ") == "Синк про роадмап")
    // Nothing but the service name is not a meeting title.
    #expect(BrowserTabs.cleanTitle("Толк") == "")
    #expect(BrowserTabs.cleanTitle(" Zoom ") == "")
    #expect(BrowserTabs.cleanTitle("") == "")
    // A meeting that genuinely mentions the service keeps its name.
    #expect(BrowserTabs.cleanTitle("Толк про роадмап") == "Толк про роадмап")
}

@Test func tabsAreParsedFromTabSeparatedLines() {
    let raw = "ОКРs Q3: общий синхрон — Толк\thttps://talk.magnit.ru/room1\nТолк\thttps://talk.magnit.ru/\nсломанная строка\n"
    #expect(BrowserTabs.parse(raw) == [BrowserTab(title: "ОКРs Q3: общий синхрон — Толк", url: "https://talk.magnit.ru/room1"),
                                       BrowserTab(title: "Толк", url: "https://talk.magnit.ru/")])
    #expect(BrowserTabs.parse("") == [])
}

@Test func unknownBrowserYieldsNoTabs() {
    #expect(BrowserTabs.tabs(ofBrowser: "com.example.nope") == [])
    #expect(BrowserTabs.meetingInfo(forBrowser: "com.example.nope") == nil)
}

@Test func isPlaceholderMatchesKnownLoadingTitles() {
    #expect(BrowserTabs.isPlaceholder("Подключение к встрече"))
    #expect(BrowserTabs.isPlaceholder("подключение к встрече"))   // case-insensitive
    #expect(BrowserTabs.isPlaceholder("Подключение"))
    #expect(BrowserTabs.isPlaceholder("Connecting"))
    #expect(BrowserTabs.isPlaceholder("CONNECTING"))
    #expect(BrowserTabs.isPlaceholder("Толк"))
    #expect(BrowserTabs.isPlaceholder("Загрузка"))
    #expect(BrowserTabs.isPlaceholder("Loading"))
    #expect(BrowserTabs.isPlaceholder(""))
    // Any «Подключение к …» reads as still-connecting, not just the exact phrase.
    #expect(BrowserTabs.isPlaceholder("Подключение к комнате"))
    // A real meeting title is not a placeholder, even one that mentions the service.
    #expect(!BrowserTabs.isPlaceholder("ОКРs Q3: общий синхрон"))
    #expect(!BrowserTabs.isPlaceholder("Толк про роадмап"))
}

@Test func meetingTabStateWaitsOutAPlaceholderTitle() {
    // Pre-join screen: URL is already a room, title has not caught up yet.
    let loading = [tab("https://talk.magnit.ru/room1", "Подключение к встрече")]
    #expect(BrowserTabs.meetingTabState(in: loading) == .placeholder)

    let ready = [tab("https://talk.magnit.ru/room1", "ОКРs Q3: общий синхрон — Толк")]
    #expect(BrowserTabs.meetingTabState(in: ready) ==
             .ready(MeetingInfo(title: "ОКРs Q3: общий синхрон", participants: [])))

    let noCallTab = [tab("https://mail.google.com/x", "Почта")]
    #expect(BrowserTabs.meetingTabState(in: noCallTab) == .none)
}
