import Foundation
import Testing
@testable import TranscriberCore

@Test func participantHeuristicKeepsPersonNamesOnly() {
    let texts = ["Егор Спирин", "Вы", "Ярослав Серов", "Включить микрофон", "Егор Спирин", "Анна-Мария Кузнецова",
                 "Участники (3)", "Владислав Слесарев", "John Smith", "чат"]
    #expect(TalkInspector.participantNames(from: texts) == ["Егор Спирин", "Ярослав Серов", "Анна-Мария Кузнецова", "Владислав Слесарев", "John Smith"])
}

@Test func titleStripsOnlyTheTrailingAppSuffix() {
    #expect(TalkInspector.cleanTitle("Синк — Толк") == "Синк")
    #expect(TalkInspector.cleanTitle("Толк про роадмап") == "Толк про роадмап")
    #expect(TalkInspector.cleanTitle("Толк") == "")
}
