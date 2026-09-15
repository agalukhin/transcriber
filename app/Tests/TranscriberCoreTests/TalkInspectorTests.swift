import Foundation
import Testing
@testable import TranscriberCore

@Test func participantHeuristicKeepsPersonNamesOnly() {
    let texts = ["Егор Спирин", "Вы", "Ярослав Серов", "Включить микрофон", "Егор Спирин", "Анна-Мария Кузнецова",
                 "Участники (3)", "Владислав Слесарев", "John Smith", "чат"]
    #expect(TalkInspector.participantNames(from: texts) == ["Егор Спирин", "Ярослав Серов", "Анна-Мария Кузнецова", "Владислав Слесарев", "John Smith"])
}
