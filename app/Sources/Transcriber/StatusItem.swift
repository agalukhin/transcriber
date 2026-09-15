import AppKit

final class StatusItem {
    private let item = NSStatusBar.system.statusItem(withLength: NSStatusItem.variableLength)
    private let statusLine = NSMenuItem(title: "Ожидание встречи", action: nil, keyEquivalent: "")
    private let toggle = NSMenuItem(title: "Начать запись", action: #selector(toggleAction), keyEquivalent: "r")
    private let openNote = NSMenuItem(title: "Открыть заметку", action: #selector(openNoteAction), keyEquivalent: "o")
    /// What the status line says when no transient hint is shown; `setHint(nil)` returns to it.
    private var baseStatus = "Ожидание встречи"

    var onToggle: (() -> Void)?
    var onOpenNote: (() -> Void)?
    var onOpenSessions: (() -> Void)?

    init() {
        let menu = NSMenu()
        menu.autoenablesItems = false   // without this AppKit re-enables items by target/action lookup
        statusLine.isEnabled = false
        menu.addItem(statusLine)
        menu.addItem(.separator())
        for m in [toggle, openNote] { m.target = self; menu.addItem(m) }
        let sessions = NSMenuItem(title: "Папка сессий", action: #selector(openSessionsAction), keyEquivalent: "")
        sessions.target = self
        menu.addItem(sessions)
        menu.addItem(.separator())
        menu.addItem(NSMenuItem(title: "Выход", action: #selector(NSApplication.terminate(_:)), keyEquivalent: "q"))
        item.menu = menu
        setRecording(false, detail: nil)
    }

    func setRecording(_ recording: Bool, detail: String?) {
        item.button?.title = recording ? "●" : "○"
        item.button?.contentTintColor = recording ? .systemRed : nil
        baseStatus = recording ? "Идёт запись" + (detail.map { " · \($0)" } ?? "") : "Ожидание встречи"
        statusLine.title = baseStatus
        toggle.title = recording ? "Остановить запись" : "Начать запись"
        openNote.isEnabled = recording
    }

    /// Transient line of text in the menu: the glyph and the menu items stay as they are.
    func setHint(_ text: String?) {
        statusLine.title = text ?? baseStatus
    }

    func setProblem(_ text: String) {
        item.button?.title = "!"
        item.button?.contentTintColor = .systemRed
        statusLine.title = text
    }

    @objc private func toggleAction() { onToggle?() }
    @objc private func openNoteAction() { onOpenNote?() }
    @objc private func openSessionsAction() { onOpenSessions?() }
}
