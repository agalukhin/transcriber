import AppKit

/// «В Толке началась встреча. Записать?» — floating panel near the top-right corner, auto-dismisses.
final class PromptPanel {
    private var panel: NSPanel?
    private var timer: Timer?

    func show(appName: String, onRecord: @escaping () -> Void, onSkip: @escaping () -> Void) {
        dismiss()
        let p = NSPanel(contentRect: NSRect(x: 0, y: 0, width: 360, height: 110),
                        styleMask: [.titled, .nonactivatingPanel, .utilityWindow], backing: .buffered, defer: false)
        p.title = "Транскрибатор"
        p.level = .floating
        p.isFloatingPanel = true
        p.hidesOnDeactivate = false
        p.collectionBehavior = [.canJoinAllSpaces, .fullScreenAuxiliary]

        let label = NSTextField(wrappingLabelWithString: "В \(appName) началась встреча. Записать и завести заметку?")
        label.frame = NSRect(x: 20, y: 60, width: 320, height: 40)
        let record = NSButton(title: "Записать", target: nil, action: nil)
        record.frame = NSRect(x: 230, y: 16, width: 110, height: 32)
        let skip = NSButton(title: "Не сейчас", target: nil, action: nil)
        skip.frame = NSRect(x: 110, y: 16, width: 110, height: 32)
        p.contentView?.addSubview(label)
        p.contentView?.addSubview(record)
        p.contentView?.addSubview(skip)

        let handler = Handler(record: { [weak self] in self?.dismiss(); onRecord() },
                              skip: { [weak self] in self?.dismiss(); onSkip() })
        record.target = handler
        record.action = #selector(Handler.record(_:))
        skip.target = handler
        skip.action = #selector(Handler.skip(_:))
        self.handler = handler

        if let screen = NSScreen.main {
            let f = screen.visibleFrame
            p.setFrameOrigin(NSPoint(x: f.maxX - 380, y: f.maxY - 140))
        }
        p.orderFrontRegardless()
        panel = p
        // .common, so an open menu or a window resize does not stall the auto-dismiss
        let t = Timer(timeInterval: 90, repeats: false) { [weak self] _ in self?.dismiss(); onSkip() }
        RunLoop.main.add(t, forMode: .common)
        timer = t
    }

    func dismiss() {
        timer?.invalidate()
        timer = nil
        panel?.orderOut(nil)
        panel = nil
        handler = nil
    }

    private var handler: Handler?

    private final class Handler: NSObject {
        let recordBlock: () -> Void
        let skipBlock: () -> Void
        init(record: @escaping () -> Void, skip: @escaping () -> Void) {
            recordBlock = record
            skipBlock = skip
        }
        @objc func record(_ sender: Any?) { recordBlock() }
        @objc func skip(_ sender: Any?) { skipBlock() }
    }
}
