import AppKit

/// «В Толке началась встреча. Записать?» — a borderless, translucent card that slides in at the
/// top-right corner, auto-dismisses after 90 s. Non-activating: it never steals focus from the call.
final class PromptPanel {
    private var panel: NSPanel?
    private var timer: Timer?
    private var handler: Handler?

    private static let width: CGFloat = 400
    private static let height: CGFloat = 118
    private static let margin: CGFloat = 16

    struct Spec {
        var symbol: String
        var tint: NSColor
        var title: String
        var subtitle: String
        var primary: String
        var secondary: String
        var timeout: TimeInterval
    }

    static func meeting(appName: String) -> Spec {
        Spec(symbol: "record.circle", tint: .systemRed, title: "Встреча в \(appName)",
             subtitle: "Записать разговор и завести заметку в вольте?", primary: "Записать", secondary: "Не сейчас", timeout: 90)
    }

    static func stop(minutes: Int) -> Spec {
        Spec(symbol: "moon.zzz", tint: .systemOrange, title: "Похоже, встреча закончилась",
             subtitle: "Тишина уже \(minutes) мин. Остановить запись?", primary: "Остановить", secondary: "Продолжить", timeout: 0)
    }

    var isShowing: Bool { panel != nil }

    /// Meeting prompt: primary = record, secondary = skip.
    func show(appName: String, onRecord: @escaping () -> Void, onSkip: @escaping () -> Void) {
        show(Self.meeting(appName: appName), onPrimary: onRecord, onSecondary: onSkip)
    }

    /// Any card. `timeout` 0 = stays until answered or dismissed; otherwise the secondary action
    /// fires when it elapses.
    func show(_ spec: Spec, onPrimary: @escaping () -> Void, onSecondary: @escaping () -> Void) {
        dismiss(animated: false)
        let onRecord = onPrimary
        let onSkip = onSecondary
        let p = NSPanel(contentRect: NSRect(x: 0, y: 0, width: Self.width, height: Self.height),
                        styleMask: [.nonactivatingPanel, .borderless], backing: .buffered, defer: false)
        p.level = .floating
        p.isFloatingPanel = true
        p.hidesOnDeactivate = false
        p.isOpaque = false
        p.backgroundColor = .clear
        p.hasShadow = true
        p.isMovableByWindowBackground = true
        p.collectionBehavior = [.canJoinAllSpaces, .fullScreenAuxiliary]

        let handler = Handler(record: { [weak self] in self?.dismiss(); onRecord() },
                              skip: { [weak self] in self?.dismiss(); onSkip() })
        self.handler = handler
        p.contentView = Card(spec: spec, handler: handler)

        guard let screen = NSScreen.main ?? NSScreen.screens.first else { return }
        let f = screen.visibleFrame
        let target = NSPoint(x: f.maxX - Self.width - Self.margin, y: f.maxY - Self.height - Self.margin)
        p.setFrameOrigin(NSPoint(x: target.x + 40, y: target.y))
        p.alphaValue = 0
        p.orderFrontRegardless()
        NSAnimationContext.runAnimationGroup { ctx in
            ctx.duration = 0.28
            ctx.timingFunction = CAMediaTimingFunction(name: .easeOut)
            p.animator().alphaValue = 1
            p.animator().setFrameOrigin(target)
        }
        panel = p

        guard spec.timeout > 0 else { return }
        // .common, so an open menu or a window resize does not stall the auto-dismiss
        let t = Timer(timeInterval: spec.timeout, repeats: false) { [weak self] _ in self?.dismiss(); onSkip() }
        RunLoop.main.add(t, forMode: .common)
        timer = t
    }

    /// Renders the card to a PNG (layout check without screen-recording permission).
    func snapshot(to url: URL) {
        guard let view = panel?.contentView,
              let rep = view.bitmapImageRepForCachingDisplay(in: view.bounds) else { return }
        view.cacheDisplay(in: view.bounds, to: rep)
        try? rep.representation(using: .png, properties: [:])?.write(to: url)
    }

    func dismiss(animated: Bool = true) {
        timer?.invalidate()
        timer = nil
        handler = nil
        guard let p = panel else { return }
        panel = nil
        if animated {
            NSAnimationContext.runAnimationGroup({ ctx in
                ctx.duration = 0.18
                p.animator().alphaValue = 0
            }, completionHandler: { p.orderOut(nil) })
        } else {
            p.orderOut(nil)
        }
    }

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

    /// Rounded translucent card: icon, title, subtitle and two buttons.
    private final class Card: NSVisualEffectView {
        init(spec: Spec, handler: Handler) {
            super.init(frame: NSRect(x: 0, y: 0, width: PromptPanel.width, height: PromptPanel.height))
            material = .popover
            blendingMode = .behindWindow
            state = .active
            wantsLayer = true
            layer?.cornerRadius = 16
            layer?.cornerCurve = .continuous
            layer?.masksToBounds = true
            layer?.borderWidth = 0.5
            layer?.borderColor = NSColor.separatorColor.cgColor

            let icon = NSImageView()
            icon.image = NSImage(systemSymbolName: spec.symbol, accessibilityDescription: spec.title)?
                .withSymbolConfiguration(.init(pointSize: 30, weight: .regular))
            icon.contentTintColor = spec.tint
            icon.translatesAutoresizingMaskIntoConstraints = false

            let title = NSTextField(labelWithString: spec.title)
            title.font = .systemFont(ofSize: 14, weight: .semibold)
            title.textColor = .labelColor
            title.translatesAutoresizingMaskIntoConstraints = false

            let subtitle = NSTextField(labelWithString: spec.subtitle)
            subtitle.font = .systemFont(ofSize: 12)
            subtitle.textColor = .secondaryLabelColor
            subtitle.translatesAutoresizingMaskIntoConstraints = false

            let skip = NSButton(title: spec.secondary, target: handler, action: #selector(Handler.skip(_:)))
            skip.bezelStyle = .rounded
            skip.controlSize = .large
            skip.translatesAutoresizingMaskIntoConstraints = false

            let record = NSButton(title: spec.primary, target: handler, action: #selector(Handler.record(_:)))
            record.bezelStyle = .rounded
            record.controlSize = .large
            record.bezelColor = .controlAccentColor
            record.translatesAutoresizingMaskIntoConstraints = false

            for v in [icon, title, subtitle, skip, record] { addSubview(v) }
            NSLayoutConstraint.activate([
                icon.leadingAnchor.constraint(equalTo: leadingAnchor, constant: 18),
                icon.topAnchor.constraint(equalTo: topAnchor, constant: 18),
                icon.widthAnchor.constraint(equalToConstant: 34),
                icon.heightAnchor.constraint(equalToConstant: 34),

                title.leadingAnchor.constraint(equalTo: icon.trailingAnchor, constant: 14),
                title.topAnchor.constraint(equalTo: topAnchor, constant: 18),
                title.trailingAnchor.constraint(lessThanOrEqualTo: trailingAnchor, constant: -18),

                subtitle.leadingAnchor.constraint(equalTo: title.leadingAnchor),
                subtitle.topAnchor.constraint(equalTo: title.bottomAnchor, constant: 3),
                subtitle.trailingAnchor.constraint(lessThanOrEqualTo: trailingAnchor, constant: -18),

                record.trailingAnchor.constraint(equalTo: trailingAnchor, constant: -16),
                record.bottomAnchor.constraint(equalTo: bottomAnchor, constant: -14),
                record.widthAnchor.constraint(greaterThanOrEqualToConstant: 104),

                skip.trailingAnchor.constraint(equalTo: record.leadingAnchor, constant: -8),
                skip.centerYAnchor.constraint(equalTo: record.centerYAnchor),
                skip.widthAnchor.constraint(greaterThanOrEqualToConstant: 104),
            ])
        }

        required init?(coder: NSCoder) { nil }
    }
}
