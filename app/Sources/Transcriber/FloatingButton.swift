import AppKit

/// Small always-on-top round button at the right edge of the screen; click opens the meeting note.
final class FloatingButton {
    private var panel: NSPanel?
    var onClick: (() -> Void)?

    /// `private` paints the dot purple — the same signal the menu and the prompt card give, so a
    /// glance at the button is enough to know where this recording is going.
    func show(private isPrivate: Bool = false) {
        if let panel, let view = panel.contentView as? DotView {
            view.isPrivate = isPrivate
            return
        }
        let size: CGFloat = 44
        let p = NSPanel(contentRect: NSRect(x: 0, y: 0, width: size, height: size),
                        styleMask: [.nonactivatingPanel, .borderless], backing: .buffered, defer: false)
        p.level = .floating
        p.isFloatingPanel = true
        p.hidesOnDeactivate = false
        p.isOpaque = false
        p.backgroundColor = .clear
        p.hasShadow = true
        p.isMovableByWindowBackground = true
        p.collectionBehavior = [.canJoinAllSpaces, .fullScreenAuxiliary]
        let view = DotView(frame: NSRect(x: 0, y: 0, width: size, height: size))
        view.isPrivate = isPrivate
        view.onClick = { [weak self] in self?.onClick?() }
        p.contentView = view
        if let screen = NSScreen.main ?? NSScreen.screens.first {   // .main is nil with no key window
            let f = screen.visibleFrame
            p.setFrameOrigin(NSPoint(x: f.maxX - size - 8, y: f.midY))
        }
        p.orderFrontRegardless()
        panel = p
    }

    func hide() {
        panel?.orderOut(nil)
        panel = nil
    }

    private final class DotView: NSView {
        var onClick: (() -> Void)?
        var isPrivate = false { didSet { if isPrivate != oldValue { needsDisplay = true } } }
        private var downAt: NSPoint = .zero

        override func draw(_ dirtyRect: NSRect) {
            let bg = NSBezierPath(ovalIn: bounds.insetBy(dx: 2, dy: 2))
            NSColor.windowBackgroundColor.withAlphaComponent(0.95).setFill()
            bg.fill()
            NSColor.separatorColor.setStroke()
            bg.lineWidth = 1
            bg.stroke()
            let dot = NSBezierPath(ovalIn: NSRect(x: bounds.midX - 7, y: bounds.midY + 3, width: 14, height: 14))
            (isPrivate ? NSColor.systemPurple : NSColor.systemRed).setFill()
            dot.fill()
            let text = "✎" as NSString
            let attrs: [NSAttributedString.Key: Any] = [.font: NSFont.systemFont(ofSize: 13), .foregroundColor: NSColor.labelColor]
            let s = text.size(withAttributes: attrs)
            text.draw(at: NSPoint(x: bounds.midX - s.width / 2, y: bounds.midY - 15), withAttributes: attrs)
        }

        override func mouseDown(with event: NSEvent) {
            downAt = NSEvent.mouseLocation
            super.mouseDown(with: event)   // lets isMovableByWindowBackground drag the panel
        }

        override func mouseUp(with event: NSEvent) {
            let up = NSEvent.mouseLocation
            if hypot(up.x - downAt.x, up.y - downAt.y) < 4 { onClick?() }
        }
    }
}
