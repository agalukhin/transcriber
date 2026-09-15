import AppKit
import TranscriberCore

final class AppDelegate: NSObject, NSApplicationDelegate {
    private let log = Log.shared
    private let status = StatusItem()
    private let prompt = PromptPanel()
    private let floating = FloatingButton()
    private let controller = SessionController()
    private let store = SessionStore(root: SessionStore.defaultRoot)
    private lazy var detector = MeetingDetector(lister: AudioProcesses.list)
    private var pollTimer: Timer?
    private var declinedApp: String?   // «Не сейчас» for this meeting: don't ask again until it ends

    func applicationDidFinishLaunching(_ notification: Notification) {
        log.info("app launched")
        PowerHold.resetAtLaunch()   // a crash mid-session could have left sleep disabled
        _ = TalkInspector.ensureAccessibility()
        status.onToggle = { [weak self] in self?.toggle() }
        status.onOpenNote = { [weak self] in self?.openNote() }
        status.onOpenSessions = { [weak self] in
            NSWorkspace.shared.open(self!.store.root)
        }
        floating.onClick = { [weak self] in self?.openNote() }
        controller.onStateChange = { [weak self] state in
            DispatchQueue.main.async { self?.render(state) }
        }
        controller.onError = { [weak self] message in
            DispatchQueue.main.async { self?.status.setProblem(message) }
        }
        pollTimer = Timer.scheduledTimer(withTimeInterval: 2, repeats: true) { [weak self] _ in self?.poll() }
        NSWorkspace.shared.notificationCenter.addObserver(forName: NSWorkspace.didWakeNotification, object: nil,
                                                          queue: .main) { [weak self] _ in
            self?.log.info("woke from sleep")
            self?.poll()
        }
        poll()
    }

    func applicationWillTerminate(_ notification: Notification) {
        controller.stop()
    }

    private func poll() {
        guard let event = detector.poll(now: Date()) else { return }
        switch event {
        case .started(let app):
            log.info("meeting detected in \(app)")
            if controller.current != nil || declinedApp == app { return }
            let name = app == TalkInspector.bundleID ? "Толке" : "Zoom"
            prompt.show(appName: name,
                        onRecord: { [weak self] in self?.controller.start(app: app) },
                        onSkip: { [weak self] in self?.declinedApp = app })
        case .ended:
            log.info("meeting ended")
            declinedApp = nil
            prompt.dismiss()
            controller.stop()
        }
    }

    private func toggle() {
        if controller.current != nil {
            controller.stop()
            detector.reset()
        } else {
            prompt.dismiss()
            controller.start(app: detector.activeApp ?? "manual")
        }
    }

    private func openNote() {
        guard let dir = controller.current else { return }
        if !Obsidian.open(path: dir.notePath()) {
            log.warn("note path not ready yet for \(dir.id)")
            status.setProblem("Заметка ещё создаётся воркером")
        }
    }

    private func render(_ state: SessionController.State) {
        switch state {
        case .idle:
            status.setRecording(false, detail: nil)
            floating.hide()
        case .recording(let dir):
            status.setRecording(true, detail: dir.id)
            floating.show()
        }
    }
}
