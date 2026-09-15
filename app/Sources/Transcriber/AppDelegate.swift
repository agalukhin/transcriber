import AppKit
import TranscriberCore

final class AppDelegate: NSObject, NSApplicationDelegate {
    private let log = Log.shared
    private let status = StatusItem()
    private let prompt = PromptPanel()
    private let floating = FloatingButton()
    private let controller = SessionController()
    private let store = SessionStore(root: SessionStore.defaultRoot)
    private let detector = MeetingDetector(lister: AudioProcesses.list)
    private var pollTimer: Timer?
    private var termSource: DispatchSourceSignal?
    /// `detector` is touched only on `pollQueue`; the main thread works off this snapshot.
    private let pollQueue = DispatchQueue(label: "transcriber.poll")
    private var activeApp: String?
    private var declinedApp: String?   // «Не сейчас» for this meeting: don't ask again until it ends
    /// Last logged set of mic holders; `pollQueue` only. `nil` = nothing logged yet.
    private var lastMicHolders: Set<String>?

    func applicationDidFinishLaunching(_ notification: Notification) {
        log.info("app launched")
        PowerHold.resetAtLaunch()   // a crash mid-session could have left sleep disabled
        _ = TalkInspector.ensureAccessibility()
        status.onToggle = { [weak self] in self?.toggle() }
        status.onOpenNote = { [weak self] in self?.openNote() }
        status.onOpenSessions = { [weak self] in
            guard let self else { return }
            NSWorkspace.shared.open(self.store.root)
        }
        floating.onClick = { [weak self] in self?.openNote() }
        controller.onStateChange = { [weak self] state in
            DispatchQueue.main.async { self?.render(state) }
        }
        controller.onError = { [weak self] message in
            DispatchQueue.main.async { self?.status.setProblem(message) }
        }
        installTermHandler()
        // .common, not the default mode: an open menu would otherwise freeze the detector.
        let timer = Timer(timeInterval: 2, repeats: true) { [weak self] _ in self?.tick() }
        RunLoop.main.add(timer, forMode: .common)
        pollTimer = timer
        NSWorkspace.shared.notificationCenter.addObserver(forName: NSWorkspace.didWakeNotification, object: nil,
                                                          queue: .main) { [weak self] _ in
            self?.log.info("woke from sleep")
            self?.controller.checkHealth(now: Date())   // the tap rarely survives sleep
            self?.poll()
        }
        poll()
    }

    /// launchd sends SIGTERM on «выход»/logout; the default disposition would kill us mid-session,
    /// leaving mic.pcm without a STOP file and the worker waiting forever.
    private func installTermHandler() {
        signal(SIGTERM, SIG_IGN)
        let source = DispatchSource.makeSignalSource(signal: SIGTERM, queue: .main)
        source.setEventHandler { [weak self] in
            self?.log.info("SIGTERM: stopping session and exiting")
            self?.controller.stop()
            exit(0)
        }
        source.resume()
        termSource = source
    }

    func applicationWillTerminate(_ notification: Notification) {
        controller.stop()
    }

    private func tick() {
        controller.checkHealth(now: Date())
        showIdleMeetingHint()
        poll()
    }

    /// Meeting running, nothing being recorded (declined or the prompt timed out) — say so in the menu.
    private func showIdleMeetingHint() {
        guard activeApp != nil, controller.current == nil else { return }
        status.setHint("Встреча идёт, запись не ведётся")
    }

    /// `detector.poll` lists audio processes — a CoreAudio round trip that has no business
    /// blocking the main thread; only the resulting event is handled there.
    private func poll() {
        pollQueue.async { [weak self] in
            guard let self else { return }
            let processes = AudioProcesses.list()
            self.logMicHolders(processes)
            let event = self.detector.poll(now: Date(), processes: processes)
            let active = self.detector.activeApp
            DispatchQueue.main.async { self.handle(event, activeApp: active) }
        }
    }

    /// Who is holding the mic right now — the only way to learn which app to watch after a
    /// meeting goes undetected. Runs every 2 s, so it logs only when the set changes.
    /// `pollQueue` only.
    private func logMicHolders(_ processes: [AudioProcess]) {
        let holders = Set(processes
            .filter { $0.isRunningInput && !$0.bundleID.hasPrefix(AudioProcesses.ownBundleID) }
            .map { $0.bundleID.isEmpty ? "pid:\($0.pid)" : $0.bundleID })
        guard holders != lastMicHolders else { return }
        lastMicHolders = holders
        let list = holders.isEmpty ? "none" : "[" + holders.sorted().joined(separator: ", ") + "]"
        log.info("mic holders: \(list)")
    }

    private func handle(_ event: MeetingEvent?, activeApp: String?) {
        self.activeApp = activeApp
        guard let event else { return }
        switch event {
        case .started(let app):
            log.info("meeting detected in \(app)")
            if controller.current != nil || declinedApp == app { return }
            prompt.show(appName: AppDelegate.promptName(for: app),
                        onRecord: { [weak self] in self?.controller.start(app: app) },
                        onSkip: { [weak self] in self?.declinedApp = app })
        case .ended:
            log.info("meeting ended")
            declinedApp = nil
            prompt.dismiss()
            controller.stop()
            render(controller.state)   // drops the «встреча идёт» hint
        }
    }

    /// A call in a browser tab (Толк web, Meet) is indistinguishable from any other tab from
    /// the outside, so the prompt names the browser rather than a meeting app.
    static func promptName(for app: String) -> String {
        switch app {
        case TalkInspector.bundleID: return "Толке"
        case "us.zoom": return "Zoom"
        default: return "браузере"
        }
    }

    private func toggle() {
        if controller.current != nil {
            controller.stop()
            // The meeting itself is still live: remember the refusal instead of resetting the
            // detector, or the next poll would offer to record again straight away.
            declinedApp = activeApp
        } else {
            prompt.dismiss()
            controller.start(app: activeApp ?? "manual")
        }
    }

    private func openNote() { openNote(retrying: true) }

    /// The worker writes `note_path` a beat after the session starts, so the first click can
    /// legitimately find nothing — that is a hint, not a problem; one retry usually lands.
    private func openNote(retrying: Bool) {
        guard let dir = controller.current else { return }
        if Obsidian.open(path: dir.notePath()) {
            if !retrying { render(controller.state) }
            return
        }
        log.warn("note path not ready yet for \(dir.id)")
        guard retrying else {
            render(controller.state)
            return
        }
        status.setHint("Заметка ещё создаётся воркером")
        DispatchQueue.main.asyncAfter(deadline: .now() + 2) { [weak self] in
            self?.openNote(retrying: false)
        }
    }

    private func render(_ state: SessionController.State) {
        switch state {
        case .idle:
            status.setRecording(false, detail: nil)   // also clears a problem glyph or a hint
            floating.hide()
        case .recording(let dir):
            status.setRecording(true, detail: dir.id)
            floating.show()
        }
    }
}
