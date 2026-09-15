import AppKit
import TranscriberCore

final class AppDelegate: NSObject, NSApplicationDelegate {
    private let log = Log.shared
    private let status = StatusItem()
    private let prompt = PromptPanel()
    private let stopPrompt = PromptPanel()
    /// «Продолжить» on the silence card mutes it until someone speaks again.
    private var stopPromptMutedUntilVoice = false
    static let silencePromptAfter: TimeInterval = 180
    static let silenceStopAfter: TimeInterval = 600
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
    /// Browser holding the mic with nothing meeting-like open yet — we keep looking instead of
    /// asking about a YouTube tab. Main thread only.
    private var pendingBrowserApp: String?
    private var pendingTicks = 0
    /// Ticks are 2 s, so five of them is the ~10 s re-check of the browser's tabs.
    static let browserRecheckTicks = 5
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
        watchSilence(now: Date())
        showIdleMeetingHint()
        recheckPendingBrowser()
        poll()
    }

    /// A browser keeps the mic for minutes after a call and plays sound outside calls, so neither
    /// system flag marks the end of a meeting reliably. Silence on both channels does: after
    /// `silencePromptAfter` seconds ask, after `silenceStopAfter` stop without asking.
    private func watchSilence(now: Date) {
        guard controller.current != nil else {
            stopPrompt.dismiss()
            stopPromptMutedUntilVoice = false
            return
        }
        let silent = controller.silentFor(now: now)
        if silent < Self.silencePromptAfter {
            stopPromptMutedUntilVoice = false
            if stopPrompt.isShowing { stopPrompt.dismiss() }   // someone spoke again
            return
        }
        if silent >= Self.silenceStopAfter {
            log.info("silence \(Int(silent))s: stopping recording")
            stopPrompt.dismiss()
            stopRecordingKeepingMeeting()
            return
        }
        guard !stopPrompt.isShowing, !stopPromptMutedUntilVoice else { return }
        stopPrompt.show(PromptPanel.stop(minutes: Int(silent / 60)),
                        onPrimary: { [weak self] in self?.stopRecordingKeepingMeeting() },
                        onSecondary: { [weak self] in self?.stopPromptMutedUntilVoice = true })
    }

    /// Stop without touching the detector: the app may still hold the mic, and re-offering to
    /// record the same meeting would be noise.
    private func stopRecordingKeepingMeeting() {
        controller.stop()
        declinedApp = activeApp
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
            guard AppDelegate.browserApps.contains(app) else {
                prompt.show(appName: AppDelegate.promptName(for: app),
                            onRecord: { [weak self] in self?.controller.start(app: app) },
                            onSkip: { [weak self] in self?.declinedApp = app })
                return
            }
            askBrowser(app: app)
        case .ended:
            log.info("meeting ended")
            declinedApp = nil
            pendingBrowserApp = nil
            pendingTicks = 0
            prompt.dismiss()
            controller.stop()
            render(controller.state)   // drops the «встреча идёт» hint
        }
    }

    /// Chrome holds the mic for minutes after a call and any site can take it, so a browser on the
    /// mic is not yet a meeting: ask the browser what it has open, and only prompt on a call tab.
    private func askBrowser(app: String) {
        pollQueue.async { [weak self] in
            let info = BrowserTabs.meetingInfo(forBrowser: app)
            DispatchQueue.main.async {
                guard let self, self.controller.current == nil, self.declinedApp != app else { return }
                guard let info else {
                    if self.pendingBrowserApp != app {
                        self.log.info("browser mic without a meeting tab, waiting")
                    }
                    self.pendingBrowserApp = app
                    return
                }
                self.pendingBrowserApp = nil
                self.pendingTicks = 0
                self.log.info("browser meeting tab: '\(info.title)'")
                self.prompt.show(PromptPanel.meeting(appName: AppDelegate.promptName(for: app),
                                                     title: info.title),
                                 onPrimary: { [weak self] in self?.controller.start(app: app, info: info) },
                                 onSecondary: { [weak self] in
                                     self?.declinedApp = app
                                     self?.pendingBrowserApp = nil
                                 })
            }
        }
    }

    /// The call tab usually appears a beat after the mic does (and the Automation permission may
    /// only be granted mid-meeting), so keep re-asking while the browser holds the mic.
    private func recheckPendingBrowser() {
        guard let app = pendingBrowserApp else {
            pendingTicks = 0
            return
        }
        guard controller.current == nil, declinedApp != app, activeApp == app, !prompt.isShowing else {
            pendingBrowserApp = nil
            pendingTicks = 0
            return
        }
        pendingTicks += 1
        guard pendingTicks >= AppDelegate.browserRecheckTicks else { return }
        pendingTicks = 0
        askBrowser(app: app)
    }

    /// Everything watched that is not a native meeting app: for those the tabs are the only
    /// source of a title, and the only way to tell a call from any other site.
    static let browserApps = MeetingDetector.defaultApps.filter {
        $0 != TalkInspector.bundleID && $0 != "us.zoom"
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
            stopRecordingKeepingMeeting()
        } else {
            prompt.dismiss()
            pendingBrowserApp = nil
            pendingTicks = 0
            let app = activeApp ?? "manual"
            guard AppDelegate.browserApps.contains(app) else {
                controller.start(app: app)
                return
            }
            // Started by hand while a browser holds the mic — the tab still knows the title.
            pollQueue.async { [weak self] in
                let info = BrowserTabs.meetingInfo(forBrowser: app)
                DispatchQueue.main.async { self?.controller.start(app: app, info: info) }
            }
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
