import Foundation
import TranscriberCore

/// Owns one recording at a time: creates the session dir, holds power, runs capture,
/// downmixes both inputs and appends them to mic.pcm / sys.pcm, and writes STOP at the end.
final class SessionController {
    enum State { case idle, recording(SessionDir) }

    private(set) var state: State = .idle
    var onStateChange: ((State) -> Void)?

    private let store: SessionStore
    private let power = PowerHold()
    private var capture: TapCapture?
    private var micMixer: Downmixer?
    private var sysMixer: Downmixer?
    private var micWriter: PcmWriter?
    private var sysWriter: PcmWriter?
    private let log = Log.shared

    init(store: SessionStore = SessionStore(root: SessionStore.defaultRoot)) {
        self.store = store
    }

    var current: SessionDir? {
        if case .recording(let dir) = state { return dir }
        return nil
    }

    func start(app: String) {
        guard case .idle = state else { return }
        let info = app == TalkInspector.bundleID ? TalkInspector.inspect() : MeetingInfo()
        do {
            let dir = try store.create(app: app, title: info.title, participants: info.participants)
            micWriter = try PcmWriter(url: dir.micURL)
            sysWriter = try PcmWriter(url: dir.sysURL)
            let cap = TapCapture()
            try cap.start { [weak self] chunk in self?.consume(chunk) }
            capture = cap
            sysMixer = try Downmixer(inputRate: cap.sampleRate, inputChannels: cap.sysChannels)
            micMixer = nil   // built on the first chunk, once the mic channel count is known
            power.acquire()
            state = .recording(dir)
            log.info("session \(dir.id) started (app \(app), title '\(info.title)')")
            onStateChange?(state)
        } catch {
            log.error("session start failed: \(error)")
            cleanup()
        }
    }

    func stop() {
        guard case .recording(let dir) = state else { return }
        capture?.stop()
        if let m = micMixer { micWriter?.write(m.finish()) }
        if let s = sysMixer { sysWriter?.write(s.finish()) }
        let micSamples = micWriter?.samplesWritten ?? 0
        cleanup()
        do { try dir.requestStop() } catch { log.error("STOP write failed: \(error)") }
        power.release()
        state = .idle
        log.info("session \(dir.id) stopped; mic \(micSamples) samples")
        onStateChange?(state)
    }

    private func cleanup() {
        capture = nil
        micWriter?.close()
        sysWriter?.close()
        micWriter = nil
        sysWriter = nil
        micMixer = nil
        sysMixer = nil
    }

    /// Runs on the capture queue.
    private func consume(_ chunk: CaptureChunk) {
        guard chunk.frames > 0, let capture else { return }
        if micMixer == nil {
            let channels = max(chunk.mic.count / chunk.frames, 1)
            micMixer = try? Downmixer(inputRate: capture.sampleRate, inputChannels: channels)
        }
        if let m = micMixer { micWriter?.write(m.process(interleaved: chunk.mic, frames: chunk.frames)) }
        if let s = sysMixer { sysWriter?.write(s.process(interleaved: chunk.sys, frames: chunk.frames)) }
    }
}
