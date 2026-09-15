import Foundation
import TranscriberCore

/// Owns one recording at a time: creates the session dir, holds power, runs capture,
/// downmixes both inputs and appends them to mic.pcm / sys.pcm, and writes STOP at the end.
///
/// Threading invariant (no locks needed beyond this):
/// writers/mixers are created before capture starts and torn down only after `TapCapture.stop()`
/// has drained its queue; `consume` is the only accessor on the capture queue.
/// The two mixers are the one exception to «created before capture starts»: their channel counts
/// are known only from the first chunk, so they are built inside `consume` itself — still on the
/// capture queue, still the only accessor. `capture`, `micWriter` and `sysWriter` are assigned
/// before `cap.start`, so the capture queue never observes them mutate once delivery begins;
/// `stop()` touches them only after `capture.stop()`'s `queue.sync {}` barrier has returned.
final class SessionController {
    enum State { case idle, recording(SessionDir) }

    private(set) var state: State = .idle
    var onStateChange: ((State) -> Void)?
    var onError: ((String) -> Void)?

    private let store: SessionStore
    private let power = PowerHold()
    private var capture: TapCapture?
    private var micMixer: Downmixer?
    private var sysMixer: Downmixer?
    private var micWriter: PcmWriter?
    private var sysWriter: PcmWriter?
    private var warnedNoMixer = false
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
        var created: SessionDir?
        do {
            let dir = try store.create(app: app, title: info.title, participants: info.participants)
            created = dir
            // Everything the capture queue reads is in place before the first chunk can arrive.
            micWriter = try PcmWriter(url: dir.micURL)
            sysWriter = try PcmWriter(url: dir.sysURL)
            micMixer = nil   // both mixers are built on the first chunk, once channel counts are known
            sysMixer = nil
            warnedNoMixer = false
            let cap = TapCapture()
            capture = cap
            try cap.start { [weak self] chunk in self?.consume(chunk) }
            power.acquire()
            state = .recording(dir)
            log.info("session \(dir.id) started (app \(app), title '\(info.title)')")
            onStateChange?(state)
        } catch {
            log.error("session start failed: \(error)")
            cleanup()
            if let dir = created { discardIfEmpty(dir) }
            state = .idle
            onStateChange?(state)
            onError?("Не удалось начать запись: \(error)")
        }
    }

    func stop() {
        guard case .recording(let dir) = state else { return }
        capture?.stop()   // returns only after the capture queue has drained: mixers/writers are ours again
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

    /// A session dir with no audio in it would keep the worker waiting forever, so drop it.
    private func discardIfEmpty(_ dir: SessionDir) {
        guard fileSize(dir.micURL) == 0, fileSize(dir.sysURL) == 0 else { return }
        do {
            try FileManager.default.removeItem(at: dir.url)
            log.warn("session \(dir.id) removed: capture never started, nothing written")
        } catch {
            log.error("session \(dir.id) removal failed: \(error)")
        }
    }

    private func fileSize(_ url: URL) -> Int {
        ((try? FileManager.default.attributesOfItem(atPath: url.path))?[.size] as? Int) ?? 0
    }

    /// Runs on the capture queue — the only place the mixers and writers are touched while recording.
    private func consume(_ chunk: CaptureChunk) {
        guard chunk.frames > 0, let capture else { return }
        if micMixer == nil {
            let channels = max(chunk.mic.count / chunk.frames, 1)
            micMixer = try? Downmixer(inputRate: capture.sampleRate, inputChannels: channels)
        }
        if sysMixer == nil {
            sysMixer = try? Downmixer(inputRate: capture.sampleRate, inputChannels: max(capture.sysChannels, 1))
        }
        // Both files are plain appends: writing one half of a chunk would shift that track against
        // the other for the rest of the session, so a missing mixer drops the whole chunk.
        guard let m = micMixer, let s = sysMixer else {
            if !warnedNoMixer {
                warnedNoMixer = true
                log.error("session: downmixer unavailable, dropping chunks (mic \(micMixer != nil), sys \(sysMixer != nil))")
            }
            return
        }
        micWriter?.write(m.process(interleaved: chunk.mic, frames: chunk.frames))
        sysWriter?.write(s.process(interleaved: chunk.sys, frames: chunk.frames))
    }
}
