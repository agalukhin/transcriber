import AppKit
import Foundation
import TranscriberCore

/// `Transcriber --record-test 8` records 8 seconds into a session and prints peak levels
/// per input to the log; used to verify capture + permissions without a meeting.
if let i = CommandLine.arguments.firstIndex(of: "--record-test") {
    let seconds = Double(CommandLine.arguments.dropFirst(i + 1).first ?? "8") ?? 8
    let log = Log.shared
    let store = SessionStore(root: SessionStore.defaultRoot.appendingPathComponent("record-test"))
    let dir = try! store.create(app: "manual", title: "record-test", participants: [])
    let mic = try! PcmWriter(url: dir.micURL)
    let sys = try! PcmWriter(url: dir.sysURL)
    let cap = TapCapture()
    var micMixer: Downmixer?
    var sysMixer: Downmixer?
    var micPeak: Float = 0
    var sysPeak: Float = 0
    try! cap.start { chunk in
        guard chunk.frames > 0 else { return }
        // Each mixer is built on its own: one failing must not leave the other permanently nil.
        if micMixer == nil {
            micMixer = try? Downmixer(inputRate: cap.sampleRate, inputChannels: max(chunk.mic.count / chunk.frames, 1))
        }
        if sysMixer == nil {
            sysMixer = try? Downmixer(inputRate: cap.sampleRate, inputChannels: max(cap.sysChannels, 1))
        }
        guard let mm = micMixer, let sm = sysMixer else {
            log.warn("record-test: downmixer unavailable (mic \(micMixer != nil), sys \(sysMixer != nil)), chunk skipped")
            return
        }
        micPeak = max(micPeak, chunk.mic.map(abs).max() ?? 0)
        sysPeak = max(sysPeak, chunk.sys.map(abs).max() ?? 0)
        mic.write(mm.process(interleaved: chunk.mic, frames: chunk.frames))
        sys.write(sm.process(interleaved: chunk.sys, frames: chunk.frames))
    }
    let say = Process()
    say.executableURL = URL(fileURLWithPath: "/usr/bin/say")
    say.arguments = ["-v", "Milena", "Проверка захвата системного звука транскрибатором, раз, два, три"]
    try? say.run()
    Thread.sleep(forTimeInterval: seconds)
    cap.stop()
    mic.close()
    sys.close()
    try? dir.requestStop()
    log.info("record-test: \(dir.url.path) mic peak \(micPeak) sys peak \(sysPeak) mic samples \(mic.samplesWritten) sys samples \(sys.samplesWritten)")
    print("record-test done: \(dir.url.path) mic peak \(micPeak) sys peak \(sysPeak)")
    exit(0)
}

/// `Transcriber --prompt-test [seconds]` shows the meeting prompt for a visual check, then exits.
if let i = CommandLine.arguments.firstIndex(of: "--prompt-test") {
    let seconds = Double(CommandLine.arguments.dropFirst(i + 1).first ?? "6") ?? 6
    let app = NSApplication.shared
    app.setActivationPolicy(.accessory)
    let prompt = PromptPanel()
    DispatchQueue.main.async {
        if CommandLine.arguments.contains("stop") {
            prompt.show(PromptPanel.stop(minutes: 3), onPrimary: { print("stop") }, onSecondary: { print("continue") })
        } else {
            prompt.show(appName: "браузере", onRecord: { print("record") }, onSkip: { print("skip") })
        }
    }
    if let j = CommandLine.arguments.firstIndex(of: "--prompt-png"), let path = CommandLine.arguments.dropFirst(j + 1).first {
        DispatchQueue.main.asyncAfter(deadline: .now() + 1.0) { prompt.snapshot(to: URL(fileURLWithPath: path)) }
    }
    DispatchQueue.main.asyncAfter(deadline: .now() + seconds) { prompt.dismiss(); exit(0) }
    app.run()
}

let app = NSApplication.shared
let delegate = AppDelegate()
app.delegate = delegate
app.setActivationPolicy(.accessory)
app.run()
