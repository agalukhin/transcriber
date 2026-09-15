import Foundation
import IOKit.pwr_mgt

/// Keeps the Mac awake while recording: a power assertion against idle sleep, plus (when the
/// sudoers rule exists) `pmset disablesleep` so a closed lid does not sleep either.
public final class PowerHold {
    private var assertion: IOPMAssertionID = 0
    private var held = false
    private let usePmset: Bool

    public init(usePmset: Bool = true) {
        self.usePmset = usePmset
    }

    public func acquire() {
        guard !held else { return }
        let status = IOPMAssertionCreateWithName(kIOPMAssertionTypePreventUserIdleSystemSleep as CFString,
                                                 IOPMAssertionLevel(kIOPMAssertionLevelOn),
                                                 "Транскрибатор пишет встречу" as CFString, &assertion)
        if status != kIOReturnSuccess { Log.shared.warn("power assertion failed: \(status)") }
        held = true
        if usePmset { pmset(disableSleep: true) }
    }

    public func release() {
        guard held else { return }
        IOPMAssertionRelease(assertion)
        assertion = 0
        held = false
        if usePmset { pmset(disableSleep: false) }
    }

    private func pmset(disableSleep: Bool) {
        let p = Process()
        p.executableURL = URL(fileURLWithPath: "/usr/bin/sudo")
        p.arguments = ["-n", "/usr/bin/pmset", "-a", "disablesleep", disableSleep ? "1" : "0"]
        p.standardOutput = FileHandle.nullDevice
        p.standardError = FileHandle.nullDevice
        do {
            try p.run()
            p.waitUntilExit()
            if p.terminationStatus != 0 {
                Log.shared.warn("pmset disablesleep \(disableSleep ? 1 : 0) skipped (exit \(p.terminationStatus)); add the sudoers rule to enable it")
            } else {
                Log.shared.info("pmset disablesleep \(disableSleep ? 1 : 0)")
            }
        } catch {
            Log.shared.warn("pmset launch failed: \(error)")
        }
    }
}
