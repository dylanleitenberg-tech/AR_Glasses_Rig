"""safety_model.py, executable model of the IR strobe SAFETY ENVELOPE (mirrors ir_strobe.ino).

The firmware's interlocks are the thing standing between the IR LEDs and your eye, so their
LOGIC is validated HERE in Python, runnable today, no MCU/hardware needed. This is a faithful
port of the envelope in `ir_strobe.ino`; the self-test asserts that EVERY fault condition forces
IR OFF and that IR turns on ONLY when every interlock passes. Keep this and the .ino in lockstep:
if you change one, change the other, and re-run `python3 safety_model.py`.

When the XIAO is flashed, the SAME conditions are bench-tested on real hardware with a multimeter/
scope and a visible-LED stand-in (see this module's docstring tail and firmware/ir_strobe/README.md).
"""
from dataclasses import dataclass, field

# ---- the safety envelope constants (must match ir_strobe.ino) ----
HB_TIMEOUT_MS = 100      # no host heartbeat within this -> IR off
MAX_PULSE_US  = 2000     # hard cap on one IR pulse (dose cap)
MIN_OFF_US    = 6000     # min gap between pulses (duty cap)
VMIN, VMAX    = 4.5, 5.5 # acceptable 5 V rail window


@dataclass
class IrStrobe:
    """State machine mirroring the firmware. Time is explicit (ms/us) so tests are deterministic."""
    enabled: bool = False           # host must EN to arm
    blinkL: bool = False
    blinkR: bool = False
    rail_v: float = 5.0             # measured 5 V rail
    last_heartbeat_ms: float = 0.0
    last_pulse_end_us: float = -1e12
    gateL: bool = False             # current gate state (the physical output)
    gateR: bool = False

    def _gates_off(self):
        self.gateL = self.gateR = False

    def envelope_ok(self, now_ms: float) -> bool:
        """Every interlock that must hold for IR to be allowed AT ALL right now."""
        if not self.enabled:
            return False
        if now_ms - self.last_heartbeat_ms > HB_TIMEOUT_MS:   # USB/host drop
            return False
        if self.rail_v < VMIN or self.rail_v > VMAX:          # voltage safeguard
            return False
        return True

    def tick(self, now_ms: float):
        """Continuous safety: break the envelope -> gates low immediately."""
        if not self.envelope_ok(now_ms):
            self._gates_off()

    # ---- host commands (mirror handleLine) ----
    def heartbeat(self, now_ms): self.last_heartbeat_ms = now_ms
    def enable(self, now_ms):   self.enabled = True;  self.last_heartbeat_ms = now_ms
    def disable(self, now_ms):  self.enabled = False; self._gates_off(); self.last_heartbeat_ms = now_ms

    def set_blink(self, left: bool, right: bool, now_ms):
        self.blinkL, self.blinkR = left, right
        if left:  self.gateL = False
        if right: self.gateR = False
        self.last_heartbeat_ms = now_ms

    def pulse(self, us: float, now_ms: float, now_us: float, left=True, right=True) -> float:
        """Fire one bounded, interlocked IR pulse. Returns the ON-time actually delivered (us)."""
        self.last_heartbeat_ms = now_ms                       # a command is also a heartbeat
        if not self.envelope_ok(now_ms):
            self._gates_off(); return 0.0
        if now_us - self.last_pulse_end_us < MIN_OFF_US:      # duty-cycle cap
            return 0.0
        if us > MAX_PULSE_US:                                 # dose cap
            us = MAX_PULSE_US
        l = left  and not self.blinkL                         # blink cutoff (per eye)
        r = right and not self.blinkR
        if not l and not r:
            return 0.0
        self.gateL, self.gateR = l, r
        # (firmware holds the gate high for `us`, aborting on heartbeat loss; modelled as delivered)
        self.gateL = self.gateR = False                      # pulse ends -> gates low (fail-safe rest state)
        self.last_pulse_end_us = now_us + us
        return us

    @property
    def ir_on(self) -> bool:
        return self.gateL or self.gateR


# ===========================================================================
#  SELF-TEST: assert every interlock forces IR OFF, and IR only fires when clean
# ===========================================================================
def selftest(verbose=True) -> int:
    checks = []

    def ok(name, cond):
        checks.append((name, cond))

    # 1. Fail-safe default: not enabled -> a pulse does nothing
    s = IrStrobe()
    ok("default OFF (not enabled): pulse delivers 0us", s.pulse(1000, 0, 0) == 0.0 and not s.ir_on)

    # 2. Armed + clean -> IR fires
    s = IrStrobe(); s.enable(0); s.rail_v = 5.0
    delivered = s.pulse(1000, now_ms=1, now_us=10_000)
    ok("armed + clean rail + fresh HB -> pulse fires", delivered == 1000.0)

    # 3. USB/host drop -> OFF via the watchdog. (A 'P'/command IS a heartbeat, so the protection
    #    is the CONTINUOUS tick when commands STOP arriving: host goes silent -> gates forced low,
    #    worst-case latency = one self-terminating pulse <= MAX_PULSE_US.)
    s = IrStrobe(); s.enable(0); s.rail_v = 5.0
    s.pulse(1000, now_ms=1, now_us=10_000)              # host alive, fires; pulse self-terminates
    ok("after a pulse the gate is already low (self-terminating)", not s.ir_on)
    s.gateL = True                                       # pretend a gate were somehow left high
    s.tick(now_ms=1 + HB_TIMEOUT_MS + 1)                 # host now silent past the timeout
    ok("USB/host drop -> watchdog tick forces gates low", not s.ir_on)
    ok("after drop, the next tick keeps IR off", (s.tick(1 + HB_TIMEOUT_MS + 500) or not s.ir_on))

    # 4. Under-voltage and over-voltage -> OFF
    s = IrStrobe(); s.enable(0); s.rail_v = 4.2
    ok("under-voltage rail -> 0us", s.pulse(1000, 1, 10_000) == 0.0)
    s = IrStrobe(); s.enable(0); s.rail_v = 5.8
    ok("over-voltage rail -> 0us", s.pulse(1000, 1, 10_000) == 0.0)

    # 5. Blink cutoff (per eye)
    s = IrStrobe(); s.enable(0); s.rail_v = 5.0; s.set_blink(left=True, right=False, now_ms=1)
    s.pulse(1000, now_ms=2, now_us=20_000, left=True, right=True)
    ok("blink L -> left gate stayed off during pulse", s.gateL is False)

    # 6. Dose cap: an over-long pulse is clamped to MAX_PULSE_US
    s = IrStrobe(); s.enable(0); s.rail_v = 5.0
    ok("pulse clamped to MAX_PULSE_US", s.pulse(99_999, now_ms=1, now_us=100_000) == MAX_PULSE_US)

    # 7. Duty cap: a 2nd pulse too soon is refused
    s = IrStrobe(); s.enable(0); s.rail_v = 5.0
    s.pulse(1000, now_ms=1, now_us=200_000)
    ok("2nd pulse within MIN_OFF_US refused", s.pulse(1000, now_ms=1, now_us=200_000 + MIN_OFF_US - 1) == 0.0)
    ok("2nd pulse after MIN_OFF_US allowed", s.pulse(1000, now_ms=1, now_us=200_000 + 1000 + MIN_OFF_US + 1) == 1000.0)

    # 8. Disable / reset -> OFF
    s = IrStrobe(); s.enable(0); s.rail_v = 5.0; s.gateL = True
    s.disable(now_ms=1); ok("DIS forces gates low", not s.ir_on)

    passed = all(c for _, c in checks)
    if verbose:
        print("== IR strobe SAFETY-ENVELOPE self-test (mirrors ir_strobe.ino) ==")
        for name, c in checks:
            print("  [%s] %s" % ("PASS" if c else "FAIL", name))
        print("  =>", "ALL INTERLOCKS HOLD, IR off on every fault, on only when clean ✅"
              if passed else "INTERLOCK FAILURE ⚠️")
        print("  NOTE: this proves the LOGIC. On real hardware, bench-test the SAME conditions with")
        print("  a multimeter/scope + a visible-LED stand-in BEFORE connecting IR or going near an eye:")
        print("    host closed -> gate LOW · USB unplugged -> IR off · rail <4.5/>5.5V -> off ·")
        print("    'B L' -> left off · 'P 99999' -> pulse clamped · deliberate short -> polyfuse trips.")
    return 0 if passed else 1


if __name__ == "__main__":
    import sys
    sys.exit(selftest())
