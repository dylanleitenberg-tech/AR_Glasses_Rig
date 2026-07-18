# firmware/ir_strobe — IR strobe + safety interlock MCU

Reference firmware for the strobe MCU (**Seeed XIAO** RP2040/SAMD21, or any Arduino) that
drives the 940 nm IR LED rings and **enforces the eye-safety envelope**. It is the concrete
implementation of the interlocks documented in `WIRING.md` and `EYE_TRACKING.md §3`.

## What it does (all fail-safe — default IR OFF)

| Interlock | How |
|---|---|
| **Strobe in sync with the shutter** | IR fires only as a bounded **pulse** on a host `P <us>` command issued at each exposure |
| **USB/host drop kills IR** | heartbeat watchdog — no host line within `HB_TIMEOUT_MS` forces both gates low (and aborts a pulse mid-flight) |
| **Voltage safeguard** | the 5 V rail is read every loop; outside `[VMIN,VMAX]` IR is disabled |
| **IR cutoff on blinks** | host `B L|R|0` flags a closed eye; that ring is held off |
| **Dose / duty caps** | pulse clamped to `MAX_PULSE_US`; `MIN_OFF_US` enforces a low duty cycle |
| **Any anomaly / reset → OFF** | gates default LOW + an external **100 k gate pull-down** guarantees OFF through power-up and MCU reset |

The **300 mA PTC polyfuse** (overcurrent) and the **TVS clamp** (transients) are hardware on
the IR branch — see `WIRING.md`. The MOSFET is a **2N7002** (logic-level, not a 2N2222 BJT).

## Host serial protocol (115200 baud, newline-terminated)

```
HB            heartbeat (send at >= 20 Hz so the watchdog stays satisfied)
EN  / DIS     arm / disarm strobing
P <us>        fire one IR pulse of <us> microseconds at the current exposure (clamped)
B L | R | 0   blink state: left / right eye closed, or 0 = both open
```

Any recognized line also refreshes the heartbeat. The host loop: send `EN` once, then each
camera frame send `P <exposure_us>` right before grabbing, send `HB` if idle, and send `B …`
from `software/blink.py`. On exit send `DIS` (and just closing the port trips the watchdog).

## Wiring (summary — full topology in WIRING.md)

- `IR_GATE_L/R` → 2N7002 gates (each with a **100 k pull-down to GND**); LED ring sources on
  the drains, rings to the 5 V rail **through the 300 mA polyfuse**.
- `RAIL_SENSE` → the 5 V rail through a **2:1 divider** into an ADC pin (set `RAIL_DIV`/ADC
  scaling to your board; XIAO ADC ref 3.3 V).
- IR power pair **twisted + star-grounded**, routed away from the USB data cables.
- IMU (ICM-20948) on I2C → the XIAO bridges it to USB (stub at the bottom of the sketch).

## Bring-up

1. Flash, set the pin numbers / ADC scaling for your board.
2. **Verify fail-safe FIRST:** with no host connected, gates read LOW (IR off). Open a serial
   monitor: `EN` then `P 1000` pulses; stop sending and the watchdog cuts IR within
   `HB_TIMEOUT_MS`.
3. Only then enable in the live pipeline, at the lowest IR current that tracks (see
   `EYE_TRACKING.md §3`).
