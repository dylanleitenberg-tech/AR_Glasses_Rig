# AR Glasses Rig

A four-camera rig clipped onto XREAL One Pro see-through glasses. It measures where real things are in front of the wearer and how the glasses sit on the face, so graphics land on the world instead of near it. Worn and running. Designed, built and measured by Dylan Leitenberg, 2026; software written with AI coding agents under his direction.

This repository is a record of what the rig does and what it measured. The design files, CAD, wiring, part list and code are not published. Photographs, including one taken through the eyepiece, are at [parahuman.net](https://parahuman.net).

## Results

- **Live translation, worn:** panels drawn on 92% of frames and the page tracked on 93% over a 3.5-minute run.
- **Overlay accuracy: about 4–5 px** (px = pixels of the glasses' 1920-pixel-wide image) when calibrated and tested in the same wearing (18 August, 33 held-out alignments).

From → to:

- **Depth to a held page: stereo depth agrees within 1–6% with the distance solved from the printed page at 41–49 cm** (before the fix: 1.48x too far), live, on a later worn run; 7–12% short at 60–65 cm. The two world cameras had moved relative to each other in the mount and were re-calibrated from the printed page (the 1.48x is the median over a worn run's recording).
- **Reading a page in daylight: 0 lines → 17–18 lines per read**, on worn runs the same day. Every frame had blown out to white; after an exposure-control fix, the page held square to the camera was read.
- **Stutter in the translation display: 1 frame in 20 took 141 ms or longer → 23 ms (95th percentile)** on worn runs (median 17 ms in both). The wearer saw the panels jump; the step causing it was moved off the drawing loop.
- **Glasses-slip sensing: 8–37 px of jitter → 0.3–1.9 px** in the eye-camera image (5–7 held-out frames; steadiness with the glasses held still. Slip-correction accuracy not yet tested). An eye-corner landmark model was replaced by tracking patches of skin that move with the skull, not the eye.

Also measured:

- **Reading:** 25–28 lines of a printed page in one read (some character errors), at about 25 cm.
- **Person replacement, worn:** the character drawn in 89% of rendered frames, with a new pose every 38 ms (one person in view).

## What it does

- **Stereo overlay** on see-through glasses: separate left- and right-eye images; black is transparent.
- **Metric depth** from two world cameras; the glasses do not measure depth on their own. A live check prints the distance solved from the printed page beside the rig's stereo depth every 5 seconds; it caught later camera shifts during worn runs.
- **Live document translation.** A printed page is read and translated offline on the laptop driving the glasses, then drawn back onto the page at its measured distance and tilt and tracked between reads. 24 languages supported; worn tests in Spanish and German.
- **Person-to-character replacement.** A tracked person is replaced by a rendered character at their position, scale and pose, in both eyes.
- **Slip sensing.** Two inward cameras watch the skin around the eyes to measure the glasses slipping on the face. The correction is wired into the overlay; the 4–5 px result above did not use it.
- **Tests.** The translation self-test grew from 23 checks to 45, all passing at its last reported run (4 October).

## Hardware

- XREAL One Pro see-through glasses, driven as a second monitor at 1920 x 1080 per eye; their built-in motion sensor is read directly to lead head rotation.
- Two ELP AR0234 global-shutter world cameras (the stereo pair).
- Two OV9281 monochrome NoIR global-shutter eye cameras.
- One 3D-printed carrier holding all four cameras; a laptop runs everything over USB.

## Status

Worn and running; latest recorded worn runs 4 October 2026. All measurements are on one wearer. One camera of the original plan never shipped; everything above was done with two eye cameras and two world cameras.

Timeline: simulation, CAD and test tooling first committed 17 July 2026; first worn measurements 3 August; overlay accuracy of about 4–5 px in one wearing on 18 August; person replacement worn 30 August; translation worn 17 September; depth re-calibrated from the page 4 October.
