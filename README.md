# AR Glasses Rig

A four-camera rig clipped onto XREAL One Pro see-through glasses. It measures how the glasses sit on the wearer's face and where real things are in front of them, so an overlay lands on the world instead of near it. Designed, built and measured by Dylan Leitenberg, 2026.

This repository is a record of what the rig does and what it measured. The design files, CAD, wiring, part list and code are not published. Photographs and renders are at [parahuman.net](https://parahuman.net).

## What it does

- **Real-world stereo overlay.** Left- and right-eye images rendered for the see-through display; black is transparent.
- **Eye tracking from inside.** Two cameras look at the wearer's eyes and track the inner eye corner with a model trained on 10,000 synthetic eye frames. From that the rig knows how the glasses are seated, every frame, and corrects the overlay for it.
- **Metric depth.** Two world cameras measure the distance to real objects, which the glasses alone cannot do.
- **Person-to-character replacement.** A tracked person is replaced by a rendered character at their position, scale and pose, in both eyes, live on the worn rig.
- **Live document translation.** A printed page is read on-device, translated, and the translation is drawn back onto the page at its measured distance and tilt. 24 languages offline.
- **Calibration that refuses.** A preflight names the precondition that failed. Depth reports its uncertainty and declines when it cannot be trusted.
- **Adaptive performance.** A frame-budget controller with an effectiveness probe that reverts any change that does not buy frame time.
- **Release gate.** 40 automated checks across the system; 23 self-tests on the translation path.

## Measured on this hardware

- Eye-corner tracking: 4 px in simulation, 5 px on hardware.
- Reading: 25 to 28 lines of a printed page at 26 to 35 cm. A tilted page stays registered: depth within 5 mm and the page normal within 3 degrees on the rehearsal clip.
- Person replacement at about 30 fps, worn.
- Loop latency 73.6 ms motion to photon before inertial reprojection.
- Four camera streams share one USB 2.0 hub. Three native streams at about 37 fps is the ceiling; four run at reduced resolution.

## Hardware, at product level

- XREAL One Pro glasses, driven as a 1920 by 1080 second display.
- Two ELP AR0234 global-shutter world cameras.
- Two OV9281 monochrome NoIR global-shutter eye cameras.
- One 3D-printed carrier holding all four cameras; one USB hub.

## Status

Worn and running. One camera of the original plan never shipped; everything above was done with two eye cameras and two world cameras.
