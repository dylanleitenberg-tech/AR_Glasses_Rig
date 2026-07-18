# Order list, every part, with links

Sourced for **Canadian delivery**. Confirmed products link to the vendor; everything else is an
**Amazon.ca search link** (always resolves to current in‑stock Canadian listings, no stale
URLs). Verify price/stock/variant at checkout. **CORE** = 6‑camera binocular build; **▲** = upgrades
(stereo 8‑cam accuracy upgrade; Jetson; kinematic mount; power meter).

> Camera variant warning: get the **USB / UVC** version of every camera (for the Mac). The
> Arducam OV9281 and ELP AR0234 also come in **MIPI/CSI** versions, those are for a Pi/Jetson.

## A. Display + compute
| ✔ | tier | part | qty | link |
|---|---|---|---|---|
| ☐ | core | **XREAL One Pro** (US$599; 1080p, 57° diag) | 1 | https://us.shop.xreal.com/products/xreal-one-pro · https://www.amazon.ca/s?k=XREAL+One+Pro |
| ☐ | core | Mac (host), runs the **6‑cam binocular CORE** over 1-2 powered USB3 hubs | have |, |
| ☐ | ▲ | NVIDIA Jetson Orin Nano dev kit, for the **8‑cam FULL** (USB bandwidth; cameras stay UVC) | 1 | https://www.amazon.ca/s?k=NVIDIA+Jetson+Orin+Nano+developer+kit |

## B. Cameras + lenses , **BINOCULAR** (XREAL One Pro has a display PER EYE; both eyes are tracked)
| ✔ | tier | part | qty | link |
|---|---|---|---|---|
| ☐ | core | **Arducam OV9281** 1MP global‑shutter **mono NoIR USB**: eye‑corner ×2 + NIR pupil ×2 (**one per eye**) | 4 | https://www.arducam.com/product/arducam_ov9281_1mp_global_shutter_usb_camera_evaluation_kit_ek026/ · https://www.amazon.ca/s?k=Arducam+OV9281+USB+global+shutter+mono |
| ☐ | ▲ | OV9281 NoIR USB, **stereo** eye‑corner pair (one per eye; the FULL accuracy upgrade) | +2 | (same link) |
| ☐ | core | **ELP AR0234** 2MP global‑shutter wide‑angle **USB** world cam (**shared** by both eyes) | 2 | https://www.amazon.ca/s?k=ELP+AR0234+global+shutter+USB+camera |
| ☐ | core | M12 lens assortment (90° eye / 55° pupil / 70° world, if stock lenses don't match) | 1 |
| ☐ | core | M12 **lens locking rings** (verify included with the camera modules first) or removable threadlocker, lock focus after calibration | 1 | https://www.amazon.ca/s?k=M12+lens+kit+camera+module |

> **Camera count:** binocular CORE = **6 cams** (2 world + 2 eye‑corner + 2 NIR pupil); FULL =
> **8 cams** (+2 stereo). Honest accuracy: ~4.3 px PERCEIVED deployed (vernier UI + per-user offset, simulated with realistic user error), 0.89 px pipeline bound; the <1 px pathway (stereo + multi-vergence kappa) awaits hardware validation. 2 world + 2 eye‑corner already register
> BOTH eyes; the 2 NIR cams add per‑eye geometry‑ID + the inter‑eye (dichoptic) alignment.

> **Module dimensions the printed holders are built for (verify with calipers before printing):**
> ELP AR0234 USB = **38 × 38 mm** PCB + M12 lens; Arducam OV9281 USB = **36 × 36 mm** PCB (≈28 mm
> mount‑hole pattern) + M12. The carrier (`cad/xreal_one_mount.scad`) holds each as a real board:
> a backing plate + 4 M2 standoffs at the mount pitch, lens clearance, cable notch, NOT an abstract
> pocket. `software/cad_fit.py` verifies every real board clears both the see‑through cone AND the
> eyeball at its rig.py position (the camera positions were nudged so the 36-38 mm boards fit; the
> 24 mm **Mini OV9281 is MIPI** and would need the Jetson, so it's not the USB/Mac path).

## C. NIR illumination
| ✔ | tier | part | qty | link |
|---|---|---|---|---|
| ☐ | core | 940 nm IR LEDs, 3 mm (NOT 850), an IR ring around **BOTH** eyes (~6 per eye = 12) | 1 pack | https://www.amazon.ca/s?k=940nm+IR+LED+3mm |
| ☐ | core | IR‑pass filter 850-1000 nm, one per NIR **pupil** cam (**2** binocular; +stereo optional) | 2 | https://www.amazon.ca/s?k=IR+pass+filter+850nm+10mm |
| ☐ | core | Resistor kit (need 330-470 Ω for the LED strings **+ one 10 kΩ MOSFET-gate pull-down**) | 1 | https://www.amazon.ca/s?k=1%2F4W+resistor+assortment+kit |
| ☐ | core | **Seeed XIAO** (RP2040/SAMD21), strobes **both** LED rings + IMU I²C→USB bridge + safety watchdog | 1 | https://www.amazon.ca/s?k=Seeed+XIAO+RP2040 |
| ☐ | core | **2N7002 logic‑level MOSFET** (low‑side IR strobe switch, NOT a 2N2222 BJT) | 1 pack | https://www.amazon.ca/s?k=2N7002+MOSFET |
| ☐ | core | **300 mA PTC resettable polyfuse** (overcurrent protection on the IR branch) | 1 pack | https://www.amazon.ca/s?k=300mA+PTC+resettable+fuse |
| ☐ | core | **470 µF 10 V electrolytic caps** + 0.1 µF ceramics (5 V rail decoupling) | 1 pack | https://www.amazon.ca/s?k=470uF+10V+electrolytic+capacitor |
| ☐ | core | **TVS diode / 5 V clamp** (rail transient protection, voltage safeguard) | 1 pack | https://www.amazon.ca/s?k=5V+TVS+diode+SMBJ5.0A |
| ☐ | core | Perfboard | 1 | https://www.amazon.ca/s?k=perfboard+prototype+board |

## D. IMU
| ✔ | tier | part | qty | link |
|---|---|---|---|---|
| ☐ | core | **SparkFun ICM‑20948** 9‑DoF IMU (Qwiic), I²C | 1 | https://www.sparkfun.com/sparkfun-9dof-imu-breakout-icm-20948-qwiic.html · https://www.amazon.ca/s?k=ICM-20948+9DoF+IMU |

## E. Rigid mount / structural
| ✔ | tier | part | qty | link |
|---|---|---|---|---|
| ☐ | core | Rigid **resin** or **PC / nylon‑CF filament** (carrier; not plain PETG/PLA) | 1 | https://www.amazon.ca/s?k=nylon+carbon+fiber+filament · https://www.amazon.ca/s?k=rigid+3D+printer+resin |
| ☐ | core | **Carbon‑fibre rod 1.5 mm** (boom reinforcement) | 1 pack | https://www.amazon.ca/s?k=carbon+fiber+rod+1.5mm |
| ☐ | core | **Structural plastic‑bonding epoxy** (e.g. JB Weld PlasticWeld) + isopropyl + abrasive | 1 | https://www.amazon.ca/s?k=plastic+bonding+epoxy+JB+Weld |
| ☐ | core | M2 SELF-TAPPING screw assortment (6-10 mm; the printed bosses are bored 1.8 mm, heat-set inserts DON'T fit the d=5 bosses) | 1 | https://www.amazon.ca/s?k=M2+self+tapping+screws+assortment |
| ☐ | core | M3 thumbscrews (~8-12 mm) + M3 nuts, the 2 brow-clamp tighteners (screw_d 3.2 in the CAD) | 1 | https://www.amazon.ca/s?k=M3+thumb+screw+knurled |
| ☐ | ▲ | Kinematic kit: 6 mm precision steel balls + neodymium magnets (removable mount) | 1 | https://www.amazon.ca/s?k=6mm+precision+steel+balls · https://www.amazon.ca/s?k=neodymium+magnets+6mm |

## F. Wiring / connectivity
| ✔ | tier | part | qty | link |
|---|---|---|---|---|
| ☐ | core | **Industrial powered USB 3.0 hub w/ its own 12 V PSU**: the 5 V/3 A (15 W) main rail for the 6‑cam CORE (1 hub) | 1 | https://www.amazon.ca/s?k=industrial+USB+3.0+hub+12V+powered |
| ☐ | ▲ | 2nd powered USB 3.0 hub **or** Jetson, only for the **8‑cam FULL** (bandwidth) | +1 | https://www.amazon.ca/s?k=powered+USB+3.0+hub+4+port |
| ☐ | core | USB‑C multiport dock (free Mac ports) | 1 | https://www.amazon.ca/s?k=USB-C+hub+multiport+adapter |
| ☐ | core | Short USB camera cables / micro‑USB leads (one per cam: 6 core / 8 full) | ~8 | https://www.amazon.ca/s?k=short+micro+USB+cable+pack |
| ☐ | core | **24 or 26 AWG silicone wire**: the main 5 V trunk (hub → carrier, full 3 A) | 1 | https://www.amazon.ca/s?k=24AWG+silicone+wire+kit |
| ☐ | core | **30 AWG silicone wire** + heat‑shrink, individual component branches only | 1 | https://www.amazon.ca/s?k=30AWG+silicone+wire+kit |
| ☐ | core | JST‑SH connector kit | 1 | https://www.amazon.ca/s?k=JST+SH+connector+kit |
| ☐ | core | Kapton / foam tape | 1 | https://www.amazon.ca/s?k=Kapton+tape |

## G. Calibration + tools
| ✔ | tier | part | qty | link |
|---|---|---|---|---|
| ☐ | core | Digital calipers | 1 | https://www.amazon.ca/s?k=digital+calipers |
| ☐ | core | Checkerboard calibration target (rigid) | 1 | https://www.amazon.ca/s?k=camera+calibration+checkerboard+target |
| ~~☐~~ | ~~core~~ | ~~Wired gamepad~~, NOT NEEDED: Mac arrow keys drive the alignment UI (vernier.py: arrows 1 px, shift 0.25 px) | 0 | https://www.amazon.ca/s?k=8BitDo+wired+controller |
| ☐ | core | Soldering iron kit (if needed) | 1 | https://www.amazon.ca/s?k=soldering+iron+kit |
| ☐ | ▲ | IR / optical power meter (eye‑safety check) | 1 | https://www.amazon.ca/s?k=optical+power+meter |

## H. Comfort / face-contact padding
| ✔ | tier | part | qty | link |
|---|---|---|---|---|
| ☐ | core | Adhesive silicone nose pads (for the bridge pad recesses) | 1 | https://www.amazon.ca/s?k=adhesive+silicone+nose+pads+eyeglasses |
| ☐ | core | **Soft silicone / PORON foam pad stock**: pad **every** printed surface that touches the face (nose pads, any brow/cheek contact) | 1 | https://www.amazon.ca/s?k=adhesive+silicone+foam+pad+sheet |

Electronic components (resistors, MOSFETs, JST, IMU) are also on **digikey.ca / mouser.ca** if you
prefer a single‑order electronics supplier with guaranteed parts.


---

## Where to buy, best sources per part class (Canadian sourcing)

| Part class | First choice | Backup | Notes |
|---|---|---|---|
| XREAL One Pro | xreal.com direct | Amazon.ca | Direct gets current firmware/revision; check DDP (duties prepaid) at checkout |
| Arducam OV9281 USB (eye/pupil, NoIR) | Arducam official store (arducam.com) | UCTRONICS storefront on Amazon.ca | Official store guarantees the exact SKU (USB, NoIR, no IR-cut); Amazon faster but verify the listing is the USB UVC variant, not MIPI |
| ELP AR0234 USB (world) | ELP/Ailipu official AliExpress store | Amazon.ca ELP listings | AliExpress = the manufacturer, ~2-3 wk; Amazon ~1 wk but fewer lens options, confirm M12 wide lens variant |
| Small electronics: XIAO MCU, ICM-20948 breakout, 2N7002, resistors, 300 mA polyfuse, 940 nm LEDs, JST, perfboard | **Digi-Key Canada (digikey.ca)**: one cart for all of it | Mouser.ca | fast Canadian shipping, CAD pricing, no surprise duties, free ship over ~$100, put every loose component in this one order |
| M12 940 nm IR-pass filters | Arducam (with the cams) | AliExpress "M12 940nm IR pass" | Ordering with the cameras avoids thread-size guessing (M12x0.5) |
| Industrial powered USB 3.0 hub | StarTech (Canadian; startech.com or Amazon.ca) | Anker on Amazon.ca | Buy the 12 V self-powered industrial model, not a travel hub, the 5 V rail is the whole power budget |
| M2 self-tapping screws, M3 thumbscrews + nuts, zip ties | Amazon.ca assortment kits | Bolt Depot (US) for exact sizes | Kits cover the sizes; nothing exotic |
| CF rods 1.5 mm, silicone pads, structural epoxy | Local RC-hobby shop (carbon rod) + Amazon.ca |, | RC shops stock 1.5 mm carbon rod cut lengths; epoxy only needed at the later bonded phase |
| SLA/rigid printing (if not printing at home) | Local print farm / library makerspace (draft PLA) | JLC3DP / Craftcloud for the resin/nylon-CF final | Draft in PLA locally first; ship out only the final rigid print |
| Odds and ends: calipers, IR power meter, Kapton | Amazon.ca | Canadian Tire (calipers) | The IR power meter is the one "trust but verify" purchase, get one that lists a 940 nm calibration point |

**Ordering strategy:** three carts, (1) Digi-Key for every component-drawer item (fast, one
shipment), (2) the two camera vendors (longest lead, order FIRST), (3) Amazon.ca for the
mechanical/consumables. The cameras gate the schedule; everything else arrives faster.
