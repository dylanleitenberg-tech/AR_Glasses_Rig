"""vernier.py — Phase 2: hyperacuity (vernier) alignment UI = the kappa-precision lever.

WHY: the whole <1px path is gated by how precisely the user can say "the overlay is on the
target" (rig.HUMAN_NOISE_SD, placeholder 0.004 norm ~ 4.3 px/correction, + 2% fat-fingers).
Human VERNIER acuity (judging collinearity / centering-in-a-gap) is 5-10x finer than
dot-on-dot covering — so the correction UI, not more sensors, is the cheapest accuracy lever
(KAPPA.md; kappa/vernier carries stereo from ~1.1 px to <1 px).

THREE PIECES, all runnable BEFORE the cameras arrive:

1) demo(): pygame practice + MEASUREMENT task on any display — including the XREAL One Pro
   itself (it's just a USB-C monitor). A simulated "world dot" appears at a hidden true
   position (optional slow drift = head sway); you align a vernier crosshair (4 segments
   converging on a central gap the dot must sit centred in) with coarse/fine nudges and
   confirm. After N trials it reports YOUR median/robust-SD alignment error in px and
   normalized units and writes data/vernier_noise.json — replacing the placeholder
   HUMAN_NOISE_SD with a measured value.       Run:  python3 main.py --vernier-demo

2) demo(sbs=True): DICHOPTIC variant for the binocular build — side-by-side 3D mode
   (set the One Pro to SBS): the REFERENCE half-pattern renders only in the left-eye half,
   the MOVABLE one only in the right-eye half; nulling the vertical offset measures/nulls
   inter-eye vertical disparity (fusion comfort needs < ~6-15 arcmin).
                                              Run:  python3 main.py --vernier-demo --sbs

3) vernier_test(): sim validation — same convergence protocol as kappa.py but with the
   per-correction noise/fat-finger rates swapped for vernier-level ones; reports overlay
   error vs #corrections for dot-nudge vs vernier. Run:  python3 main.py --vernier-test

The live run_loop integration point: replace the "nudge red dot, press approve" step with
VernierPattern.render + the same nudge keys; each confirm yields exactly the same
(features, label) sample the calibrator already consumes — only with less noise.
"""
import json
import os
import sys

import numpy as np

import autosim
import pixel_map
import rig
from autotrain import WarmCalibrator
from config import Config

_DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")

# vernier-level per-correction model (until demo() measures the real user):
# conservative 3x better than dot-nudge, and confirm-twice kills most gross errors.
VERNIER_NOISE_SD = 0.0013
VERNIER_GROSS_PROB = 0.002


# ======================================================================
#  sim validation: what does lower correction noise buy end-to-end?
# ======================================================================
def _convergence_arm(label, noise_sd, gross_prob, prior, poses, dots,
                     Ks, n_users, seed_base):
    """kappa.py-style convergence with the human-correction constants overridden."""
    saved = (rig.HUMAN_NOISE_SD, rig.HUMAN_GROSS_PROB)
    rig.HUMAN_NOISE_SD, rig.HUMAN_GROSS_PROB = noise_sd, gross_prob
    rows = []
    try:
        for K in Ks:
            geo, perc = [], []
            for i in range(n_users):
                s = autosim.Simulator(seed_base + i)
                subj = s.new_subject()
                X, Y, dev, guard = [], [], s.seat(), 0
                while len(X) < K and guard < K * 6:
                    guard += 1
                    dev = s.slip(dev)
                    o = s.observe(subj, dev, s.world_point())
                    if o is not None:
                        X.append(o[0]); Y.append(o[1])
                cal = WarmCalibrator(prior, Config())
                cal.fit(np.array(X), np.array(Y))
                ge, pe = [], []
                for dev in poses:
                    for P in dots:
                        g = s.ground_truth(subj, dev, P)
                        if g is None:
                            continue
                        pred = cal.predict(g[0])
                        ge.append(np.linalg.norm(pred - g[1]))
                        pe.append(np.linalg.norm(pred - (g[1] + subj.human_bias)))
                geo.append(np.median(ge)); perc.append(np.median(pe))
            rows.append((label, K, np.median(geo) * 1080, np.median(perc) * 1080))
    finally:
        rig.HUMAN_NOISE_SD, rig.HUMAN_GROSS_PROB = saved
    return rows


def vernier_test(n_users=12, Ks=(4, 8, 16, 32), seed_base=7100, verbose=True):
    """Dot-nudge baseline vs vernier-level correction noise, same protocol/prior."""
    if verbose:
        print("== vernier UI value test: overlay error vs #corrections ==")
        print("   (training the warm-start prior ...)")
    prior = pixel_map.train_prior(Config())
    poses = pixel_map.pose_grid()[::3]
    dots = pixel_map.dot_grid()[::10]
    arms = [
        ("dot-nudge (placeholder)", rig.HUMAN_NOISE_SD, rig.HUMAN_GROSS_PROB),
        ("vernier   (3x + no-gross)", VERNIER_NOISE_SD, VERNIER_GROSS_PROB),
    ]
    # a measured personal value (from demo()) beats both assumptions if present
    mp = os.path.join(_DATA, "vernier_noise.json")
    if os.path.exists(mp):
        with open(mp) as f:
            m = json.load(f)
        arms.append(("vernier   (MEASURED user)", float(m["noise_sd_norm"]), VERNIER_GROSS_PROB))
    rows = []
    for label, sd, gp in arms:
        rows += _convergence_arm(label, sd, gp, prior, poses, dots, Ks, n_users, seed_base)
        if verbose:
            for r in [r for r in rows if r[0] == label]:
                print("  %-26s K=%3d  vs-truth %6.2f px   vs-perceived %6.2f px"
                      % (r[0], r[1], r[2], r[3]))
    return rows


# ======================================================================
#  the pygame UI
# ======================================================================
class VernierPattern:
    """The render: 4 line segments converging on (x, y) with a central GAP — the target
    must sit centred in the gap; any offset breaks the segments' collinearity with it,
    which the eye judges at hyperacuity. Sub-pixel: drawn into a 4x supersampled tile
    blitted with smoothscale, so 0.25-px nudges are really displayed."""
    SS = 4          # supersample factor
    TILE = 96       # on-screen tile size (px) around the estimate

    def __init__(self, gap=9.0, arm=28.0, thick=1.6):
        self.gap, self.arm, self.thick = gap, arm, thick

    def draw(self, screen, pos, color=(255, 60, 60)):
        import pygame
        ss, T = self.SS, self.TILE
        tile = pygame.Surface((T * ss, T * ss), pygame.SRCALPHA)
        cx = (pos[0] % 1.0) * ss + (T // 2) * ss
        cy = (pos[1] % 1.0) * ss + (T // 2) * ss
        g, a, w = self.gap * ss, (self.gap + self.arm) * ss, max(1, int(self.thick * ss))
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            p0 = (cx + dx * g, cy + dy * g)
            p1 = (cx + dx * a, cy + dy * a)
            pygame.draw.line(tile, color, p0, p1, w)
        small = pygame.transform.smoothscale(tile, (T, T))
        screen.blit(small, (int(pos[0]) - T // 2, int(pos[1]) - T // 2))


def _measure_report(residuals_px, w):
    r = np.asarray(residuals_px)
    err = np.linalg.norm(r, axis=1)
    med = float(np.median(err))
    # robust per-axis SD (MAD): what rig.HUMAN_NOISE_SD models
    mad = float(np.median(np.abs(r - np.median(r, axis=0)))) * 1.4826
    sd_norm = mad / w
    print("\n== your measured alignment precision (%d trials) ==" % len(r))
    print("  median error : %.2f px" % med)
    print("  robust SD    : %.2f px/axis  ->  HUMAN_NOISE_SD ~ %.5f (placeholder %.5f)"
          % (mad, sd_norm, rig.HUMAN_NOISE_SD))
    out = {"n_trials": len(r), "median_err_px": med, "robust_sd_px": mad,
           "noise_sd_norm": sd_norm, "screen_w": w}
    os.makedirs(_DATA, exist_ok=True)
    with open(os.path.join(_DATA, "vernier_noise.json"), "w") as f:
        json.dump(out, f, indent=1)
    print("  saved -> data/vernier_noise.json (vernier_test picks it up automatically)")
    return out


def demo(trials=20, drift=True, sbs=False, fullscreen=True):
    """Practice + measure alignment precision. Put the window on the XREAL display for the
    real test. Keys: arrows = 1 px, SHIFT+arrows = 0.25 px, ENTER/SPACE = confirm,
    R = re-randomize, ESC/Q = quit early. SBS mode: reference dot on the LEFT half
    (left eye), your movable pattern on the RIGHT half (right eye) — null the VERTICAL
    offset you perceive when the two halves fuse."""
    import pygame
    pygame.init()
    flags = pygame.FULLSCREEN if fullscreen else 0
    screen = pygame.display.set_mode((0, 0) if fullscreen else (1280, 720), flags)
    W, H = screen.get_size()
    pygame.display.set_caption("vernier alignment — " + ("DICHOPTIC SBS" if sbs else "monocular"))
    font = pygame.font.SysFont(None, 22)
    clock = pygame.time.Clock()
    pat = VernierPattern()
    rng = np.random.default_rng()
    half = W // 2

    residuals = []
    t = 0
    while t < trials:
        # hidden true position (kept away from edges); in SBS the truth lives on the LEFT
        # half and the estimate on the RIGHT half at the same in-eye coordinates
        span_w = half if sbs else W
        true0 = np.array([rng.uniform(0.22, 0.78) * span_w, rng.uniform(0.22, 0.78) * H])
        est = true0 + rng.uniform(-25, 25, 2)
        phase = rng.uniform(0, 2 * np.pi, 2)
        t0 = pygame.time.get_ticks()
        confirming = True
        while confirming:
            dt = (pygame.time.get_ticks() - t0) / 1000.0
            truth = true0 + (3.0 * np.sin(0.4 * dt + phase) if drift else 0.0)
            step = 1.0
            mods = pygame.key.get_mods()
            if mods & pygame.KMOD_SHIFT:
                step = 0.25
            for ev in pygame.event.get():
                if ev.type == pygame.QUIT:
                    pygame.quit(); return None
                if ev.type == pygame.KEYDOWN:
                    if ev.key in (pygame.K_ESCAPE, pygame.K_q):
                        pygame.quit()
                        return _measure_report(residuals, W) if len(residuals) >= 5 else None
                    if ev.key == pygame.K_LEFT:   est[0] -= step
                    if ev.key == pygame.K_RIGHT:  est[0] += step
                    if ev.key == pygame.K_UP:     est[1] -= step
                    if ev.key == pygame.K_DOWN:   est[1] += step
                    if ev.key == pygame.K_r:
                        est = truth + rng.uniform(-25, 25, 2)
                    if ev.key in (pygame.K_RETURN, pygame.K_SPACE):
                        residuals.append(est - truth)
                        t += 1
                        confirming = False
            # held-key auto-repeat for smooth coarse motion
            keys = pygame.key.get_pressed()
            rep = 0.12 * step
            est[0] += rep * (keys[pygame.K_RIGHT] - keys[pygame.K_LEFT])
            est[1] += rep * (keys[pygame.K_DOWN] - keys[pygame.K_UP])

            screen.fill((8, 8, 10))
            if sbs:
                pygame.draw.line(screen, (40, 40, 48), (half, 0), (half, H), 1)
                dot_pos = truth                          # LEFT half -> left eye
                pat_pos = est + np.array([half, 0.0])    # RIGHT half -> right eye
            else:
                dot_pos = truth
                pat_pos = est
            pygame.draw.circle(screen, (190, 190, 185),
                               (int(dot_pos[0]), int(dot_pos[1])), 3)
            pat.draw(screen, pat_pos)
            msg = ("trial %d/%d   arrows=1px  shift=0.25px  enter=confirm  q=quit"
                   % (t + 1, trials))
            screen.blit(font.render(msg, True, (120, 120, 130)), (12, 10))
            pygame.display.flip()
            clock.tick(60)
    pygame.quit()
    return _measure_report(residuals, W)


if __name__ == "__main__":
    if "--demo" in sys.argv:
        demo(sbs="--sbs" in sys.argv)
    else:
        vernier_test()
