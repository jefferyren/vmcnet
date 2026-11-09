#!/usr/bin/env python3
"""H1 report: Kaczmarz++ (the SS-SPRING probe) on the linear system frozen at snapshots
of a diverging stretched-N2 replay, next to the replay itself.

Needs, in order (Savio login node; see slurm/h1_log_to_wandb.py):
    wandb sync <the two h1-n2-dense-replay offline runs>
    python slurm/h1_log_to_wandb.py
Run:  python slurm/report_h1_frozen.py            (creates, prints the URL)
      python slurm/report_h1_frozen.py --url URL  (rebuilds in place)
Then: python ~/.claude/skills/wandb-experiment-report/scripts/verify_report.py <url>
"""

import argparse

import numpy as np
import wandb
import wandb_workspaces.reports.v2 as wr

ENTITY = "ren27-university-of-california-berkeley"
PH = "vmcnet-phase-h"
RUN = "e14s0"
E14_REPORT = (
    "https://wandb.ai/ren27-university-of-california-berkeley/vmcnet-phase-e/reports/"
    "Ada-SPRING-divergence-on-stretched-N2--VmlldzoxNzk0MDIzNw"
)
G_REPORT = (
    "https://wandb.ai/ren27-university-of-california-berkeley/vmcnet-phase-g/reports/"
    "Phase-G-—-original-SS-SPRING-on-strongly-correlated-systems--VmlldzoxODIwNTIxNQ=="
)
LEG = ["config:x_snapshot", "config:x_epoch"]
ARMS = (
    ("b0990", "beta = 0.99"),
    ("b0995", "beta = 0.995"),
    ("brun", "beta = the run's own (~0.996)"),
    ("adaptive", "fresh adaptive controller"),
)


def frozen_meta():
    """Marker y-ranges and the event epoch, from the logged frozen runs' config."""
    runs = list(
        wandb.Api(timeout=60).runs(
            f"{ENTITY}/{PH}", filters={"config.x_kind": "frozen", "config.x_run": RUN}
        )
    )
    if not runs:
        raise SystemExit(f"no frozen {RUN} runs in {PH}: run slurm/h1_log_to_wandb.py")
    c = runs[0].config
    ranges = {
        k: (c[f"x_marker_{k}_lo"], c[f"x_marker_{k}_hi"])
        for k in ("energy", "variance", "mu", "r_ip")
        if f"x_marker_{k}_lo" in c
    }
    return ranges, int(c["x_event"]), sorted(int(r.config["x_epoch"]) for r in runs)


def zoom_ranges(event):
    """y-ranges for the zoom panels from the replay's last 700 epochs before the event,
    so the approach is visible and the event itself runs off the top."""
    runs = list(
        wandb.Api(timeout=60).runs(
            f"{ENTITY}/{PH}", filters={"config.x_kind": "replay", "config.x_run": RUN}
        )
    )
    if not runs:
        raise SystemExit(f"no {RUN} replay run tagged in {PH}: run h1_log_to_wandb.py")
    keys = ["energy_noclip", "variance_noclip"]
    rows = list(runs[0].scan_history(keys=keys, min_step=event - 700, max_step=event))
    out = {}
    for k in keys:
        x = np.array([r[k] for r in rows if r.get(k) is not None], dtype=float)
        x = x[np.isfinite(x)]
        if not len(x):
            continue
        lo, hi = np.percentile(x, [0.5, 99.5])
        if k == "variance_noclip":
            out[k] = (float(lo / 2), float(hi * 5))
        else:
            out[k] = (float(lo - 0.5 * (hi - lo)), float(hi + 1.5 * (hi - lo)))
    return out


def rs(name, kind):
    f = (
        f"Config('x_experiment') == 'H1' and Config('x_kind') == '{kind}' "
        f"and Config('x_run') == '{RUN}'"
    )
    return wr.Runset(
        ENTITY,
        PH,
        name=name,
        filters=f,
        order=[wr.OrderBy(wr.Config("x_epoch"), ascending=True)],
    )


def grid(panels, sets, ncol=2):
    w = 24 // ncol
    for i, p in enumerate(panels):
        p.layout = wr.Layout(x=w * (i % ncol), y=8 * (i // ncol), w=w, h=8)
    return wr.PanelGrid(runsets=sets, panels=panels)


def line(title, y, x="Step", rx=None, ry=None, log=False, xlab=None, ylab=None):
    kw = {
        k: v
        for k, v in (
            ("range_x", rx),
            ("range_y", ry),
            ("title_x", xlab),
            ("title_y", ylab),
        )
        if v is not None
    }
    return wr.LinePlot(
        x=x,
        y=y if isinstance(y, list) else [y],
        title=title,
        log_y=log,
        legend_fields=LEG,
        legend_position="east",
        **kw,
    )


def bar(title, metrics, xlab):
    return wr.BarPlot(metrics=metrics, title=title, orientation="h", title_x=xlab)


def run_section(ranges, event, zr):
    full, zoom = (9900.0, 16050.0), (float(event - 700), float(event + 150))
    m = "vertical lines = where theta was frozen"
    return grid(
        [
            line(
                f"Energy of the replay; {m}",
                ["energy_noclip", "freeze/energy"],
                rx=full,
                ry=ranges.get("energy"),
                xlab="epoch",
                ylab="energy [Ha]",
            ),
            line(
                f"Zoom: the 700 epochs before the first event ({event}); {m}",
                ["energy_noclip", "freeze/energy"],
                rx=zoom,
                ry=zr.get("energy_noclip"),
                xlab="epoch",
                ylab="energy [Ha]",
            ),
            line(
                f"Local-energy variance (log); {m}",
                ["variance_noclip", "freeze/variance"],
                rx=full,
                ry=ranges.get("variance"),
                log=True,
                xlab="epoch",
                ylab="variance [Ha^2]",
            ),
            line(
                f"Zoom: variance (log) before the event; {m}",
                ["variance_noclip", "freeze/variance"],
                rx=zoom,
                ry=zr.get("variance_noclip"),
                log=True,
                xlab="epoch",
                ylab="variance [Ha^2]",
            ),
            line(
                f"The run's adaptive beta (logged as mu); {m}",
                ["mu", "freeze/mu"],
                rx=full,
                ry=ranges.get("mu"),
                xlab="epoch",
                ylab="beta",
            ),
            line(
                f"The run's own probe window ratio r_ip (log, > 1 = growing); {m}",
                ["probe_r_ip", "freeze/r_ip"],
                rx=full,
                ry=ranges.get("r_ip"),
                log=True,
                xlab="epoch",
                ylab="r_ip",
            ),
        ],
        [
            rs(f"{RUN} replay (the VMC run)", "replay"),
            rs("freeze points (one run per snapshot)", "frozen"),
        ],
    )


def probe_section():
    xs = dict(x="frozen_step", xlab="frozen step (theta fixed, fresh MCMC samples)")
    res = [
        line(
            f"Held-out residual, relative to step 0 (log): {t}",
            f"probe/rel_res_hold_{k}",
            log=True,
            ylab="||A(z - x*)|| / step 0",
            **xs,
        )
        for k, t in ARMS
    ]
    return grid(
        res
        + [
            line(
                "Fresh adaptive controller: the beta it chooses",
                "probe/beta_adaptive",
                ry=(0.95, 1.0),
                ylab="beta",
                **xs,
            ),
            line(
                "Window ratio r_ip (log): fresh adaptive controller",
                "probe/r_ip_adaptive",
                log=True,
                ylab="r_ip",
                **xs,
            ),
            line(
                "Window ratio r_ip (log): beta = the run's own",
                "probe/r_ip_brun",
                log=True,
                ylab="r_ip",
                **xs,
            ),
            line(
                "Training-block residual, relative (log): beta = the run's own "
                "(what the controller sees)",
                "probe/rel_res_brun",
                log=True,
                ylab="relative residual",
                **xs,
            ),
            line(
                "Gram max eigenvalue of the frozen operator (log)",
                "frozen/gram_max",
                log=True,
                ylab="eigenvalue",
                **xs,
            ),
            line(
                "Gram min eigenvalue (linear: float32 can push it below 0)",
                "frozen/gram_min",
                ylab="eigenvalue",
                **xs,
            ),
            line(
                "Gram eigenvalues below the probe damping",
                "frozen/gram_n_below_damping",
                ylab="count",
                **xs,
            ),
            bar(
                "Per-step factor g of the held-out residual, 2nd half (> 1 = diverging)",
                [f"x_g_{k}" for k, _ in ARMS],
                "g per step",
            ),
            bar(
                "r_ip: the run's own probe over the same epochs vs the frozen probe at the "
                "run's beta",
                ["x_inrun_rip", "x_rip_brun"],
                "r_ip (median)",
            ),
        ],
        [rs("frozen snapshots of E14 s0", "frozen")],
    )


DESCRIPTION = f"""\
**Follows from Phase F (F4d) and E14/E15** ([E14-E19 report]({E14_REPORT}),
[Phase G report]({G_REPORT})). SS-SPRING diverges on N2 at 4.0 Bohr (6/8 seeds) where
SPRING(0.99) never does. In the F4d replays the in-run probe started growing over the
last ~30 rows before every event (r_ip 1.0-2.2 rising to 12.6-930) while beta stayed
at 0.9957-0.9971. That left open whether the probe grows because the linear system
moves under it (theta changes every step) or because the system itself goes bad.

**Hypothesis (H1), framed before the runs.** In VMC, A_k changes from step to step,
which slows the probe; the adaptive controller reads a slow probe as "needs more
momentum" and pushes beta up; the high beta diverges. **What would confirm it:**
Kaczmarz++ on the system frozen at a snapshot (theta fixed, fresh MCMC rows each step)
converges at the run's beta near the event, while the run's own probe over the same
epochs stalls or grows, and the run's beta rises before the event.

**What ran.**
- E14 s0 was replayed from its 10k checkpoint on the original SS-SPRING code (tag
  `ss-spring-original`, E14's flags), with a checkpoint every 50 epochs. Its first
  event (energy > trailing median + 1 Ha and variance > 20x, as in
  `n2_failure_anatomy.py`) came at epoch 14,319. It never went to NaN; it ran to 16k
  about 3 Ha too high.
- theta was frozen at 5 snapshots: the reload state (9,999, "healthy") and the
  checkpoints at or before event-500, -200, -50 and -20.
- At each snapshot the probe recursion and controller, line for line the module's, ran
  for 3,000 steps on 1,000 walkers moved by the run's own MCMC each step. Arms: beta
  fixed at 0.99, 0.995 and the run's beta at that checkpoint, plus a FRESH adaptive
  controller (beta starts at 0), each with 2 random targets x*. Convergence is read on
  a fixed held-out walker set; r_ip uses the fresh block, as in the run.

**How the panels work.** Each frozen snapshot is its own wandb run. It draws a vertical
line at its epoch on the replay's panels (two points spanning the y-range), and its
probe curves are plotted against `frozen_step`.

**Scope.** One failing replay. E15 s7 was replayed the same way but did not diverge by
16k, so it is not shown. Float32, as in the runs. g is the slope of the log held-out
residual over the second half of the 3,000 steps, the geometric mean over the 2
targets.
"""

TAKEAWAYS = """\
- **H1 is not supported: the frozen system is worse than the moving one.** From 500
  epochs before the event, the probe diverges on the frozen system at every beta. Over
  the same epochs the run's own probe still contracts or holds (r_ip 0.95 and 1.02 at
  event-500 and -200); it only grows in the last ~50 epochs:

  | snapshot | epoch | run's probe r_ip | frozen g at 0.99 / 0.995 / run's beta | fresh controller's beta |
  |---|---|---|---|---|
  | healthy | 9,999 | 1.000 | 0.9980 / 0.9980 / 0.9980 | 0.992 |
  | event-500 | 13,799 | 0.949 | 1.044 / 1.045 / 1.046 | 1.000 |
  | event-200 | 14,099 | 1.017 | 1.040 / 1.041 / 1.041 | 0.999 |
  | event-50 | 14,249 | 2.01 | 1.051 / 1.059 / 1.061 | 1.000 |
  | event-20 | 14,299 | 21.8 | 1.134 / 1.132 / 1.133 | 1.000 |

  At the healthy snapshot the frozen probe drops 2.6 decades in 3,000 steps. From
  event-500 on, it grows 4-13% per step until it overflows float32.
- **beta is not the lever.** At each snapshot g differs by at most 0.02 between beta
  0.99 and the run's ~0.996. The fresh controller reacts to the growing residual exactly
  as the hypothesis said it would: r_ip > 1 is clipped to 1, so it drives beta to 1.0.
  But that happens on the frozen system, and lower beta would not have helped.
- **The run's beta never rose.** Its median over each snapshot's window stayed at
  0.9955-0.9959 from 10k to the event; the controller's EMA was already frozen. So the "controller pushes beta up" step did not
  happen in the run.
- **Unexplained: why the moving system is fine where the frozen one is not.** At 13,799
  the run's own probe had r_ip 0.95 on the same theta where the frozen probe grows at
  4% per step. A linear recursion's growth rate should not depend on where z starts, so
  this has to be explained before reading "the system goes bad 500 epochs out" as
  fact. Next checks: (1) the frozen run in float64 at event-500 (this code lacks F4d's
  centred Gram; without it Phase F measured Gram minimum eigenvalues of -0.005 to
  -0.010, below the 1e-3 damping); (2) frozen snapshots between 10k and 13.8k to find the onset;
  (3) the time series and Gram spectrum below.
- *How computed:* `slurm/h1_frozen_probe.py` (summary table) and
  `slurm/h1_log_to_wandb.py` (these runs). The run's probe r_ip and beta are medians of
  its .txt metrics from the snapshot epoch to min(epoch + 3000, event).
"""


def build(url=None):
    ranges, event, epochs = frozen_meta()
    print(f"event {event}, snapshots {epochs}, marker ranges {ranges}")
    blocks = [
        wr.H1("The experiment"),
        wr.MarkdownBlock(DESCRIPTION),
        wr.H1("Takeaways — H1"),
        wr.MarkdownBlock(TAKEAWAYS),
        wr.H1("The diverging run and where theta was frozen"),
        run_section(ranges, event, zoom_ranges(event)),
        wr.H1("Kaczmarz++ on the frozen systems"),
        probe_section(),
    ]
    if url:
        rep = wr.Report.from_url(url)
        rep.blocks = blocks
    else:
        rep = wr.Report(entity=ENTITY, project=PH, blocks=blocks)
    rep.title = "H1 — frozen-theta Kaczmarz++ on a diverging stretched-N2 run"
    rep.description = (
        "E14 s0 replay frozen at 5 snapshots: the probe converges on the "
        "healthy system but diverges at every beta from 500 epochs "
        "before the event, while the run's own probe does not."
    )
    rep.width = "fluid"  # from_url reads 'readable'; must be re-set before every save
    rep.save()
    print(rep.url)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--url", default=None, help="existing report to rebuild in place")
    build(ap.parse_args().url)
