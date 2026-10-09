#!/usr/bin/env python3
"""F5 report: the beta* guard replays against PRIME-SR, SPRING and the old SS-SPRING,
on every molecule. Same four sections as the E14-E19 report ("Ada-SPRING divergence on
stretched N2"), each split into one subsection per molecule.

Needs the F5 runs synced (`wandb sync --legacy`) and stamped first:
    python slurm/f5_backfill_wandb.py
Run:  python slurm/report_f5_guard.py            (creates the report, prints its URL)
      python slurm/report_f5_guard.py --url URL  (rebuilds that report in place)
Then: python ~/.claude/skills/wandb-experiment-report/scripts/verify_report.py <url>
"""

import argparse

import wandb_workspaces.reports.v2 as wr

ENTITY = "ren27-university-of-california-berkeley"
PD, PE, PF = "vmcnet-phase-d", "vmcnet-phase-e", "vmcnet-phase-f"
HOST = PE  # next to the E14-E19 report it follows from
TEMPLATE = (
    "https://wandb.ai/ren27-university-of-california-berkeley/vmcnet-phase-e/reports/"
    "Ada-SPRING-divergence-on-stretched-N2--VmlldzoxNzk0MDIzNw"
)
LEG = ["config:x_experiment", "config:x_arm", "config:x_seed", "config:x_replay_of"]
FULL = (0.0, 100000.0)

# molecule -> (section title, comparison project, source experiment, plain-SPRING arm,
#              energy range full, zoom window, energy range in the zoom)
# Energy ranges come from the SS-SPRING s4 histories (1st-99th percentile + margin).
SYSTEMS = {
    "N2_4.0": ("N2 at 4.0 Bohr (stretched) — the failing case", PE, None,
               "spring_mu0.99", (-112.0, -94.0), (8000.0, 30000.0), (-110.5, -107.0)),
    "N2_eq": ("N2 at equilibrium", PD, "D10", "spring_mu0.99",
              (-109.6, -109.0), (45000.0, 60000.0), (-109.57, -109.49)),
    "CO": ("CO", PD, "D10", "spring_mu0.99",
           (-113.37, -112.8), (45000.0, 60000.0), (-113.36, -113.28)),
    "carbon": ("carbon atom", PD, "D7", "spring_default",
               (-37.87, -37.6), (45000.0, 60000.0), (-37.852, -37.836)),
    "N": ("N atom", PD, "D9", "spring_mu0.99",
          (-54.62, -54.3), (45000.0, 60000.0), (-54.597, -54.580)),
    "O": ("O atom", PD, "D9", "spring_mu0.99",
          (-75.10, -74.8), (45000.0, 60000.0), (-75.082, -75.050)),
    "H4": ("H4", PD, "D7", "spring_default",
           (-2.037, -2.02), (45000.0, 60000.0), (-2.0352, -2.0330)),
}


def rs(project, name, filters):
    return wr.Runset(ENTITY, project, name=name, filters=filters,
                     order=[wr.OrderBy(wr.Config("x_seed"), ascending=True)])


def runsets(sysname, ss_only=False):
    """F5 first, then the old SS-SPRING; PRIME-SR and SPRING unless ss_only."""
    _, proj, exp, spring, *_ = SYSTEMS[sysname]
    sy = f"Config('x_system') == '{sysname}'"
    out = [rs(PF, "SS-SPRING + beta* guard (F5 replays)",
              f"Config('x_experiment') == 'F5' and {sy}")]
    if sysname == "N2_4.0":
        ss = f"{sy} and Config('x_arm') == 'ssu_defaults'"
        out += [rs(PE, "SS-SPRING eta=0.002 (E14)", f"{ss} and Config('x_experiment') == 'E14'"),
                rs(PE, "SS-SPRING eta=0.002 (E15)", f"{ss} and Config('x_experiment') == 'E15'"),
                rs(PE, "SS-SPRING eta=0.0015 (E18)", f"{ss} and Config('x_experiment') == 'E18'")]
        if not ss_only:
            out += [rs(PE, "SPRING mu=0.99", f"{sy} and Config('x_arm') == 'spring_mu0.99'"),
                    rs(PE, "SPRING mu=0.995 (E16)", f"{sy} and Config('x_arm') == 'spring_mu0.995'"),
                    rs(PE, "PRIME-SR", f"{sy} and Config('x_arm') == 'prime_sr'")]
        return out
    base = f"Config('x_experiment') == '{exp}' and {sy}"
    out.append(rs(proj, f"SS-SPRING ({exp})", f"{base} and Config('x_arm') == 'ssu_defaults'"))
    if not ss_only:
        out += [rs(proj, f"SPRING mu=0.99 ({exp})", f"{base} and Config('x_arm') == '{spring}'"),
                rs(proj, f"PRIME-SR ({exp})", f"{base} and Config('x_arm') == 'prime_sr'")]
    return out


def line(title, y, rx=FULL, ry=None, log=False):
    kw = {"range_y": ry} if ry is not None else {}
    return wr.LinePlot(x="Step", y=y if isinstance(y, list) else [y], title=title,
                       log_y=log, range_x=rx, legend_fields=LEG, legend_position="east",
                       **kw)


def grid(panels, sets):
    """Two across, like the template."""
    for i, p in enumerate(panels):
        p.layout = wr.Layout(x=12 * (i % 2), y=8 * (i // 2), w=12, h=8)
    return wr.PanelGrid(runsets=sets, panels=panels)


def energy(s):
    _, _, _, _, full_e, zoom, zoom_e = SYSTEMS[s]
    z = f"{int(zoom[0] / 1000)}k-{int(zoom[1] / 1000)}k"
    return grid([
        line("Energy over the full 100k", "energy_noclip", ry=full_e),
        line(f"Zoom: epochs {z} (the F5 replay window)", "energy_noclip", rx=zoom, ry=zoom_e),
        line("Local-energy variance (log)", "variance_noclip", log=True),
        line("MCMC acceptance ratio", "accept_ratio"),
    ], runsets(s))


def momentum(s):
    zoom = SYSTEMS[s][5]
    z = f"{int(zoom[0] / 1000)}k-{int(zoom[1] / 1000)}k"
    return grid([
        line("mu: SPRING fixed, PRIME-SR and SS-SPRING adaptive (F5: the controller's "
             "beta, NOT the applied momentum)", "mu", ry=(0.0, 1.05)),
        line(f"Zoom: epochs {z}", "mu", rx=zoom, ry=(0.9, 1.01)),
        line("F5 only: momentum actually applied, min(beta, 2 beta*)",
             "diag_beta_applied", rx=zoom, ry=(-0.05, 1.05)),
        line("F5 only: beta* and cos(eps, A phi) (guard binds iff carried/eps > 2 cos)",
             ["diag_beta_star", "diag_momentum_cos"], rx=zoom, ry=(-0.3, 1.0)),
    ], runsets(s))


def probe(s):
    return grid([
        line("beta (mu): old SS-SPRING seeds against the F5 replays", "mu", ry=(0.0, 1.05)),
        line("r_hat -- the probe ratio beta is derived from", "r_hat", ry=(0.0, 1.05)),
        line("probe_r_ip (clipped at 1 in r_hat; F5 sanitises non-finite values)",
             "probe_r_ip"),
        line("probe residual norm (log)", "probe_res_norm", log=True),
    ], runsets(s, ss_only=True))


def step(s):
    zoom = SYSTEMS[s][5]
    z = f"{int(zoom[0] / 1000)}k-{int(zoom[1] / 1000)}k"
    return grid([
        line("Squared update norm before the cap (log)", "update_sq_norm_preclip", log=True),
        line("Norm cap binding (1 = rescaled this step)", "norm_cap_applied",
             ry=(-0.05, 1.05)),
        line(f"Zoom: epochs {z} (log)", "update_sq_norm_preclip", rx=zoom, log=True),
        line(f"Zoom: epochs {z}", "norm_cap_applied", rx=zoom, ry=(-0.05, 1.05)),
    ], runsets(s))


DESCRIPTION = f"""\
**Follows from F4d** (docs/CAMPAIGN_LOG.md, Phase F) and the
[E14-E19 report]({TEMPLATE}), which has the same four sections for stretched N2 only.
F4d centred the SR Gram before the float32 contraction. That fixed the numerics: the
Gram's negative eigenvalues fell below the damping, and the step stayed inside SPRING's
exact bound. But 6/6 stretched-N2 replays still failed. In the last ~30 rows before each
event the probe residual grew 4-12% per step, while the beta controller sat frozen at
~0.997. So the momentum recurrence itself was expanding, and the controller could not
see it.

**What F5 changes.** SS-SPRING's main solve uses momentum min(beta, 2 beta*), and 0 if
beta* <= 0. Here beta* = <eps, A phi> / ||A phi||^2 is the momentum that best fits this
step's target on this step's walkers. 2 beta* is the largest momentum for which the
carried term does not increase the residual the solve must fit,
||eps - beta A phi|| <= ||eps||. The rule has no tunable constant, and the probe and the
beta controller are untouched. F4d (`gram_center_first`) is on too, and the probe
sanitiser is on.

**What ran: 12 checkpoint replays, one arm.**

| | states | replay window |
|---|---|---|
| failing N2 at 4.0 Bohr | E14 s0, s1, s2 and E15 s7 (from 10k); E15 s6 (from 60k); E18 s2 at eta 0.0015 (from 10k) | 10k steps |
| stable systems, seed 0 | carbon and H4 (D7), N and O (D9), N2-eq and CO (D10), all from 50k | 5k steps |

Each F5 curve starts at its replay epoch, on top of the original run it was replayed
from (the same seed in the old SS-SPRING runset). The comparison runsets are the
original 100k runs of those experiments: the old SS-SPRING (no guard, no F4d),
PRIME-SR and SPRING at mu = 0.99 (the published default). Stretched N2 also has SPRING
at mu = 0.995 (E16) and SS-SPRING at eta 0.0015 (E18).

**What would confirm it (framed before the run).** The N2 replays reach their end epoch
at or below the pre-replay energy, with the guard binding mainly where carried/eps used
to climb. On the stable systems the guard binds rarely (cos(eps, A phi) > 0.5 was
expected on healthy rows), and the energy is unchanged.

**Scope.** The F5 runs are short windows from checkpoints, so read them in the zoom
panels. There is one seed per stable system and no eval phase. The F5 runs log every
2nd-7th epoch, the originals every 10th. For F5, `mu` is the controller's beta, not the
momentum applied; the applied momentum is in Momentum panels 3-4 of each molecule.
"""

TAKEAWAYS = """\
- **The guard stops the stretched-N2 catastrophe: 6/6 replays reach their end epoch with
  no event.** The baseline is 11/12 failures (F3b no-cap + F4d replays of the same
  states; Fisher exact one-sided p ~ 4e-4). E15 s6 is the sharpest case: F4d failed 80
  rows into the same 60k state, and F5 ran all 10k. E18 s2 counts less, because it also
  survived both F3b arms.
- **But it is not a rare brake. It binds on ~100% of rows on every system**, and cuts the
  applied momentum from beta ~ 0.997 to a median of ~0.5 on N2-4.0, N2-eq and CO, and
  to 0.19-0.27 on H4, N, O and carbon. On the atoms and H4, 6-14% of rows run with no
  momentum at all. The prediction that it would bind rarely failed.
- **Why: cos(eps, A phi) is only 0.08-0.28 on healthy rows**, not > 0.5. A likely
  reading, not verified: on fresh walkers eps is mostly Monte Carlo noise, so a single
  batch cannot credit the momentum for averaging that noise out. N2-4.0 has the same cos
  (0.24-0.26) as N2-eq and CO (0.26-0.28), so beta* does not single out the failing
  geometry.
- **Energy.** The N2 replays end 4-76 mHa below their pre-replay level, and none dips
  below the -109.2021 reference (the F4d approach dipped 0.5-0.9 Ha below it). There is
  no fair same-epoch baseline yet, because every original had failed by then; compare
  with the E15 survivors s4/s5 in the zoom panel. Over the 50k-55k window the stable
  systems are within 0.4 mHa of their 50k level, and CO is 2.9 mHa lower. Late in
  training this is a weak test.
- **So F5 shows that momentum ~0.5 is stable where ~0.997 is not.** This fits E16 (SPRING
  at mu = 0.995 fails 2/3, at 0.99 it survives 3/3). It does not show that the guard's
  adaptivity matters. The next step is a fixed-momentum control with the guard off.
- *How computed:* guard and energy figures are from `slurm/f2_replay_summary.py
  --carried --guard` on the per-epoch logdir files (every row), not on these sampled
  wandb curves. "Healthy rows" means the whole replay, since no F5 run had an event.
"""


def build(url=None):
    blocks = [wr.H1("The experiment"), wr.MarkdownBlock(DESCRIPTION),
              wr.H1("Takeaways — F5"), wr.MarkdownBlock(TAKEAWAYS)]
    for title, fn in (("Energy", energy), ("Momentum", momentum),
                      ("SS-SPRING probe internals", probe),
                      ("Step size and the norm cap", step)):
        blocks.append(wr.H1(title))
        for s in SYSTEMS:
            blocks += [wr.H2(f"{title} — {SYSTEMS[s][0]}"), fn(s)]
    if url:
        rep = wr.Report.from_url(url)
        rep.blocks = blocks
    else:
        rep = wr.Report(entity=ENTITY, project=HOST, blocks=blocks)
    rep.title = "F5 beta* guard vs PRIME-SR, SPRING and SS-SPRING — all molecules"
    rep.description = ("Phase F, F5: the beta* momentum guard, replayed from checkpoints "
                       "on the 6 failing stretched-N2 states and 6 stable systems.")
    rep.width = "fluid"  # from_url reads 'readable'; must be re-set before every save
    rep.save()
    print(rep.url)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--url", default=None, help="existing report to rebuild in place")
    build(ap.parse_args().url)
