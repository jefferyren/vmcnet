#!/usr/bin/env python3
"""Phase G report: the original SS-SPRING against SPRING(0.99) on more strongly
correlated systems (N2 bond scan, C2, stretched CO, H10 chains, H4 square), with the
stretched-N2 runs of E14/E15 as the end of the N2 scan.

Needs, in order (see docs/CAMPAIGN_LOG.md, Phase G):
    python slurm/parse_eval_energies.py --experiment G1 G2 G3 --backfill
    python slurm/g_backfill_divergence.py
Run:  python slurm/report_g_strong_correlation.py            (creates, prints the URL)
      python slurm/report_g_strong_correlation.py --url URL  (rebuilds in place)
Then: python ~/.claude/skills/wandb-experiment-report/scripts/verify_report.py <url>
"""

import argparse

import wandb_workspaces.reports.v2 as wr

ENTITY = "ren27-university-of-california-berkeley"
PE, PG = "vmcnet-phase-e", "vmcnet-phase-g"
E14_REPORT = (
    "https://wandb.ai/ren27-university-of-california-berkeley/vmcnet-phase-e/reports/"
    "Ada-SPRING-divergence-on-stretched-N2--VmlldzoxNzk0MDIzNw"
)
F6_REPORT = (
    "https://wandb.ai/ren27-university-of-california-berkeley/vmcnet-phase-e/reports/"
    "F6-beta*-guard-from-scratch-vs-PRIME-SR,-SPRING-and-SS-SPRING-—-all-molecules--"
    "VmlldzoxODA5ODE0NQ=="
)
LEG = ["config:x_experiment", "config:x_arm", "config:x_seed"]
FULL = (0.0, 100000.0)
FAIL_WINDOW = (5000.0, 40000.0)  # every Phase G divergence starts in 13k-30k
LATE = (50000.0, 100000.0)

# system -> (section title, experiment, eval energy scale [Ha], SS-SPRING diverged?,
#            full-run energy margins below/above, zoom margins below/above).
# The energy scale is the arms' eval mean, used only to place the axes.
SYSTEMS = {
    "N2_2.4": ("N2 at 2.4 Bohr", "G1", -109.481, False, (0.05, 0.5), (0.01, 0.02)),
    "N2_2.8": ("N2 at 2.8 Bohr", "G1", -109.370, False, (0.05, 0.5), (0.01, 0.02)),
    "N2_3.2": ("N2 at 3.2 Bohr", "G1", -109.275, True, (0.05, 0.5), (0.02, 0.3)),
    "N2_3.6": ("N2 at 3.6 Bohr", "G1", -109.214, True, (0.05, 0.5), (0.02, 0.3)),
    "N2_4.0": ("N2 at 4.0 Bohr (E14 + E15)", "E14+E15", -109.19, True, (0.05, 0.5),
               (0.02, 0.3)),
    "C2": ("C2 at equilibrium (2.348 Bohr)", "G2", -75.915, False, (0.05, 0.5),
           (0.01, 0.02)),
    "CO_4.0": ("CO at 4.0 Bohr", "G2", -112.969, True, (0.05, 0.5), (0.02, 0.3)),
    "H10_1.8": ("H10 chain at 1.8 Bohr", "G3", -5.6633, False, (0.005, 0.1),
                (0.001, 0.003)),
    "H10_3.6": ("H10 chain at 3.6 Bohr", "G3", -5.1550, False, (0.005, 0.1),
                (0.001, 0.003)),
    "H4_1.0": ("H4 square at 1.0 Bohr", "G3", -1.43883, False, (0.002, 0.05),
               (0.0005, 0.001)),
}


def rs(project, name, filters):
    return wr.Runset(ENTITY, project, name=name, filters=filters,
                     order=[wr.OrderBy(wr.Config("x_seed"), ascending=True)])


def runsets(sysname):
    """SS-SPRING, then SPRING(0.99), for one system."""
    sy = f"Config('x_system') == '{sysname}'"
    ss, sp = "Config('x_arm') == 'ssu_defaults'", "Config('x_arm') == 'spring_mu0.99'"
    if sysname == "N2_4.0":
        return n2_40_runsets()
    return [rs(PG, "SS-SPRING", f"{sy} and {ss}"),
            rs(PG, "SPRING mu=0.99", f"{sy} and {sp}")]


def n2_40_runsets():
    """E14 + E15 on N2_4.0 in vmcnet-phase-e. E17/E18 share the arm name ssu_defaults
    at other etas, so eta == 0.002 is what isolates E14 + E15 (no `or` needed); the
    only spring_mu0.99 cells on N2_4.0 are E14's."""
    n2 = "Config('x_system') == 'N2_4.0'"
    return [rs(PE, "SS-SPRING, N2 4.0 (E14 s0-2, E15 s3-7)",
               f"{n2} and Config('x_arm') == 'ssu_defaults' and Config('x_eta') == 0.002"),
            rs(PE, "SPRING mu=0.99, N2 4.0 (E14)",
               f"{n2} and Config('x_arm') == 'spring_mu0.99'")]


def overview_runsets():
    # vmcnet-phase-g holds only Phase G runs, so the arm alone scopes these.
    return [rs(PG, "SS-SPRING (Phase G)", "Config('x_arm') == 'ssu_defaults'"),
            rs(PG, "SPRING mu=0.99 (Phase G)", "Config('x_arm') == 'spring_mu0.99'"),
            *n2_40_runsets()]


def grid(panels, sets):
    """Two across."""
    for i, p in enumerate(panels):
        p.layout = wr.Layout(x=12 * (i % 2), y=8 * (i // 2), w=12, h=8)
    return wr.PanelGrid(runsets=sets, panels=panels)


def line(title, y, rx=FULL, ry=None, log=False, smooth=0.0):
    kw = {"range_y": ry} if ry is not None else {}
    if smooth:
        kw.update(smoothing_factor=smooth, smoothing_type="exponential",
                  smoothing_show_original=True)
    return wr.LinePlot(x="Step", y=[y], title=title, log_y=log, range_x=rx,
                       legend_fields=LEG, legend_position="east", **kw)


def bar(title, metric, xlab, groupby="x_arm"):
    return wr.BarPlot(metrics=[metric], title=title, orientation="h", groupby=groupby,
                      groupby_aggfunc="mean", groupby_rangefunc="stderr", title_x=xlab)


def overview():
    return grid([
        bar("Fraction of seeds diverged, by system", "x_diverged",
            "fraction diverged (0 = none, 1 = all)", groupby="x_system"),
        bar("Norm cap binding over epochs 0-12k, mean over seeds", "x_cap_frac_0_12k",
            "fraction of steps capped", groupby="x_system"),
        wr.ScatterPlot(title="Cap fraction 0-12k vs healthy-until epoch (one point per "
                       "run; 100k = survived)",
                       x=wr.SummaryMetric("x_cap_frac_0_12k"),
                       y=wr.SummaryMetric("x_healthy_until")),
        bar("Norm cap binding over epochs 8-10k (late), mean over seeds",
            "x_cap_frac_8_10k", "fraction of steps capped", groupby="x_system"),
    ], overview_runsets())


def system(s):
    title, exp, e0, failed, full_m, zoom_m = SYSTEMS[s]
    zoom = FAIL_WINDOW if failed else LATE
    z = f"{int(zoom[0] / 1000)}k-{int(zoom[1] / 1000)}k"
    why = "where the divergences happen" if failed else "late training"
    eval_note = " (diverged seeds have no eval: survivors only)" if failed else ""
    if s == "H4_1.0":
        ev = bar("Eval error [mHa above FCI/CBS -1.4390], mean ± s.e.m.", "x_eval_mHa",
                 "mHa above reference (lower is better)")
    else:
        ev = bar(f"Eval energy [Ha], mean ± s.e.m.{eval_note}", "x_eval_energy",
                 "energy [Ha] (lower is better)")
    return grid([
        line("Energy over the full 100k", "energy_noclip",
             ry=(e0 - full_m[0], e0 + full_m[1])),
        line(f"Zoom: epochs {z} ({why}), smoothed", "energy_noclip", rx=zoom,
             ry=(e0 - zoom_m[0], e0 + zoom_m[1]), smooth=0.9),
        ev,
        bar("Eval local-energy variance [Ha^2]", "x_eval_variance",
            "variance [Ha^2] (lower is better)"),
        line("Norm cap binding (1 = rescaled this step), smoothed, epochs 0-40k",
             "norm_cap_applied", rx=(0.0, 40000.0), ry=(-0.05, 1.05), smooth=0.95),
        line("Squared update norm before the cap (log)", "update_sq_norm_preclip",
             log=True),
        line("Momentum: SS-SPRING's adaptive beta vs SPRING's fixed 0.99", "mu",
             ry=(0.98, 1.0)),
        line("Local-energy variance (log)", "variance_noclip", log=True),
    ], runsets(s))


DESCRIPTION = f"""\
**Follows from E14/E15** ([E14-E19 report]({E14_REPORT})) **and Phase F**
([F6 report]({F6_REPORT})). SS-SPRING diverged on N2 at 4.0 Bohr on 6/8 seeds, where
SPRING(0.99) and PRIME-SR completed 3/3, and on no other system in the campaign. Phase F
tried to stabilise it there and did not find a variant that is both stable and accurate.
One strongly correlated system was already a pass, though: the H4 square at 4.0 Bohr
(D7, 0/5 diverged, -2.03421 Ha against the -2.0342 FCI/CBS of Lin, Goldshlager & Lin,
arXiv:2112.03491). So the question here is whether the N2 failure is **common on
strongly correlated systems** or specific to that one.

**The optimizer is the original SS-SPRING.** The code is tag `ss-spring-original`,
identical to the commit E14 ran: every Phase F flag (including the F4d centred Gram) is
gone. Each job refuses to start on any other code.

**What ran: 72 runs**, SS-SPRING (`ssu_defaults`, 5 seeds) and SPRING mu = 0.99
(3 seeds), E14's settings (norm cap 1e-3, inverse-time decay 1e-4), 100k epochs + 20k
eval. SPRING(0.99) is the comparison arm, since most of these systems have no reference
energy, and the stability control.

| | systems | eta | why |
|---|---|---|---|
| G1 | N2 at 2.4, 2.8, 3.2, 3.6 Bohr | 0.002 | bond-length dose-response between N2-eq (2.016, clean) and 4.0 (6/8 fail) |
| G2 | C2 at 2.348 Bohr; CO at 4.0 Bohr | 0.002 | multireference without stretching; the isoelectronic twin of N2-4.0 |
| G3 | H10 chain at 1.8 and 3.6 Bohr; H4 square at 1.0 Bohr | 0.02 | strong correlation without heavy nuclei; the other H4 geometry of 2112.03491 |

N2 at 4.0 Bohr (E14 + E15, same code and settings) is shown as the end of the scan.

**What would answer it (framed before the runs).** If the failure is common, SS-SPRING
diverges on a fraction of seeds on several of these systems. If it is specific, it
stays at 0/5 off stretched N2. With 5 seeds, a system that failed as often as N2-4.0
would show 0/5 only 0.1% of the time; 0/5 bounds the per-seed rate below ~45%. A second
pre-registered readout tests the campaign's warning sign out of sample: runs whose norm
cap binds on more than ~60% of epochs 0-12k should be the ones that diverge.

**Scope.** 5 and 3 seeds per cell. Eval energies are the publishable numbers; wandb
drops the eval phase, so they were parsed from the slurm .out files and backfilled
(`x_eval_energy`, `x_eval_variance`). Divergence also comes from the .out files
(`x_diverged`, `x_healthy_until`), not from wandb's sampled history. Only H4 at 1.0
Bohr has a reference energy, so the others compare arms in raw Ha.
"""

TAKEAWAYS = """\
- **The failure is specific to stretched heavy-atom multiple bonds, not to strong
  correlation.** SS-SPRING divergence by system (SPRING(0.99) in brackets):

  | N2 2.4 | N2 2.8 | N2 3.2 | N2 3.6 | N2 4.0 | CO 4.0 | C2 | H10 1.8 | H10 3.6 | H4 1.0 |
  |---|---|---|---|---|---|---|---|---|---|
  | 0/5 (0/3) | 0/5 (0/3) | **1/5** (0/3) | **5/5** (0/3) | **6/8** (0/3) | **1/5** (0/3) | 0/5 (0/3) | 0/5 (0/3) | 0/5 (0/3) | 0/5 (0/3) |

  On N2 it is a dose-response in bond length: onset between 2.8 and 3.2 Bohr, every seed
  by 3.6. C2 (multireference at equilibrium), the stretched H10 chain and the compressed
  H4 square are clean. SPRING(0.99) never diverges (0/30).
- **Where SS-SPRING survives, it ties SPRING(0.99) without tuning.** SS-SPRING minus
  SPRING eval energy, mHa: N2 2.4 -1.03 ± 0.44, N2 2.8 -0.44 ± 0.57, N2 3.2 -0.02 ± 1.56
  (4 survivors), C2 +1.23 ± 3.58, CO 4.0 +0.08 ± 6.79 (4 survivors), H10 1.8 -0.02,
  H10 3.6 +0.50 ± 0.39; H4 1.0 0.175 vs 0.176 mHa above FCI/CBS. Nothing separates by
  more than ~2.3σ.
- **The survivors are sound.** Eval variance is 0.17-0.30 Ha² on the N2 and CO 4.0
  survivors (C2 0.08-0.10), the same as SPRING, unlike F6's 2.6-53. Surviving here means
  a healthy wavefunction.
- **The cap-binding warning sign holds out of sample, but only as a risk factor.** Over
  the 45 SS-SPRING runs, cap fraction over epochs 0-12k ranks diverged above survived
  with AUC 0.97, but "> 60% diverges" catches 6/7 with 4 false alarms in 38: CO 4.0 s2
  died at 58%, and N2 3.2 s3 survived at 75%. Late binding (epochs 8-10k) is the
  sharper sign: 22-51% only on the N2 geometries that fail (3.2, 3.6, 4.0), 0-1%
  everywhere else, though CO 4.0 failed once at 0%.
- **SS-SPRING takes bigger steps than SPRING everywhere, and more so as N2 stretches.**
  Median squared pre-cap step at 8-10k: 4.0e-4 vs 2.5e-4 at N2 2.4, rising to 9.8e-4 vs
  3.6e-4 at 3.6. Its beta is 0.996 on the heavy-atom systems and 0.993-0.994 on hydrogen,
  against SPRING's fixed 0.99.
- *How computed:* divergence and healthy-until from the slurm .out files
  (`divergence_check.py`: NaN marker; 100-epoch blocks within 0.5 Ha of the run's best).
  Eval energies from `parse_eval_energies.py` (20k eval epochs, blocked errors; mean ±
  s.e.m. over seeds, combined s.e.m. for differences). Cap fractions from wandb history
  (`norm_cap_applied`, logged ~every 10 steps; `slurm/g_backfill_divergence.py`).
"""


def build(url=None):
    blocks = [wr.H1("The experiment"), wr.MarkdownBlock(DESCRIPTION),
              wr.H1("Takeaways — Phase G"), wr.MarkdownBlock(TAKEAWAYS),
              wr.H1("Overview: divergence and the norm cap across all systems"),
              overview()]
    groups = (("G1 — N2 bond-length scan", ["N2_2.4", "N2_2.8", "N2_3.2", "N2_3.6",
                                            "N2_4.0"]),
              ("G2 — heavy-atom strong correlation", ["C2", "CO_4.0"]),
              ("G3 — hydrogen-only strong correlation", ["H10_1.8", "H10_3.6",
                                                         "H4_1.0"]))
    for title, systems in groups:
        blocks.append(wr.H1(title))
        for s in systems:
            blocks += [wr.H2(SYSTEMS[s][0]), system(s)]
    if url:
        rep = wr.Report.from_url(url)
        rep.blocks = blocks
    else:
        rep = wr.Report(entity=ENTITY, project=PG, blocks=blocks)
    rep.title = "Phase G — original SS-SPRING on strongly correlated systems"
    rep.description = ("72 runs: SS-SPRING diverges only on stretched heavy-atom "
                       "multiple bonds (N2 from 3.2 Bohr, CO at 4.0), not on C2, H10 "
                       "or H4; SPRING(0.99) never does.")
    rep.width = "fluid"  # from_url reads 'readable'; must be re-set before every save
    rep.save()
    print(rep.url)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--url", default=None, help="existing report to rebuild in place")
    build(ap.parse_args().url)
