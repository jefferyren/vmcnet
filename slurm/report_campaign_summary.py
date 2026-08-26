#!/usr/bin/env python3
"""Campaign-wide summary report, written for a reader outside the project.

Covers all three stages (screening / mechanism / generality) with no internal
experiment codenames. Edits the existing report IN PLACE via from_url, so the link
stays stable — see docs/CAMPAIGN_LOG.md section 7 for why a bare Report() + save()
would silently mint a duplicate instead.

Run:  python slurm/report_campaign_summary.py
Then: python ~/.claude/skills/wandb-experiment-report/scripts/verify_report.py <url>
"""
import wandb_workspaces.reports.v2 as wr

ENTITY = "ren27-university-of-california-berkeley"
PA, PB, PC = "vmcnet-phase-a", "vmcnet-phase-b", "vmcnet-phase-c"
URL = ("https://wandb.ai/ren27-university-of-california-berkeley/vmcnet-phase-c/"
       "reports/Phase-C-summary-%E2%80%94-SS-SPRING-vs-PRIME-SR--VmlldzoxNzc1MDYwNA")

LABEL = {"ssu_defaults": "SS-SPRING (ours)", "prime_sr": "PRIME-SR (competitor)",
         "spring_mu0.995": "SPRING mu=0.995", "spring_mu0.99": "SPRING mu=0.99",
         "spring_mu0.95": "SPRING mu=0.95", "spring_mu0.9": "SPRING mu=0.9",
         "spring_default": "SPRING (published default)", "spring_tuned": "SPRING (tuned)"}
LEG = ["config:x_arm", "config:x_seed"]


def arms(project, base, keys):
    return [wr.Runset(ENTITY, project, name=LABEL[a],
                      filters=f"{base} and Config('x_arm') == '{a}'",
                      order=[wr.OrderBy(wr.Config("x_seed"), ascending=True)])
            for a in keys]


C6 = ["ssu_defaults", "prime_sr", "spring_mu0.995", "spring_mu0.99",
      "spring_mu0.95", "spring_mu0.9"]
C4 = ["ssu_defaults", "prime_sr", "spring_mu0.995", "spring_mu0.99"]
B4 = ["ssu_defaults", "prime_sr", "spring_tuned", "spring_default"]


def curve(title, y, xlab, ylab, rng=None, sm=0.9, w=12):
    kw = {"range_x": rng} if rng is not None else {}   # pydantic rejects range_x=None
    return wr.LinePlot(x="Step", y=[y], title=title,
                       title_x=xlab, title_y=ylab, smoothing_factor=sm,
                       smoothing_type="exponential", ignore_outliers=True,
                       legend_position="east", legend_fields=LEG,
                       groupby="x_arm", groupby_aggfunc="mean",
                       groupby_rangefunc="stderr", layout=wr.Layout(w=w, h=9), **kw)


INTRO = """\
**What the project is about.** A neural-network wavefunction is trained by an optimizer,
and the three optimizers compared here are variants of the same method
(stochastic reconfiguration) that differ in exactly one respect: **how much of the
previous update they carry forward — the momentum.** That single number matters a great
deal. On carbon, sweeping it from 0 to 0.999 moves the final energy error by a factor of
17, so in practice it has to be hand-tuned per system, and a tuning sweep costs as much
as the production run it enables.

Three methods are on trial:

| | how momentum is set | needs tuning? |
|---|---|---|
| **SPRING** | a fixed number chosen by hand | **yes** |
| **PRIME-SR** | computed automatically each step from spectral properties of the sampled data (published competitor, Wang & Liu, arXiv:2604.18357) | no |
| **SS-SPRING** | computed automatically each step from a probe that measures how fast the solver is actually converging (**ours**) | no |

**The two questions.** (1) Does our method, run untuned, match SPRING that has been
*optimally* tuned? (2) Does it match or beat the published competitor? Both were framed
before any runs, and results are reported against them either way.

**How anything is scored.** Energies come from a separate evaluation phase after
training, never from the training curve, averaged over 3–5 independent random seeds and
quoted as mean ± standard error. Lower is better throughout. Errors are in
**milli-Hartree (mHa)**; chemical accuracy is ~1.6 mHa, and the differences that decide
these comparisons are between 0.03 and 3 mHa.
"""

RAN = """\
The campaign ran in three stages, **325 runs and roughly 1050 GPU-hours** in total, all
on the same hardware and protocol (random initialisation, 1 GPU per run, identical
network and sampler; only the optimizer differs).

| stage | what it was for | systems | runs |
|---|---|---|---|
| **1 — screening** | How much does momentum matter? Does step size or sample size change the ranking? Which settings are even stable? | carbon, H4 | 109 |
| **2 — mechanism and main comparison** | A careful 5-seed head-to-head, plus two controls asking whether our method's advantage is something more boring than adaptivity | carbon, H4 | 70 |
| **3 — generality and stress** | Does it hold on more systems, at other learning rates, and on the **competitor's own benchmark molecules at the competitor's own settings**? | N, O, carbon, N2, CO | 110 |

Stage 3 was designed specifically to *break* our own result — new systems we had never
run, a 10× learning-rate range, and finally the competitor's home ground where it should
have been strongest.
"""

SUMMARY = """\
We tested whether the momentum in these optimizers can be set automatically instead of by
hand, and whether our way of doing that beats the published competitor. Across three
stages, 325 runs and ~1050 GPU-hours spanning six chemical systems and three learning
rates, **the comparison against the competitor is unambiguous: our method never lost to
it in any condition tested** — eight conditions, ties in two, clear wins in six — and its
largest margin came on the competitor's *own* benchmark molecules at the competitor's
*own* learning rate, where it is ahead by about **2.9 mHa on every seed** (13 and 38
standard errors). We also found *why* the competitor fails, which is the cleanest result
of the campaign: **its momentum is computed without any reference to the learning rate**,
so at small learning rates it takes an effective step roughly **ten times shorter** than
ours and simply has not converged within the same budget — it is 2.7× worse than every
other method at the smallest learning rate we tried. The other question came out weaker,
and we are scoping it honestly rather than asserting it: untuned, our method **matches
optimally tuned SPRING on five of seven conditions and loses on two** (oxygen, and carbon
at the largest learning rate). What does survive is a robustness statement — across all
seven conditions our method has the **lowest average and lowest worst-case shortfall of
any single setting tested**, and matches per-condition oracle tuning on average. So the
defensible claim is "the best default if you are not going to tune", not "tuning is
unnecessary". Three of our own hypotheses died along the way, including the one the
project started from. One caveat to carry into the molecule numbers: **those runs are not
converged** at the budget used, so they compare methods at a fixed cost, not at their
best achievable energies.
"""

MAIN = """\
This is the result the paper should lead with. Our method was compared against the
competitor in **eight distinct conditions** — six chemical systems across three learning
rates — and **did not lose a single one**.

| condition | outcome |
|---|---|
| CO molecule, competitor's own setting | ✅ **win**, 38 s.e., every seed |
| N2 molecule, competitor's own setting | ✅ **win**, 13 s.e., every seed |
| carbon, small learning rate | ✅ **win**, 12.5 s.e. |
| carbon, standard learning rate | ✅ win, 7 s.e. |
| nitrogen atom | ✅ win, 3.9 s.e., 5 of 5 seeds |
| H4 chain | ✅ win, 3.4 s.e. |
| oxygen atom | ⚪ tie |
| carbon, large learning rate | ⚪ tie |

**The strongest evidence is on the competitor's home ground.** The last stage put both
methods on N2 and CO — the molecules the competitor's own paper benchmarks — at the
learning rate that paper uses. Final energies, in mHa above the reference (lower better,
3 seeds):

| method | momentum | N2 | CO |
|---|---|---|---|
| SPRING mu=0.9 | 0.90 | 20.018 ± 0.105 | 18.270 ± 0.132 |
| SPRING mu=0.95 | 0.95 | 14.900 ± 0.328 | 12.872 ± 0.248 |
| **PRIME-SR (competitor)** | ~0.953 | 14.260 ± 0.123 | 12.063 ± 0.160 |
| SPRING mu=0.99 | 0.99 | 11.818 ± 0.095 | 9.607 ± 0.024 |
| SPRING mu=0.995 | 0.995 | 11.668 ± 0.048 | 9.210 ± 0.025 |
| **SS-SPRING (ours)** | ~0.995 | **11.410 ± 0.108** | **9.157 ± 0.118** |

Two honesty notes on this table, both of which we built in deliberately:

- **The high-momentum SPRING rows were added on purpose.** Against only the grid the
  competitor's paper uses (momentum ≤ 0.95), our method would have appeared to beat
  "tuned SPRING" by ~3.5 mHa — an artifact of effective step size, not of adaptivity.
  Including momentum 0.99 and 0.995 removes that artifact and shrinks our apparent win to
  an honest one. Any careful referee would have caught this.
- **Nothing here is converged.** Every method was still improving when the budget ran
  out. This is a fixed-cost comparison; the competitor's deficit would partly close given
  a longer run.
"""

WHY = """\
The competitor's momentum is **computed without reference to the learning rate at all**.
Across a 10× change in learning rate its momentum moves by 0.001, settling at ~0.953
regardless. Since what actually determines how far the optimizer travels is roughly
`learning rate / (1 - momentum)`, holding momentum fixed while the learning rate shrinks
leaves it taking an **effective step about ten times shorter** than ours, and it simply
does not converge in the budget available.

Carbon, final error in mHa across a 10× learning-rate range (3 seeds):

| method | small lr | standard lr | large lr | swing |
|---|---|---|---|---|
| SPRING mu=0.995 | **0.162 ± 0.001** | 0.161 | 0.191 ± 0.001 | 0.030 |
| **SS-SPRING (ours)** | 0.172 ± 0.027 | 0.148 | 0.192 ± 0.010 | 0.044 |
| SPRING mu=0.99 | 0.201 ± 0.012 | 0.147 | **0.155 ± 0.004** | 0.054 |
| **PRIME-SR (competitor)** | **0.435 ± 0.025** | 0.217 | 0.193 ± 0.004 | **0.242** |

The competitor swings 4–8× more than anything else, and **the ranking of methods does not
survive a change of learning rate** — which matters because its paper advertises
robustness and contains no learning-rate study. The same mechanism reproduces
independently on both molecules, where the ordering of all six methods correlates with
effective step size at r = -0.89.

This is a *mechanism*, not just an observation: it predicts where the competitor will
fail before running it, and it did.
"""

TUNING = """\
This is the claim we cannot make as strongly as we hoped, and the report is scoped
accordingly.

**Per condition it fails.** Untuned, our method matches optimally tuned SPRING on carbon,
H4, nitrogen, CO and N2 — but **loses on oxygen** (behind on 0 of 5 seeds won) and on
**carbon at the largest learning rate** (0 of 3). It is also the *least* seed-stable
method on oxygen, the reverse of what we saw on carbon.

**As a robustness statement it holds.** Scoring every method by how far it lands behind
the best method in each of the seven conditions:

| method | average shortfall | worst case |
|---|---|---|
| **SS-SPRING (ours)** | **0.036 mHa** | **0.178 mHa** |
| SPRING mu=0.995 | 0.055 | 0.257 |
| SPRING mu=0.99 | 0.131 | 0.450 |
| PRIME-SR (competitor) | 0.913 | 2.906 |

Ours has the lowest average **and** the lowest worst case of any single setting, and comes
within 0.009 mHa of what you would get by tuning perfectly for each system separately.
The honest framing is therefore: **the best choice if you are not going to tune**, not a
proof that tuning is unnecessary.
"""

RULED_OUT = """\
Recording these because they cost real GPU-hours and because two of them cut against us.

- **The hypothesis the project started from was wrong, and backwards.** We predicted the
  competitor would fail at small sample sizes, as its underlying idea does in a
  neighbouring field. At the smallest sample size tested it is the **best** method and
  ours is the worst.
- **The advantage is not a momentum warm-up.** Ours happens to start at zero momentum for
  the first few dozen steps, which looked like an automatic warm-up. Removing that
  changes the result by nothing at all.
- **It is not simply "finding a better constant".** Pinning SPRING at exactly the momentum
  our method converges to does **not** reproduce our method's behaviour.
- **Short tuning sweeps mis-rank momentum.** The setting our own screening stage picked
  for H4 turned out to be the *worst* of all options once run properly with multiple seeds
  and a full budget. This is now an argument *for* removing the tuning step rather than an
  embarrassment.
- **We still cannot fully explain one result.** Our method beats well-tuned SPRING on
  carbon by ~0.10 mHa at ~2 s.e., and neither of the two obvious explanations above
  accounts for it. What remains is the *path* the momentum takes over training. None of
  the claims depend on resolving this, and it is flagged as future work rather than
  hidden.

**One further practical finding.** All three methods rely on a safety cap on the update
size. With that cap removed, **10 of 15 runs diverge** within a few hundred steps
regardless of method — so the cap is load-bearing infrastructure, not a tuning detail.
"""

CLOSE = """\
**The campaign is finished — there are no outstanding runs.** In the order the evidence
supports, the paper's spine is:

1. **Our method is at least as good as the published competitor** on six systems and
   three learning rates, never losing, and beats it by ~2.9 mHa on the competitor's own
   benchmark molecules at the competitor's own settings.
2. **We know why** — the competitor's momentum ignores the learning rate, costing it a
   ~10× smaller effective step wherever the safety cap is not active. This predicts its
   failures rather than merely describing them.
3. **Untuned matches tuned, stated as robustness** — best average and worst-case shortfall
   of any single setting; not per-condition parity, which fails on two of seven.

One optional experiment (~45 GPU-hours) would tighten a caveat rather than change
anything: on nitrogen the best momentum sits at the edge of the grid we swept, so the
tuned baseline there is slightly understated.

Full detail, including the operational history and every result that went against us, is
in `docs/CAMPAIGN_LOG.md` in the repo.
"""

blocks = [
    wr.H1("What this is"), wr.MarkdownBlock(INTRO),
    wr.H1("Summary"), wr.MarkdownBlock(SUMMARY),
    wr.H1("What was run"), wr.MarkdownBlock(RAN),

    wr.H1("Result 1 — our method vs the published competitor"),
    wr.MarkdownBlock(MAIN),
    wr.MarkdownBlock(
        "**Below:** training on the competitor's own two molecules. The methods separate "
        "in momentum order, which is the whole point — the second panel shows the "
        "momentum each one actually used. Ours (top line in the legend) tracks the best "
        "hand-tuned settings; the competitor sits down with momentum 0.95."),
    wr.PanelGrid(runsets=arms(PC, "Config('x_experiment') == 'E10' and Config('x_system') == 'N2_eq'", C6),
                 panels=[curve("N2 molecule — energy during training", "energy_noclip",
                               "training step", "energy [Ha]"),
                         curve("N2 — final stretch, where methods separate", "energy_noclip",
                               "training step", "energy [Ha]", rng=(40000, 50000), sm=0.95)]),
    wr.PanelGrid(runsets=arms(PC, "Config('x_experiment') == 'E10' and Config('x_system') == 'CO'", C6),
                 panels=[curve("CO molecule — energy during training", "energy_noclip",
                               "training step", "energy [Ha]"),
                         curve("CO — momentum actually used by each method", "mu",
                               "training step", "momentum", sm=0.8)]),

    wr.H1("Result 2 — why the competitor breaks"),
    wr.MarkdownBlock(WHY),
    wr.MarkdownBlock(
        "**Left:** final error against learning rate on carbon. Everything sits near "
        "0.15–0.20 mHa except the competitor, which blows up to 0.435 at the smallest "
        "learning rate. **Right:** the cause — its momentum barely moves across a 10× "
        "change in learning rate, while the methods that scale properly do not."),
    wr.PanelGrid(runsets=arms(PC, "Config('x_experiment') == 'E11'", C4),
                 panels=[wr.ScatterPlot(title="carbon — final error vs learning rate (lower is better)",
                                        x=wr.Config("x_eta"), y=wr.SummaryMetric("x_eval_mHa"),
                                        log_x=True, layout=wr.Layout(w=12, h=9)),
                         wr.LinePlot(x="Step", y=["mu"], title="momentum vs training step",
                                     title_x="training step", title_y="momentum",
                                     smoothing_factor=0.8, smoothing_type="exponential",
                                     legend_position="east",
                                     legend_fields=["config:x_arm", "config:x_eta"],
                                     layout=wr.Layout(w=12, h=9))]),

    wr.H1("Result 3 — does it generalise?"),
    wr.MarkdownBlock(
        "Two atoms neither method had been run on before, 5 seeds each. **Nitrogen** is a "
        "clean win over the competitor (5 of 5 seeds) and a tie with tuned SPRING. "
        "**Oxygen is the campaign's worst case for us**: a tie with the competitor, but a "
        "loss to tuned SPRING on all 5 seeds. Note also that the best hand-tuned momentum "
        "*differs between the two atoms* — which is itself the argument for not having to "
        "choose it."),
    wr.PanelGrid(runsets=[wr.Runset(ENTITY, PC, name="nitrogen atom",
                                    filters="Config('x_experiment') == 'E9' and Config('x_system') == 'N'",
                                    order=[wr.OrderBy(wr.Config("x_arm"), ascending=True)])],
                 panels=[wr.BarPlot(metrics=["x_eval_mHa"], title="nitrogen — final error by method [mHa]",
                                    orientation="h", groupby="x_arm", groupby_aggfunc="mean",
                                    groupby_rangefunc="stderr", title_x="mHa above reference (lower better)",
                                    layout=wr.Layout(w=12, h=9))]),
    wr.PanelGrid(runsets=[wr.Runset(ENTITY, PC, name="oxygen atom",
                                    filters="Config('x_experiment') == 'E9' and Config('x_system') == 'O'",
                                    order=[wr.OrderBy(wr.Config("x_arm"), ascending=True)])],
                 panels=[wr.BarPlot(metrics=["x_eval_mHa"], title="oxygen — final error by method [mHa]",
                                    orientation="h", groupby="x_arm", groupby_aggfunc="mean",
                                    groupby_rangefunc="stderr", title_x="mHa above reference (lower better)",
                                    layout=wr.Layout(w=12, h=9))]),

    wr.H1("Result 4 — untuned vs optimally tuned"),
    wr.MarkdownBlock(TUNING),
    wr.MarkdownBlock(
        "**Below:** the original careful head-to-head, 5 seeds on two systems. On carbon "
        "our method ties the best hand-tuned setting and clearly beats the competitor. On "
        "H4 it is narrowly best overall — and note that **SPRING (tuned)** is the worst "
        "bar there: that setting was chosen by our own short screening sweep, and it did "
        "not survive a proper run. H4 is shown as raw energy because it has no reliable "
        "literature reference."),
    wr.PanelGrid(runsets=[wr.Runset(ENTITY, PB, name="carbon, 5-seed head-to-head",
                                    filters="Config('x_experiment') == 'E7' and Config('x_system') == 'carbon'",
                                    order=[wr.OrderBy(wr.Config("x_arm"), ascending=True)])],
                 panels=[wr.BarPlot(metrics=["x_eval_error_mha"], title="carbon — final error by method [mHa]",
                                    orientation="h", groupby="x_arm", groupby_aggfunc="mean",
                                    groupby_rangefunc="stderr", title_x="mHa above reference (lower better)",
                                    layout=wr.Layout(w=12, h=9))]),
    wr.PanelGrid(runsets=[wr.Runset(ENTITY, PB, name="H4 chain, 5-seed head-to-head",
                                    filters="Config('x_experiment') == 'E7' and Config('x_system') == 'H4'",
                                    order=[wr.OrderBy(wr.Config("x_arm"), ascending=True)])],
                 panels=[wr.BarPlot(metrics=["x_eval_energy"], title="H4 — final energy by method [Ha]",
                                    orientation="h", groupby="x_arm", groupby_aggfunc="mean",
                                    groupby_rangefunc="stderr", title_x="energy [Ha] (lower better)",
                                    layout=wr.Layout(w=12, h=9))]),

    wr.H1("Result 5 — how much momentum matters, and what is load-bearing"),
    wr.MarkdownBlock(
        "**Left:** the screening sweep that motivated the whole project — final error on "
        "carbon against hand-chosen momentum, single seed. Error falls by a factor of ~17 "
        "from no momentum to the best value, which is why this parameter gets tuned at "
        "all. The fine detail of this curve is *not* trustworthy — the settings it picks "
        "were later overturned by longer, multi-seed runs — but the scale of the effect "
        "is. **Right:** every method depends on a safety cap on the update size; with it "
        "removed, 10 of 15 runs diverge within a few hundred steps."),
    wr.PanelGrid(runsets=[wr.Runset(ENTITY, PA, name="carbon, momentum sweep",
                                    filters="Config('x_experiment') == 'E0' and Config('x_system') == 'carbon'",
                                    order=[wr.OrderBy(wr.Config("x_mu_fixed"), ascending=True)])],
                 panels=[wr.ScatterPlot(title="carbon — final error vs hand-chosen momentum",
                                        x=wr.Config("x_mu_fixed"),
                                        y=wr.SummaryMetric("x_final_error_mha"),
                                        log_y=True, layout=wr.Layout(w=12, h=9))]),
    wr.PanelGrid(runsets=[wr.Runset(ENTITY, PA, name="safety cap ON",
                                    filters="Config('x_experiment') == 'E2' and Config('x_constraint') == 'ON'",
                                    order=[wr.OrderBy(wr.Config("x_eta"), ascending=True)]),
                          wr.Runset(ENTITY, PA, name="safety cap OFF",
                                    filters="Config('x_experiment') == 'E2' and Config('x_constraint') == 'OFF'",
                                    order=[wr.OrderBy(wr.Config("x_eta"), ascending=True)])],
                 panels=[wr.BarPlot(metrics=["x_diverged"], title="fraction of runs that diverged, by method",
                                    orientation="h", groupby="x_optimizer", groupby_aggfunc="mean",
                                    title_x="fraction diverged", layout=wr.Layout(w=12, h=9))]),

    wr.H1("What we ruled out along the way"), wr.MarkdownBlock(RULED_OUT),
    wr.H1("Where this leaves the paper"), wr.MarkdownBlock(CLOSE),
]

report = wr.Report.from_url(URL)
print("editing report id:", report.id)
report.title = "SS-SPRING — full campaign summary"
report.description = "What we ran, what we tested, and what we found: three stages, 325 runs, six systems."
report.blocks = blocks
report.width = "fluid"          # from_url always reads 'readable'; must re-set
report.save()
print(report.url)
