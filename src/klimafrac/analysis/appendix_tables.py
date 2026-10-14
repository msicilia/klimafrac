"""
The appendix tables that are too long to type, as LaTeX, from the result
tables of analysis/thin_benchmark.py, analysis/size.py, analysis/tiers.py and
analysis/thin_robustness.py.

  benchmark_tokens.tex      the 45 conventional benchmark pools
  ladder_cleaning.tex       the ladder comparison under the 32 cleaning settings
  benchmark_cleaning.tex    the benchmark contrasts under the cleaning grid and
                            the dollar-floor family
  benchmark_variants.tex    the benchmark contrasts under every other family

Output: ../paper/tables/ (KLIMAFRAC_TABLE_DIR overrides)

Usage:
    uv run python -m klimafrac.analysis.appendix_tables
"""

import os
from pathlib import Path

import pandas as pd

from klimafrac.fetch.pools import POOLS
from klimafrac.paths import CODE_ROOT
from klimafrac.results import read_table

GROUP_TITLE = {"deep": "the deepest pools", "thin": "as thin as the carbon pools",
               "active": "busier than any carbon pool"}
DEX = {"sushiswap": "Sushiswap", "quickswap": "Quickswap"}


def out_dir() -> Path:
    out = Path(os.environ.get("KLIMAFRAC_TABLE_DIR", CODE_ROOT.parent / "paper" / "tables"))
    out.mkdir(parents=True, exist_ok=True)
    return out


def fmt(x: float, digits: int = 3, sign: bool = False) -> str:
    return ("$%+.*f$" if sign else "$%.*f$") % (digits, x)


def pval(x: float) -> str:
    return "$<0.001$" if x < 0.001 else "$%.3f$" % x


def ladder_cleaning() -> None:
    c = read_table("tiers_cleaning")
    raw = {"BCT": 476_404, "NCT": 50_760, "MCO2": 61_477}
    lines = [r"\begin{table}[H]", r"\centering\footnotesize",
             r"\caption{The ladder comparison under the 32 cleaning settings: the smallest "
             r"trade kept in dollars and in base units, and the factor of the rolling-median "
             r"filter (off: no filter). Kept: share of each pool's raw swaps that survive. "
             r"Gaps and $p$-values as in Table~\ref{tab:robust}; the paper's setting is "
             r"marked.}", r"\label{tab:ladder_cleaning}", r"\setlength{\tabcolsep}{3.5pt}",
             r"\begin{tabular}{rrlrrrrrrr}", r"\toprule",
             r"\$ & base & filter & BCT kept & NCT kept & MCO2 kept & NCT gap & $p$ & MCO2 gap & $p$ \\",
             r"\midrule"]
    c = c.assign(max_deviation=c.max_deviation.fillna(-1.0))
    for (usd, base_u, dev), rows in c.groupby(["min_usd", "min_base", "max_deviation"], sort=True):
        by = rows.set_index("comparison")
        n = by.loc["NCT vs BCT"]
        dev_s = "off" if dev < 0 else "%g" % dev
        mark = r"$^{*}$" if bool(n.default) else ""
        lines.append("%g%s & %g & %s & %.1f & %.1f & %.1f & %s & %s & %s & %s \\\\" % (
            usd, mark, base_u, dev_s,
            100 * n.trades_BCT / raw["BCT"], 100 * n.trades_NCT / raw["NCT"],
            100 * n.trades_MCO2 / raw["MCO2"],
            fmt(n.gap, sign=True), pval(n.p),
            fmt(by.loc["MCO2 vs BCT", "gap"], sign=True), pval(by.loc["MCO2 vs BCT", "p"])))
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    path = out_dir() / "ladder_cleaning.tex"
    path.write_text("\n".join(lines).replace("\\\\\\\\", "\\\\") + "\n")
    print("wrote", path)


def benchmark_variants() -> None:
    r = read_table("thin_robustness")
    g = read_table("thin_robustness_groups")
    kept = g.pivot_table(index="spec", columns="group", values="removed_share") if "removed_share" in g else None
    contrasts = (("thinness", "thin vs deep"), ("busy", "thin vs busy"), ("asset", "carbon vs thin"))
    for name, families, caption in (
            ("benchmark_cleaning", ("baseline", "cleaning", "floor"),
             "The benchmark contrasts under the cleaning grid and the dollar floor alone. "
             "For each contrast: the gap, the reassignment $p$ and the rank $p$."),
            ("benchmark_variants", ("baseline", "deep", "endpoint", "window", "s_min", "s_max",
                                    "scales", "order", "q", "fit"),
             "The benchmark contrasts under every other family of variants: the deep group "
             "with WETH counted once, the endpoint of the sample, the window length, the "
             "smallest and largest scale, the number of scales, the detrending order, the "
             "$q$ range and the fit filter. Widths under another $q$ range, window or order "
             "are not on the baseline's scale, so the gaps' signs and the tests are what is "
             "compared.")):
        lines = [r"\begin{table}[H]", r"\centering\footnotesize",
                 r"\caption{%s}" % caption, r"\label{tab:%s}" % name,
                 r"\setlength{\tabcolsep}{3pt}", r"\resizebox{\linewidth}{!}{%",
                 r"\begin{tabular}{llrrrrrrrrr}", r"\toprule",
                 r"family & variant & \multicolumn{3}{c}{thin vs deep} & "
                 r"\multicolumn{3}{c}{thin vs busy} & \multicolumn{3}{c}{carbon vs thin} \\",
                 r"\cmidrule(lr){3-5}\cmidrule(lr){6-8}\cmidrule(lr){9-11}",
                 r" & & gap & reass. $p$ & rank $p$ & gap & reass. $p$ & rank $p$ & gap & reass. $p$ & rank $p$ \\",
                 r"\midrule"]
        sub = r[r.family.isin(families)]
        for spec, rows in sub.groupby("spec", sort=False):
            by = rows.set_index("contrast")
            cells = []
            for c, _ in contrasts:
                cells += [fmt(by.loc[c, "gap"], sign=True), pval(by.loc[c, "token_p"]),
                          pval(by.loc[c, "rank_p"])]
            fam = rows.family.iloc[0].replace("_", r"\_")
            label = spec.replace("_", r"\_").replace("min_usd", "min\_usd").replace(">=", r"$\geq$")
            lines.append("%s & %s & %s \\\\" % (fam, label, " & ".join(cells)))
        lines += [r"\bottomrule", r"\end{tabular}}", r"\end{table}"]
        path = out_dir() / ("%s.tex" % name)
        path.write_text("\n".join(lines).replace("\\\\\\\\", "\\\\") + "\n")
        print("wrote", path)


def main() -> None:
    tokens = read_table("thin_benchmark_tokens")
    size = read_table("size_tokens")[["token", "snapshot_rank", "check_rank"]]
    t = tokens.merge(size, on="token", how="left")
    dex = {p.symbol: DEX.get(p.dex, p.dex) for p in POOLS if p.chain == "polygon"}
    lines = [r"\begin{table}[H]", r"\centering\small",
             r"\caption{The conventional benchmark pools. Usable trades and the no-trade "
             r"share of hours are over the estimation sample. Rank is the asset's "
             r"market-capitalisation rank on 3 October 2021, before the sample, and in "
             r"parentheses on 13 February 2022 where it differs; a dash marks an asset "
             r"not in the ranking. WETH, WBTC and WPOL take the rank of the coin they "
             r"wrap.}", r"\label{tab:benchmark}", r"\setlength{\tabcolsep}{4pt}",
             r"\begin{tabular}{llrrr}", r"\toprule",
             r"Token & Exchange & Usable trades & No-trade \% & Rank \\", r"\midrule"]
    for group in ("deep", "thin", "active"):
        g = t[t.group == group].sort_values("no_trade")
        lines.append(rf"\multicolumn{{5}}{{l}}{{\emph{{{GROUP_TITLE[group]} ({len(g)})}}}} \\")
        for r in g.itertuples():
            name = r.token.replace("_S", "").replace("_Q", "")
            rank = "--" if pd.isna(r.snapshot_rank) else f"{int(r.snapshot_rank):,}"
            if not pd.isna(r.check_rank) and (pd.isna(r.snapshot_rank)
                                              or int(r.check_rank) != int(r.snapshot_rank)):
                rank += f" ({int(r.check_rank):,})"
            lines.append(f"{name} & {dex[r.token]} & {r.trades:,} & {100 * r.no_trade:.1f} & "
                         f"{rank} \\\\".replace(",", "{,}"))
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    path = out_dir() / "benchmark_tokens.tex"
    path.write_text("\n".join(lines) + "\n")
    print("wrote", path, f"({len(t)} tokens)")
    ladder_cleaning()
    benchmark_variants()


if __name__ == "__main__":
    main()
