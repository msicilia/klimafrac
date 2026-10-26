"""
What pricing a swap by execution rather than from the pool's reserves does to
the hourly positive-side width: the paper's paragraph on the price itself.

For every pool whose swap file carries both prices, the hourly grid and the
non-overlapping windows of trades.py are built twice, once from the price
paths.PRICE names (the pool price in the paper's configuration) and once from
the other, and the median window width of the token is recorded under each.
The aggregated Bitcoin and Ether series of analysis/asset_class.py, which
blend venues and carry no fee, are the outside reference for the deep pools.

Outputs (via klimafrac.results):
  data/results/price_definition.csv    one row per token: median width under
                                       each price, trades, no-trade share
  data/results/price_definition.json   the numbers for the manuscript

Usage:
    uv run python -m klimafrac.analysis.price_definition
"""

import numpy as np
import pandas as pd

from klimafrac import trades
from klimafrac.analysis.thin_benchmark import GROUPS
from klimafrac.paths import PRICE
from klimafrac.results import numbers, read_table, table

OTHER = {"pool": "execution", "execution": "pool"}[PRICE]


def widths(df: pd.DataFrame) -> float:
    grid = trades.calendar(df, freq="1h")
    w = trades.windows(grid.ret.to_numpy())
    return float(w.width_pos.median()) if len(w) else np.nan


def main() -> None:
    rows = []
    for group, tokens in GROUPS.items():
        for token, key in tokens.items():
            raw = trades.load(key)
            if "execution_price" not in raw:
                raise SystemExit("run under KLIMAFRAC_PRICE=pool, where both prices are loaded")
            d = trades.sample(trades.clean(raw, key))
            other = d.assign(price=d.execution_price if PRICE == "pool" else d.pool_price)
            rows.append({"token": token, "group": group, "trades": len(d),
                         f"width_{PRICE}": widths(d), f"width_{OTHER}": widths(other)})
    t = pd.DataFrame(rows)
    t["difference"] = t[f"width_{OTHER}"] - t[f"width_{PRICE}"]
    table("price_definition", t)

    agg = read_table("asset_class_windows")
    agg = agg[agg.group == "aggregated"].groupby("token").width_pos.median()
    out = {"price_definition.price": PRICE, "price_definition.other": OTHER}
    for r in t.itertuples():
        pre = f"price_definition.{r.token}."
        out[pre + PRICE] = getattr(r, f"width_{PRICE}")
        out[pre + OTHER] = getattr(r, f"width_{OTHER}")
        out[pre + "difference"] = r.difference
    for token, v in agg.items():
        out[f"price_definition.aggregated.{token}"] = float(v)
    for group in GROUPS:
        g = t[t.group == group]
        out[f"price_definition.{group}.median_{PRICE}"] = float(g[f"width_{PRICE}"].median())
        out[f"price_definition.{group}.median_{OTHER}"] = float(g[f"width_{OTHER}"].median())
        out[f"price_definition.{group}.abs_difference_max"] = float(g.difference.abs().max())
    print("wrote", numbers("price_definition", out))
    pd.set_option("display.width", 200)
    print(t[t.group.isin(("carbon", "deep"))].round(3).to_string(index=False))
    print("aggregated:", agg.round(3).to_dict())


if __name__ == "__main__":
    main()
