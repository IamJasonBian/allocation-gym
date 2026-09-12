"""
ETF Flow Indicator — Estimated BTC spot ETF net flows.

Pre-fetches ETF + BTC data in __init__, computes flows via
premium/discount method, serves values via date lookup in next().
"""

import backtrader as bt
import numpy as np
import pandas as pd

from allocation_gym.signals.btc_dashboard import (
    BTC_ETFS,
    fetch_btc_daily,
    fetch_etf_data,
)


class ETFFlowIndicator(bt.Indicator):
    """
    Estimated BTC ETF net flows from premium/discount vs spot BTC.

    Lines:
        daily_flow:      Single-day estimated net flow (USD)
        cumulative_flow:  Running cumulative net flow (USD)
    """

    lines = ("daily_flow", "cumulative_flow")

    params = (
        ("fetch_days", 730),
    )

    plotinfo = dict(subplot=True)

    def __init__(self):
        super().__init__()
        self._daily_map = {}       # date -> daily flow
        self._cumulative_map = {}  # date -> cumulative flow

        btc_df = fetch_btc_daily(days=self.p.fetch_days)
        etf_data = fetch_etf_data(days=self.p.fetch_days)

        if not etf_data:
            return

        btc_ret = btc_df["Close"].pct_change()
        agg_flows = None

        for ticker, df in etf_data.items():
            if len(df) < 5:
                continue
            etf_ret = df["Close"].pct_change()
            dollar_vol = (df["Close"] * df["Volume"]).fillna(0)

            common = etf_ret.index.intersection(btc_ret.index)
            if len(common) < 5:
                continue

            premium = etf_ret.loc[common] - btc_ret.loc[common]
            flow_sign = np.sign(premium).fillna(0)
            daily_flow = (dollar_vol.loc[common] * flow_sign).fillna(0)

            if agg_flows is None:
                agg_flows = daily_flow.copy()
            else:
                agg_flows = agg_flows.add(daily_flow, fill_value=0)

        if agg_flows is None:
            return

        cum_flows = agg_flows.cumsum()

        for ts, val in agg_flows.items():
            dt = ts.date() if hasattr(ts, "date") else ts
            self._daily_map[dt] = float(val)

        for ts, val in cum_flows.items():
            dt = ts.date() if hasattr(ts, "date") else ts
            self._cumulative_map[dt] = float(val)

    def next(self):
        dt = self.data.datetime.date(0)
        self.lines.daily_flow[0] = self._daily_map.get(dt, 0.0)
        self.lines.cumulative_flow[0] = self._cumulative_map.get(dt, 0.0)
