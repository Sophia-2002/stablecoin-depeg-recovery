# Data

The notebook expects `stablecoin_events_dataset.csv` in this folder.

- One row = one depeg event (135 events, 9 stablecoins)
- Built from daily stablecoin, BTC and ETH price, volume and market-cap data by a separate event-detection script
- Source: CoinGecko 

Key columns: `coin`, `type`, `event_start`, `event_end`, `duration_days`, `initial_deviation_pct`, `direction`, `label` (fast ≤3d / slow >3d), plus 33 feature columns measured at t-1.
