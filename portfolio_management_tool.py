import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from datetime import datetime
import json

# Load Configuration
print("=" * 60)
print("PORTFOLIO MANAGEMENT TOOL")
print("=" * 60)

# Load optimal weights from file
try:
    with open('optimal_weights.json', 'r') as f:
        weights_data = json.load(f)
    
    tickers = weights_data['tickers']
    w_mvp = np.array(weights_data['weights'])
    risk_free_rate = weights_data['risk_free_rate']
    
    print("\nLoaded optimal weights from previous optimization:")
    print(f"Optimization Date: {weights_data['optimization_date']}")
    print(f"Expected Return: {weights_data['expected_return']:.2%}")
    print(f"Expected Volatility: {weights_data['expected_volatility']:.2%}")
    print(f"Expected Sharpe: {weights_data['sharpe_ratio']:.2f}")
    print("\nWeights:")
    for t, w in zip(tickers, w_mvp):
        if w > 0.001:
            print(f"  {t:10s}: {w:6.2%}")
    
except FileNotFoundError:
    print("\nERROR: 'optimal_weights.json' not found!")
    print("Please run the stock_weighting_tool.py first to generate optimal weights.")
    exit(1)

# Date configuration
simulation_start = weights_data['optimization_date']
simulation_end = datetime.today().strftime("%Y-%m-%d")

# Stop-loss parameters
POSITION_DD_THRESHOLD = -0.20  # -20% drawdown
POSITION_CUT_PCT = 0.30  # Cut to 30% (reduce by 70%)
POSITION_RECOVERY_DD = -0.10  # Drawdown must be within -10%
POSITION_RESTORE_50 = 0.50  # First restoration to 50%

PORTFOLIO_VOL_MULTIPLIER = 1.3  # Volatility breach at 1.3x target
TARGET_VOLATILITY = 0.17  # 17% target volatility
VOL_LOOKBACK = 20  # 20-day realized volatility

PORTFOLIO_DD_HARD = -0.15  # -15% drawdown
PORTFOLIO_CUT_PCT = 0.50  # Cut to 50%
PORTFOLIO_RECOVERY_DD = -0.05  # Must recover to -5%

# Data Acquisition - Simulation Period
print("\n" + "=" * 60)
print("DOWNLOADING SIMULATION DATA")
print("=" * 60)
print(f"Simulation period: {simulation_start} to {simulation_end}")

prices_sim = yf.download(tickers, start=simulation_start, end=simulation_end, auto_adjust=True)["Close"]
returns_sim = prices_sim.pct_change().dropna()

print(f"Simulation prices: {len(prices_sim)} days")
print(f"Simulation returns: {len(returns_sim)} days")
print(f"Date range: {prices_sim.index[0].date()} to {prices_sim.index[-1].date()}")

if len(returns_sim) < 2:
    print("\nWARNING: Insufficient simulation data. Need at least 2 days of returns.")
    print("This may be because the simulation start date is very recent.")

# Portfolio Manager Class
class PortfolioManager:
    def __init__(self, weights, returns, tickers):
        self.base_weights = weights.copy()
        self.returns = returns
        self.tickers = tickers
        self.n_assets = len(tickers)

        # Position tracking
        self.position_highs = np.ones(self.n_assets)
        self.position_exposure = np.ones(self.n_assets)  # 1.0 = 100% exposure
        self.position_frozen = np.zeros(self.n_assets, dtype=bool)
        self.position_restored_50 = np.zeros(self.n_assets, dtype=bool)
        self.position_pre_stop_vol = np.zeros(self.n_assets)

        # Portfolio tracking
        self.portfolio_high = 1.0
        self.portfolio_exposure = 1.0

        # History
        self.history = []

    def calculate_position_drawdown(self, current_values, position_idx):
        dd = (current_values[position_idx] / self.position_highs[position_idx]) - 1
        return dd

    def calculate_portfolio_drawdown(self, portfolio_value):
        dd = (portfolio_value / self.portfolio_high) - 1
        return dd

    def calculate_realized_volatility(self, returns_array, window=20):
        #Rolling volatility
        if len(returns_array) < window:
            return np.std(returns_array) * np.sqrt(252)
        return np.std(returns_array[-window:]) * np.sqrt(252)

    def apply_position_stops(self, t, position_values, position_returns_history):
        for i in range(self.n_assets):
            if self.position_frozen[i]:
                continue

            # Update high watermark
            if position_values[i] > self.position_highs[i]:
                self.position_highs[i] = position_values[i]

            dd = self.calculate_position_drawdown(position_values, i)

            # Check for stop trigger
            if dd <= POSITION_DD_THRESHOLD and self.position_exposure[i] == 1.0:
                # Store pre-stop volatility
                if len(position_returns_history[i]) >= VOL_LOOKBACK:
                    self.position_pre_stop_vol[i] = self.calculate_realized_volatility(
                        position_returns_history[i], VOL_LOOKBACK
                    )

                self.position_exposure[i] = POSITION_CUT_PCT
                print(f"  [{self.returns.index[t].date()}] {self.tickers[i]}: "
                      f"STOP triggered at {dd:.2%} DD, cut to {POSITION_CUT_PCT:.0%}")

            # Check for recovery and restoration
            elif self.position_exposure[i] < 1.0:
                # Calculate 20-day volatility
                if len(position_returns_history[i]) >= VOL_LOOKBACK:
                    vol_now = self.calculate_realized_volatility(
                        position_returns_history[i], VOL_LOOKBACK
                    )

                    # Recovery conditions
                    if dd >= POSITION_RECOVERY_DD and vol_now < self.position_pre_stop_vol[i]:
                        if not self.position_restored_50[i]:
                            # First restoration to 50%
                            self.position_exposure[i] = POSITION_RESTORE_50
                            self.position_restored_50[i] = True
                            print(f"  [{self.returns.index[t].date()}] {self.tickers[i]}: "
                                  f"Restored to 50% (DD={dd:.2%}, Vol={vol_now:.2%} < Pre-stop={self.position_pre_stop_vol[i]:.2%})")
                        elif dd >= 0 and self.position_exposure[i] == POSITION_RESTORE_50:
                            # Full restoration only if no further drawdown
                            self.position_exposure[i] = 1.0
                            self.position_restored_50[i] = False
                            self.position_pre_stop_vol[i] = 0
                            print(f"  [{self.returns.index[t].date()}] {self.tickers[i]}: "
                                  f"Fully restored (DD={dd:.2%})")

                    # Freeze if drawdown during re-entry
                    elif self.position_restored_50[i] and dd < POSITION_RECOVERY_DD:
                        self.position_frozen[i] = True
                        print(f"  [{self.returns.index[t].date()}] {self.tickers[i]}: "
                              f"FROZEN due to drawdown during re-entry (DD={dd:.2%})")

    def apply_portfolio_stops(self, t, portfolio_value, portfolio_returns_history):
        # Update portfolio high watermark
        if portfolio_value > self.portfolio_high:
            self.portfolio_high = portfolio_value

        portfolio_dd = self.calculate_portfolio_drawdown(portfolio_value)

        # Hard kill stop
        if portfolio_dd <= PORTFOLIO_DD_HARD and self.portfolio_exposure == 1.0:
            self.portfolio_exposure = PORTFOLIO_CUT_PCT
            print(f"\n  *** [{self.returns.index[t].date()}] PORTFOLIO HARD STOP ***")
            print(f"  *** {portfolio_dd:.2%} DD, cutting to {PORTFOLIO_CUT_PCT:.0%} ***\n")

        # Recovery from hard stop
        elif self.portfolio_exposure == PORTFOLIO_CUT_PCT and portfolio_dd >= PORTFOLIO_RECOVERY_DD:
            self.portfolio_exposure = 1.0
            print(f"\n  *** [{self.returns.index[t].date()}] PORTFOLIO RESTORED ***")
            print(f"  *** DD recovered to {portfolio_dd:.2%} ***\n")

        # Volatility breach stop
        if len(portfolio_returns_history) >= VOL_LOOKBACK:
            realized_vol = self.calculate_realized_volatility(
                portfolio_returns_history, VOL_LOOKBACK
            )

            if realized_vol > PORTFOLIO_VOL_MULTIPLIER * TARGET_VOLATILITY:
                scale_factor = TARGET_VOLATILITY / realized_vol
                old_exposure = self.portfolio_exposure
                self.portfolio_exposure = min(old_exposure, scale_factor)  # Don't scale up
                if self.portfolio_exposure < old_exposure:
                    print(f"  [{self.returns.index[t].date()}] VOLATILITY BREACH: "
                          f"{realized_vol:.2%} > {PORTFOLIO_VOL_MULTIPLIER * TARGET_VOLATILITY:.2%}, "
                          f"scaled from {old_exposure:.2%} to {self.portfolio_exposure:.2%}")

            # Recovery from volatility breach (requires 1 month below target)
            elif realized_vol < TARGET_VOLATILITY and self.portfolio_exposure < 1.0:
                # Check if vol has been below target for rolling 1-month window (20 days)
                if len(portfolio_returns_history) >= VOL_LOOKBACK + 20:
                    # Check last 20 days of 20-day rolling vol
                    all_below = True
                    for i in range(20):
                        idx_start = len(portfolio_returns_history) - i - VOL_LOOKBACK
                        idx_end = len(portfolio_returns_history) - i
                        if idx_start < 0:
                            all_below = False
                            break
                        window_vol = self.calculate_realized_volatility(
                            portfolio_returns_history[idx_start:idx_end], VOL_LOOKBACK
                        )
                        if window_vol >= TARGET_VOLATILITY:
                            all_below = False
                            break

                    if all_below:
                        old_exposure = self.portfolio_exposure
                        self.portfolio_exposure = 1.0
                        print(f"  [{self.returns.index[t].date()}] VOLATILITY RECOVERED: "
                              f"Vol={realized_vol:.2%}, restored from {old_exposure:.2%} to 100%")

    def get_active_weights(self):
        # Apply position-level exposure
        adjusted = self.base_weights * self.position_exposure

        # Apply portfolio-level exposure
        adjusted = adjusted * self.portfolio_exposure

        # Renormalize to sum to portfolio_exposure
        if adjusted.sum() > 0:
            adjusted = adjusted / adjusted.sum() * self.portfolio_exposure

        return adjusted

    def simulate(self):
        n_returns = len(self.returns)

        # Initialize arrays
        portfolio_values = np.zeros(n_returns + 1)
        portfolio_values[0] = 1.0

        position_values = np.ones((n_returns + 1, self.n_assets))
        position_returns_list = [[] for _ in range(self.n_assets)]
        portfolio_returns_list = []

        print("\n" + "=" * 60)
        print("RUNNING STOP-LOSS SIMULATION")
        print("=" * 60)
        print(f"Simulating {n_returns} days of returns")

        for t in range(n_returns):
            # Update individual position values
            for i in range(self.n_assets):
                position_values[t + 1, i] = position_values[t, i] * (1 + self.returns.iloc[t, i])
                position_returns_list[i].append(self.returns.iloc[t, i])

            # Convert to numpy arrays for stop-loss checks
            position_returns_history = [np.array(prl) for prl in position_returns_list]

            # Apply position-level stops
            self.apply_position_stops(t, position_values[t + 1], position_returns_history)

            # Get active weights
            active_weights = self.get_active_weights()

            # Calculate portfolio return
            port_return = np.sum(active_weights * self.returns.iloc[t].values)
            portfolio_values[t + 1] = portfolio_values[t] * (1 + port_return)
            portfolio_returns_list.append(port_return)

            # Apply portfolio-level stops
            self.apply_portfolio_stops(t, portfolio_values[t + 1],
                                       np.array(portfolio_returns_list))

            # Record history
            self.history.append({
                'date': self.returns.index[t],
                'portfolio_value': portfolio_values[t + 1],
                'portfolio_exposure': self.portfolio_exposure,
                'position_exposure': self.position_exposure.copy(),
                'weights': active_weights.copy(),
                'portfolio_dd': self.calculate_portfolio_drawdown(portfolio_values[t + 1])
            })

        # Convert to pandas Series for easier handling
        dates = self.returns.index
        portfolio_returns_series = pd.Series(portfolio_returns_list, index=dates)

        return portfolio_values, portfolio_returns_series

# Run Simulation
if len(returns_sim) >= 2:
    print("\n>>> Minimum Variance Portfolio")
    pm_mvp = PortfolioManager(w_mvp, returns_sim, tickers)
    pv_mvp, pr_mvp = pm_mvp.simulate()

    # Performance Summary
    print("\n" + "=" * 60)
    print("SIMULATION RESULTS")
    print("=" * 60)
    print(f"Simulation Period: {returns_sim.index[0].date()} to {returns_sim.index[-1].date()}")
    print(f"Number of Trading Days: {len(returns_sim)}")

    def print_performance(name, portfolio_values, returns_series, manager):
        total_return = portfolio_values[-1] - 1

        if len(returns_series) > 1:
            ann_return = returns_series.mean() * 252
            ann_vol = returns_series.std() * np.sqrt(252)
            sharpe = (ann_return - risk_free_rate) / ann_vol if ann_vol > 0 else 0
        else:
            ann_return = 0
            ann_vol = 0
            sharpe = 0

        max_dd = (portfolio_values / np.maximum.accumulate(portfolio_values) - 1).min()

        # Count stop events
        position_stops = sum(manager.position_exposure < 1.0)
        position_frozen = sum(manager.position_frozen)
        portfolio_stopped = manager.portfolio_exposure < 1.0

        print(f"\n{name}:")
        print(f"  Total Return:          {total_return:7.2%}")
        print(f"  Annualized Return:     {ann_return:7.2%}")
        print(f"  Annualized Volatility: {ann_vol:7.2%}")
        print(f"  Sharpe Ratio:          {sharpe:7.2f}")
        print(f"  Max Drawdown:          {max_dd:7.2%}")
        print(f"  Final Value:           ${portfolio_values[-1]:7.2f}")
        print(f"  Current Exposure:      {manager.portfolio_exposure:7.2%}")
        print(f"\n  Stop-Loss Summary:")
        print(f"    Positions stopped:   {position_stops}/{manager.n_assets}")
        print(f"    Positions frozen:    {position_frozen}/{manager.n_assets}")
        print(f"    Portfolio stopped:   {'Yes' if portfolio_stopped else 'No'}")

        print(f"\n  Current Position Exposures:")
        for t, exp in zip(tickers, manager.position_exposure):
            if exp < 1.0 or manager.position_frozen[manager.tickers.index(t)]:
                status = " (FROZEN)" if manager.position_frozen[manager.tickers.index(t)] else ""
                print(f"    {t:10s}: {exp:6.2%}{status}")

    print_performance("Minimum Variance Portfolio", pv_mvp, pr_mvp, pm_mvp)

    # Visualization
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Create date index for portfolio values (one extra point for initial value)
    dates_with_initial = pd.DatetimeIndex([returns_sim.index[0] - pd.Timedelta(days=1)] +
                                          list(returns_sim.index))

    # Portfolio values
    axes[0, 0].plot(dates_with_initial, pv_mvp, label='Min Variance', linewidth=2, color='green')
    axes[0, 0].set_title(f'Portfolio Value: {returns_sim.index[0].date()} to {returns_sim.index[-1].date()}',
                         fontweight='bold')
    axes[0, 0].set_ylabel('Portfolio Value ($)')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Drawdowns
    dd_mvp = pv_mvp / np.maximum.accumulate(pv_mvp) - 1
    axes[0, 1].fill_between(dates_with_initial, dd_mvp * 100, 0, alpha=0.5, label='Min Variance', color='green')
    axes[0, 1].axhline(y=PORTFOLIO_DD_HARD * 100, color='r', linestyle='--',
                       linewidth=2, label='Hard Stop (-15%)')
    axes[0, 1].axhline(y=POSITION_DD_THRESHOLD * 100, color='orange', linestyle='--',
                       linewidth=1, label='Position Stop (-20%)')
    axes[0, 1].set_title('Portfolio Drawdown', fontweight='bold')
    axes[0, 1].set_ylabel('Drawdown (%)')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # Rolling volatility
    if len(pr_mvp) >= VOL_LOOKBACK:
        roll_vol_mvp = pr_mvp.rolling(VOL_LOOKBACK).std() * np.sqrt(252) * 100
        axes[1, 0].plot(returns_sim.index[VOL_LOOKBACK - 1:], roll_vol_mvp.iloc[VOL_LOOKBACK - 1:],
                        label='Min Variance', linewidth=2, color='green')
        axes[1, 0].axhline(y=TARGET_VOLATILITY * 100, color='g', linestyle='--',
                           linewidth=2, label='Target Vol (17%)')
        axes[1, 0].axhline(y=TARGET_VOLATILITY * PORTFOLIO_VOL_MULTIPLIER * 100,
                           color='r', linestyle='--', linewidth=2, label='Breach (22.1%)')
        axes[1, 0].set_title(f'{VOL_LOOKBACK}-Day Rolling Volatility', fontweight='bold')
        axes[1, 0].set_ylabel('Volatility (%)')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
    else:
        axes[1, 0].text(0.5, 0.5, 'Insufficient data for volatility calculation',
                        ha='center', va='center', transform=axes[1, 0].transAxes)
        axes[1, 0].set_title(f'{VOL_LOOKBACK}-Day Rolling Volatility', fontweight='bold')

    # Exposure over time
    mvp_exposure = [h['portfolio_exposure'] for h in pm_mvp.history]
    exposure_dates = [h['date'] for h in pm_mvp.history]

    axes[1, 1].plot(exposure_dates, np.array(mvp_exposure) * 100,
                    label='Min Variance', linewidth=2, color='green')
    axes[1, 1].axhline(y=100, color='g', linestyle='--', alpha=0.5, label='Full Exposure')
    axes[1, 1].axhline(y=50, color='r', linestyle='--', alpha=0.5, label='Hard Stop Level')
    axes[1, 1].set_title('Portfolio Exposure Over Time', fontweight='bold')
    axes[1, 1].set_ylabel('Exposure (%)')
    axes[1, 1].set_ylim([0, 105])
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('portfolio_simulation.png', dpi=300, bbox_inches='tight')
    print("\n" + "=" * 60)
    print("Simulation plots saved as 'portfolio_simulation.png'")
    print("=" * 60)

    # Individual Stock Drawdown Analysis
    print("\n" + "=" * 60)
    print("INDIVIDUAL STOCK DRAWDOWN ANALYSIS")
    print("=" * 60)

    # Calculate individual stock cumulative values starting from 1.0
    individual_values = np.ones((len(returns_sim) + 1, len(tickers)))
    for t in range(len(returns_sim)):
        for i in range(len(tickers)):
            individual_values[t + 1, i] = individual_values[t, i] * (1 + returns_sim.iloc[t, i])

    # Calculate drawdowns for each stock
    individual_dds = np.zeros_like(individual_values)
    for i in range(len(tickers)):
        cummax = np.maximum.accumulate(individual_values[:, i])
        individual_dds[:, i] = (individual_values[:, i] / cummax - 1) * 100

    def find_drawdown_periods(portfolio_values, threshold_pct=-5.0):
        """Find all periods where drawdown exceeds threshold"""
        dd = (portfolio_values / np.maximum.accumulate(portfolio_values) - 1) * 100
        in_drawdown = dd < threshold_pct

        periods = []
        start_idx = None

        for i in range(len(in_drawdown)):
            if in_drawdown[i] and start_idx is None:
                start_idx = i
            elif not in_drawdown[i] and start_idx is not None:
                periods.append((start_idx, i - 1))
                start_idx = None

        # If still in drawdown at end
        if start_idx is not None:
            periods.append((start_idx, len(in_drawdown) - 1))

        return periods, dd

    # Analyze Minimum Variance Portfolio
    dd_periods_mvp, portfolio_dd_mvp = find_drawdown_periods(pv_mvp, threshold_pct=-5.0)

    if len(dd_periods_mvp) > 0:
        print(f"\nFound {len(dd_periods_mvp)} significant drawdown period(s) (>5%)")

        for period_num, (start_idx, end_idx) in enumerate(dd_periods_mvp, 1):
            start_date = dates_with_initial[start_idx]
            end_date = dates_with_initial[end_idx]
            max_dd = portfolio_dd_mvp[start_idx:end_idx + 1].min()

            print(f"\n{'=' * 60}")
            print(f"DRAWDOWN PERIOD #{period_num}")
            print(f"{'=' * 60}")
            print(f"Period: {start_date.date()} to {end_date.date()}")
            print(f"Duration: {(end_date - start_date).days} days")
            print(f"Max Portfolio Drawdown: {max_dd:.2f}%")

            # Calculate individual stock performance during this period
            print(f"\nIndividual Stock Performance During Drawdown:")
            print(f"{'Ticker':<10} {'Start DD%':<12} {'Max DD%':<12} {'End DD%':<12} {'Total Return%':<15}")
            print("-" * 65)

            stock_performance = []
            for i, ticker in enumerate(tickers):
                start_dd = individual_dds[start_idx, i]
                max_dd_stock = individual_dds[start_idx:end_idx + 1, i].min()
                end_dd = individual_dds[end_idx, i]

                # Calculate total return during period
                if start_idx > 0:
                    period_return = (individual_values[end_idx, i] / individual_values[start_idx, i] - 1) * 100
                else:
                    period_return = (individual_values[end_idx, i] - 1) * 100

                stock_performance.append({
                    'ticker': ticker,
                    'start_dd': start_dd,
                    'max_dd': max_dd_stock,
                    'end_dd': end_dd,
                    'return': period_return
                })

                print(f"{ticker:<10} {start_dd:>10.2f}% {max_dd_stock:>10.2f}% {end_dd:>10.2f}% {period_return:>13.2f}%")

            # Sort by worst performance
            stock_performance.sort(key=lambda x: x['max_dd'])

            print(f"\nWorst Performers (by max drawdown):")
            for i, perf in enumerate(stock_performance[:3], 1):
                print(f"  {i}. {perf['ticker']}: {perf['max_dd']:.2f}% max DD, {perf['return']:.2f}% return")

            print(f"\nBest Performers (by max drawdown):")
            for i, perf in enumerate(reversed(stock_performance[-3:]), 1):
                print(f"  {i}. {perf['ticker']}: {perf['max_dd']:.2f}% max DD, {perf['return']:.2f}% return")
    else:
        print("\nNo significant drawdown periods (>5%) detected.")

    # Visualize Individual Stock Drawdowns
    fig2, axes2 = plt.subplots(3, 1, figsize=(16, 14))

    # Plot 1: Individual Stock Cumulative Performance
    for i, ticker in enumerate(tickers):
        axes2[0].plot(dates_with_initial, individual_values[:, i],
                      label=ticker, linewidth=1.5, alpha=0.7)

    axes2[0].axhline(y=1.0, color='black', linestyle='--', linewidth=1, alpha=0.5)
    axes2[0].set_title('Individual Stock Cumulative Performance', fontweight='bold', fontsize=12)
    axes2[0].set_ylabel('Cumulative Value ($)', fontsize=11)
    axes2[0].legend(loc='best', ncol=2)
    axes2[0].grid(True, alpha=0.3)

    # Plot 2: Individual Stock Drawdowns
    for i, ticker in enumerate(tickers):
        axes2[1].plot(dates_with_initial, individual_dds[:, i],
                      label=ticker, linewidth=1.5, alpha=0.7)

    axes2[1].axhline(y=0, color='black', linestyle='-', linewidth=1, alpha=0.5)
    axes2[1].axhline(y=POSITION_DD_THRESHOLD * 100, color='red', linestyle='--',
                     linewidth=2, label=f'Position Stop ({POSITION_DD_THRESHOLD * 100:.0f}%)')
    axes2[1].set_title('Individual Stock Drawdowns', fontweight='bold', fontsize=12)
    axes2[1].set_ylabel('Drawdown (%)', fontsize=11)
    axes2[1].legend(loc='best', ncol=2)
    axes2[1].grid(True, alpha=0.3)

    # Plot 3: Portfolio vs Average Stock Drawdown
    avg_stock_dd = individual_dds.mean(axis=1)

    axes2[2].plot(dates_with_initial, (pv_mvp / np.maximum.accumulate(pv_mvp) - 1) * 100,
                  label='Min Variance Portfolio', linewidth=2.5, color='green')
    axes2[2].plot(dates_with_initial, avg_stock_dd,
                  label='Average Stock Drawdown', linewidth=2, linestyle='--',
                  color='gray', alpha=0.7)

    axes2[2].axhline(y=0, color='black', linestyle='-', linewidth=1, alpha=0.5)
    axes2[2].axhline(y=PORTFOLIO_DD_HARD * 100, color='red', linestyle='--',
                     linewidth=2, label=f'Portfolio Hard Stop ({PORTFOLIO_DD_HARD * 100:.0f}%)')
    axes2[2].set_title('Portfolio Drawdown vs Average Stock Drawdown', fontweight='bold', fontsize=12)
    axes2[2].set_ylabel('Drawdown (%)', fontsize=11)
    axes2[2].set_xlabel('Date', fontsize=11)
    axes2[2].legend(loc='best')
    axes2[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('individual_stock_drawdowns.png', dpi=300, bbox_inches='tight')
    print("\nIndividual stock drawdown charts saved as 'individual_stock_drawdowns.png'")

    plt.show()

else:
    print("\nSimulation cannot be performed with insufficient data.")

print("\n" + "=" * 60)
print("PORTFOLIO MANAGEMENT ANALYSIS COMPLETE")
print("=" * 60)
