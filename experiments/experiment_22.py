
from edgeguard.historical_data import load_historical_data
from edgeguard.historical_strategy import moving_average_strategy
from edgeguard.data_split import chronological_split
from edgeguard.backtest import run_backtest


DATA_PATH = "data/raw/SPY.csv"
WINDOW = 50
TRANSACTION_COST_BPS = 5.0


def print_metrics(name, metrics):
    print(f"\n{name}")
    print("-" * 40)

    print(
        f"Total return: "
        f"{metrics['total_return']:.2%}"
    )
    print(
        f"Maximum drawdown: "
        f"{metrics['max_drawdown']:.2%}"
    )
    print(
        f"Annualized Sharpe ratio: "
        f"{metrics['annualized_sharpe']:.3f}"
    )
    print(
        f"Total turnover: "
        f"{metrics['total_turnover']:.2f}"
    )


def main():
    print("EDGEGUARD - Experiment 22")
    print("Historical Moving-Average Backtest")
    print("=" * 50)

    # Calculate signals using chronological history.
    # No future prices are used for exposure decisions.
    prices = load_historical_data(DATA_PATH)
    strategy = moving_average_strategy(
        prices,
        window=WINDOW,
    )

    # Split after calculating causal signals so that
    # rolling history carries across boundaries.
    train, validation, _ = chronological_split(strategy)

    for name, dataset in [
        ("Training (2010-2018)", train),
        ("Validation (2019-2022)", validation),
    ]:
        print(f"\n{'=' * 50}")
        print(name)
        print(f"Observations: {len(dataset)}")

        # Buy-and-hold exposure is always 100%.
        baseline = dataset.copy()
        baseline["Exposure"] = 1.0

        # Each period is evaluated independently.
        baseline_initial = 0.0

        # Strategy's first exposure is established
        # at the beginning of each evaluation period.
        strategy_initial = 0.0

        _, baseline_metrics = run_backtest(
            baseline,
            transaction_cost_bps=TRANSACTION_COST_BPS,
            initial_exposure=baseline_initial,
        )

        _, strategy_metrics = run_backtest(
            dataset,
            transaction_cost_bps=TRANSACTION_COST_BPS,
            initial_exposure=strategy_initial,
        )

        print_metrics("Buy-and-Hold Baseline", baseline_metrics)
        print_metrics("50-Day Moving-Average Strategy", strategy_metrics)

        print("\nComparison")
        print("-" * 40)
        print(
            "Return difference: "
            f"{strategy_metrics['total_return'] - baseline_metrics['total_return']:+.2%}"
        )
        print(
            "Drawdown reduction: "
            f"{baseline_metrics['max_drawdown'] - strategy_metrics['max_drawdown']:+.2%}"
        )


if __name__ == "__main__":
    main()
