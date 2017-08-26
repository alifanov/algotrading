import time
import queue

from data import HistoricCSVDataHandler
from portfolio import NaivePortfolio
from strategy import BuyAndHoldStrategy
from execution import SimulatedExecutionHandler

events = queue.Queue(100)

broker = SimulatedExecutionHandler(events)
bars = HistoricCSVDataHandler(events, '../datasets/', ['btc_etc'])
strategy = BuyAndHoldStrategy(bars, events)
port = NaivePortfolio(bars, events, None, 1000.0)


def run(heartbeat=3):
    while True:
        # Update the bars (specific backtest code, as opposed to live trading)
        if bars.continue_backtest:
            bars.update_bars()
        else:
            break

        # Handle the events
        while True:
            try:
                event = events.get(False)
            except queue.Empty:
                break
            else:
                if event is not None:
                    if event.type == 'MARKET':
                        strategy.calculate_signals(event)
                        port.update_timeindex(event)

                    elif event.type == 'SIGNAL':
                        port.update_signal(event)

                    elif event.type == 'ORDER':
                        broker.execute_order(event)

                    elif event.type == 'FILL':
                        port.update_fill(event)

        # 3-sec heartbeat
        # time.sleep(heartbeat)
    port.create_equity_curve_dataframe()
    print(port.output_summary_stats())

if __name__ == "__main__":
    run()