import time
import queue

from data import HistoricCSVDataHandler
from portfolio import NaivePortfolio
from strategy import BuyAndHoldStrategy, StatArbitrageStrategy
from execution import SimulatedExecutionHandler

events_queue = queue.Queue(100)

broker = SimulatedExecutionHandler(events_queue)
bars = HistoricCSVDataHandler(events_queue, './datasets/', ['BTC_ETC', 'BTC_LTC'], fields=[
    'open',
    'high',
    'low',
    'close',
])
strategy = StatArbitrageStrategy(bars, events_queue)
port = NaivePortfolio(bars, events_queue, None, 1000.0)


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
                event = events_queue.get(False)
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