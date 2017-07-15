from pyalgotrade import strategy
from pyalgotrade.barfeed import csvfeed

class MyStrategy(strategy.BacktestingStrategy):
    def __init__(self, feed, instrument):
        super(MyStrategy, self).__init__(feed)
        self.__instrument = instrument

    def onBars(self, bars):
        bar = bars[self.__instrument]
        self.info(bar.getClose())

# Load the yahoo feed from the CSV file
feed = csvfeed.BarFeed(300)
feed.addBarsFromCSV("btc_etc", "btc_etc.csv")

# Evaluate the strategy with the feed's bars.
myStrategy = MyStrategy(feed, "btc_etc")
myStrategy.run()