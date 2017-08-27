import datetime

from abc import ABCMeta, abstractmethod

from event import FillEvent, OrderEvent


class ExecutionHandler(object):
    __metaclass__ = ABCMeta

    @abstractmethod
    def execute_order(self, queue):
        raise NotImplementedError("Should implement execute_order()")


class SimulatedExecutionHandler(ExecutionHandler):
    def __init__(self, queue, balance=1000.0):
        self.queue = queue
        self.balance = balance

    def execute_order(self, event):
        if event.type == 'ORDER':
            fill_event = FillEvent(datetime.datetime.utcnow(), event.symbol,
                                   'BACKTEST', event.quantity, event.direction)
            self.queue.put(fill_event)