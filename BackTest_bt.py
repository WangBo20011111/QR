# -*- coding: utf-8 -*-
"""
Created on 2025/6/28 15:15
@author: Wang bo
"""
import pandas as pd
import BackTest_bt as bt
import datetime

class PandasData_more(bt.feed.PandasData):
    lines = ('ROE', 'EP', )
    params = dict(
        ROE = -1,
        EP = -1 #-1表示自动按列名匹配数据
    )



class StockSelectStrategy(bt.Strategy):
    def __init__(self):
        # 读取调仓表
        super().__init__()
        self.buy_stock = pd.read_csv('./data/trade_info.csv', parse_dates=['trading_date'])
        #调仓日期每月的最后一个交易日，回测时会在这一天下单，然后再下一个交易日，以开盘价买入
        self.trade_dates = pd.to_datetime(self.buy_stock['trade_date'].unique()).tolist()
        self.buy_stocks_pre = [] #记录上一期持仓
        self.stop_loss_pct = 0.05
        self.take_profit_pct = 0.10
        
    def next(self):
        dt = self.datas[0].datetime.date(0)
        if dt in self.trade_dates:
            print('-------------{} 为调仓日 -------------'.format(dt))
            
            buy_stocks_data = self.buy_stock.query(f"trade_date=='{dt}'")
            long_list = buy_stocks_data.loc[buy_stocks_data['signal'] == 1, 'sec_code'].tolist()
            print('long_list', long_list)

            #对现有持仓中，调仓后不再继续持有的股票进行卖出平仓
            sell_stock = [i for i in self.buy_stocks_pre if i not in long_list]
            print('sell_stock', sell_stock)
            if len(sell_stock) > 0:
                print('----------对不再持有的股票进行平仓-------------')
                for stock in sell_stock:
                    data = self.getdatabyname(stock)
                    if self.getposition(data).size > 0:
                       self.order_target_percent(data=data, target=0.0)
            print('-----------买入此次调仓期的股票----------------')
            if len(long_list) > 0:    
                for stock in long_list:
                        w = 1 / max(3, len(long_list))
                        data = self.getdatabyname(stock)
                        self.order_target_percent(data=data, target=w * 0.95) #留5%的现金备用

            self.buy_stocks_pre = long_list
         # 止盈止损
        else:
            to_remove = []
            for stock in self.buy_stocks_pre:
                data = self.getdatabyname(stock
                position = self.getposition(data)
                if position.size > 0:  # 如果有持仓
                    current_price = data.close[0]
                    avg_price = position.price  # 持仓的平均买入价格
                    stop_loss_price = avg_price * (1 - self.stop_loss_pct)
                    take_profit_price = avg_price * (1 + self.take_profit_pct)
    
                    if current_price <= stop_loss_price:
                        print(f'止损平仓 {data._name}，当前价格：{current_price}, 止损价格：{stop_loss_price}')
                        self.order_target_percent(data=data, target=0.0)
                    elif current_price >= take_profit_price:
                        print(f'止盈平仓 {data._name}，当前价格：{current_price}, 止盈价格：{take_profit_price}')
                        self.order_target_percent(data=data, target=0.0) 


if __name__ == '__main__':
    cerebro = bt.Cerebro()
    #读取行情数据
    daily_price = pd.read_csv('./data/daily_price.csv', parse_dates=['datetime'])
    daily_price = daily_price.set_index(['datetime'])
    #按股票代码依次循环传入数据
    for stock in daily_price['sec_code'].unique():
        data = pd.DataFrame(index=daily_price.index.unique())
        df = daily_price.query(f"sec_code=='{stock}")[['open', 'high', 'low', 'close', 'volume', 'openinterest', 'ROE', 'EP']]
        data_ = pd.merge(data, df, left_index=True, right_index=True, how='left')
        # 缺失值处理：日期对齐时会使得有些交易日的数据为空，所以需要对缺失数据进行填充
        data_.loc[:, ['volume', 'openinterest']] = data_.loc[:, ['volume', 'openinterest']].fillna(0)
        data_.loc[:, ['open', 'high', 'low', 'close', 'ROE', 'EP']] = data_.loc[:, ['open', 'high', 'low', 'close', 'ROE', 'EP']].fillna(method='pad')
        data_.loc[:, ['open', 'high', 'low', 'close', 'ROE', 'EP']] = data_.loc[:, ['open', 'high', 'low', 'close', 'ROE', 'EP']].fillna(0)
        # 导入数据
        datafeed = PandasData_more(dataname=data_, fromdate=datetime.datetime(2019,1,2), todate=datetime.datetime(2021,1,28))
        cerebro.adddata(datafeed, name=stock)
        print(f"{stock} Done")
    cerebro.broker.setcash(100000000.0)
    cerebro.broker.setcommission(commission=0.0003) #双边佣金
    cerebro.broker.set_slippage_perc(perc=0.0001) # 双边滑点
    cerebro.addstrategy(StockSelectStrategy)
    cerebro.addnanlyzer(bt.analyzers.PyFolio, _name='pyfolio')
    result = cerebro.run()

    pyfolio = result[0].analyzers.pyfolio # 注意：后边不要调用.get_analysis()方法
    returns, positions, transactions, gross_lev = pyfolio.get_pf_items()

    import pyfolio as pf
    pf.create_full_tear_sheet(returns)
