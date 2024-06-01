import time
import numpy as np
import os
import sys
import random
import math
import pandas as pd
import matplotlib.pyplot as plt

class FXEnv:
    def __init__(self, currency_pair = "USDJPY", spread = 0):
        self.CURRENCY_PAIR = currency_pair
        self.SPREAD = spread

        self.price_data = self.load_price_data_from_csv()
        self.watching_data_count = 100
        self.price_data_idx = self.watching_data_count
        self.watching_price_data = self.price_data[0:self.watching_data_count]

        self.ma_data = [self.moving_average(self.price_data, 25)
                        ,self.moving_average(self.price_data, 75)
                        ,self.moving_average(self.price_data, 100)]
        # print(np.shape(self.ma_data))
        self.watching_ma_data = self.ma_data[0][0:self.watching_data_count]
        self.watching_ma_data += self.ma_data[1][0:self.watching_data_count]
        self.watching_ma_data += self.ma_data[2][0:self.watching_data_count]
        print(np.shape(self.watching_ma_data))

        self.portfolio = Portfolio(currency_pair)



    def load_price_data_from_csv(self):
        data = pd.read_csv(f"../data/{self.CURRENCY_PAIR}/21_1H.csv", delimiter=',').iloc[:, 1]
        if len(data) != 0:
            print(f"\n{self.CURRENCY_PAIR} has loaded.")
            print(f"size of data {len(data)}\n")
            return data
        else:
            return None


    def plot_price_data(self, start=0, end=500):
        start_price = self.price_data[start:end]

        plt.plot(start_price)

        plt.title('始値の推移')
        plt.xlabel('日時')
        plt.ylabel('始値')
        plt.grid(True)

        plt.show()


    def plot_ma(self, start=0, end=500, period = 200):
        start_price = self.price_data[start:end]

        moving_avg = env.moving_average(start_price, 25)
        plt.plot(moving_avg[end-period:end])

        moving_avg = env.moving_average(start_price, 50)
        plt.plot(moving_avg[end-period:end])

        moving_avg = env.moving_average(start_price, 75)
        plt.plot(moving_avg[end-period:end])


        plt.title('moving avarage')
        plt.xlabel(' ')
        plt.ylabel('slope')
        plt.grid(True)

        plt.show()




    def step(self, action):
        if action > 2:
            raise Exception(str(action) + " is invalid.")

        current_price = self.price_data[self.price_data_idx]

        next_state = None
        reward = None

        if action == 0: # BUY
            reward = self.portfolio.buy(current_price, self.SPREAD)
            print("BUY ", end=' ')
        elif action == 1: # HOLD
            reward = self.portfolio.hold(self.price_data[self.price_data_idx + 1], self.SPREAD)
            print("HOLD ", end=' ')
        elif action == 2: # SELL
            reward = self.portfolio.sell(current_price, self.SPREAD)
            print("SELL ", end=' ')
        else:
            raise Exception("action is changed in step. something wrong.")
        
        #reward = self.calculate_reward()

        self.price_data_idx += 1
        next_state = self.ma_data[0][self.price_data_idx - self.watching_data_count : self.price_data_idx]
        next_state += self.ma_data[1][self.price_data_idx - self.watching_data_count : self.price_data_idx]
        next_state += self.ma_data[2][self.price_data_idx - self.watching_data_count : self.price_data_idx]
        # next_state = self.price_data[self.price_data_idx - self.watching_data_count : self.price_data_idx]
       
        return next_state, reward, False
    
    def calculate_rewards(self):
        pass


    def moving_average(self, data, period):
        data = [float(x) if isinstance(x, str) else x for x in data]

        moving_averages = []
        for i in range(len(data) - 1):
            if i < period:
                moving_averages.append(data[i])
                continue
            window = data[i-period:i]
            ma = sum(window) / period
            moving_averages.append(ma)
        slopes = np.diff(moving_averages).tolist()
            
        return slopes






class Portfolio:
    def __init__(self, currency_pair):
        self.LONG = 0
        self.NOT = 1
        self.SHORT = 2
        
        self.CURRENCY_PAIR = currency_pair
        self.__money = 1000000
        self.position_status = self.NOT 
        self.__position_price = None

        self.money_history = []
        self.money_history.append(self.__money)

        self.pips = 0
        self.pips_history = []


    def get_position_status(self):
        return self.position_status

    def set_money(self, money):
        self.__money = money

    def show_money_history(self):
        pass

    def show_earned_pips_history(self):
        plt.plot(self.pips_history)
        plt.title('earned pips')
        plt.xlabel('time')
        plt.ylabel('pips')
        plt.grid(True)
        plt.show()

    def calculate_pips(self, price):
        if self.CURRENCY_PAIR == "EURUSD":
            pips = price * 10000
            return pips
        else:
            return price

    def buy(self, price, spread):
        pips_rewards = 0
        if self.position_status == self.NOT:
            self.position_status = self.LONG
            self.__position_price = price
        elif self.position_status == self.SHORT: # close position
            self.position_status = self.NOT
            pips_rewards = price - self.__position_price - spread
            pips_rewards = self.calculate_pips(pips_rewards)
            self.pips += pips_rewards
            self.pips_history.append(self.pips)
            return pips_rewards
        elif self.position_status == self.LONG:
            pass
        else:
            raise Exception("ERROR : wrong position_status")
        self.pips_history.append(self.pips)
        return pips_rewards

    def hold(self, next_price, spread):
        if self.position_status == self.NOT:
            self.pips_history.append(self.pips)
            return 0
        elif self.position_status == self.LONG or self.position_status == self.SHORT:
            self.pips_history.append(self.pips) #ここでは損益は増減しない
            expect_pips = next_price - self.__position_price - spread
            expect_pips = self.calculate_pips(expect_pips)
            return expect_pips
        else:
            raise Exception("ERROR : wrong position_status")

    def sell(self, price, spread):
        pips_rewards = 0
        if self.position_status == self.NOT:
            self.position_status = self.SHORT
            self.__position_price = price
        elif self.position_status == self.LONG: # close position
            self.position_status = self.NOT
            pips_rewards = price - self.__position_price - spread
            pips_rewards = self.calculate_pips(pips_rewards)
            self.pips += pips_rewards
            self.pips_history.append(self.pips)
            return pips_rewards
        elif self.position_status == self.SHORT:
            pass
        else:
            raise Exception("ERROR : wrong position_status")
        self.pips_history.append(self.pips)
        return pips_rewards






if __name__ == "__main__":
    print("------------------------------------------------")
    print("-                                              -")
    print("-               debugging env.py               -")
    print("-                                              -")
    print("------------------------------------------------")

    env = FXEnv("EURUSD")
    env.load_price_data_from_csv()
    env.plot_price_data(0, 6000)

    moving_avg = env.moving_average(env.price_data, 200)
    plt.plot(moving_avg[201:])

    plt.title('始値の推移')
    plt.xlabel('日時')
    plt.ylabel('始値')
    plt.grid(True)

    plt.show()

    print('successfully finished.')
