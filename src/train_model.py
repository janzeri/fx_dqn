import datetime
import os

from agent import FXAgt
from environment import FXEnv



EPISODE = 1

if __name__ == '__main__':
    currency = "EURUSD"
    period = None
    num_period = None
    watching_width = 100
    length_of_data = 0

    current_datetime = datetime.datetime.now()
    start_day = current_datetime.strftime("%m%d_%H%M")


    print("\n------   Start Training   ------")


    env = FXEnv(currency)
    agt = FXAgt()
    state = env.watching_ma_data

    for episode in range(EPISODE):
        step = 1
        env.price_data_idx = env.watching_data_count
        while step + env.watching_data_count < len(env.price_data) - 1:
            print(f"[episode {episode + 1}] [step {step}]", end = ' ')
            
            action = agt.select_action(state, env.portfolio.position_status)
            next_state, reward, done = env.step(action)
            print(f"[reward {round(reward, 2)}]")
            agt.model.train(state, action, reward, done, next_state)
            state = next_state

            step += 1
        
        print()

    agt.model.save_weights(f"../weights/{start_day}")

    env.portfolio.show_earned_pips_history()

    print("\n [ Training has successfully completed. ]")
