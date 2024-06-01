import os
import sys
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


class PriorityReplayBuffer:
    def __init__(self, capacity = 1000, alpha=0.6):
        self.capacity = capacity
        self.alpha = alpha
        self.buffer = []
        self.priorities = np.zeros(capacity, dtype=np.float32)
        self.pos = 0
        self.size = 0

    def _len(self):
        return len(self.buffer)

    def _add(self, experience, priority):
        if len(self.buffer) < self.capacity:
            self.buffer.append(experience)
        else:
            self.buffer[self.pos] = experience
        self.priorities[self.pos] = priority
        self.pos = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size):
        priorities = self.priorities[:self.size]
        probabilities = priorities ** self.alpha / np.sum(priorities ** self.alpha)
        indices = np.random.choice(self.size, batch_size, p=probabilities)
        samples = [self.buffer[i] for i in indices]
        weights = (self.size * probabilities[indices]) ** (-self.alpha)
        weights /= np.max(weights)
        return samples, indices, weights

    def update_priorities(self, indices, priorities):
        self.priorities[indices] = priorities



class QNetwork:
    def __init__(self, input_size = 300):
        self.BATCH_SIZE = 32
        self.input_size = input_size
        self.gamma = 0.9

        self.experience_memory = PriorityReplayBuffer()

        self.model = self.build_model()

    def build_model(self):
        model = nn.Sequential(
            nn.Linear(self.input_size, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 32),
            nn.ReLU(),
            nn.Linear(32, 5),
        )
        return model
    
    def get_Q(self, state):
        state_copy = np.array(state, dtype=np.float32, copy=True)
        state_tensor = torch.tensor(state_copy, dtype=torch.float32).squeeze(0)
        state_tensor = state_tensor.reshape(1, -1)
        Q = self.model(state_tensor).detach().numpy()

        return Q
    
    def calculate_prioriry(self, state, act, reward, done, next_state):
        with torch.no_grad():
            target = reward
            if not done:
                next_y = self.get_Q(next_state)
                target_act = reward + self.gamma * max(next_y.squeeze(0))
            else:
                target_act = reward

            y = self.get_Q(state).squeeze(0)
            td_error = abs(target_act - y[act]) + 1e-5

        return td_error
    
    
    def train(self, state, act, reward, done, next_state):
        if reward is None:
            return

        priority = self.calculate_prioriry(state, act, reward, done, next_state)

        self.experience_memory._add((state, act, reward, done, next_state), priority)

        if self.experience_memory._len() >= self.BATCH_SIZE:
            sampled_experiences, indices, weights = self.experience_memory.sample(self.BATCH_SIZE)

            for i, exp in enumerate(sampled_experiences):
                state, act, reward, done, next_state = exp
                q_values = self.get_Q(state).squeeze(0)
                #q_values = q_values.gather(1, act_batch.unsqueeze(1))
                target = q_values.copy()

                with torch.no_grad():
                    if not done:
                        target_q_values = self.get_Q(next_state)
                        target_act = reward + self.gamma * max(target_q_values.squeeze(0))
                    else:
                        target_act = reward
                target[act] = target_act

                state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
                target_tensor = torch.tensor(target, dtype=torch.float32).unsqueeze(0)

                loss_function = nn.MSELoss()
                optimizer = optim.Adam(self.model.parameters(), lr = 0.001)
                loss = (weights[i] * loss_function(self.model(state_tensor), target_tensor))

                optimizer.zero_grad()
                loss.backward()
                optimizer.step() 

            for i, exp in enumerate(sampled_experiences):
                state, act, reward, done, _ = exp
                new_priority = self.calculate_prioriry(state, act, reward, done, state)
                self.experience_memory.update_priorities(indices[i], new_priority)
        return
    

    def save_weights(self, filepath=None):
        #モデルの重みデータの保存
        if filepath is None:
            filepath = self.filepath
        torch.save(self.model.state_dict(), filepath + '.pt')
        print(f"{filepath} has saved.")


    def load_weights(self, filepath=None):
        #モデルの重みデータの読み込み
        if filepath is None:
            filepath = self.filepath
        self.model.load_state_dict(torch.load(filepath + '.pt'))
        print(f"{filepath} has loaded.")




class FXAgt:
    def __init__(self):
        self.MEMORY_CAPACITY = 1000
        self.ALPHA = 0.6
        self.model = QNetwork()
        self.experience_memory = PriorityReplayBuffer(self.MEMORY_CAPACITY, self.ALPHA)
        self.epsilon = 0.6

    def select_action(self, state, position):
        # position 
        # NO_POSITION = 0
        # LONG        = 1
        # HOLD_LONG   = 2
        # SHORT       = 3
        # HOLD_SHORT  = 4
        if position == 0:
            if self.epsilon <= np.random.rand():
                q = self.model.get_Q(state)
                q[0][2] = -1e5
                q[0][4] = -1e5
                action = np.argmax(q)
            else:
                action = np.random.choice([0, 1, 3])
        elif position == 1:
            if self.epsilon <= np.random.rand():
                q = self.model.get_Q(state)
                q[0][0] = -1e5
                q[0][1] = -1e5
                q[0][4] = -1e5
                action = np.argmax(q)
            else:
                action = np.random.choice([2, 3])
        elif position == 2:
            if self.epsilon <= np.random.rand():
                q = self.model.get_Q(state)
                q[0][0] = -1e5
                q[0][1] = -1e5
                q[0][4] = -1e5
                action = np.argmax(q)
            else:
                action = np.random.choice([2, 3])
        elif position == 3:
            if self.epsilon <= np.random.rand():
                q = self.model.get_Q(state)
                q[0][0] = -1e5
                q[0][3] = -1e5
                q[0][2] = -1e5
                action = np.argmax(q)
            else:
                action = np.random.choice([1, 4])
        elif position == 4:
            if self.epsilon <= np.random.rand():
                q = self.model.get_Q(state)
                q[0][0] = -1e5
                q[0][3] = -1e5
                q[0][2] = -1e5
                action = np.argmax(q)
            else:
                action = np.random.choice([1, 4])
        else:
            raise Exception("Error: wrong position_status at select_action")

        return action
    


if __name__ == '__main__':#
    print("------------------------------------------------")
    print("-                                              -")
    print("-             debugging agent.py               -")
    print("-                                              -")
    print("------------------------------------------------")


    print('successfully finished.')
