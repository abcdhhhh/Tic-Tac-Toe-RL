import numpy as np
from collections import defaultdict
from tqdm import tqdm
import matplotlib.pyplot as plt
from pylab import mpl

mpl.rcParams['font.sans-serif'] = ['SimHei']
n = 3  # 井字棋阶数


def checkState(state):
    if n == 1:
        return state[0]
    for index in [1, 2]:  # 1 先手胜 2 后手胜
        line = np.full(n, index)
        for i in range(n):
            if np.array_equal(state[i::n], line) or np.array_equal(state[i * n:(i + 1) * n], line):
                return index
        if np.array_equal(state[0::n + 1], line) or np.array_equal(state[n - 1:n * n - 1:n - 1], line):
            return index
    if len(np.where(state == 0)[0]):
        return 0  # 没下完
    else:
        return -1  # 平局


class Agent():

    def __init__(self, index, eps, lr):
        self.value = defaultdict(float)  # Q 函数，表示自己回合结束时的状态的收益（原始形式是开始时状态+动作，这里可以简化）
        self.currentState = np.zeros(n**2, dtype=int)
        self.previousState = np.zeros(n**2, dtype=int)
        self.index = index
        self.eps = eps
        self.lr = lr

    def reset(self):
        self.currentState = np.zeros(n**2, dtype=int)
        self.previousState = np.zeros(n**2, dtype=int)

    def actionTake(self, State):
        state = State.copy()
        available = np.where(state == 0)[0]  # 空格
        length = len(available)
        assert (length)
        r = np.random.uniform(0, 1)
        if r < self.eps:
            choice = np.random.choice(available)  # 随机策略
        else:
            tmpValue = np.zeros(length)  # 查找各决策收益
            for i, choice in enumerate(available):
                tmpState = State.copy()
                assert (tmpState[choice] == 0)
                tmpState[choice] = self.index
                tmpValue[i] = self.value[tuple(tmpState)]
            choice = available[np.random.choice(np.where(tmpValue == tmpValue.max())[0])]  # 选择最大收益
        assert (state[choice] == 0)
        state[choice] = self.index
        return state

    def valueUpdate(self, State, r):  # sarsa 算法
        self.currentState = State.copy()
        self.value[tuple(self.previousState)] += self.lr * (r + self.value[tuple(self.currentState)] - self.value[tuple(self.previousState)])
        self.previousState = self.currentState.copy()


def play(agent1, agent2, verbose=False):  # agent1 先手，agent2 后手
    state = np.zeros(n**2, dtype=int)
    agent1.reset()
    agent2.reset()
    # 胜利获得奖励 1，失败获得奖励 -1，平局无奖励
    reward1 = {1: 1, 2: -1, -1: 0, 0: 0}
    reward2 = {2: 1, 1: -1, -1: 0, 0: 0}
    while True:
        state = agent1.actionTake(state)
        result = checkState(state)
        agent1.valueUpdate(state, reward1[result])
        if verbose:
            print(state, result)
        if result:
            agent2.valueUpdate(state, reward2[result])
            return result

        state = agent2.actionTake(state)
        result = checkState(state)
        agent2.valueUpdate(state, reward2[result])
        if verbose:
            print(state, result)
        if result:
            agent1.valueUpdate(state, reward1[result])
            return result


def train(indexes: list = [1], iters=30000, turning=20000, step=500):  # indexes 表示需要训练的，比如 [1] 表示只训练先手，后手随机
    assert (0 < turning <= iters)
    assert (turning % step == 0)
    agent1 = Agent(1, eps=0.1, lr=0.1) if 1 in indexes else Agent(1, eps=1, lr=0)
    agent2 = Agent(2, eps=0.1, lr=0.1) if 2 in indexes else Agent(2, eps=1, lr=0)
    x = []
    y = {1: [], 2: [], -1: []}
    cnt = {1: 0, 2: 0, -1: 0}
    for i in tqdm(range(iters)):
        cnt[play(agent1, agent2)] += 1
        if (i + 1) % step == 0:
            x.append(i + 1)
            for result in cnt:
                y[result].append(cnt[result] / step)
            cnt = {1: 0, 2: 0, -1: 0}
            if i + 1 == turning:
                agent1.eps = 0
                agent2.eps = 0
    plt.xlabel('局数')
    plt.ylabel('比率')
    plt.grid(True)
    plt.plot(x, y[1])
    plt.plot(x, y[2])
    plt.plot(x, y[-1])
    plt.legend(['先手胜', '后手胜', '平局'])
    # plt.show()
    plt.savefig('./stats.png')
    # play(agent1, agent2, verbose=True)


if __name__ == '__main__':
    train([1])
