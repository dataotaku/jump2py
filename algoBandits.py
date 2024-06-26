import numpy as np
import matplotlib.pyplot as plt

np.random.seed(0)

rewards = []

for n in range(1, 11):
    reward = np.random.rand()  # 보상
    rewards.append(reward)
    Q = sum(rewards) / n
    print(Q)

Q = 0

# Qn과 Qn-1에 해당하는 변수를 Q하나로 처리했음
# 등호 왼쪽의 Q(새로운 추정치)와
# 오른쪽의 Q(1회전의 추정치)의 의미가 다름에 유의할 것!
for n in range(1, 11):
    reward = np.random.rand()
    # Q = Q + (reward - Q) / n  # 식 1.5
    Q += (reward - Q) / n  # 증분구현이라고도 함.
    print(Q)


class Bandit:
    def __init__(self, arms=10):  # arms : 슬롯머신 댓수
        self.rates = np.random.rand(arms)  # 슬롯머신 승률 설정 무작위

    def play(self, arm):
        rate = self.rates[arm]
        # print(rate)
        prob = np.random.rand()
        # print(prob)
        if rate > prob:
            return 1
        else:
            return 0


bandit = Bandit()
for i in range(5):
    print(bandit.play(0))

# 에이전트 구현
bandit = Bandit()
Q = 0

for n in range(1, 11):
    reward = bandit.play(0)
    Q += (reward - Q) / n
    print(Q)

bandit = Bandit()
Qs = np.zeros(10)  # 각 슬롯머신의 가치 추정치
ns = np.zeros(10)  # 각 슬롯머신의 play 횟수

for n in range(10):
    action = np.random.randint(0, 10)  # 무작위 행동(임의의 슬롯머신 선택)
    reward = bandit.play(action)

    ns[action] += 1  # action번째 슬롯머신을 플레이한 횟수 증가
    Qs[action] += (reward - Qs[action]) / ns[action]
    print(Qs)


class Agent:
    def __init__(self, epsilon, action_size=10):
        self.epsilon = epsilon
        self.Qs = np.zeros(action_size)
        self.ns = np.zeros(action_size)

    def update(self, action, reward):
        self.ns[action] += 1
        self.Qs[action] += (reward - self.Qs[action]) / self.ns[action]

    def get_action(self):  # 행동선택(엡실론 탐욕 정책)
        if np.random.rand() < self.epsilon:
            return np.random.randint(0, len(self.Qs))  # 무작위 행동선택
        return np.argmax(self.Qs)  # 탐욕행동 선택 - 리스트 Qs의 가장 높은 값의 index값


steps = 1000
epsilon = 0.1

bandit = Bandit()
agent = Agent(epsilon)
total_reward = 0
total_rewards = []
rates = []

for step in range(steps):
    action = agent.get_action()  # 행동선택
    reward = bandit.play(action)  # 실제 플레이 및 보상 획득
    agent.update(action, reward)  # 행동과 보상을 통해 학습
    total_reward += reward

    total_rewards.append(total_reward)
    rates.append(total_reward / (step + 1))

print(total_reward)

plt.ylabel("Total reward")
plt.xlabel("Steps")
plt.plot(total_rewards)
plt.show()

plt.ylabel("Rates")
plt.xlabel("Steps")
plt.plot(rates)
plt.show()

# 실험 200번 실행후 평균으로 알고리즘 성능을 평가함.
runs = 200
steps = 1000
epsilon = 0.3
all_rates = np.zeros((runs, steps))

for run in range(runs):
    bandit = Bandit()
    agent = Agent(epsilon)
    total_reward = 0
    rates = []

    for step in range(steps):
        action = agent.get_action()
        reward = bandit.play(action)
        agent.update(action, reward)
        total_reward += reward
        rates.append(total_reward / (step + 1))

    all_rates[run] = rates

avg_rates = np.average(all_rates, axis=0)

plt.ylabel("Rates")
plt.xlabel("Steps")
plt.plot(avg_rates)
plt.show()


# 비정상 문제 형태의 bandit
class NonStatBandit:
    def __init__(self, arms=10):
        self.arms = arms
        self.rates = np.random.rand(arms)

    def play(self, arm):
        rate = self.rates[arm]
        self.rates += 0.1 * np.random.randn(self.arms)
        if rate > np.random.rand():
            return 1
        else:
            return 0


class AlphaAgent:
    def __init__(self, epsilon, alpha, actions=10):
        self.epsilon = epsilon
        self.Qs = np.zeros(actions)
        self.alpha = alpha

    def update(self, action, reward):
        self.Qs[action] += (reward - self.Qs[action]) * self.alpha

    def get_action(self):
        if np.random.rand() < self.epsilon:
            return np.random.randint(0, len(self.Qs))
        return np.argmax(self.Qs)


# Stat vs Nonstat 실험 200번 실행후 평균으로 알고리즘 성능을 평가함.
runs = 200
steps = 1000
epsilon = 0.3
alpha = 0.1

stat_rates = np.zeros((runs, steps))

for run in range(runs):
    bandit = Bandit()
    agent = Agent(epsilon)
    total_reward = 0
    rates = []

    for step in range(steps):
        action = agent.get_action()
        reward = bandit.play(action)
        agent.update(action, reward)
        total_reward += reward
        rates.append(total_reward / (step + 1))

    stat_rates[run] = rates

avg_stat_rates = np.average(stat_rates, axis=0)

plt.ylabel("Rates")
plt.xlabel("Steps")
plt.plot(avg_stat_rates)
plt.show()

nstat_rates = np.zeros((runs, steps))

for run in range(runs):
    bandit = NonStatBandit()
    agent = AlphaAgent(epsilon, alpha)
    total_reward = 0
    rates = []

    for step in range(steps):
        action = agent.get_action()
        reward = bandit.play(action)
        agent.update(action, reward)
        total_reward += reward
        rates.append(total_reward / (step + 1))

    nstat_rates[run] = rates

avg_nstat_rates = np.average(nstat_rates, axis=0)

plt.ylabel("Rates")
plt.xlabel("Steps")
plt.plot(avg_nstat_rates)
plt.show()

plt.ylabel("Rates")
plt.xlabel("Steps")
plt.plot(avg_stat_rates)
plt.plot(avg_nstat_rates)
plt.show()
