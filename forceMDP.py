# 정책1, 상태 L1
V = 1
for i in range(1, 100):
    V += -1 * (0.9**i)
print(V)

# 정책1, 상태 L2
V = -1
for i in range(1, 100):
    V += -1 * (0.9**i)
print(V)

# 정책2, 상태 L1
V = 1
for i in range(1, 100):
    V += 0.9 ** (2 * i)
print(V)

# 정책2, 상태L2
V = 0
for i in range(1, 100):
    V += 0.9 ** (2 * i - 1)
print(V)
