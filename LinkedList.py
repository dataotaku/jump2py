import queue

data_queue = queue.Queue()

data_queue.put('funcoding')
data_queue.put(1)

data_queue.qsize()

data_queue.get()

data_queue.get()

# 파이썬으로 enqueue, dequeue 기능 구현하기

queue_list = list()

def enqueue(data):
    queue_list.append(data)

def dequeue():
    data = queue_list[0]
    del queue_list[0]
    return data

for index in range(10):
    enqueue(index)

len(queue_list)

# 재귀함수
def recursive(data):
    if data < 0:
        print('ended')
    else:
        print(data)
        recursive(data-1)
        print('returned', data)

recursive(4)