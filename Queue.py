import queue

d_data = queue.Queue()

d_data.put("dataotaku")
d_data.put("python")

d_data.qsize()

d_data.get()

# import queue

data_queue = queue.Queue()

data_queue.put("funcoding")
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
        print("ended")
    else:
        print(data)
        recursive(data - 1)
        print("returned", data)


recursive(4)

# 대문자 M이 몇번이나 나오는가?
dataset = [
    "Braund, Mr. Owen Harris",
    "Cumings, Mrs. John Bradley (Florence Briggs Thayer)",
    "Heikkinen, Miss. Laina",
    "Futrelle, Mrs. Jacques Heath (Lily May Peel)",
    "Allen, Mr. William Henry",
    "Moran, Mr. James",
    "McCarthy, Mr. Timothy J",
    "Palsson, Master. Gosta Leonard",
    "Johnson, Mrs. Oscar W (Elisabeth Vilhelmina Berg)",
    "Nasser, Mrs. Nicholas (Adele Achem)",
    "Sandstrom, Miss. Marguerite Rut",
    "Bonnell, Miss. Elizabeth",
    "Saundercock, Mr. William Henry",
    "Andersson, Mr. Anders Johan",
    "Vestrom, Miss. Hulda Amanda Adolfina",
    "Hewlett, Mrs. (Mary D Kingcome) ",
    "Rice, Master. Eugene",
    "Williams, Mr. Charles Eugene",
    "Vander Planke, Mrs. Julius (Emelia Maria Vandemoortele)",
    "Masselmani, Mrs. Fatima",
    "Fynney, Mr. Joseph J",
    "Beesley, Mr. Lawrence",
    'McGowan, Miss. Anna "Annie"',
    "Sloper, Mr. William Thompson",
    "Palsson, Miss. Torborg Danira",
    "Asplund, Mrs. Carl Oscar (Selma Augusta Emilia Johansson)",
    "Emir, Mr. Farred Chehab",
    "Fortune, Mr. Charles Alexander",
    'Dwyer, Miss. Ellen "Nellie"',
    "Todoroff, Mr. Lalio",
]

dataset[1]

m_cnt = 0
for el in dataset:
    for i in range(len(el)):
        if el[i] == "M":
            m_cnt += 1

print(m_cnt)
