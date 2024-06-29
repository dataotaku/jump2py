# 재귀함수
def recursive(data):
    if data < 0:
        print("ended")
    else:
        print(data)
        recursive(data - 1)
        print("returned", data)


recursive(4)

d_stack = []
d_stack.append("Dataotaku")
d_stack.append(1)
d_stack
d_stack.pop()
d_stack

stack_list = []


def push(data):
    stack_list.append(data)


def pop():
    data = stack_list[-1]
    del stack_list[-1]
    return data


push("data_otaku")
push(100)
stack_list
pop()
