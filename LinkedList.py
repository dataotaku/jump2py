class Node:
    def __init__(self, data, next=None):
        self.data = data
        self.next = next


node1 = Node(1)
node2 = Node(2)
node1.next = node2
head = node1

print(head.data)
print(node1.next)
print(node2.data)
print(node2.next)


def add(data):
    node = head
    while node.next:  # 마지막 None (falthy)이 나오기 전까지...
        node = node.next
    node.next = Node(data)


for i in range(3, 10):
    add(i)

node = head
while node.next:
    print(node.data)
    node = node.next
print(node.data)

node3 = Node(1.5)
node = head
search = True
while search:
    if node.data == 1:
        search = False
    else:
        node = node.next
next_dum = node.next
node.next = node3
node3.next = next_dum

node = head
while node.next:
    print(node.data)
    node = node.next
