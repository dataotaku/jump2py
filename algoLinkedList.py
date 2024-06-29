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


class NodeMgmt:
    def __init__(self, data):
        self.head = Node(data)

    def add(self, data):
        if self.head == "":
            self.head = Node(data)
        else:
            node = self.head
            while node.next:
                node = node.next
            node.next = Node(data)

    def desc(self):
        node = self.head
        while node:
            print(node.data)
            node = node.next

    def delete(self, data):
        if self.head == "":
            print("찾고 계시는 값이 포함된 노드가 없습니다.")
            return

        if self.head.data == data:
            temp = self.head
            self.head = temp.next
            del temp
        else:
            node = self.head
            while node.next:
                if node.next.data == data:
                    temp = node.next
                    node.next = temp.next
                    del temp
                else:
                    node = node.next


l_list1 = NodeMgmt(0)
l_list1.desc()

l_list1.head
l_list1.delete(0)
l_list1.head

l_list2 = NodeMgmt(0)
for data in range(1, 10):
    l_list2.add(data)

l_list2.delete(5)
l_list2.desc()


# 링크드 리스트내 값 검색 메서트 추가
class NodeMgmt:
    def __init__(self, data):
        self.head = Node(data)

    def add(self, data):
        if self.head == "":
            self.head = Node(data)
        else:
            node = self.head
            while node.next:
                node = node.next
            node.next = Node(data)

    def desc(self):
        node = self.head
        while node:
            print(node.data)
            node = node.next

    def delete(self, data):
        if self.head == "":
            print("찾고 계시는 값이 포함된 노드가 없습니다.")
            return

        if self.head.data == data:
            temp = self.head
            self.head = temp.next
            del temp
        else:
            node = self.head
            while node.next:
                if node.next.data == data:
                    temp = node.next
                    node.next = temp.next
                    del temp
                else:
                    node = node.next

    def findprint(self, data):
        if self.head == "":
            print("빈 링크드 리스트입니다.")
            return
        else:
            node = self.head
            while node:
                if node.data == data:
                    print(node.data)
                    return
                else:
                    node = node.next


l_list3 = NodeMgmt(0)
for i in range(1, 10):
    l_list3.add(i)
l_list3.desc()

l_list3.findprint(5)
l_list3.findprint(4)
