from llib import Llib
import sys
rc = sys.getrefcount

a = [1,2,3]
ll = Llist(a)
print(rc(ll))
print(rc(1), rc(2), rc(3))
del ll
print(rc(1), rc(2), rc(3))

ll = Llist(a)

n1 = ll.first
n2 = ll.first.next
n3 = ll.first.next.next

print(n1.data, n2.data, n3.data)