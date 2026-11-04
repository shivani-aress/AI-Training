import array as arr

a = arr.array('i', [1, 2, 3, 1, 4])

a.insert(1, 4)
print(*a)

a.remove(1)
a.pop(2)
print(*a)

res = a[0:1]
print(*res)

b = arr.array('i', [1, 2, 3])
print(*b)