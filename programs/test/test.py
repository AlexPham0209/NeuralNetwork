import cupy as cp

a = cp.array([
    [1, 2, 3, 5, 6],
    [4, 5, 6, 5, 6]
]).T

b = cp.array([
    [1, 2, 3, 4, 5],
    [2, 4, 5, 4, 5],
]).T


k = cp.array([[1, 2, 3, 4, 5], [2, 2, 3, 4, 5], [3, 2, 3, 4, 5]]).T

# print(a @ k)
c = cp.array([1, 2, 3])
d = cp.array([1, 1, 1])

e = cp.array([4, 5, 6])
f = cp.array([2, 2, 2])

# res = cp.transpose(cp.tile(a, (3, 1, 1)).T, (1, 0, 2))
# bs = cp.tile(b.T, (3, 1, 1))

# print(res)
# print(res * bs.transpose(1, 2, 0))

# print(cp.repeat(a.transpose(1, 0), 2, 0))

k = cp.tile(c, (3,1)).T
print(k)

j = cp.tile(d, (3,1)).T
print(j)

# print(k)
# print(j)
# print(k + j)
# print()

# cp.repeat(b.T[:, :, cp.newaxis], 3, 2)
#res = cp.repeat(a[:, :, cp.newaxis], 3, 2) * cp.repeat(b.T[:, :, cp.newaxis], 3, 2)
# print(cp.repeat(a[:, :, cp.newaxis], 3, 2))
l = cp.array([c, d])[:, :, cp.newaxis]
print(cp.repeat(l, b.shape[0], 2))

#print(cp.repeat(b.T[:, cp.newaxis, :], a.shape[0], 1))
# print(res.sum(0))

 
# print(cp.einsum('bkw,hw->hkw', res, b))
