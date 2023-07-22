import torch
import numpy as np
# a1 = torch.randn((2, 3, 4))
# print(a1)
# a1 = a1.view(a1.size(0), -1)
# print(a1)
l = [1, 2]
a1 = np.array([[1, 2, 3], [6, 5, 4], [7, 8, 9]])
a2 = np.argmax(a1, axis=1)

l += a2.tolist()
l += a2.tolist()
print(l)
def pad4(i):
    return '0' * (4 - len(str(i))) + str(i)
print(pad4(32))