import torch

# Create a tensor
x = torch.tensor([1, 2, 3])
print(x)

# Create a random tensor
y = torch.rand(3, 3)
print(y)

# Addition
a = torch.tensor([1, 2, 3])
b = torch.tensor([4, 5, 6])
c = a + b
print(c)

# Matrix multiplication
d = torch.matmul(a.view(3, 1), b.view(1, 3))
print(d)