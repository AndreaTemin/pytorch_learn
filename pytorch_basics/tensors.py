import torch

data = [[1,2,3], [1,2,3]]

x_data = torch.tensor(data)

# print(f"{type(x_data) = } \n {x_data = }")

print(f"{x_data.shape = }")
print(f"{x_data.dtype = }")
print(f"{x_data.device = }")

# We move our tensor to the GPU if available
if torch.cuda.is_available():
    tensor = x_data.to('cuda')
    print(f"{tensor.device = }")
    print(f"{tensor = }")


print("xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx")

x_ones = torch.ones_like(x_data)
print(f"Ones Tensor: \n {x_ones} \n")

x_rand = torch.rand_like(x_data, dtype=torch.float)
print(f"Ones Tensor: \n {x_rand} \n")


print("xxxxxxxxxx Tensor slicing xxxxxxxxxxxxxxxxxx")

tensor = torch.ones(4,4)
tensor[:,1] = 0
tensor[1] = 0
print(f"First row: {tensor[0]}")
print(f"First row: {tensor[1]}")
print(f"Last row: {tensor[-1]}")
print(f"First column: {tensor[:,0]}")
print(f"second column: {tensor[:,1]}")
print(f"Last column: {tensor[...,-1]}")
print(tensor)


print("xxxxxxxxx Tensor concat xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx")

t1 = torch.cat([tensor, tensor, tensor], dim=1)
print(t1)


print("xxxxxxxxxxxxxxxxxxxx Arithmetic Operations xxxxxxxxxxxxxxxxxx")

""" This computes the matrix multiplication between two tensors. y1, y2, y3 will have the same value """
# tensor.T returns the transposition of the tensor
y1 = tensor @ tensor.T
y2 = tensor.matmul(tensor.T)

y3 = torch.rand_like(y1)

print(y1,y2,y3)
print(torch.matmul(tensor, tensor.T, out=y3))

""" This computes the element-wise product. z1, z2, z3 will have the same value """
z1 = tensor * tensor
z2 = tensor.mul(tensor.T)
z3 = torch.rand_like(tensor)
print(z1,z2,z3)
print(torch.mul(tensor, tensor.T, out=z3))


print(f"xxxxxxxxxxx Single element tensor xxxxxxxxxxxxxx")

agg = tensor.sum()
print(agg, type(agg))
agg_item = agg.item()
print(agg_item, type(agg_item))

print(f"{tensor} \n")


print(f"xxxxxxxxxx In place operations xxxxxxxxxxxxxxx")

print(f"{tensor} \n")
tensor.add_(5)
print(tensor)






