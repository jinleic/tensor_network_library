import torch

device = 'cuda'
bond_dim = 100
A = torch.randn(bond_dim,6,16,bond_dim).to(device)
B = torch.randn(bond_dim,16,6,bond_dim).to(device)
y = torch.tensordot(A, B, dims=([2], [1]))
y = y.permute(0,3,1,4,2,5)
# y = y.reshape(100*100, 6, 6, 100*100)
y = y.view(100*100, 6, 6, 100*100)