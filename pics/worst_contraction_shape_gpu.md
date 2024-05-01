| Bond Dimension | Shape A | Shape B | Computation Scale | torch.matmul Scale |
| --- | --- | --- | --- | --- |
| 1 | torch.Size([1, 16, 16, 1]) | torch.Size([1, 16, 16, 1]) | 4.10e+03 | 6.87e+10 |
| 10 | torch.Size([1, 16, 16, 10]) | torch.Size([1, 16, 16, 10]) | 4.10e+05 | 6.87e+10 |
| 20 | torch.Size([20, 16, 16, 20]) | torch.Size([20, 16, 16, 20]) | 6.55e+08 | 6.87e+10 |
| 30 | torch.Size([30, 16, 16, 30]) | torch.Size([30, 16, 16, 30]) | 3.32e+09 | 6.87e+10 |
| 40 | torch.Size([40, 16, 16, 40]) | torch.Size([40, 16, 16, 40]) | 1.05e+10 | 6.87e+10 |
| 50 | torch.Size([50, 16, 16, 50]) | torch.Size([50, 16, 16, 50]) | 2.56e+10 | 6.87e+10 |
