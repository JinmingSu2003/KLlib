import torch 
from python import kl

x=torch.randn((4096,4096),dtype=torch.float32,device="cuda")
y=torch.randn((4096,4096),dtype=torch.float32,device="cuda")

o1=torch.matmul(x,y)
o2=kl.gemm4096(x,y)
ok=torch.allclose(
    o1,o2,1e-3,1e-3
)
print(ok)