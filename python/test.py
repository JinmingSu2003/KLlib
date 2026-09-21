import torch 
import kl

A=torch.randn((4096,4096),dtype=torch.float32,device="cuda")
B=torch.randn((4096,4096),dtype=torch.float32,device="cuda")

o1=kl.siluxmul(A,B)
o2=torch.nn.functional.silu(A)*B
ok=torch.allclose(
    o1,o2,1e-4,1e-4
)
print(ok)