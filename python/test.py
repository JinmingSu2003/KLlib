import torch 
import kl

A=torch.randn((4096,4096),dtype=torch.float32,device="cuda")
B=torch.randn((4096,4096),dtype=torch.float32,device="cuda")
W=torch.randn(4096,dtype=torch.float32,device="cuda")
o1=kl.rmsnorm_add(A,B,W)
o2=torch.rms_norm(A+B,(4096,),W,1e-6)

ok=torch.allclose(
    o1,o2,1e-4,1e-4
)
print(ok)