import torch
from transformers import DynamicCache

def make(n):
    c=DynamicCache()
    # 用 update 填充一层
    k=torch.zeros(1,1,n,8); v=torch.zeros(1,1,n,8)
    for i in range(n):
        c.update(k[:,:,i:i+1],v[:,:,i:i+1],layer_idx=0)
    return c

c=make(11); print("start",c.get_seq_length())
c.crop(-2); print("after crop(-2):",c.get_seq_length(),"(expect 9)")
c=make(11); c.crop(-5); print("after crop(-5):",c.get_seq_length(),"(expect 6)")
