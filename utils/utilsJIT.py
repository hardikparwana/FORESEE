import numpy as np
import torch
    
def wrap_angle_tensor_JIT(angle):
    factor = torch.tensor(2*3.14157,dtype=torch.float)
    if angle>torch.tensor(3.14157):
        angle = angle - factor
    if angle<torch.tensor(-3.14157):
        angle = angle + factor
    return angle

def symsqrt(a):
    """Computes the symmetric square root of a positive definite matrix"""

    s, u = torch.symeig(a, eigenvectors=True)
    cond_dict = {torch.float32: 1e3 * 1.1920929e-07, torch.float64: 1E6 * 2.220446049250313e-16}

    cond = cond_dict[a.dtype]

    above_cutoff = (abs(s) > cond * torch.max(abs(s)))

    psigma_diag = torch.sqrt(s[above_cutoff])
    u = u[:, above_cutoff]

    B = u @ torch.diag(psigma_diag) @ u.t()

    return B