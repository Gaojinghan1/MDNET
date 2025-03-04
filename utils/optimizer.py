# optimizer.py
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler

def get_optimizer(net, lr=0.001, weight_decay=0.05):
    optimizer = optim.AdamW(
        net.parameters(),
        lr=lr,
        betas=(0.9, 0.999),
        eps=1e-08,
        weight_decay=weight_decay
    )
    return optimizer

def get_lr_schedulers(optimizer, warmup_epochs=20, total_epochs=300):
    warmup_scheduler = lr_scheduler.LinearLR(
        optimizer,
        start_factor=0.001,
        end_factor=1.0,
        total_iters=warmup_epochs
    )
    cosine_scheduler = lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=total_epochs - warmup_epochs,
        eta_min=1e-5
    )
    return warmup_scheduler, cosine_scheduler
