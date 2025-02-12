import torch
import torch.optim as optim
import math
import matplotlib.pyplot as plt
import numpy as np

class CosineWarmupScheduler:
    def __init__(self, optimizer, warmup_epochs, total_epochs):
        """
        Cosine learning rate scheduler with warmup.
        
        Args:
            optimizer: PyTorch optimizer
            warmup_epochs: Number of epochs for warmup
            total_epochs: Total number of training epochs
        """
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.current_epoch = 0
        
        # Store base learning rate
        self.base_lr = [group['lr'] for group in optimizer.param_groups]
        
    def step(self):
        """Update learning rate based on current epoch"""
        self.current_epoch += 1
        
        for i, group in enumerate(self.optimizer.param_groups):
            if self.current_epoch <= self.warmup_epochs:
                # Linear warmup
                lr = self.base_lr[i] * (self.current_epoch / self.warmup_epochs)
            else:
                # Cosine decay
                progress = (self.current_epoch - self.warmup_epochs) / (self.total_epochs - self.warmup_epochs)
                lr = self.base_lr[i] * 0.5 * (1 + math.cos(math.pi * progress))
            
            group['lr'] = lr

def one_cycle_scheduler(optimizer, max_lr, total_steps, pct_start=0.3, div_factor=25., final_div_factor=1e4):
    """
    Creates One Cycle learning rate scheduler.
    
    Args:
        optimizer: PyTorch optimizer
        max_lr: Maximum learning rate
        total_steps: Total number of training steps
        pct_start: Percentage of total steps spent in increasing LR
        div_factor: Initial LR = max_lr/div_factor
        final_div_factor: Final LR = initial_lr/final_div_factor
    """
    return optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=max_lr,
        total_steps=total_steps,
        pct_start=pct_start,
        div_factor=div_factor,
        final_div_factor=final_div_factor
    )

def plot_lr_schedule(scheduler, steps):
    """
    Visualize the learning rate schedule.
    
    Args:
        scheduler: Learning rate scheduler
        steps: Number of steps to plot
    """
    lrs = []
    for i in range(steps):
        scheduler.step()
        lrs.append(scheduler.optimizer.param_groups[0]['lr'])
    
    plt.figure(figsize=(10, 5))
    plt.plot(lrs)
    plt.title('Learning Rate Schedule')
    plt.xlabel('Steps')
    plt.ylabel('Learning Rate')
    plt.grid(True)
    plt.show()


# Initialize
# model = YourModel()
# optimizer = optim.AdamW(model.parameters(), lr=0.001)
# scheduler = CosineWarmupScheduler(optimizer, warmup_epochs=5, total_epochs=30)

# # Training loop
# for epoch in range(num_epochs):
#     train_one_epoch(model, optimizer)
#     scheduler.step()

# scheduler = one_cycle_scheduler(
#     optimizer,
#     max_lr=0.01,
#     total_steps=len(train_dataloader) * num_epochs
# )

# # Training loop
# for epoch in range(num_epochs):
#     for batch in train_dataloader:
#         train_step(model, batch, optimizer)
#         scheduler.step()