import torch
import torch.nn as nn

class MiniTransformer(nn.Module):
    def __init__(self, dim=512, heads=8):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.ff = nn.Sequential(
            nn.Linear(dim, 2048),
            nn.ReLU(),
            nn.Linear(2048, dim)
        )

    def forward(self, x):
        x,_ = self.attn(x, x, x)
        x = x + self.ff(x)
        return x

model = MiniTransformer(dim=512, heads=8)

x = torch.randn(2, 10, 512)

# Forward pass
output = model(x)

print("Output shape:", output.shape)
print("Output:", output)
