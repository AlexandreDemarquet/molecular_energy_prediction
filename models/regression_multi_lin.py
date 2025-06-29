
import torch
import torch.nn as nn

# from torch_cluster import radius_graph
# import torch_cluster

# Module ElementwiseProd
class ElementwiseProd(nn.Module):
    def __init__(self, N, q, k):
        super().__init__()
        self.N = N
        self.q = q
        self.k = k
        self.W = nn.ParameterList([nn.Parameter(torch.randn(N, q)) for _ in range(k)])
        self.b = nn.ParameterList([nn.Parameter(torch.randn(q)) for _ in range(k)])
        self.act = torch.sigmoid

    def forward(self, x):
        outs = []
        for i in range(self.k):
            y = x @ self.W[i] + self.b[i]
            y = self.act(y)
            outs.append(y)
        prod = outs[0]
        for i in range(1, self.k):
            prod = prod * outs[i]
        return prod

# Modèle complet de régression
class RegressionModel(nn.Module):
    def __init__(self, input_dim, q, k):
        super().__init__()
        self.elem = ElementwiseProd(input_dim, q, k)
        self.fc = nn.Linear(q, 1)
        self.norm = nn.BatchNorm1d(input_dim)

    def forward(self, x):
        x = self.norm(x)
        x = self.elem(x)
        out = self.fc(x)
        return out