import torch
import torch.nn as nn
import torch.optim as optim

X = torch.tensor([
    [2.0, 3.0],
    [3.0, 4.0],
    [2.0, 1.0],
    [1.0, 1.0]
])

y = torch.tensor([
    [5.0],
    [7.0],
    [3.0],
    [2.0],
])

model = nn.Sequential(
    nn.Linear(2, 8),
    nn.ReLU(),
    nn.Linear(8, 1)
)

loss_fn = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.05)

for epoch in range(10001):
    y_pred = model(X)
    loss = loss_fn(y_pred, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 1000 == 0:
        print(f"epoch {epoch}, loss {loss.item():.4f}")

f1 = float(input("Enter the first float: "))
f2 = float(input("Enter the second float: "))

a = torch.tensor([[f1, f2]])

with torch.no_grad():
    inpPred = model(a)

print(round(inpPred.item(),1))