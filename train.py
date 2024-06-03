from src.data import MADataset
from src.simlpe import SiMLPe, DCTM, IDCTM

import torch
from torch.utils.data import DataLoader


def compute_loss(y_pred, y):
    # 计算L_re
    L_re = torch.nn.functional.mse_loss(y_pred, y)

    # 计算L_v
    v_pred = y_pred[:, 1:, :] - y_pred[:, :-1, :]
    v_true = y[:, 1:, :] - y[:, :-1, :]
    L_v = torch.nn.functional.mse_loss(v_pred, v_true)

    # 总损失
    loss = L_re + L_v
    return loss


# 加载数据集
train_dataset = MADataset('./data', train=True)
test_dataset = MADataset('./data', train=False)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# 定义模型
model = SiMLPe()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# 优化器
optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
# 训练和验证循环
num_epochs = 1000
for epoch in range(num_epochs):
    model.train()
    train_loss = 0.0
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        dctm = DCTM.expand(x.shape[0], -1, -1).to(device)
        idctm = IDCTM.expand(y.shape[0], -1, -1).to(device)
        x = torch.bmm(dctm, x)
        optimizer.zero_grad()
        y_pred = model(x)
        y_pred = torch.bmm(idctm, y_pred)
        loss = compute_loss(y_pred, y)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    train_loss /= len(train_loader)

    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            dctm = DCTM.expand(x.shape[0], -1, -1).to(device)
            idctm = IDCTM.expand(y.shape[0], -1, -1).to(device)
            x = torch.bmm(dctm, x)
            y_pred = model(x)
            y_pred = torch.bmm(idctm, y_pred)
            loss = compute_loss(y_pred, y)
            val_loss += loss.item()

    val_loss /= len(test_loader)
    if (epoch + 1) % 10 == 0:
        print(f"Epoch [{epoch + 1}/{num_epochs}], Train Loss: {train_loss*1000:.4f}, Val Loss: {val_loss*1000:.4f}")
