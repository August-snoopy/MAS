import argparse
import logging
import os
import time

import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from src.data import MADataset
from src.simlpe import SiMLPe, DCTM, IDCTM

# 使用数据集
dataset = MADataset(root='./data', train=True)
print(len(dataset))

# 划分训练集和测试集
train_dataset, test_dataset = train_test_split(dataset, test_size=0.2, random_state=42)
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

# 配置管理
parser = argparse.ArgumentParser(description='Training Config')
parser.add_argument('--epochs', default=4000, type=int, help='number of total epochs')
parser.add_argument('--lr', default=3e-4, type=float, help='learning rate')
parser.add_argument('--checkpoint', default='checkpoint.pth', help='path to checkpoint')
args = parser.parse_args()

# 日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# TensorBoard Summary
summary_writer = SummaryWriter()


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


def train(model, train_loader, optimizer, epoch):
    model.train()
    running_loss = 0.0
    rmse = 0.0
    proc_data = tqdm(train_loader, desc=f'Train Epoch {epoch}')
    for i, (x, y) in enumerate(proc_data):
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

        running_loss += loss.item()
        rmse += float(torch.sqrt(loss))
        proc_data.set_postfix(loss=running_loss / (i + 1), rmse=rmse / (i + 1))

    # 记录训练损失和准确率
    train_loss = running_loss / len(train_loader)
    average_rmse = rmse / len(train_loader)
    summary_writer.add_scalar('Loss/train', train_loss, epoch)
    summary_writer.add_scalar('RMSE/train', average_rmse, epoch)

    # 保存模型
    if epoch % 10 == 0:
        checkpoint = {
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'best_loss': best_loss
        }
        torch.save(checkpoint, args.checkpoint)
        logging.info(f'Checkpoint saved to {args.checkpoint}')

    return train_loss, average_rmse


def test(model, loader):
    model.eval()
    running_loss = 0.0
    rmse = 0.0
    outputs = []
    proc_data = tqdm(loader, desc='Test')
    with torch.no_grad():
        for i, (x, y) in enumerate(proc_data):
            x, y = x.to(device), y.to(device)
            dctm = DCTM.expand(x.shape[0], -1, -1).to(device)
            idctm = IDCTM.expand(y.shape[0], -1, -1).to(device)

            x = torch.bmm(dctm, x)
            y_pred = model(x)
            y_pred = torch.bmm(idctm, y_pred)
            loss = compute_loss(y_pred, y)

            outputs.append(loss)

            running_loss += loss.item()
            rmse += torch.sqrt(loss)
            proc_data.set_postfix(loss=running_loss / (i + 1), rmse=rmse / (i + 1))

    summary_writer.add_scalar('Loss/test', running_loss / len(loader), epoch)
    summary_writer.add_scalar('RMSE/test', rmse / len(loader), epoch)

    return running_loss / len(loader), rmse / len(loader)


# 创建模型、优化器和学习率调度器
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = SiMLPe().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

# 主循环
start_epoch = 1
best_loss = 0.0

# 加载检查点
if os.path.exists(args.checkpoint):
    checkpoint = torch.load(args.checkpoint)
    start_epoch = checkpoint['epoch'] + 1
    best_loss = checkpoint.get('best_loss', 0.0)
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    logger.info(f'Loaded checkpoint from {args.checkpoint}')

for epoch in range(start_epoch, args.epochs + 1):
    start_time = time.time()

    # 训练和评估
    train_loss, train_acc = train(model, train_loader, optimizer, epoch)
    test_loss, test_rmse = test(model, test_loader)

    # 保存最佳模型
    if test_loss < best_loss:
        best_loss = test_loss
        torch.save(model.state_dict(), 'best_model.pth')
        logger.info(f'Best model saved, test loss: {best_loss:.4f}')
    if epoch % 10 == 0:
        logger.info(f'Epoch {epoch} - Train Loss: {train_loss:.4f} Train RMSE: {train_acc:.4f} '
                    f'Test Loss: {test_loss:.4f} Test RMSE: {test_rmse:.4f} Time: {time.time() - start_time:.2f}s')

# 关闭TensorBoard
summary_writer.close()
