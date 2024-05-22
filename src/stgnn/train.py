import argparse
import logging
import os
import time

import torch
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from torch.utils.tensorboard import SummaryWriter
from torch_geometric.loader import DataLoader
from tqdm import tqdm

from src.stgnn.data import HumanBodyDataset
from src.stgnn.model import HumanBodyGNN

os.chdir(os.path.join(os.path.dirname(__file__), '..\\..\\'))
# 使用数据集
dataset = HumanBodyDataset(root='data')
print(len(dataset))

# 划分训练集和测试集
train_dataset, test_dataset = train_test_split(dataset, test_size=0.2, random_state=42)
train_loader = DataLoader(train_dataset, batch_size=20000, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=20000, shuffle=False)

# 配置管理
parser = argparse.ArgumentParser(description='Training Config')
parser.add_argument('--epochs', default=10, type=int, help='number of total epochs')
parser.add_argument('--lr', default=1e-5, type=float, help='learning rate')
parser.add_argument('--checkpoint', default='checkpoint.pth', help='path to checkpoint')
args = parser.parse_args()

# 日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# TensorBoard Summary
summary_writer = SummaryWriter()


def train(model, train_loader, optimizer, scheduler, epoch):
    model.train()
    running_loss = 0.0
    correct = 0
    proc_data = tqdm(train_loader, desc=f'Train Epoch {epoch}')
    for i, data in enumerate(proc_data):
        data = data.to(device)
        optimizer.zero_grad()
        output = model(data)
        output = output.squeeze(1)
        loss = F.mse_loss(output, data.y)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        correct += ((output - data.y).abs() < 0.1).sum().item()
        proc_data.set_postfix(loss=running_loss / (i + 1), acc=correct / ((i + 1) * data.y.size(0)))

    # 记录训练损失和准确率
    train_loss = running_loss / len(train_loader)
    summary_writer.add_scalar('Loss/train', train_loss, epoch)
    summary_writer.add_scalar('Acc/train', correct / len(train_loader.dataset), epoch)

    # 保存模型
    if epoch % 10 == 0:
        checkpoint = {
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'best_acc': best_acc
        }
        torch.save(checkpoint, args.checkpoint)
        logging.info(f'Checkpoint saved to {args.checkpoint}')

    return train_loss, correct / len(train_loader.dataset)


def test(model, loader):
    model.eval()
    running_loss = 0.0
    correct = 0
    proc_data = tqdm(loader, desc='Test')
    with torch.no_grad():
        for i, data in enumerate(proc_data):
            data = data.to(device)
            output = model(data)
            output = output.squeeze(1)
            loss = F.mse_loss(output, data.y)
            running_loss += loss.item()
            correct += ((output - data.y).abs() < 0.1).sum().item()
            proc_data.set_postfix(loss=running_loss / (i + 1), acc=correct / (i + 1 * data.y.size(0)))

    summary_writer.add_scalar('Loss/test', running_loss / len(loader), epoch)
    summary_writer.add_scalar('Acc/test', correct / len(loader.dataset), epoch)

    return running_loss / len(loader), correct / len(loader.dataset)


# 创建模型、优化器和学习率调度器
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = HumanBodyGNN().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, verbose=True)

# 主循环
start_epoch = 1
best_acc = 0.0

# 加载检查点
if os.path.exists(args.checkpoint):
    checkpoint = torch.load(args.checkpoint)
    start_epoch = checkpoint['epoch'] + 1
    best_acc = checkpoint.get('best_acc', 0.0)
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    logger.info(f'Loaded checkpoint from {args.checkpoint}')

for epoch in range(start_epoch, args.epochs + 1):
    start_time = time.time()

    # 训练和评估
    train_loss, train_acc = train(model, train_loader, optimizer, scheduler, epoch)
    test_loss, test_acc = test(model, test_loader)

    # 调整学习率
    scheduler.step(train_loss)

    # 保存最佳模型
    if test_acc > best_acc:
        best_acc = test_acc
        torch.save(model.state_dict(), 'best_model.pth')
        logger.info(f'Best model saved, test acc: {best_acc:.4f}')

    logger.info(f'Epoch {epoch} - Train Loss: {train_loss:.4f} Train Acc: {train_acc:.4f} '
                f'Test Loss: {test_loss:.4f} Test Acc: {test_acc:.4f} Time: {time.time() - start_time:.2f}s')

# 关闭TensorBoard
summary_writer.close()
