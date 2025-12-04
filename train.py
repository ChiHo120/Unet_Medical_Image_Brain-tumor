import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from pycocotools.coco import COCO
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import random
import os
import time
from net import UNet
from data import COCOSegmentationDataset

# 获取当前脚本所在目录
base_dir = os.path.dirname(os.path.abspath(__file__))

# 数据路径设置
train_dir = os.path.join(base_dir, 'dataset', 'train')
val_dir = os.path.join(base_dir, 'dataset', 'valid')
test_dir = os.path.join(base_dir, 'dataset', 'test')

train_annotation_file = os.path.join(base_dir, 'dataset', 'train', '_annotations.coco.json')
test_annotation_file = os.path.join(base_dir, 'dataset', 'test', '_annotations.coco.json')
val_annotation_file = os.path.join(base_dir, 'dataset', 'valid', '_annotations.coco.json')

# 检查文件是否存在
def check_files_exist():
    required_files = [
        train_annotation_file,
        val_annotation_file,
        test_annotation_file
    ]
    
    for file_path in required_files:
        if not os.path.exists(file_path):
            print(f"错误: 找不到文件 {file_path}")
            print(f"当前工作目录: {os.getcwd()}")
            print(f"基础目录: {base_dir}")
            return False
        else:
            print(f"找到文件: {file_path}")
    return True

print("检查数据文件...")
if not check_files_exist():
    print("请确保数据集已正确解压且路径正确")
    print("数据集应该包含: dataset/train/, dataset/valid/, dataset/test/ 文件夹")
    print("每个文件夹中应该有 _annotations.coco.json 文件")
    exit(1)

# 加载COCO数据集
print("加载COCO数据集...")
train_coco = COCO(train_annotation_file)
val_coco = COCO(val_annotation_file)
test_coco = COCO(test_annotation_file)
print("COCO数据集加载完成!")

# 配置参数
config = {
    "batch_size": 4,  # 暂时减小批次大小以便调试
    "learning_rate": 1e-4,
    "num_epochs": 40,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
}

# 定义损失函数
def dice_loss(pred, target, smooth=1e-6):
    pred_flat = pred.view(-1)
    target_flat = target.view(-1)
    intersection = (pred_flat * target_flat).sum()
    return 1 - ((2. * intersection + smooth) / (pred_flat.sum() + target_flat.sum() + smooth))

def combined_loss(pred, target):
    dice = dice_loss(pred, target)
    bce = nn.BCELoss()(pred, target)
    return 0.6 * dice + 0.4 * bce

def main():
    print("=" * 50)
    print("开始训练程序")
    print("=" * 50)
    
    print("配置信息:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    
    # 设置设备
    device = torch.device(config["device"])
    print(f"使用设备: {device}")
    
    # 数据预处理
    print("创建数据预处理转换...")
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((256, 256)),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # 创建数据集
    print("创建数据集中...")
    try:
        train_dataset = COCOSegmentationDataset(train_coco, train_dir, transform=transform)
        print(f"训练集创建成功，样本数: {len(train_dataset)}")
        
        val_dataset = COCOSegmentationDataset(val_coco, val_dir, transform=transform)
        print(f"验证集创建成功，样本数: {len(val_dataset)}")
        
        test_dataset = COCOSegmentationDataset(test_coco, test_dir, transform=transform)
        print(f"测试集创建成功，样本数: {len(test_dataset)}")
    except Exception as e:
        print(f"创建数据集时出错: {e}")
        return
    
    # 创建数据加载器
    print("创建数据加载器中...")
    try:
        BATCH_SIZE = config["batch_size"]
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, num_workers=0)
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, num_workers=0)
        print("数据加载器创建成功!")
    except Exception as e:
        print(f"创建数据加载器时出错: {e}")
        return
    
    # 测试数据加载
    print("测试数据加载...")
    try:
        start_time = time.time()
        for i, (images, masks) in enumerate(train_loader):
            print(f"批次 {i+1}: 图像形状 {images.shape}, 掩码形状 {masks.shape}")
            if i >= 2:  # 只测试前3个批次
                break
        end_time = time.time()
        print(f"数据加载测试成功! 耗时: {end_time - start_time:.2f}秒")
    except Exception as e:
        print(f"数据加载测试失败: {e}")
        return
    
    # 初始化模型
    print("初始化模型中...")
    try:
        model = UNet(n_filters=16).to(device)  # 使用更小的模型进行测试
        print(f"模型初始化成功! 参数数量: {sum(p.numel() for p in model.parameters())}")
    except Exception as e:
        print(f"模型初始化失败: {e}")
        return
    
    # 设置优化器
    print("设置优化器...")
    try:
        optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"])
        print("优化器设置成功!")
    except Exception as e:
        print(f"优化器设置失败: {e}")
        return
    
    # 开始训练
    print("开始训练...")
    try:
        train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            criterion=combined_loss,
            optimizer=optimizer,
            num_epochs=config["num_epochs"],
            device=device,
        )
    except Exception as e:
        print(f"训练过程中出错: {e}")
        return
    
    print("训练完成!")

# 训练函数
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, device):
    print("进入训练循环...")
    best_val_loss = float('inf')
    patience = 8
    patience_counter = 0
    
    # 记录训练历史
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }

    for epoch in range(num_epochs):
        print(f"开始第 {epoch+1}/{num_epochs} 轮训练...")
        model.train()
        train_loss = 0
        train_acc = 0
        
        batch_count = 0
        for images, masks in train_loader:
            batch_count += 1
            if batch_count % 10 == 0:  # 每10个批次打印一次进度
                print(f"  处理第 {batch_count} 个批次...")
                
            images, masks = images.to(device), masks.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            train_acc += (outputs.round() == masks).float().mean().item()

        train_loss /= len(train_loader)
        train_acc /= len(train_loader)
        
        # 验证
        print(f"开始第 {epoch+1} 轮验证...")
        model.eval()
        val_loss = 0
        val_acc = 0
        
        with torch.no_grad():
            for images, masks in val_loader:
                images, masks = images.to(device), masks.to(device)
                outputs = model(images)
                loss = criterion(outputs, masks)
                
                val_loss += loss.item()
                val_acc += (outputs.round() == masks).float().mean().item()
        
        val_loss /= len(val_loader)
        val_acc /= len(val_loader)
        
        # 记录历史
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        print(f'Epoch {epoch+1}/{num_epochs}:')
        print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}')
        print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')
        print('-' * 50)
        
        # 早停
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), 'best_model.pth')
            print(f"保存最佳模型，验证损失: {val_loss:.4f}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("早停触发")
                break
    
    return history

if __name__ == '__main__':
    main()