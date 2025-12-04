import torch
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
from net import UNet
import numpy as np
import os
import glob
import time
import datetime

# 设置matplotlib使用支持中文的字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

def load_model(model_path='readme_files/best_model.pth', device='cpu'):
    """加载训练好的模型"""
    try:
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"模型文件不存在: {model_path}")
            
        print(f"从 {model_path} 加载模型...")
        
        # 使用与训练时相同的配置
        model = UNet(n_filters=16).to(device)
        
        # 加载权重
        state_dict = torch.load(model_path, map_location=device, weights_only=True)
        model.load_state_dict(state_dict)
        model.eval()
        
        print("模型加载成功!")
        print(f"模型参数数量: {sum(p.numel() for p in model.parameters())}")
        return model
        
    except Exception as e:
        print(f"加载模型时出错: {e}")
        raise

def preprocess_image(image_path):
    """预处理输入图像"""
    image = Image.open(image_path).convert('RGB')
    display_image = image.resize((256, 256), Image.Resampling.BILINEAR)
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((256, 256)),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    image_tensor = transform(image)
    return image_tensor.unsqueeze(0), display_image

def predict_mask(model, image_tensor, device='cpu', threshold=0.5):
    """预测分割掩码"""
    with torch.no_grad():
        image_tensor = image_tensor.to(device)
        prediction = model(image_tensor)
        prediction = (prediction > threshold).float()
    return prediction

def generate_unique_filename(image_path, suffix):
    """生成唯一的文件名，避免覆盖"""
    # 获取原始图像的文件名（不含扩展名）
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    
    # 添加时间戳确保唯一性
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 组合成新文件名
    return f"{base_name}_{suffix}_{timestamp}.png"

def visualize_result(original_image, predicted_mask, image_path, save_dir='./results'):
    """可视化预测结果"""
    # 创建结果目录
    os.makedirs(save_dir, exist_ok=True)
    
    # 生成唯一文件名
    save_path = os.path.join(save_dir, generate_unique_filename(image_path, "predictions"))
    
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.imshow(original_image)
    plt.title('原始脑部图像', fontsize=12)
    plt.axis('off')
    
    plt.subplot(1, 3, 2)
    plt.imshow(predicted_mask.squeeze(), cmap='gray')
    plt.title('预测肿瘤区域', fontsize=12)
    plt.axis('off')
    
    plt.subplot(1, 3, 3)
    plt.imshow(np.array(original_image))
    plt.imshow(predicted_mask.squeeze(), cmap='Reds', alpha=0.4)
    plt.title('叠加显示 (红色=肿瘤)', fontsize=12)
    plt.axis('off')
        
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"可视化结果已保存为: {save_path}")
    
    # 显示图像（可选）
    plt.show()
    
    return save_path

def save_mask_as_image(mask_array, image_path, save_dir='./results'):
    """将分割掩码保存为图像文件"""
    # 创建结果目录
    os.makedirs(save_dir, exist_ok=True)
    
    # 生成唯一文件名
    output_path = os.path.join(save_dir, generate_unique_filename(image_path, "mask"))
    
    mask_uint8 = (mask_array.squeeze() * 255).astype(np.uint8)
    mask_image = Image.fromarray(mask_uint8, mode='L')
    mask_image.save(output_path)
    print(f"分割掩码已保存为: {output_path}")
    
    return output_path

def save_mask_as_numpy(mask_array, image_path, save_dir='./results'):
    """将分割掩码保存为numpy文件"""
    # 创建结果目录
    os.makedirs(save_dir, exist_ok=True)
    
    # 生成唯一文件名
    output_path = os.path.join(save_dir, generate_unique_filename(image_path, "data").replace('.png', '.npy'))
    
    np.save(output_path, mask_array.squeeze())
    print(f"分割数据已保存为: {output_path}")
    
    return output_path

def find_any_image():
    """在当前目录和子目录中查找任何图像文件"""
    print("在当前目录和子目录中查找图像文件...")
    
    # 支持的图像格式
    extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff']
    
    for ext in extensions:
        # 在当前目录查找
        files = glob.glob(f"*{ext}")
        if files:
            print(f"在当前目录找到图像: {files[0]}")
            return files[0]
        
        # 在所有子目录中查找
        files = glob.glob(f"**/*{ext}", recursive=True)
        if files:
            print(f"在子目录找到图像: {files[0]}")
            return files[0]
    
    return None

def main():
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    try:
        # 加载模型
        model_path = "./best_model.pth"
        model = load_model(model_path, device)
        
        # 查找图像文件
        image_path = None
        
        # 首先尝试使用您原来的路径
        original_path =r"D:\桌面\Things_have_been_completed\xiae_Study\UNet-Medical-master\UNet-Transformer\dataset\test\2603_jpg.rf.5e3809e5081d5f1a7f30ba781331c4b2.jpg"
        if os.path.exists(original_path):
            image_path = original_path
            print(f"使用原始路径图像: {image_path}")
        else:
            print(f"原始路径不存在: {original_path}")
            
            # 尝试在当前目录和子目录中查找任何图像
            image_path = find_any_image()
            
            if image_path is None:
                # 如果还是找不到，让用户输入
                print("\n无法自动找到图像文件，请选择以下选项:")
                print("1. 手动输入图像文件路径")
                print("2. 退出程序")
                
                choice = input("请选择 (1 或 2): ").strip()
                if choice == "1":
                    image_path = input("请输入图像文件的完整路径: ").strip()
                    # 去除可能的引号
                    image_path = image_path.strip('"').strip("'")
                else:
                    print("程序退出")
                    return
        
        # 验证图像文件是否存在
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"图像文件不存在: {image_path}")
            
        print(f"处理图像: {image_path}")
        
        # 预处理图像
        image_tensor, original_image = preprocess_image(image_path)
        print(f"输入图像形状: {image_tensor.shape}")
        
        # 进行预测
        print("进行预测...")
        predicted_mask = predict_mask(model, image_tensor, device)
        
        # 将预测结果转回CPU并转换为numpy数组
        predicted_mask = predicted_mask.cpu().numpy()
        print(f"预测掩码形状: {predicted_mask.shape}")
        
        # 计算肿瘤区域统计
        tumor_pixels = np.sum(predicted_mask > 0)
        total_pixels = predicted_mask.size
        tumor_ratio = (tumor_pixels / total_pixels) * 100
        print(f"肿瘤区域统计: {tumor_pixels}/{total_pixels} 像素 ({tumor_ratio:.2f}%)")
        
        # 创建结果目录
        results_dir = './results'
        os.makedirs(results_dir, exist_ok=True)
        
        # 可视化结果
        print("生成可视化结果...")
        prediction_path = visualize_result(original_image, predicted_mask, image_path, results_dir)
        
        # 保存各种格式的结果
        mask_path = save_mask_as_image(predicted_mask, image_path, results_dir)
        data_path = save_mask_as_numpy(predicted_mask, image_path, results_dir)
        
        print("\n" + "="*50)
        print("所有结果已成功保存到 results 目录:")
        print(f"- 可视化对比图: {os.path.basename(prediction_path)}")
        print(f"- 二值分割掩码: {os.path.basename(mask_path)}")
        print(f"- 原始数据文件: {os.path.basename(data_path)}")
        print(f"- 肿瘤区域占比: {tumor_ratio:.2f}%")
        print("="*50)
        
    except Exception as e:
        print(f"预测过程中出错: {str(e)}")
        print("\n故障排除建议:")
        print("1. 确保当前目录或子目录中有图像文件")
        print("2. 图像文件格式支持: JPG, JPEG, PNG, BMP, TIFF")
        print("3. 您可以手动将图像文件复制到当前目录")
        print("4. 或者运行程序时手动输入图像文件完整路径")

if __name__ == '__main__':
    main()