import torchvision
import numpy as np
import os


def prepare_mnist():
    print("正在下载并加载 MNIST 数据集...")
    # 使用 torchvision 下载训练集和测试集
    train_set = torchvision.datasets.MNIST(root='./data', train=True, download=True)
    test_set = torchvision.datasets.MNIST(root='./data', train=False, download=True)

    # 将 28x28 的图像展平为 784 维的向量
    # train_set.data 形状是 (60000, 28, 28) -> (60000, 784)
    X_train = train_set.data.numpy().reshape(-1, 28 * 28)
    y_train = train_set.targets.numpy()

    # test_set.data 形状是 (10000, 28, 28) -> (10000, 784)
    X_test = test_set.data.numpy().reshape(-1, 28 * 28)
    y_test = test_set.targets.numpy()

    # 创建保存目录
    save_dir = './data/mnist_npy'
    os.makedirs(save_dir, exist_ok=True)

    # 保存为通用的 .npy 格式
    np.save(os.path.join(save_dir, 'train_data.npy'), X_train)
    np.save(os.path.join(save_dir, 'train_labels.npy'), y_train)
    np.save(os.path.join(save_dir, 'test_data.npy'), X_test)
    np.save(os.path.join(save_dir, 'test_labels.npy'), y_test)

    print(f"✅ MNIST 数据已成功保存到 {save_dir}/ 目录下！")
    print(f"训练集特征形状: {X_train.shape}, 标签形状: {y_train.shape}")
    print(f"测试集特征形状: {X_test.shape}, 标签形状: {y_test.shape}")


if __name__ == "__main__":
    prepare_mnist()