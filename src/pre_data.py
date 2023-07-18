import os
import random
import shutil

# 设置随机种子，以确保结果可重现
random.seed(42)

# 数据集文件夹路径
dataset_folder = "../data"

# 训练集和测试集的比例
train_ratio = 0.7

# 创建训练集和测试集文件夹
train_folder = os.path.join(dataset_folder, "train")
test_folder = os.path.join(dataset_folder, "test")
os.makedirs(train_folder, exist_ok=True)
os.makedirs(test_folder, exist_ok=True)

# 分别创建猫和狗的文件夹
cat_train_folder = os.path.join(train_folder, "cat")
dog_train_folder = os.path.join(train_folder, "dog")
cat_test_folder = os.path.join(test_folder, "cat")
dog_test_folder = os.path.join(test_folder, "dog")
os.makedirs(cat_train_folder, exist_ok=True)
os.makedirs(dog_train_folder, exist_ok=True)
os.makedirs(cat_test_folder, exist_ok=True)
os.makedirs(dog_test_folder, exist_ok=True)

# 获取数据集文件夹中的所有文件
files = os.listdir(dataset_folder)

# 遍历数据集中的每个文件
for file in files:
    if file.endswith(".jpg"):
        # 获取文件名和类别（猫或狗）
        file_name, extension = os.path.splitext(file)
        category, index = file_name.split(".")

        # 随机分配到训练集或测试集
        if random.random() < train_ratio:
            # 复制文件到训练集文件夹
            if category == "cat":
                shutil.copyfile(
                    os.path.join(dataset_folder, file),
                    os.path.join(cat_train_folder, file)
                )
            elif category == "dog":
                shutil.copyfile(
                    os.path.join(dataset_folder, file),
                    os.path.join(dog_train_folder, file)
                )
        else:
            # 复制文件到测试集文件夹
            if category == "cat":
                shutil.copyfile(
                    os.path.join(dataset_folder, file),
                    os.path.join(cat_test_folder, file)
                )
            elif category == "dog":
                shutil.copyfile(
                    os.path.join(dataset_folder, file),
                    os.path.join(dog_test_folder, file)
                )
