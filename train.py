import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from main import resnet18, resnet152
from tqdm import tqdm
import argparse
import torch.distributed as dist
import torch.multiprocessing as mp
import os
import logging
from tensorboardX import SummaryWriter
import swanlab

os.environ['MASTER_ADDR'] = '172.16.6.2'
os.environ['MASTER_PORT'] = '2222'
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
torch.backends.cudnn.enabled = False


def parse_args():
    """
    添加参数，比如用几块gpu训练、训练的epochs有多少等
    :return:
    """
    parser = argparse.ArgumentParser(description='Training script for your model')  # 添加些参数

    # 添加超参数
    parser.add_argument('--epochs', type=int, default=100, help='Epoch num for training')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=1e-3, help='Learning rate')

    parser.add_argument('-n', '--nodes', default=1, type=int, metavar='N')  # 定义gpu运行的节点数量，这里设定为1
    parser.add_argument('-g', '--gpus', default=2, type=int, help='number of gpus per node')  # 每个节点gpu数量，这里用我自己电脑，所以用1快
    parser.add_argument('-nr', '--nr', default=0, type=int, help='ranking within the nodes')  # 节点内排名，这里设定是1

    # 更多超参数可以根据需要添加
    parser.add_argument('-resnet_num', default=152, type=int, help='num for resnet length')  # 节点内排名，这里设定是1

    args = parser.parse_args()
    args.world_size = args.gpus * args.nodes
    return args


def train(rank, args):
    """

    :param rank: 几块gpy
    :param args: args参数
    :return:
    """

    # 1、设置logger日志
    # 确保日志目录存在
    log_dir = "./results"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    # 设置logger日志
    log_filename = f"{args.resnet_num}_log.txt"
    logging.basicConfig(filename=os.path.join(log_dir, log_filename), filemode='w', level=logging.INFO)
    # 把日志保存在txt文件里，并且w表示覆盖模式，每一次训练得到的日志都会覆盖上一个结果
    # 日志级别被设置为 INFO，这意味着只有 INFO 级别及以上的日志消息才会被记录
    # 确保SummaryWriter的日志目录存在
    tensorboard_log_dir = './logs'
    if not os.path.exists(tensorboard_log_dir):
        os.makedirs(tensorboard_log_dir)
    writer = SummaryWriter(tensorboard_log_dir)
    # 存在上面这个文件夹里

    # 2、设置分布式训练环境
    dist.init_process_group(backend='nccl', init_method='env://', world_size=args.world_size, rank=rank)
    # nccl表示后端通信方式
    # env://初始化方法，这里使用了环境变量来进行初始化，可能是指定了一些环境变量来设置进程间的通信方式和地址。
    # 指定了分布式训练的总进程数
    # 排名（或编号）

    # 3、加载数据
    # 参数：下载保存路径、train=训练集(True)或者测试集(False)、download=在线(True) 或者 本地(False)、数据类型转换
    # 准备数据集并预处理

    # 它使用了 torchvision.transforms 中的一些函数来定义数据增强（data augmentation）和预处理的操作
    # transforms.Compose 允许用户将多个数据转换步骤按顺序应用在数据上，以便在训练或推理过程中进行数据增强、标准化或其他处理。

    # 导入数据
    train_data = torchvision.datasets.CIFAR10("./datas",
                                              train=True,
                                              download=True,
                                              transform=torchvision.transforms.ToTensor())
    test_data = torchvision.datasets.CIFAR10("./datas",
                                             train=False,
                                             download=True,
                                             transform=torchvision.transforms.ToTensor())
    train_len = len(train_data)
    val_len = len(test_data)
    print("训练数据集合{} = 50000".format(train_len))
    print("测试数据集合{} = 10000".format(val_len))

    # 4、格式打包
    # 参数：数据、1组几个、下一轮轮是否打乱、进程个数、最后一组是否凑成一组
    # train_sampler = torch.utils.data.distributed.DistributedSampler(train_data, num_replicas=args.world_size, rank=rank, shuffle=True)
    # 它用于在分布式环境下对训练数据进行采样
    # num_replicas=args.world_size 指定了采样的副本数
    # rank 是当前进程的排名
    # shuffle=True 表示每个 epoch 是否重新洗牌数据

    train_loader = DataLoader(dataset=train_data, batch_size=args.batch_size,
                              shuffle=True, num_workers=0, drop_last=True)
    # num_workers=4 是用于数据加载的子进程数
    # drop_last=True 表示如果最后一个批次样本数量不足一个批次，则丢弃该批次

    test_loader = DataLoader(dataset=test_data, batch_size=args.batch_size,
                             shuffle=True, num_workers=0, drop_last=True)

    # 初始化swanlab
    lxy = swanlab.init(
        project="ResNet_Test",
        experiment_name="ResNet152",
        workspace=None,
        description="ResNet训练",
        config={'epochs': args.epochs, 'learning_rate': args.learning_rate},  # 通过config参数保存输入或超参数
        logdir="./logs",  # 指定日志文件的保存路径
    )

    # 5、导入网络
    tudui = resnet152()
    # 使用GPU
    tudui = tudui.cuda(rank)

    # 损失函数
    loss_fn = nn.CrossEntropyLoss()
    # 使用GPU
    loss_fn = loss_fn.cuda(rank)

    # 优化器
    optimizer = torch.optim.SGD(tudui.parameters(), lr=args.learning_rate)
    # 下面这句具体不知道是为什么，但是下面这句会导致训练的时候每次的梯度都会下降，即每次梯度都会×0.1，最后会有梯度消失的现象，导致准确率不会提高
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)
    # 训练轮数700
    epoch = args.epochs

    print(f'Using GPU: {torch.cuda.current_device()}')  # 打印当前使用的 GPU 设备

    # 保存权重文件
    folder_name = 'output_pth'

    # 检查文件夹是否存在
    if not os.path.exists(folder_name):
        # 如果不存在，则创建文件夹
        os.makedirs(folder_name)
        print(f"文件夹 '{folder_name}' 创建成功！")
    else:
        print(f"文件夹 '{folder_name}' 已经存在。")

    for i in range(epoch):
        print()
        print("第{}轮训练开始".format(i + 1))

        # 训练开关
        tudui.train(mode=True)
        # 准确率总和
        acc_ = 0
        # 训练
        pbar = tqdm(enumerate(train_loader), total=len(train_loader))  # 创建一个可视化进度条
        for batch_idx, data in pbar:
            imgs, targets = data
            # 使用GPU
            imgs = imgs.cuda(rank)
            targets = targets.cuda(rank)

            # 数据输入模型
            outputs = tudui(imgs)
            loss = loss_fn(outputs, targets)

            # 优化模型  清零、反向传播、优化器开始优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # scheduler.step()
            # 更新日志
            pbar.set_description(f'Epoch [{i + 1}/{epoch}] Loss: {loss / (batch_idx + 1):.4f}')
            # 准确率
            accuracy = (outputs.argmax(1) == targets).sum()
            acc_ += accuracy

        print("EPOCH数{},Loss:{}, 准确率：{}".format(i, loss, acc_ / train_len))
        # log metrics to wandb
        lxy.log({"tain_loss": loss, "train_acc": acc_ / train_len})
        # 将损失值添加到SummaryWriter中
        writer.add_scalar('Loss', loss.item(), i)
        # 测试脚本
        acc_test = test(rank, tudui, test_loader, loss_fn, lxy)
        # 将精度写入日志
        logging.info("Epoch: {}, Accuracy_train: {}, Accuracy_test: {}".format(i, acc_ / train_len, acc_test))
        # 每10轮保存模型
        if (i + 1) % 20 == 0:
            torch.save(tudui, "./output_pth/{}_{}_{}_{}.pth".format(args.resnet_num, args.epochs, args.batch_size, i))
            print("模型已保存")

    writer.close()
    # 销毁分布式训练环境
    dist.destroy_process_group()
    # [optional] finish the wandb run, necessary in notebooks


def test(gpu, tudui, test_loader, loss_fn, lxy):
    # 测试开关
    tudui.eval()
    val_len = len(test_loader.dataset)
    # 测试
    total_test_loss = 0
    acc_val = 0
    with torch.no_grad():
        for data in test_loader:
            imgs, targets = data
            # 使用GPU
            imgs = imgs.cuda(gpu)
            targets = targets.cuda(gpu)

            outputs = tudui(imgs)
            loss = loss_fn(outputs, targets)

            # 准确率
            accuracy_val = (outputs.argmax(1) == targets).sum()
            acc_val += accuracy_val

            total_test_loss += loss
            print("\r测试集的Loss:{}".format(total_test_loss), end="")
    print()
    print("整体测试集的Loss:{}, 准确率{}".format(total_test_loss, acc_val / val_len))
    lxy.log({"test_loss": loss, "test_acc": acc_val / val_len})
    return acc_val / val_len


if __name__ == "__main__":
    args = parse_args()
    mp.spawn(train, nprocs=args.gpus, args=(args,))  # 后半部分意味着把args里的参数传递给了train函数
