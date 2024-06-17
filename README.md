# ResNet训练可视化

[![swanlab](https://img.shields.io/badge/ResNet-SwanLab-007BFF)](https://swanlab.cn/@LiXinYu/ResNet_Test/runs/749974s1uexn9i57m8l1n/chart)

## 环境安装

需要安装以下内容：

```
torch
gradio
swanlab
```

> 本文的代码测试于torch==2.2.2、swanlab==0.3.0，更多库版本可查看[SwanLab记录的Python环境](https://swanlab.cn/@LiXinYu/ResNet_Test/runs/749974s1uexn9i57m8l1n/environment/requirements)

## 下载数据集

```
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
 ```

## 使用swanlab可视化结果

```python
# 可视化部署
swanlab.init(
    project="ResNet",
    experiment_name="epoch-100",
    workspace=None,
    description="基于BERT的问答模型",
    config={'epochs': args.epochs, 'learning_rate': args.lr},  # 通过config参数保存输入或超参数
    logdir="./logs",  # 指定日志文件的保存路径
)
```

## 训练

训练过程可视化：[BERT-QA-Swanlab](https://swanlab.cn/@LiXinYu/ResNet_Test/runs/749974s1uexn9i57m8l1n/chart)

在首次使用SwanLab时，需要去[官网](https://swanlab.cn)注册一下账号，然后在[用户设置](https://swanlab.cn/settings)复制一下你的API Key。

然后在终端输入`swanlab login`:

```bash
swanlab login
```

把API Key粘贴进去即可完成登录，之后就不需要再次登录了。
 
