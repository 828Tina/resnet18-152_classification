import gradio as gr
from PIL import Image
import numpy as np
import torch
import torchvision
import pathlib

# 检查是否有可用的GPU
device = torch.device('cpu')

# 定义类别对应字典
dist = {0: "飞机", 1: "汽车", 2: "鸟", 3: "猫", 4: "鹿", 5: "狗", 6: "青蛙", 7: "马", 8: "船", 9: "卡车"}

# 初始化全局模型变量
global_model = None


def choose_model(choice):
    """
    选择模型
    """
    global global_model
    if choice == "18":
        model = torch.load('./output_pth/18_100_16_99.pth', map_location=device)
    elif choice == "152":
        model = torch.load('./output_pth/152_100_16_99.pth', map_location=device)
    else:
        model = torch.load('./output_pth/152_100_16_99.pth', map_location=device)
    model = model.to(device)
    global_model = model
    return f"已加载模型：ResNet-{choice}"


def set_example_image(example: list) -> dict:
    # 读取图像文件路径并加载为 PIL 图像
    image_path = example[0]["path"]
    image = Image.open(image_path)
    return gr.update(value=image)


def predict(image):
    global global_model
    if global_model is None:
        return "请先选择模型"

    # 如果图像是numpy.ndarray类型，将其转换为PIL Image
    if isinstance(image, np.ndarray):
        image = Image.fromarray(np.uint8(image))

    # 数据预处理
    transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize((32, 32)),
        torchvision.transforms.ToTensor()
    ])
    image = transform(image)
    image = image.to(device)
    image = torch.reshape(image, (1, 3, 32, 32))

    # 模型测试开关
    global_model.eval()

    with torch.no_grad():
        output = global_model(image)

    # 转numpy格式, 列表内取第一个
    a = dist[output.argmax(1).cpu().numpy()[0]]
    return a


if __name__ == "__main__":
    # 标题
    title = "<h1 id='title'> ResNet图像分类 </h1>"
    css = '''
    h1#title {
      text-align:center;
      font-size:20px;
    }
    '''
    demo = gr.Blocks(css=css)
    with demo:
        gr.Markdown(title)
        with gr.Row():
            with gr.Column():
                model_size = gr.Radio(["18", "152"], label="模型网络深度", value="152")
                load_model_btn = gr.Button("加载模型")
                model_status = gr.Textbox(label="模型状态", interactive=False)
                input = gr.Image(label="输入图")
                submit = gr.Button("Submit")
                example_images = gr.Dataset(components=[input], samples=[[str(path)] for path in sorted(
                    pathlib.Path('./examples').rglob('*.jpg'))])
            with gr.Column():
                output = gr.Textbox(label="输出类别")

        load_model_btn.click(fn=choose_model, inputs=[model_size], outputs=[model_status])
        example_images.click(fn=set_example_image, inputs=[example_images], outputs=[input])
        submit.click(fn=predict, inputs=[input], outputs=[output])

    demo.launch(server_name="0.0.0.0", share=True)
