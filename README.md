# 前路声声（BlindPath）

面向视障人士的智能出行感知原型：通过第一视角摄像头识别盲道、交通与常见道路障碍物，并将检测结果转化为简短、可执行的语音提示。

> 本仓库是算法与实时感知原型，不替代盲杖、导盲犬、无障碍设施或专业出行训练；实际部署必须进行充分的安全测试与人工评估。

## 项目能力

- **盲道识别**：横向导向盲道、纵向导向盲道、警示盲道。
- **道路目标识别**：行人、车辆、路桩、锥桶、井盖、警示牌等可配置类别。
- **行动导向反馈**：将结果转为“左侧/前方/右侧”的简短提示，并抑制重复播报；可选本地语音输出。
- **双模型协同**：盲道模型与道路障碍物模型可同时加载，类别和置信度由 YAML 配置。

## 仓库结构

```text
configs/                 模型配置示例（不含权重）
scripts/train.py         通用 YOLO 训练入口
src/blindpath/           实时推理与提示逻辑
project/                 本地历史数据、训练产物和权重（默认不提交）
```

## 快速开始

要求：Python 3.10+；训练建议使用 NVIDIA GPU 与匹配的 PyTorch/CUDA 环境。

```bash
python -m venv .venv
.venv\\Scripts\\activate
pip install -r requirements.txt
copy configs\\models.example.yaml configs\\models.yaml
```

将训练好的权重放到 `models/`，然后在 `configs/models.yaml` 中填写路径。

```bash
set PYTHONPATH=src
python -m blindpath.cli --config configs/models.yaml --source 0 --show --speak
```

`--speak` 需要另行安装 `pyttsx3`；按 `Q` 或 `Esc` 退出预览窗口。

## 训练

数据集、训练缓存和输出均不纳入版本控制。准备好 YOLO 格式的 `data.yaml` 后：

```bash
python scripts/train.py --data path\\to\\data.yaml --model yolo11n.pt --epochs 100 --imgsz 640
```

模型权重请通过 GitHub Releases、对象存储或模型平台单独分发；公开前请确认数据集与模型授权。

## 安全与限制

- 当前“左/前/右”仅由目标在图像中的相对位置估计，不等于真实空间方位。
- 单目 2D 检测框不能可靠给出米级距离；距离和碰撞风险判断应接入双目/深度相机并完成标定后再使用。
- 光照、反光、遮挡、盲道磨损和未覆盖场景都会影响识别；禁止将原型作为唯一安全保障。
