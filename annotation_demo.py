"""
FiftyOne 标注数据快速上手 Demo
================================

运行方式：
    conda activate fiftyone
    export CUDA_VISIBLE_DEVICES="MIG-2f3489ad-b8ef-5d7c-8e85-243e121a7724"
    python annotation_demo.py

这个脚本演示 FiftyOne 里「标注数据」的三种方式：
  1. 在 App 里给样本 / 标签打 tag（零配置，最快）
  2. 用 Python 直接创建 / 修改标签（程序化标注）
  3. 连接专业标注工具（CVAT / Label Studio）画框、分割（需安装对应后端）

为了不污染原始 quickstart 数据，这里克隆出一个独立的沙盒数据集 annotation-demo。
"""

import fiftyone as fo
import fiftyone.zoo as foz

SANDBOX = "annotation-demo"

# ---------------------------------------------------------------------------
# 0) 准备数据：载入 quickstart（200 张图，含 ground_truth / predictions 检测框）
#    并克隆一份独立的沙盒，方便随便改、随时删
# ---------------------------------------------------------------------------
base = foz.load_zoo_dataset("quickstart")

if fo.dataset_exists(SANDBOX):
    fo.delete_dataset(SANDBOX)  # 让脚本可以重复运行

dataset = base.clone(SANDBOX)
dataset.persistent = True  # 关闭进程后数据仍保留在数据库里
print(f"[准备完成] 沙盒数据集 = {dataset.name} | 样本数 = {len(dataset)}")

# ---------------------------------------------------------------------------
# 1) 程序化标注：用 Python 直接新增 / 修改标签
#    Detection 的 bounding_box = [x, y, w, h]，都是 0~1 的相对坐标（左上角 + 宽高）
# ---------------------------------------------------------------------------
sample = dataset.first()
before = len(sample.ground_truth.detections)

# 新增一个检测框
new_box = fo.Detection(label="demo_object", bounding_box=[0.1, 0.1, 0.3, 0.4])
sample["ground_truth"].detections.append(new_box)

# 也可以顺手修改已有标签（例如把第一个框的类别改名）
sample.ground_truth.detections[0].label = "renamed_label"

sample.save()  # 一定要 save() 才会写回数据库
after = len(sample.ground_truth.detections)
print(f"[程序化标注] 第一张图检测框：{before} -> {after}（已新增 demo_object）")

# ---------------------------------------------------------------------------
# 2) tag 标注：给样本 / 标签批量打标记（常用于「待复核」「标错了」等）
# ---------------------------------------------------------------------------
# 给前 10 张图打上样本级 tag
dataset.take(10).tag_samples("to_review")
# 给所有 confidence < 0.2 的预测框打上标签级 tag
low_conf = dataset.filter_labels("predictions", fo.ViewField("confidence") < 0.2)
low_conf.tag_labels("low_confidence", label_fields=["predictions"])

print("[tag 标注] sample tags =", dataset.count_sample_tags())
print("[tag 标注] label tags =", dataset.count_label_tags())

# ---------------------------------------------------------------------------
# 3) 专业标注工具（画框 / 分割）—— 以 Label Studio 为例（本地运行，无需云账号）
#    先安装：pip install label-studio
#    下面这段默认注释掉，想用时取消注释即可。
# ---------------------------------------------------------------------------
# anno_key = "demo_run_1"
# view = dataset.take(5)  # 只取 5 张做演示，避免上传太多
# view.annotate(
#     anno_key,
#     backend="labelstudio",           # 也可换成 "cvat"（需 Docker 或 cvat.ai 账号）
#     label_field="ground_truth",
#     label_type="detections",
#     classes=["bird", "cat", "dog", "demo_object"],
#     launch_editor=True,              # 自动打开标注工具
# )
# # 在 Label Studio 里画完框、保存后，把标注结果拉回 FiftyOne：
# dataset.load_annotations(anno_key)
# print("[后端标注] 已从 Label Studio 拉回标注结果")

# ---------------------------------------------------------------------------
# 4) 在 App 里查看结果
# ---------------------------------------------------------------------------
print(
    "\n下一步：用下面命令在 App 里查看刚才的标注效果：\n"
    f"    fiftyone app launch {SANDBOX}\n"
    "在浏览器（VS Code 的 PORTS 面板会转发 5151 端口）里：\n"
    "  - 勾选样本 -> 顶部 Tag samples 给样本打 tag\n"
    "  - 点开某个检测框 -> Tag labels 给标签打 tag\n"
)
