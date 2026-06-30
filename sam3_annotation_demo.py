"""
用本地 SAM3 checkpoint 在 FiftyOne 里做「自动 / 辅助」分割标注
=================================================================

环境（注意：用 sam3 env，里面已经装好 fiftyone + torch + sam3）：
    conda activate sam3
    export CUDA_VISIBLE_DEVICES="MIG-2f3489ad-b8ef-5d7c-8e85-243e121a7724"
运行：
    python sam3_annotation_demo.py

两种标注模式：
  1) concept（自动标注）：只给文本概念（如 "bird"），SAM3 自动找出并分割所有匹配目标
  2) visual （辅助标注）：用已有的检测框作为提示，SAM3 为每个框生成精确分割掩码

结果写入新数据集 `sam3-demo`，包含字段：
    ground_truth  -> 原始检测框
    sam3_concept  -> 文本自动分割（实例掩码）
    sam3_visual   -> 框 -> 掩码（辅助标注）
之后用  `fiftyone app launch sam3-demo`  在浏览器查看。
"""

import os

# 全程离线：只用本地权重，绝不联网（公司代理会拦截 HuggingFace 下载）
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault(
    "CUDA_VISIBLE_DEVICES", "MIG-2f3489ad-b8ef-5d7c-8e85-243e121a7724"
)

import torch
import fiftyone as fo
import fiftyone.zoo as foz

# --------------------------------------------------------------------------
# 本地 SAM3 权重（checkpoint + CLIP BPE 词表）
# --------------------------------------------------------------------------
SAM3_DIR = (
    "/fs/scratch/rb_bd_dlp_rng_dl01_cr_PRS_employees/"
    "rai_models/hf-base-models/sam3"
)
ENTRYPOINT_ARGS = {
    # 指定 checkpoint_path 后，FiftyOne 不会再去 HuggingFace 下载
    "checkpoint_path": os.path.join(SAM3_DIR, "sam3.pt"),
    "bpe_path": os.path.join(SAM3_DIR, "bpe_simple_vocab_16e6.txt.gz"),
}

MODEL_NAME = "segment-anything-3-image-torch"
NUM_SAMPLES = 10                      # 演示用的图片数量
CONCEPTS = ["bird", "cat", "person"]  # concept 模式要分割的文本概念（可自行修改）


def main():
    # ---- 准备一个干净的小数据集（从 quickstart 克隆一个子集）----
    if fo.dataset_exists("sam3-demo"):
        fo.delete_dataset("sam3-demo")
    dataset = fo.load_dataset("quickstart").limit(NUM_SAMPLES).clone("sam3-demo")
    dataset.persistent = True
    print(f"[准备] sam3-demo: {len(dataset)} 张图")

    # ---- 模式一：concept 自动标注（文本 -> 分割所有匹配目标）----
    print(f"[concept] 加载 SAM3，文本概念 = {CONCEPTS} ...")
    model = foz.load_zoo_model(
        MODEL_NAME,
        classes=CONCEPTS,
        operation_mode="concept",
        entrypoint_args=ENTRYPOINT_ARGS,
    )
    dataset.apply_model(model, label_field="sam3_concept")
    del model
    torch.cuda.empty_cache()
    print(
        "[concept] 完成 ->",
        dataset.count("sam3_concept.detections"),
        "个自动分割实例",
    )

    # ---- 模式二：visual 辅助标注（已有框 -> 掩码）----
    print("[visual] 加载 SAM3 visual 模式 ...")
    model = foz.load_zoo_model(
        MODEL_NAME,
        operation_mode="visual",
        entrypoint_args=ENTRYPOINT_ARGS,
    )
    dataset.apply_model(
        model, label_field="sam3_visual", prompt_field="ground_truth"
    )
    del model
    torch.cuda.empty_cache()
    print(
        "[visual] 完成 ->",
        dataset.count("sam3_visual.detections"),
        "个分割掩码（每个 GT 框一个）",
    )

    print("\n查看结果：")
    print("    conda activate sam3 && fiftyone app launch sam3-demo")
    print(
        "字段：ground_truth(原始框) / sam3_concept(文本自动分割) / "
        "sam3_visual(框->掩码)"
    )


if __name__ == "__main__":
    main()
