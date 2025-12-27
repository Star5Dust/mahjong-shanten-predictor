# ==============================
# 向听预测模型 - 极简版（仅用舍牌计数）
# 输入：37 维（每种牌打了多少张）
# 输出：0=听牌, 1=一向听, 2=两向听及以上
# ==============================

import os
import random
import json
import gzip
from pathlib import Path
from collections import defaultdict
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from datasets import Dataset, load_from_disk
from mahjong.shanten import Shanten
from tqdm import tqdm
import hashlib

# ==============================
# 🔧 配置（请按需修改）
# ==============================
MJSON_DIR = r"D:\archive\2024\2024"  # ← 你的 .mjson 文件目录
NUM_MJSON_FILES = 5000              # 使用多少个 mjson 文件
EPOCH_NUM = 10                      # 训练轮数
RANDOM_SEED = 42
BASE_CACHE_DIR = "cache"            # 基础缓存目录（可改为 D:\xxx）

# 固定随机种子
random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

# === GPU 检查 ===
print("🔥 正在运行最新版 pipeline_new.py！")
if not torch.cuda.is_available():
    raise RuntimeError("❌ CUDA 不可用！请安装 GPU 版本 PyTorch。")
device = torch.device("cuda")
print(f"🚀 使用设备: {device}")
print(f"GPU: {torch.cuda.get_device_name(0)}")

shanten_calculator = Shanten()

def tile_to_index(tile_str):
    """将 '1m', '5mr', 'E' 等转换为 0~36 的索引"""
    if not tile_str:
        return None
    if tile_str.endswith('r'):
        tile_str = tile_str[:-1]
    try:
        num_part = tile_str[0]
        suit_part = tile_str[1]
        num = int(num_part)
        if suit_part == 'm':
            return num - 1
        elif suit_part == 'p':
            return num - 1 + 9
        elif suit_part == 's':
            return num - 1 + 18
        elif suit_part in 'ESWNRGB':
            return {'E': 27, 'S': 28, 'W': 29, 'N': 30, 'R': 31, 'G': 32, 'B': 33}[suit_part]
        else:
            return None
    except:
        return None

def calculate_shanten_with_melds(hand_tiles, melds):
    """根据手牌和副露计算向听数"""
    hand_34 = [0] * 34
    for t in hand_tiles:
        if t < 27:
            hand_34[t] += 1
        elif t < 34:
            hand_34[t] += 1
    shanten_basic = shanten_calculator.calculate_shanten(hand_34)
    reduction = sum(len(m) - 1 for m in melds)
    return max(0, shanten_basic - reduction)

def extract_moments_from_file(file_path):
    """从单个 mjson 文件提取所有打牌时刻的状态"""
    file_path = Path(file_path)
    moments = []

    # 安全读取文件
    try:
        with open(file_path, 'rb') as f:
            header = f.read(2)
        if header == b'\x1f\x8b':
            with gzip.open(file_path, 'rt', encoding='utf-8') as f:
                lines = f.readlines()
        else:
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
    except UnicodeDecodeError:
        try:
            if header == b'\x1f\x8b':
                with gzip.open(file_path, 'rt', encoding='gbk') as f:
                    lines = f.readlines()
            else:
                with open(file_path, 'r', encoding='gbk') as f:
                    lines = f.readlines()
        except:
            print(f"❌ 无法读取文件: {file_path}")
            return []
    except Exception as e:
        print(f"❌ 读取文件出错: {file_path}, 错误: {e}")
        return []

    print(f"✅ 成功读取文件: {file_path}, 行数: {len(lines)}")

    hands = None
    discards = None
    melds = None

    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            event = json.loads(line)
        except:
            continue

        if event['type'] == 'start_kyoku':
            hands = []
            for i in range(4):
                hand_ints = []
                for tile_str in event['tehais'][i]:
                    idx = tile_to_index(tile_str)
                    if idx is not None:
                        hand_ints.append(idx)
                hands.append(hand_ints)
            discards = [[] for _ in range(4)]
            melds = [[] for _ in range(4)]

        elif event['type'] == 'dahai':
            actor = event['actor']
            tile_str = event['pai']
            tile_idx = tile_to_index(tile_str)
            if tile_idx is None:
                continue
            discards[actor].append(tile_idx)
            if tile_idx in hands[actor]:
                hands[actor].remove(tile_idx)
            moment = {
                "discard_seq": discards[actor].copy(),
                "true_shanten": calculate_shanten_with_melds(hands[actor], melds[actor])
            }
            moments.append(moment)

        elif event['type'] in ('chi', 'pon', 'daiminkan', 'kakan', 'ankan'):
            pass  # 简化处理

    return moments

def encode_state(moment):
    """仅使用舍牌计数：37 维"""
    discard_count = [0] * 37
    for tile in moment["discard_seq"]:
        if 0 <= tile < 37:
            discard_count[tile] += 1
    return discard_count

# ==============================
# 🚀 主流程
# ==============================

def main():
    # 生成唯一缓存路径（避免配置变更后加载旧缓存）
    config_str = f"{MJSON_DIR}_{NUM_MJSON_FILES}_{RANDOM_SEED}"
    config_hash = hashlib.md5(config_str.encode()).hexdigest()[:8]
    CACHE_DIR = Path(BASE_CACHE_DIR) / config_hash
    cache_train = CACHE_DIR / "train"
    cache_val = CACHE_DIR / "val"

    if cache_train.exists() and cache_val.exists():
        print(f"📦 加载缓存数据集 (hash={config_hash})...")
        train_ds = load_from_disk(str(cache_train))
        val_ds = load_from_disk(str(cache_val))
    else:
        print("🔍 查找 mjson 文件...")
        root = Path(MJSON_DIR)
        all_paths = list(root.glob("*.mjson")) + list(root.glob("*.mjson.gz"))
        if not all_paths:
            raise FileNotFoundError(f"在 {MJSON_DIR} 中未找到 .mjson 或 .mjson.gz 文件")
        selected_paths = random.sample(all_paths, min(NUM_MJSON_FILES, len(all_paths)))

        print(f"📄 解析 {len(selected_paths)} 个 mjson 文件...")
        all_moments = []
        for i, fp in enumerate(selected_paths):
            print(f"[{i + 1}/{len(selected_paths)}] 正在处理: {fp}")
            moments = extract_moments_from_file(fp)
            all_moments.extend(moments)

        print(f"🧮 共提取 {len(all_moments)} 个训练样本")
        valid_moments = [m for m in all_moments if isinstance(m["true_shanten"], int) and 0 <= m["true_shanten"] <= 8]
        labels = [min(m["true_shanten"], 2) for m in valid_moments]
        input_ids = [encode_state(m) for m in valid_moments]

        ds = Dataset.from_dict({"input_ids": input_ids, "labels": labels})
        ds = ds.train_test_split(test_size=0.1, seed=RANDOM_SEED)
        train_ds = ds["train"]
        val_ds = ds["test"]

        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        train_ds.save_to_disk(str(cache_train))
        val_ds.save_to_disk(str(cache_val))

    input_dim = len(train_ds[0]["input_ids"])
    print(f"✅ 输入维度: {input_dim} (应为 37)")
    print(f"📊 数据集大小 - 训练: {len(train_ds)}, 验证: {len(val_ds)}")

    def collate_fn(batch):
        input_ids = torch.tensor([item["input_ids"] for item in batch], dtype=torch.float32)
        labels = torch.tensor([item["labels"] for item in batch], dtype=torch.long)
        return {"input_ids": input_ids, "labels": labels}

    train_loader = DataLoader(train_ds, batch_size=1024, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=2048, shuffle=False, collate_fn=collate_fn)

    class SimpleMLP(nn.Module):
        def __init__(self, input_dim, num_classes=3):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(input_dim, 128),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(64, num_classes)
            )
        def forward(self, x):
            return self.net(x)

    model = SimpleMLP(input_dim=input_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()

    # === 调试：确认模型在 GPU ===
    print("🔍 模型设备:", next(model.parameters()).device)

    print("🚀 开始训练...")
    for epoch in range(EPOCH_NUM):
        # ===== 训练 =====
        model.train()
        total_loss = 0
        for i, batch in enumerate(train_loader):
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            # === 调试：只在第一个 batch 打印设备 ===
            if epoch == 0 and i == 0:
                print("🔍 输入设备:", input_ids.device)
                print("🔍 标签设备:", labels.device)
            logits = model(input_ids)
            loss = loss_fn(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_train_loss = total_loss / len(train_loader)

        # ===== 验证 =====
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device)
                labels = batch["labels"].to(device)
                logits = model(input_ids)
                preds = logits.argmax(dim=-1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
        val_acc = correct / total

        print(f"Epoch {epoch + 1}/{EPOCH_NUM}, Loss: {avg_train_loss:.4f}, Val Acc: {val_acc:.4f}")

    print("🎉 训练完成！")

if __name__ == "__main__":
    main()