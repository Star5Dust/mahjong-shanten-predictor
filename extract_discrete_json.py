# -*- coding: utf-8 -*-
"""
从 Tenhou 牌谱 (.mjson / .mjson.gz) 提取非立直阶段的打牌状态，
每个样本保存为一个独立的 JSON 文件。
"""

import json
import gzip
from pathlib import Path
import os
import re
import time
from multiprocessing import Pool, cpu_count
from tqdm import tqdm

# ----------------------------
# 配置区（请根据你的路径修改）
# ----------------------------
INPUT_ROOT_PATH = r"D:\archive\2024\2024"  # 输入目录（含 .mjson 或 .mjson.gz）
OUTPUT_BASE_PATH = r"D:\archive\paipu_json_discrete"  # 输出根目录
NUM_FILES_TO_PROCESS = 5000  # 处理前 N 个文件（设为 None 则处理全部）
MAX_WORKERS = None  # 并行进程数（None = 自动）


# ----------------------------
# 牌编码函数
# ----------------------------
def pai_to_id(pai: str) -> int:
    """将 Tenhou 牌名转为 0~36 的内部 ID"""
    if pai == '5mr':
        return 0
    elif pai == '5pr':
        return 10
    elif pai == '5sr':
        return 20
    elif pai.endswith('m'):
        return int(pai[0])
    elif pai.endswith('p'):
        return int(pai[0]) + 9
    elif pai.endswith('s'):
        return int(pai[0]) + 19
    else:
        honor_map = {'E': 30, 'S': 31, 'W': 32, 'N': 33, 'P': 34, 'F': 35, 'C': 36}
        return honor_map.get(pai, -1)




def get_dora_tiles(indicator: int) -> list:
    """
    将宝牌指示牌 ID 转换为对应的真实宝牌 ID 列表（支持红5）
    输入: indicator (0~36)
    输出: list of dora tile IDs (0~36)
    """
    if indicator < 0 or indicator > 36:
        return []

    dora_list = []

    # 万子 (0-9): 红5万=0, 1m=1, ..., 9m=9
    if 0 <= indicator <= 9:
        if indicator == 0:  # 红5万 → 宝牌是 6m (ID=6)
            next_tile = 6
        elif 1 <= indicator <= 8:
            next_tile = indicator + 1
        elif indicator == 9:  # 9m → 1m
            next_tile = 1
        else:
            next_tile = None

        if next_tile is not None:
            dora_list.append(next_tile)
            # 如果是 5m (ID=5)，还要加上红5万 (ID=0)
            if next_tile == 5:
                dora_list.append(0)

    # 筒子 (10-19): 红5筒=10, 1p=11, ..., 9p=19
    elif 10 <= indicator <= 19:
        if indicator == 10:  # 红5筒 → 宝牌是 6p (ID=16)
            next_tile = 16
        elif 11 <= indicator <= 18:
            next_tile = indicator + 1
        elif indicator == 19:  # 9p → 1p
            next_tile = 11
        else:
            next_tile = None

        if next_tile is not None:
            dora_list.append(next_tile)
            # 如果是 5p (ID=15)，还要加上红5筒 (ID=10)
            if next_tile == 15:
                dora_list.append(10)

    # 索子 (20-29): 红5索=20, 1s=21, ..., 9s=29
    elif 20 <= indicator <= 29:
        if indicator == 20:  # 红5索 → 宝牌是 6s (ID=26)
            next_tile = 26
        elif 21 <= indicator <= 28:
            next_tile = indicator + 1
        elif indicator == 29:  # 9s → 1s
            next_tile = 21
        else:
            next_tile = None

        if next_tile is not None:
            dora_list.append(next_tile)
            # 如果是 5s (ID=25)，还要加上红5索 (ID=20)
            if next_tile == 25:
                dora_list.append(20)

    # 字牌 (30-36): 东=30, 南=31, ..., 中=36
    elif 30 <= indicator <= 36:
        if indicator == 36:  # 中 → 东
            next_tile = 30
        else:
            next_tile = indicator + 1
        dora_list.append(next_tile)

    return dora_list



def hand_to_vec(hand):
    """将手牌列表转为长度为 37 的向量"""
    vec = [0] * 37
    for tile in hand:
        if 0 <= tile < 37:
            vec[tile] += 1
    return vec


# ----------------------------
# 向听数计算（使用 mahjong 库）
# ----------------------------
try:
    from mahjong.shanten import Shanten

    SHANTEN_CALC = Shanten()
except ImportError:
    raise ImportError("请先安装 mahjong: pip install mahjong")


def calculate_shanten_with_melds(hand_tiles_37, meld_counts):
    """
    计算带副露的向听数（简化版：仅用 meld_counts 判断是否门清）
    注意：mahjong.Shanten 不支持传入 Meld 对象时自动处理暗杠，
          但若只关心“有无副露”，可强制设 melds=[] 并用门清逻辑。
    本实现采用：只要有任何副露（chi/pon/kan），就视为非门清 → 向听+1？
    但更准确做法是：用 TilesConverter 构造 136 格式 + Meld 对象。

    然而，为简化且避免 Meld 兼容问题，此处采用：
      - 若无任何副露 → 门清向听
      - 否则 → 非门清向听（实际向听可能相同，但保险起见我们仍调用标准算法）

    实际上，Shanten.calculate_shanten(tiles_136, melds=[]) 已能处理非门清手牌结构。
    所以我们只需传入当前手牌（不含副露部分），因为副露已从 hand 中移除。
    因此，直接计算即可。
    """
    # 转换为 34 张牌表示（mahjong 内部格式）
    tiles_34 = [0] * 34
    for tid in hand_tiles_37:
        if tid == 0:  # 5mr → 5m (index 4)
            tiles_34[4] += 1
        elif tid == 10:  # 5pr → 5p (index 13)
            tiles_34[13] += 1
        elif tid == 20:  # 5sr → 5s (index 22)
            tiles_34[22] += 1
        elif 1 <= tid <= 9:
            tiles_34[tid - 1] += 1
        elif 11 <= tid <= 19:
            tiles_34[tid - 2] += 1
        elif 21 <= tid <= 29:
            tiles_34[tid - 3] += 1
        elif 30 <= tid <= 36:
            tiles_34[tid - 5] += 1
        # 忽略无效牌
    try:
        shanten = SHANTEN_CALC.calculate_shanten(tiles_34)
        return max(0, min(shanten, 6))
    except:
        return 6


# ----------------------------
# 单文件处理函数（直接写小 JSON）
# ----------------------------
def process_single_file(args):
    input_path, output_root = args
    path = Path(input_path)
    if not path.exists():
        return 0

    # 自动检测 gzip
    try:
        with open(path, 'rb') as f:
            is_gzipped = f.read(2) == b'\x1f\x8b'
    except:
        return 0

    try:
        if is_gzipped:
            with gzip.open(path, 'rt', encoding='utf-8') as f:
                lines = f.readlines()
        else:
            with open(path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
    except Exception:
        return 0

    game_name = path.stem.replace('.mjson', '').replace('.gz', '')
    game_output_dir = Path(output_root) / f"game_{game_name}"
    game_output_dir.mkdir(parents=True, exist_ok=True)

    # 游戏状态初始化
    hands = None
    discards = None
    tsumogiri_flags = None
    num_chi = None
    num_pon = None
    num_kan = None
    is_riichi = None
    kyoku_count = 0
    dora_indicators = []
    sample_count = 0

    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            msg = json.loads(line)
        except:
            continue

        msg_type = msg.get("type")
        if not msg_type:
            continue

        if msg_type == "start_kyoku":
            kyoku_count += 1
            hands = [[pai_to_id(p) for p in msg["tehais"][i]] for i in range(4)]
            discards = [[] for _ in range(4)]
            tsumogiri_flags = [[] for _ in range(4)]
            num_chi = [0] * 4
            num_pon = [0] * 4
            num_kan = [0] * 4
            is_riichi = [False] * 4
            meld_positions = [[] for _ in range(4)]  # ← 新增这一行！
            dora_marker = msg.get("dora_marker")
            dora_indicators = [dora_marker] if dora_marker else []

        elif msg_type == "tsumo":
            actor = msg.get("actor", -1)
            pai_str = msg.get("pai")
            if actor < 0 or pai_str is None:
                continue
            tile = pai_to_id(pai_str)
            if tile != -1:
                hands[actor].append(tile)

        elif msg_type in ("chi", "pon", "daiminkan", "ankan", "kakan"):
            actor = msg.get("actor", -1)
            if actor < 0:
                continue

            meld_positions[actor].append(len(discards[actor]))

            if msg_type == "chi":
                num_chi[actor] += 1
            elif msg_type == "pon":
                num_pon[actor] += 1
            elif msg_type in ("daiminkan", "ankan", "kakan"):
                num_kan[actor] += 1

            consumed_pais = msg.get("consumed")
            if consumed_pais:
                for p in consumed_pais:
                    tid = pai_to_id(p)
                    if tid != -1 and tid in hands[actor]:
                        hands[actor].remove(tid)
            elif msg_type == "ankan":
                pai_str = msg.get("pai")
                if pai_str:
                    tid = pai_to_id(pai_str)
                    if tid != -1:
                        for _ in range(4):
                            if tid in hands[actor]:
                                hands[actor].remove(tid)

        elif msg_type == "reach_accepted":
            actor = msg.get("actor", -1)
            if 0 <= actor < 4:
                is_riichi[actor] = True

        elif msg_type == "dahai":
            actor = msg.get("actor", -1)
            pai_str = msg.get("pai")
            if actor < 0 or pai_str is None:
                continue
            if is_riichi[actor]:
                continue  # 跳过立直后打牌

            tile = pai_to_id(pai_str)
            if tile == -1:
                continue

            is_tsumogiri = msg.get("tsumogiri", False)
            discards[actor].append(tile)
            tsumogiri_flags[actor].append(1 if is_tsumogiri else 0)

            # 从手牌中移除打出的牌
            if tile in hands[actor]:
                hands[actor].remove(tile)

            # 构造样本
            T = len(discards[actor])
            current_hand_vec = hand_to_vec(hands[actor])
            true_shanten = calculate_shanten_with_melds(hands[actor], [num_chi[actor], num_pon[actor], num_kan[actor]])

            # 计算 dora_onehot (37维)
            dora_onehot = [0] * 37
            for ind in dora_indicators:
                ind_id = pai_to_id(ind)
                if ind_id < 0:
                    continue
                for dora_tile in get_dora_tiles(ind_id):
                    if 0 <= dora_tile < 37:
                        dora_onehot[dora_tile] = 1

            sample = {
                "game_id": game_name,
                "kyoku": kyoku_count,
                "player": actor,
                "turn_index": T,
                "hand_vec": current_hand_vec,
                "discard_seq": discards[actor].copy(),
                "tsumogiri_flags": tsumogiri_flags[actor].copy(),
                "meld_counts": [num_chi[actor], num_pon[actor], num_kan[actor]],
                "meld_positions": meld_positions[actor].copy(),
                "dora_indicators": [pai_to_id(d) for d in dora_indicators if d],  # 保留用于 debug
                "dora_onehot": dora_onehot,  # ← 新增！
                "true_shanten": true_shanten
            }

            # 写入独立 JSON 文件
            fname = f"p{actor}_kyoku{kyoku_count}_t{T}.json"
            (game_output_dir / fname).write_text(
                json.dumps(sample, ensure_ascii=False),
                encoding='utf-8'
            )
            sample_count += 1

        # 忽略其他消息类型（hora, ryukyoku, dora 等）

    return sample_count


# ----------------------------
# 主函数
# ----------------------------
def main():
    input_root = Path(INPUT_ROOT_PATH)
    output_base = Path(OUTPUT_BASE_PATH)

    # 从输入路径中提取年份
    year_match = re.search(r'(\d{4})', str(input_root))
    year = year_match.group(1) if year_match else "unknown"

    # 构建带年份和文件数量的输出目录名
    n_files = NUM_FILES_TO_PROCESS if NUM_FILES_TO_PROCESS is not None else "all"
    output_dir = output_base / f"{year}_{n_files}paipu_json"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"📁 输入目录: {input_root}")
    print(f"🎯 提取前 {n_files} 个牌谱文件")
    print(f"📂 输出目录: {output_dir}")

    # 获取所有 .mjson 和 .mjson.gz 文件
    all_files = sorted(
        list(input_root.glob("*.mjson")) +
        list(input_root.glob("*.mjson.gz"))
    )
    if NUM_FILES_TO_PROCESS is not None:
        all_files = all_files[:NUM_FILES_TO_PROCESS]

    if not all_files:
        print("❌ 未找到任何 .mjson 或 .mjson.gz 文件！")
        return

    # 准备任务：每个任务是 (文件路径, 输出目录)
    tasks = [(str(f), str(output_dir)) for f in all_files]
    max_workers = MAX_WORKERS or min(cpu_count() - 1, len(tasks), 8)

    total_samples = 0
    start_time = time.time()

    with Pool(processes=max_workers) as pool:
        with tqdm(total=len(tasks), desc="Processing games") as pbar:
            for count in pool.imap_unordered(process_single_file, tasks):
                total_samples += count
                pbar.update(1)

    elapsed = time.time() - start_time
    print(f"\n🎉 完成！共生成 {total_samples:,} 个样本文件。")
    print(f"📁 输出目录: {output_dir}")


if __name__ == "__main__":
    main()