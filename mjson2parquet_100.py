#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
批量 mjson → Parquet 预处理脚本 (最终修正版：副露逻辑 + 多进程)
基于天凤 mjson 标准格式验证
"""

import json
import os
import gzip
from pathlib import Path
from typing import List, Dict, Tuple
import pyarrow as pa
import pyarrow.parquet as pq
from mahjong.shanten import Shanten
from multiprocessing import Pool, cpu_count
import time

# ==================================================
# 配置
# ==================================================

INPUT_DIR = '/home/yyt/Desktop/shanten/archive/2024/2024'
OUTPUT_DIR = '/home/yyt/Desktop/shanten/data_preprocessing/test_10000mjson'
NUM_FILES = 10000
NUM_WORKERS = 16  # 根据你的 CPU 核心数调整


# ==================================================
# 工具函数
# ==================================================

def is_gzip_file(filepath: str) -> bool:
    try:
        with open(filepath, 'rb') as f:
            return f.read(2) == b'\x1f\x8b'
    except:
        return False


def parse_tile(tile_str: str) -> int:
    """天凤字符串 → 37维ID"""
    try:
        if not tile_str:
            return -1
        if len(tile_str) == 1:
            return {'E': 30, 'S': 31, 'W': 32, 'N': 33, 'P': 34, 'F': 35, 'C': 36}.get(tile_str, -1)

        is_aka = tile_str.endswith('r')
        tile_clean = tile_str[:-1] if is_aka else tile_str

        if len(tile_clean) < 2:
            return -1

        num = int(tile_clean[:-1])
        suit = tile_clean[-1]

        if suit == 'm':
            if is_aka and num == 5: return 0
            return num if 1 <= num <= 9 else -1
        elif suit == 'p':
            if is_aka and num == 5: return 10
            return num + 10 if 1 <= num <= 9 else -1
        elif suit == 's':
            if is_aka and num == 5: return 20
            return num + 20 if 1 <= num <= 9 else -1
        return -1
    except:
        return -1


def to_34dim(tile_id: int) -> int:
    """37维ID → 34维索引 (向听数计算用)"""
    if tile_id == 0:   return 4
    if tile_id == 10:  return 13
    if tile_id == 20:  return 22
    if 1 <= tile_id <= 9:    return tile_id - 1
    if 11 <= tile_id <= 19:  return tile_id - 11 + 9
    if 21 <= tile_id <= 29:  return tile_id - 21 + 18
    if 30 <= tile_id <= 36:  return tile_id - 30 + 27
    return -1


def calculate_shanten(hand_tiles: List[int], melds: List[List[int]]) -> int:
    """计算向听数 (带容错)"""
    # 强制修正手牌数量，防止 mahjong 库报错
    if len(hand_tiles) > 14:
        hand_tiles = hand_tiles[:14]
    if len(hand_tiles) < 1:
        return 6

    hand_34 = [0] * 34
    for tile_id in hand_tiles:
        idx = to_34dim(tile_id)
        if idx >= 0:
            hand_34[idx] += 1

    melds_34 = []
    for meld in melds:
        meld_34 = [to_34dim(t) for t in meld if to_34dim(t) >= 0]
        if meld_34:
            melds_34.append(meld_34)

    try:
        shanten_val = Shanten().calculate_shanten(hand_34, melds_34)
        return max(0, min(6, shanten_val))
    except Exception as e:
        # 如果计算失败，返回最差向听数
        return 6


def split_kyokus(events: List[Dict]) -> List[List[Dict]]:
    """按小局分割事件流"""
    kyokus = []
    current_kyoku = []
    in_kyoku = False
    for event in events:
        event_type = event.get('type', '')
        if event_type == 'start_kyoku':
            in_kyoku = True
            current_kyoku = [event]
        elif event_type == 'end_kyoku':
            if in_kyoku:
                current_kyoku.append(event)
                kyokus.append(current_kyoku)
            in_kyoku = False
            current_kyoku = []
        elif in_kyoku:
            current_kyoku.append(event)
    return kyokus


def process_kyoku(kyoku_events: List[Dict], game_id: str = "", kyoku_num: int = 0) -> List[Dict]:
    """处理一个小局，生成样本列表"""
    samples = []
    players = {
        0: {'hand': [], 'melds': [], 'reach': False, 'discard_seq': []},
        1: {'hand': [], 'melds': [], 'reach': False, 'discard_seq': []},
        2: {'hand': [], 'melds': [], 'reach': False, 'discard_seq': []},
        3: {'hand': [], 'melds': [], 'reach': False, 'discard_seq': []}
    }

    # 用于调试计数
    turn_counter = {0: 0, 1: 0, 2: 0, 3: 0}

    for event in kyoku_events:
        event_type = event.get('type', '')

        # === 小局开始：初始化配牌 ===
        if event_type == 'start_kyoku':
            tehais = event.get('tehais', [[], [], [], []])
            for i in range(4):
                players[i]['hand'] = [parse_tile(t) for t in tehais[i] if t]
            continue

        # === 摸牌：加入手牌 ===
        if event_type == 'tsumo':
            actor = event.get('actor', -1)
            pai = event.get('pai', '')
            if actor >= 0 and pai:
                tile_id = parse_tile(pai)
                if tile_id >= 0:
                    players[actor]['hand'].append(tile_id)
            continue

        # === 打牌：生成样本 ===
        if event_type == 'dahai':
            actor = event.get('actor', -1)
            pai = event.get('pai', '')

            if actor < 0 or not pai:
                continue

            tile_id = parse_tile(pai)
            if tile_id < 0:
                continue

            turn_counter[actor] += 1

            # 立直后不生成样本
            if players[actor]['reach']:
                if tile_id in players[actor]['hand']:
                    players[actor]['hand'].remove(tile_id)
                players[actor]['discard_seq'].append(tile_id)
                continue

            # ★★★ 计算向听数 (此时副露已更新，手牌已减除) ★★★
            shanten = calculate_shanten(
                players[actor]['hand'].copy(),
                players[actor]['melds']
            )

            sample = {
                'discard_seq': players[actor]['discard_seq'].copy(),
                'meld_count': len(players[actor]['melds']),
                'true_shanten': shanten
            }
            samples.append(sample)

            # 更新状态：打出手牌
            if tile_id in players[actor]['hand']:
                players[actor]['hand'].remove(tile_id)
            players[actor]['discard_seq'].append(tile_id)
            continue

        # === 副露处理 (核心修正) ===
        if event_type in ['chi', 'pon', 'kan', 'daiminkan', 'kakan', 'ankan']:
            actor = event.get('actor', -1)
            consumed = event.get('consumed', [])

            if actor >= 0 and consumed:
                # 1. 解析 consumed 里的牌
                meld_tiles = [parse_tile(t) for t in consumed if t]

                # 2. 从手牌移除 consumed 里的牌
                current_hand = players[actor]['hand']
                removed_count = 0
                for tile in meld_tiles:
                    if tile in current_hand:
                        current_hand.remove(tile)
                        removed_count += 1
                    else:
                        # 理论上 consumed 里的牌都在手牌里，除非数据异常
                        pass
                players[actor]['hand'] = current_hand

                # 3. 记录完整副露 (用于向听数计算)
                full_meld = meld_tiles.copy()

                # 特殊处理：大明杠需要把别人打的那张牌 (pai) 补进去，凑成4张
                if event_type == 'daiminkan':
                    pai = event.get('pai')
                    if pai:
                        full_meld.append(parse_tile(pai))

                players[actor]['melds'].append(full_meld)

                # 调试输出 (可选)
                # print(f"[{game_id}] Kyoku:{kyoku_num} Player:{actor} Type:{event_type} Removed:{removed_count} HandSize:{len(current_hand)}")

            continue

        # === 立直：标记 ===
        if event_type == 'reach':
            actor = event.get('actor', -1)
            if actor >= 0:
                players[actor]['reach'] = True
            continue

        # === 小局结束 ===
        if event_type in ['hora', 'ryukyoku']:
            break

    return samples


def save_to_parquet(samples: List[Dict], output_path: str):
    """保存为 Parquet 文件"""
    if not samples:
        return
    table = pa.table({
        'discard_seq': [s['discard_seq'] for s in samples],
        'meld_count': [s['meld_count'] for s in samples],
        'true_shanten': [s['true_shanten'] for s in samples]
    })
    pq.write_table(table, output_path, compression='snappy')


def process_single_mjson(args: Tuple[str, str]) -> Tuple[str, int, bool, str]:
    """处理单个 mjson 文件 (多进程 worker)"""
    input_path, output_dir = args
    error_msg = ""

    try:
        events = []
        if is_gzip_file(input_path):
            with gzip.open(input_path, 'rt', encoding='utf-8', errors='ignore') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            events.append(json.loads(line))
                        except:
                            continue
        else:
            with open(input_path, 'r', encoding='utf-8', errors='ignore') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            events.append(json.loads(line))
                        except:
                            continue

        if not events:
            return (Path(input_path).name, 0, True, "Empty")

        game_id = Path(input_path).stem
        if game_id.endswith('.mjson'):
            game_id = game_id[:-6]

        kyokus = split_kyokus(events)
        total_samples = 0

        for kyoku_idx, kyoku_events in enumerate(kyokus):
            samples = process_kyoku(kyoku_events, game_id=game_id, kyoku_num=kyoku_idx + 1)
            if samples:
                output_path = os.path.join(output_dir, f'{game_id}_kyoku_{kyoku_idx + 1}.parquet')
                save_to_parquet(samples, output_path)
                total_samples += len(samples)

        return (Path(input_path).name, total_samples, True, "OK")

    except Exception as e:
        error_msg = str(e)[:100]
        return (Path(input_path).name, 0, False, error_msg)


def batch_process_parallel(input_dir: str, output_dir: str, num_files: int = 100, num_workers: int = 16):
    print(f"\n{'=' * 70}")
    print(f"批量处理 mjson → Parquet (最终修正版)")
    print(f"{'=' * 70}")
    print(f"输入目录：{input_dir}")
    print(f"输出目录：{output_dir}")
    print(f"处理数量：前 {num_files} 个文件")
    print(f"进程数量：{num_workers}")
    print(f"{'=' * 70}\n")

    all_files = []
    for f in os.listdir(input_dir):
        if f.endswith('.mjson') or f.endswith('.mjson.gz'):
            all_files.append(f)
    all_files.sort()
    selected_files = all_files[:num_files]

    print(f"目录中共有 {len(all_files)} 个文件")
    print(f"本次处理 {len(selected_files)} 个文件\n")

    os.makedirs(output_dir, exist_ok=True)
    tasks = [(os.path.join(input_dir, f), output_dir) for f in selected_files]

    start_time = time.time()

    with Pool(processes=num_workers) as pool:
        results = list(pool.imap(process_single_mjson, tasks, chunksize=5))

    elapsed_time = time.time() - start_time

    total_samples = 0
    success_count = 0
    fail_count = 0
    failed_files = []

    for filename, samples, success, msg in results:
        if success:
            success_count += 1
            total_samples += samples
        else:
            fail_count += 1
            failed_files.append((filename, msg))

    print(f"\n{'=' * 70}")
    print(f"处理完成！")
    print(f"{'=' * 70}")
    print(f"处理时间：{elapsed_time:.2f} 秒 ({elapsed_time / 60:.2f} 分钟)")
    print(f"处理速度：{len(selected_files) / elapsed_time:.2f} 文件/秒")
    print(f"成功：{success_count} 文件 | 失败：{fail_count} 文件")
    print(f"总样本数：{total_samples:,}")

    if failed_files:
        print(f"\n⚠️ 失败文件:")
        for f, err in failed_files[:5]:
            print(f"  - {f}: {err}")

    print(f"{'=' * 70}\n")

    with open(os.path.join(output_dir, 'processed_files.txt'), 'w') as f:
        for filename in selected_files:
            f.write(filename + '\n')


if __name__ == '__main__':
    print(f"系统可用 CPU 核心数：{cpu_count()}")
    batch_process_parallel(INPUT_DIR, OUTPUT_DIR, NUM_FILES, NUM_WORKERS)