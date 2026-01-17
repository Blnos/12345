# -*- coding: utf-8 -*-
import os
import redo
import json
import time
import glob
import requests
import math

# ================= 配置区域 =================
# 1. 填入您的百度 AK
BAIDU_AK = "UmJFvBxkmtPryMVPcRvxlAN5ng2DXHCy"

# 2. 输入文件路径 (您之前的 CSV 文件夹)
INPUT_DIR = r'output_road_network/road_points_wgs84'

# 3. 输出文件路径 (转换好后存到哪里)
OUTPUT_DIR = r'output_road_network/road_points_mc'


# ===========================================

class BatchConverter:
    def __init__(self, ak):
        self.ak = ak
        self.api_url = "http://api.map.baidu.com/geoconv/v1/"

    def convert_chunk(self, points_list):
        """
        批量转换函数
        points_list: list of (lng, lat) tuples
        return: list of {'x': int, 'y': int}
        """
        # 1. 拼接坐标字符串，格式: x1,y1;x2,y2;...
        # 注意: 百度要求经纬度最多保留6位小数，避免超长
        coords_str = ";".join([f"{float(p[0]):.6f},{float(p[1]):.6f}" for p in points_list])

        params = {
            "coords": coords_str,
            "from": 1,  # 1 = WGS84 (GPS设备采集)
            "to": 6,  # 6 = 百度墨卡托 (直接用于街景)
            "ak": self.ak,
            "output": "json"
        }

        try:
            resp = requests.get(self.api_url, params=params, timeout=10)
            data = resp.json()

            if data['status'] == 0:
                # 转换成功，提取结果
                results = []
                for item in data['result']:
                    results.append({
                        'x': float(item['x']),  # 转为整数
                        'y': float(item['y'])
                    })
                return results
            else:
                print(f"    ⚠️ API报错 (Code {data['status']}): {data.get('message')}")
                # 如果整批失败，返回空列表对应的 None
                return [None] * len(points_list)

        except Exception as e:
            print(f"    ⚠️ 请求异常: {e}")
            return [None] * len(points_list)


def process_files():
    if BAIDU_AK == "您的AK":
        print("❌ 错误：请先在代码顶部填入您的 BAIDU_AK")
        return

    # 准备路径
    input_path = os.path.normpath(INPUT_DIR)
    output_path = os.path.normpath(OUTPUT_DIR)
    os.makedirs(output_path, exist_ok=True)

    csv_files = glob.glob(os.path.join(input_path, "*.csv"))
    if not csv_files:
        print(f"❌ 未找到CSV文件，请检查路径: {input_path}")
        return

    print(f"🚀 开始处理 {len(csv_files)} 个文件...")
    print(f"📂 结果将保存至: {output_path}\n")

    total_converted = 0

    for idx, file_path in enumerate(csv_files):
        filename = os.path.basename(file_path)
        print(f"[{idx + 1}/{len(csv_files)}] 处理文件: {filename}")

        # 读取原始 CSV
        rows = []
        with open(file_path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            rows = list(reader)

        if not rows:
            continue

        # 分离表头和数据
        header = rows[0]
        data = rows[1:]

        # 新表头：追加两列
        new_header = header + ['mc_x', 'mc_y']

        # 准备批量处理
        # 假设 CSV 结构: [ID, Area, Lng, Lat, ...]
        # Lng在索引2, Lat在索引3

        # 将数据分块，每100条一组 (百度API上限)
        batch_size = 100
        new_data_rows = []

        # 循环处理每一批
        for i in range(0, len(data), batch_size):
            chunk = data[i: i + batch_size]

            # 提取这一批的经纬度
            batch_coords = []
            valid_indices = []  # 记录哪些行是有效坐标

            for row_idx, row in enumerate(chunk):
                try:
                    lng = float(row[2])
                    lat = float(row[3])
                    batch_coords.append((lng, lat))
                    valid_indices.append(row_idx)
                except (ValueError, IndexError):
                    # 如果坐标无效，填None占位
                    pass

            if not batch_coords:
                # 如果这一批全是无效数据，直接填空
                for row in chunk:
                    new_data_rows.append(row + ['', ''])
                continue

            # === 调用 API (核心) ===
            converter = BatchConverter(BAIDU_AK)
            results = converter.convert_chunk(batch_coords)

            # 将结果回填到 chunk 中
            result_ptr = 0
            for row_idx, row in enumerate(chunk):
                if row_idx in valid_indices:
                    res = results[result_ptr]
                    if res:
                        # 追加转换后的坐标
                        new_row = row + [str(res['x']), str(res['y'])]
                    else:
                        # 转换失败
                        new_row = row + ['', '']
                    result_ptr += 1
                else:
                    # 原始坐标无效
                    new_row = row + ['', '']

                new_data_rows.append(new_row)

            # 稍微延时，避免QPS过高（虽然批量处理通常很快）
            time.sleep(0.2)

        # 写入新文件
        save_path = os.path.join(output_path, filename)
        with open(save_path, 'w', encoding='utf-8-sig', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(new_header)
            writer.writerows(new_data_rows)

        total_converted += len(new_data_rows)
        print(f"    ✅ 已保存 {len(new_data_rows)} 条数据")

    print(f"\n🎉 全部完成！共处理 {total_converted} 个点。")
    print("👉 下一步：请使用这些生成好的新 CSV 运行爬虫脚本（无需再进行坐标转换）。")


if __name__ == "__main__":
    process_files()
