# -*- coding: utf-8 -*-
import re
import os
import csv
import time
import json
import requests
import glob
import traceback

# ==========================================
# 配置区域
# ==========================================
# 在此处填入您购买了配额的百度 AK
BAIDU_AK = "zwCZdF4xg9oU1FywO0WQH6mivt9MPLVs"  # 您的 AK


# ==========================================
# 第一部分：官方 API 转换类 (已优化)
# ==========================================

class BaiduCoordConverter:
    """
    使用百度官方 API 进行坐标转换
    优化策略：使用 geoconv/v1 接口直接从 WGS84 转 百度墨卡托 (to=6)
    优势：精度完美，且相比两步转换节省一半配额。
    """

    def __init__(self, ak):
        self.ak = ak
        # 官方文档：http://api.map.baidu.com/geoconv/v1/
        self.api_url = "http://api.map.baidu.com/geoconv/v1/"

    def wgs84_to_mc(self, lng, lat):
        """
        输入: WGS84 经纬度 (GPS原始坐标)
        输出: 百度墨卡托坐标 (x, y) 整数
        """
        params = {
            "coords": f"{lng},{lat}",
            "from": 1,  # 1 = WGS84
            "to": 6,  # 6 = 百度墨卡托 (直接米制)
            "ak": self.ak,
            "output": "json"
        }

        try:
            # 这里的 timeout 稍微设长一点，防止网络波动
            response = requests.get(self.api_url, params=params, timeout=5)

            # 检查 HTTP 状态码
            if response.status_code != 200:
                print(f"      API HTTP错误: {response.status_code}")
                return None, None

            data = response.json()

            # status=0 代表成功
            if data.get("status") == 0:
                result = data["result"][0]
                # 百度墨卡托通常取整数即可
                return int(result["x"]), int(result["y"])
            else:
                # status=210 代表 IP 校验失败，240 代表配额用尽 等
                print(f"      API 业务错误码: {data.get('status')} - {data.get('message')}")
                return None, None

        except Exception as e:
            print(f"      API 请求异常: {e}")
            return None, None


# ==========================================
# 第二部分：工具函数 (保持不变)
# ==========================================

def write_csv(filepath, data, head=None):
    if head and not os.path.exists(filepath):
        data = [head] + data
    elif head:
        data = [head] + data
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, mode='w', encoding='UTF-8-sig', newline='') as f:
        writer = csv.writer(f)
        for i in data:
            writer.writerow(i)


def read_csv(filepath):
    data = []
    if os.path.exists(filepath):
        with open(filepath, mode='r', encoding='utf-8') as f:
            lines = csv.reader(f)
            for line in lines:
                data.append(line)
        return data
    else:
        return []


def grab_img_baidu(_url):
    headers = {
        "Referer": "https://map.baidu.com/",
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/90.0.4430.212 Safari/537.36"
    }
    try:
        response = requests.get(_url, headers=headers, timeout=10)
        if response.status_code == 200 and response.headers.get('Content-Type') == 'image/jpeg':
            return response.content
        return None
    except Exception:
        return None


def openUrl(_url):
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/90.0.4430.212 Safari/537.36"
    }
    try:
        response = requests.get(_url, headers=headers, timeout=10)
        if response.status_code == 200:
            return response.content
        return None
    except Exception:
        return None


def getPanoId(_mc_x, _mc_y):
    """
    通过墨卡托坐标查询 Panoid
    """
    url = f"https://mapsv0.bdimg.com/?&qt=qsdata&x={_mc_x}&y={_mc_y}&l=17&action=0&mode=day&t=1530956939770"
    response = openUrl(url)
    if response is None:
        return None
    try:
        response_str = response.decode("utf8")
        # 使用 JSON 解析比正则更稳定
        data = json.loads(response_str)
        if 'content' in data and 'id' in data['content']:
            return data['content']['id']
    except:
        pass
    return None


# ==========================================
# 主程序入口
# ==========================================

if __name__ == "__main__":
    # 初始化 API 转换器
    if not BAIDU_AK:
        print("❌ 请先在代码顶部填入您的百度 AK！")
        exit()

    converter = BaiduCoordConverter(BAIDU_AK)
    print("✅ API 转换器初始化成功，已启用官方 AK 模式。")

    current_dir = os.path.dirname(os.path.abspath(__file__))
    input_points_dir = os.path.join(current_dir, r'../路网提取/output_road_network/road_points')
    input_points_dir = os.path.normpath(input_points_dir)

    output_root_name = 'image_dir'
    base_output_dir = os.path.join(current_dir, output_root_name)
    base_error_dir = os.path.join(base_output_dir, 'error_points')

    csv_pattern = os.path.join(input_points_dir, "point_*.csv")
    csv_files = glob.glob(csv_pattern)

    if not csv_files:
        print(f"❌ 未找到任何CSV文件，请检查路径: {input_points_dir}")
        exit()

    print(f"📂 发现 {len(csv_files)} 个任务文件，准备开始处理...\n")
    os.makedirs(base_error_dir, exist_ok=True)

    for index, csv_path in enumerate(csv_files):
        file_name = os.path.basename(csv_path)
        print(f"[{index + 1}/{len(csv_files)}] 正在读取文件: {file_name}")

        try:
            parts = file_name.split('_')
            street_name = parts[1] if len(parts) >= 2 else file_name.replace('.csv', '')
        except:
            street_name = "unknown_street"

        current_img_dir = os.path.join(base_output_dir, f"{street_name}_images")
        os.makedirs(current_img_dir, exist_ok=True)
        current_error_csv = os.path.join(base_error_dir, f"{street_name}_error.csv")

        data = read_csv(csv_path)
        if not data:
            continue

        header = data[0]
        data_rows = data[1:]

        filenames_exist = set()
        if os.path.exists(current_img_dir):
            for f in os.listdir(current_img_dir):
                if f.endswith('.png'):
                    filenames_exist.add(f)

        error_img = []
        headings = ['0', '90', '180', '270']

        # 缓存：避免同一个坐标点重复调用 API 扣费
        coord_cache = {}

        print(f"   >>> 开始处理 {len(data_rows)} 个点...")

        for i, row in enumerate(data_rows):
            if (i + 1) % 20 == 0:
                print(f'      进度: {i + 1}/{len(data_rows)}')

            try:
                # 【请确认CSV列是否正确】 假设：ID, Area, Lng, Lat
                longitude = row[2]
                latitude = row[3]
                Area = row[1]
                ID = row[0]
            except IndexError:
                continue

            # 检查是否已存在
            all_exist = True
            for heading in headings:
                img_name = f"{ID}_{Area}_{longitude}_{latitude}_{heading}_0.png"
                if img_name not in filenames_exist:
                    all_exist = False
                    break
            if all_exist:
                continue

            # -------------------------------------------------
            # 核心修改：使用 AK 进行坐标转换
            # -------------------------------------------------
            coord_key = f"{longitude}_{latitude}"

            if coord_key in coord_cache:
                mc_x, mc_y = coord_cache[coord_key]
            else:
                # 调用官方 API
                mc_x, mc_y = converter.wgs84_to_mc(longitude, latitude)

                # 如果转换成功，存入缓存
                if mc_x is not None:
                    coord_cache[coord_key] = (mc_x, mc_y)
                    # 稍微 sleep 一下，虽然官方并发高，但稳一点更好
                    time.sleep(0.05)
                else:
                    # 转换失败（可能是坐标非法或配额耗尽）
                    error_img.append(row + ['API_Convert_Fail'])
                    continue

            # -------------------------------------------------
            # 后续逻辑保持不变：拿 Panoid -> 下载图片
            # -------------------------------------------------
            svid = getPanoId(mc_x, mc_y)
            if not svid:
                # print(f"      ❌ 无街景: {ID}")
                error_img.append(row + ['No_SV_ID'])
                continue

            for heading in headings:
                save_name = f"{ID}_{Area}_{longitude}_{latitude}_{heading}_0.png"
                save_file_abs = os.path.join(current_img_dir, save_name)

                if save_name in filenames_exist:
                    continue

                url = f'https://mapsv0.bdimg.com/?qt=pr3d&fovy=90&quality=100&panoid={svid}&heading={heading}&pitch=0&width=480&height=320'
                img_data = grab_img_baidu(url)

                if img_data:
                    with open(save_file_abs, "wb") as f:
                        f.write(img_data)
                    filenames_exist.add(save_name)
                    # print(f"      已保存: {save_name}")
                else:
                    error_img.append(row + [heading])

                # 下载间隔
                time.sleep(0.2)

        if error_img:
            write_csv(current_error_csv, error_img, header + ['error_info'])
            print(f"   ⚠️ {street_name} 完成，有 {len(error_img)} 个异常。")
        else:
            print(f"   ✅ {street_name} 全部成功。")

    print("\n🎉 所有任务处理完毕！")
