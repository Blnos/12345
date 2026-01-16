# -*- coding: utf-8 -*-
import re
import os
import csv
import math
import time
import json
import requests
import glob  # 用于查找所有文件
import traceback

# ==========================================
# 第一部分：本地坐标转换算法 (保持不变)
# ==========================================

x_pi = 3.14159265358979324 * 3000.0 / 180.0
pi = 3.1415926535897932384626
a = 6378245.0
ee = 0.00669342162296594323


def transform_lat(lng, lat):
    ret = -100.0 + 2.0 * lng + 3.0 * lat + 0.2 * lat * lat + 0.1 * lng * lat + 0.2 * math.sqrt(math.fabs(lng))
    ret += (20.0 * math.sin(6.0 * lng * pi) + 20.0 * math.sin(2.0 * lng * pi)) * 2.0 / 3.0
    ret += (20.0 * math.sin(lat * pi) + 40.0 * math.sin(lat / 3.0 * pi)) * 2.0 / 3.0
    ret += (160.0 * math.sin(lat / 12.0 * pi) + 320 * math.sin(lat * pi / 30.0)) * 2.0 / 3.0
    return ret


def transform_lng(lng, lat):
    ret = 300.0 + lng + 2.0 * lat + 0.1 * lng * lng + 0.1 * lng * lat + 0.1 * math.sqrt(math.fabs(lng))
    ret += (20.0 * math.sin(6.0 * lng * pi) + 20.0 * math.sin(2.0 * lng * pi)) * 2.0 / 3.0
    ret += (20.0 * math.sin(lng * pi) + 40.0 * math.sin(lng / 3.0 * pi)) * 2.0 / 3.0
    ret += (150.0 * math.sin(lng / 12.0 * pi) + 300.0 * math.sin(lng / 30.0 * pi)) * 2.0 / 3.0
    return ret


def wgs84_to_gcj02(lng, lat):
    if lng < 72.004 or lng > 137.8347 or lat < 0.8293 or lat > 55.8271:
        return lng, lat
    dlat = transform_lat(lng - 105.0, lat - 35.0)
    dlng = transform_lng(lng - 105.0, lat - 35.0)
    radlat = lat / 180.0 * pi
    magic = math.sin(radlat)
    magic = 1 - ee * magic * magic
    sqrtmagic = math.sqrt(magic)
    dlat = (dlat * 180.0) / ((a * (1 - ee)) / (magic * sqrtmagic) * pi)
    dlng = (dlng * 180.0) / (a / sqrtmagic * math.cos(radlat) * pi)
    mglat = lat + dlat
    mglng = lng + dlng
    return mglng, mglat


def gcj02_to_bd09(lng, lat):
    z = math.sqrt(lng * lng + lat * lat) + 0.00002 * math.sin(lat * x_pi)
    theta = math.atan2(lat, lng) + 0.000003 * math.cos(lng * x_pi)
    bd_lng = z * math.cos(theta) + 0.0065
    bd_lat = z * math.sin(theta) + 0.006
    return bd_lng, bd_lat


def bd09_to_mc(lng, lat):
    x = lng * 20037508.34 / 180
    try:
        y = math.log(math.tan((90 + lat) * pi / 360)) / (pi / 180)
    except ValueError:
        y = 0
    y = y * 20037508.34 / 180
    return x, y


def wgs2bd09mc_no_ak(wgs_x, wgs_y):
    try:
        lng = float(wgs_x)
        lat = float(wgs_y)
        g_lng, g_lat = wgs84_to_gcj02(lng, lat)
        b_lng, b_lat = gcj02_to_bd09(g_lng, g_lat)
        mc_x, mc_y = bd09_to_mc(b_lng, b_lat)
        return mc_x, mc_y
    except Exception as e:
        return 0, 0


# ==========================================
# 第二部分：工具函数 (保持不变)
# ==========================================

def write_csv(filepath, data, head=None):
    if head and not os.path.exists(filepath):
        # 只有文件不存在时才写入表头，防止追加模式下重复写表头(虽然这里是'w'模式)
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


def grab_img_baidu(_url, _headers=None):
    if _headers is None:
        headers = {
            "sec-ch-ua": '" Not A;Brand";v="99", "Chromium";v="90", "Google Chrome";v="90"',
            "Referer": "https://map.baidu.com/",
            "sec-ch-ua-mobile": "?0",
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/90.0.4430.212 Safari/537.36"
        }
    else:
        headers = _headers
    try:
        response = requests.get(_url, headers=headers, timeout=10)
        if response.status_code == 200 and response.headers.get('Content-Type') == 'image/jpeg':
            return response.content
        else:
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
        else:
            return None
    except Exception:
        return None


def getPanoId(_lng, _lat):
    url = "https://mapsv0.bdimg.com/?&qt=qsdata&x=%s&y=%s&l=17.031000000000002&action=0&mode=day&t=1530956939770" % (
        str(_lng), str(_lat))
    response = openUrl(url)
    if response is None:
        return None
    try:
        response_str = response.decode("utf8")
        reg = r'"id":"(.+?)",'
        pat = re.compile(reg)
        result = re.findall(pat, response_str)
        if result:
            return result[0]
        else:
            return None
    except:
        return None


# ==========================================
# 主程序入口 (修改版：支持批量处理)
# ==========================================

if __name__ == "__main__":
    # 0. 基础路径配置
    current_dir = os.path.dirname(os.path.abspath(__file__))

    # 输入目录：路网数据的 point 文件夹
    input_points_dir = os.path.join(current_dir, r'../路网提取/output_road_network/road_points')
    input_points_dir = os.path.normpath(input_points_dir)

    # 输出总目录：image_dir
    output_root_name = 'image_dir'
    base_output_dir = os.path.join(current_dir, output_root_name)

    # 错误日志总目录：image_dir/error_points
    base_error_dir = os.path.join(base_output_dir, 'error_points')

    # 1. 查找所有CSV文件
    # 使用 glob 查找该文件夹下所有以 "point_" 开头的 csv 文件
    csv_pattern = os.path.join(input_points_dir, "point_*.csv")
    csv_files = glob.glob(csv_pattern)

    if not csv_files:
        print(f"❌ 未找到任何CSV文件，请检查路径: {input_points_dir}")
        exit()

    print(f"📂 发现 {len(csv_files)} 个任务文件，准备开始处理...\n")

    # 创建错误日志的总文件夹
    os.makedirs(base_error_dir, exist_ok=True)

    # ----------------------------------------------------
    # 开始遍历每个 CSV 文件
    # ----------------------------------------------------
    for index, csv_path in enumerate(csv_files):
        file_name = os.path.basename(csv_path)
        print(f"[{index + 1}/{len(csv_files)}] 正在读取文件: {file_name}")

        # ==============================================
        # 步骤 A: 解析文件名，提取街道名称
        # 例子: point_万柳地区_110108023000.csv -> 提取 "万柳地区"
        # ==============================================
        try:
            # 用 "_" 分割，取中间部分
            parts = file_name.split('_')
            if len(parts) >= 2:
                street_name = parts[1]  # 获取 "万柳地区"
            else:
                # 如果文件名格式不对，就用整个文件名去掉后缀
                street_name = file_name.replace('.csv', '')
        except:
            street_name = "unknown_street"

        # ==============================================
        # 步骤 B: 动态配置当前任务的 输出路径
        # ==============================================

        # 1. 图片保存路径: .../image_dir/万柳地区_images
        current_img_dir = os.path.join(base_output_dir, f"{street_name}_images")
        os.makedirs(current_img_dir, exist_ok=True)

        # 2. 错误文件路径: .../image_dir/error_points/万柳地区_error.csv
        current_error_csv = os.path.join(base_error_dir, f"{street_name}_error.csv")

        print(f"   >>> 图片存放: {street_name}_images")
        print(f"   >>> 错误日志: {street_name}_error.csv")

        # ==============================================
        # 步骤 C: 执行核心下载逻辑
        # ==============================================

        # 读取数据
        data = read_csv(csv_path)
        if not data:
            print("   ⚠️ 文件为空或无法读取，跳过。")
            continue

        # 处理表头
        header = data[0]
        data_rows = data[1:]

        # 获取当前文件夹已存在的图片（断点续传）
        filenames_exist = set()
        if os.path.exists(current_img_dir):
            for f in os.listdir(current_img_dir):
                if f.endswith('.png'):
                    filenames_exist.add(f)

        error_img = []
        headings = ['0', '90', '180', '270']
        pitchs = '0'

        print(f"   >>> 开始下载 {len(data_rows)} 个坐标点...")

        for i in range(len(data_rows)):
            # 简化进度显示：每50个点打印一次
            if (i + 1) % 50 == 0:
                print(f'      进度: {i + 1}/{len(data_rows)}')

            row = data_rows[i]
            try:
                # 注意：这里需要根据你的CSV实际列位置调整
                # 假设 CSV 结构: [ID, Area, Longitude, Latitude, ...]
                longitude = row[2]
                latitude = row[3]
                Area = row[1]
                ID = row[0]
            except IndexError:
                continue

            # 坐标转换
            bd09mc_x, bd09mc_y = wgs2bd09mc_no_ak(longitude, latitude)
            if bd09mc_x == 0 and bd09mc_y == 0:
                continue

            # 检查是否所有方向都已下载
            all_exist = True
            for heading in headings:
                img_name = f"{ID}_{Area}_{longitude}_{latitude}_{heading}_{pitchs}.png"
                if img_name not in filenames_exist:
                    all_exist = False
                    break

            if all_exist:
                continue

            # 获取街景ID
            svid = getPanoId(bd09mc_x, bd09mc_y)
            if not svid:
                print(f"      ❌ 无街景: {ID}_{Area}_{longitude}_{latitude}")
                error_img.append(row + ['No_SV_ID'])
                continue

            # 下载图片
            for heading in headings:
                save_name = f"{ID}_{Area}_{longitude}_{latitude}_{heading}_{pitchs}.png"
                save_file_abs = os.path.join(current_img_dir, save_name)

                if save_name in filenames_exist:
                    continue

                url = f'https://mapsv0.bdimg.com/?qt=pr3d&fovy=90&quality=100&panoid={svid}&heading={heading}&pitch=0&width=480&height=320'
                img_data = grab_img_baidu(url)

                if img_data is None:
                    error_img.append(row + [heading])
                else:
                    with open(save_file_abs, "wb") as f:
                        f.write(img_data)
                    filenames_exist.add(save_name)
                    print(f"      已保存: {save_name}") # 如果嫌刷屏太多可以注释这行
                    time.sleep(0.2)  # 稍微快一点

            time.sleep(0.5)

            # 保存该街道的错误记录
        if error_img:
            write_csv(current_error_csv, error_img, header + ['error_info'])
            print(f"   ❌ 该街道处理完成，生成 {len(error_img)} 条错误记录。")
        else:
            print("   ✅ 该街道处理完成，无错误。")

        print("-" * 50)  # 分隔线

    print("\n🎉 所有任务文件处理完毕！")
