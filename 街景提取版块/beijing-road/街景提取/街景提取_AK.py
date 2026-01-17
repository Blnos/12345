# -*- coding: utf-8 -*-
import os
import csv
import time
import json
import requests
import glob

# ==========================================
# 配置区域
# ==========================================
# 输入路径：存放包含百度墨卡托坐标(mc_x, mc_y)的CSV文件夹
# 这里的路径根据您提供的文件进行了保留
INPUT_REL_PATH = r'../路网提取/output_road_network/road_points_mc'

# 输出路径：图片保存位置
OUTPUT_ROOT_NAME = 'image_dir'


# ==========================================
# 工具函数
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
    """下载图片二进制数据"""
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
    """通用请求函数"""
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/90.0.4430.212 Safari/537.36"
    }
    try:
        # 这个请求访问的是百度公共接口，不需要AK
        response = requests.get(_url, headers=headers, timeout=10)
        if response.status_code == 200:
            return response.content
        return None
    except Exception:
        return None


def getPanoId(_mc_x, _mc_y):
    """
    通过墨卡托坐标查询 Panoid (街景ID)
    此接口免费，利用 Web 端公开接口
    """
    url = f"https://mapsv0.bdimg.com/?&qt=qsdata&x={_mc_x}&y={_mc_y}&l=17&action=0&mode=day&t=1530956939770"
    response = openUrl(url)
    if response is None:
        return None
    try:
        response_str = response.decode("utf8")
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
    current_dir = os.path.dirname(os.path.abspath(__file__))
    input_points_dir = os.path.join(current_dir, INPUT_REL_PATH)
    input_points_dir = os.path.normpath(input_points_dir)

    base_output_dir = os.path.join(current_dir, OUTPUT_ROOT_NAME)
    base_error_dir = os.path.join(base_output_dir, 'error_points')

    # 检查输入目录
    csv_pattern = os.path.join(input_points_dir, "point_*.csv")
    csv_files = glob.glob(csv_pattern)

    if not csv_files:
        print(f"❌ 未找到任何CSV文件，请检查路径: {input_points_dir}")
        exit()

    print(f"📂 发现 {len(csv_files)} 个任务文件，准备开始下载...\n")
    os.makedirs(base_error_dir, exist_ok=True)

    for index, csv_path in enumerate(csv_files):
        file_name = os.path.basename(csv_path)
        print(f"[{index + 1}/{len(csv_files)}] 正在处理: {file_name}")

        # 提取街道名称
        try:
            parts = file_name.split('_')
            street_name = parts[1] if len(parts) >= 2 else file_name.replace('.csv', '')
        except:
            street_name = "unknown_street"

        # 设置输出目录
        current_img_dir = os.path.join(base_output_dir, f"{street_name}_images")
        os.makedirs(current_img_dir, exist_ok=True)
        current_error_csv = os.path.join(base_error_dir, f"{street_name}_error.csv")

        # 读取数据
        data = read_csv(csv_path)
        if not data:
            continue

        header = data[0]
        data_rows = data[1:]

        # 扫描已存在的图片，支持断点续传
        filenames_exist = set()
        if os.path.exists(current_img_dir):
            for f in os.listdir(current_img_dir):
                if f.endswith('.png'):
                    filenames_exist.add(f)

        error_img = []
        headings = ['0', '90', '180', '270']

        print(f"   >>> 共 {len(data_rows)} 个点")

        for i, row in enumerate(data_rows):
            if (i + 1) % 20 == 0:
                print(f'      进度: {i + 1}/{len(data_rows)}')

            try:
                # -----------------------------------------------------------
                # 读取预处理好的墨卡托坐标
                # 您的CSV结构应该是：ID, Area, Lng, Lat, mc_x, mc_y
                # -----------------------------------------------------------
                mc_x = row[4]
                mc_y = row[5]

                # 读取原始信息用于文件命名
                ID = row[0]
                Area = row[1]
                longitude = row[2]
                latitude = row[3]
            except IndexError:
                # 行数据不完整，跳过
                continue

            # 检查文件是否已存在 (如果4个方向都有了，就跳过这个点)
            all_exist = True
            for heading in headings:
                img_name = f"{ID}_{Area}_{longitude}_{latitude}_{heading}_0.png"
                if img_name not in filenames_exist:
                    all_exist = False
                    break
            if all_exist:
                continue

            # 校验坐标有效性
            try:
                if float(mc_x) == 0:
                    error_img.append(row + ['No_mc_coord'])
                    continue
            except ValueError:
                # 如果坐标不是数字（比如是'fail'或空字符串）
                error_img.append(row + ['Invalid_coord'])
                continue

            # -------------------------------------------------
            # 步骤 1: 获取 Panoid (街景ID)
            # -------------------------------------------------
            svid = getPanoId(mc_x, mc_y)
            if not svid:
                # print(f"      ❌ 无街景: {ID}")
                error_img.append(row + ['No_SV_ID'])
                continue

            # -------------------------------------------------
            # 步骤 2: 下载图片 (4个方向)
            # -------------------------------------------------
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
                    print(f"      已保存: {save_name}")
                else:
                    error_img.append(row + [heading])

                # 下载间隔 (0.2秒比较安全)
                time.sleep(0.2)

        # 记录错误信息
        if error_img:
            write_csv(current_error_csv, error_img, header + ['error_info'])
            print(f"   ⚠️ {street_name} 完成，有 {len(error_img)} 个点下载失败或无街景。")
        else:
            print(f"   ✅ {street_name} 全部成功。")

    print("\n🎉 所有任务处理完毕！")
