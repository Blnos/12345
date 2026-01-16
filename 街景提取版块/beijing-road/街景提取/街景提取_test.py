import re, os
import json
import requests
import time, glob
import csv
import traceback


# 写入CSV文件
def write_csv(filepath, data, head=None):
    if head:
        data = [head] + data
    with open(filepath, mode='w', encoding='UTF-8-sig', newline='') as f:
        writer = csv.writer(f)
        for i in data:
            writer.writerow(i)


# 读取CSV文件
def read_csv(filepath):
    data = []
    if os.path.exists(filepath):
        with open(filepath, mode='r', encoding='utf-8') as f:
            lines = csv.reader(f)
            for line in lines:
                data.append(line)
        return data
    else:
        print('文件路径错误：{}'.format(filepath))
        return []


# 获取百度街景图片
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
    response = requests.get(_url, headers=headers)

    if response.status_code == 200 and response.headers.get('Content-Type') == 'image/jpeg':
        return response.content
    else:
        return None


# 通用HTTP请求
def openUrl(_url):
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/90.0.4430.212 Safari/537.36"
    }
    response = requests.get(_url, headers=headers)
    if response.status_code == 200:
        return response.content
    else:
        return None


# 获取街景ID（svid）
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
        svid = re.findall(pat, response_str)[0]
        return svid
    except:
        return None


# WGS84转百度墨卡托坐标
def wgs2bd09mc(wgs_x, wgs_y):
    # 百度地图API密钥（ak），建议替换为自己的密钥
    ak = 'zwCZdF4xg9oU1FywO0WQH6mivt9MPLVs'
    url = f'http://api.map.baidu.com/geoconv/v1/?coords={wgs_x},{wgs_y}&from=1&to=6&output=json&ak={ak}'
    res = openUrl(url)
    if res is None:
        return 0, 0
    try:
        temp = json.loads(res.decode())
        if temp['status'] == 0:
            return temp['result'][0]['x'], temp['result'][0]['y']
        else:
            print(f"坐标转换失败：{temp.get('message', '未知错误')}")
            return 0, 0
    except Exception as e:
        print(f"坐标转换异常：{str(e)}")
        return 0, 0


if __name__ == "__main__":
    # 配置路径
    root = r'.\dir_test'  # 根目录
    read_fn = r'point.csv'  # 输入CSV文件名（包含经纬度的文件）
    error_fn = r'万寿路街道_error.csv'  # 错误记录CSV文件名
    img_dir = r'images'  # 图片保存目录

    # 创建图片目录（如果不存在）
    os.makedirs(os.path.join(root, img_dir), exist_ok=True)

    # 获取已存在的图片文件名（用于去重）
    filenames_exist = glob.glob1(os.path.join(root, img_dir), "*.png")

    # 读取输入CSV
    data = read_csv(os.path.join(root, read_fn))
    if not data:
        print("未读取到数据，程序退出")
        exit()

    # 表头处理
    header = data[0]
    data = data[1:]  # 去掉表头行

    # 记录错误信息
    error_img = []

    # 街景方向（北、东、南、西）
    headings = ['0', '90', '180', '270']
    pitchs = '0'  # 俯仰角

    # 遍历每个坐标点
    for i in range(len(data)):
        print(f'正在处理第 {i + 1}/{len(data)} 个点...')
        # 从表头可知，经纬度在第4、5列（索引3、4，因为CSV读取后是0开始的列表）
        try:
            longitude = data[i][2]  # Longitude（经度）
            latitude = data[i][3]   # Latitude（纬度）
            Area = data[i][1]
        except IndexError:
            print(f"第 {i+1} 行数据格式错误，跳过")
            continue

        # 转换坐标为百度墨卡托
        try:
            bd09mc_x, bd09mc_y = wgs2bd09mc(longitude, latitude)
            if bd09mc_x == 0 and bd09mc_y == 0:
                print("坐标转换失败，跳过该点")
                continue
        except Exception as e:
            print(f"坐标转换出错：{str(e)}，跳过该点")
            continue

        # 检查该点的四个方向图片是否已存在
        all_exist = True
        for heading in headings:
            img_name = f"{Area}_{longitude}_{latitude}_{heading}_{pitchs}.png"
            if img_name not in filenames_exist:
                all_exist = False
                break
        if all_exist:
            print("该点所有方向图片已存在，跳过")
            continue

        # 获取街景ID
        svid = getPanoId(bd09mc_x, bd09mc_y)
        if not svid:
            print("未获取到街景ID，跳过该点")
            continue
        print(f"街景ID：{svid}")

        # 下载四个方向的街景图片
        for heading in headings:
            save_path = os.path.join(root, img_dir, f"{i}_{Area}_{longitude}_{latitude}_{heading}_{pitchs}.png")
            # 构建街景图片URL
            url = f'https://mapsv0.bdimg.com/?qt=pr3d&fovy=90&quality=100&panoid={svid}&heading={heading}&pitch=0&width=480&height=320'
            img_data = grab_img_baidu(url)

            if img_data is None:
                print(f"下载失败：{longitude},{latitude} 方向 {heading}°")
                error_img.append(data[i] + [heading])  # 记录错误点信息+失败方向
            else:
                with open(save_path, "wb") as f:
                    f.write(img_data)
                print(f"下载成功：{save_path}")

        # 休眠避免请求过于频繁
        time.sleep(6)

    # 保存错误记录
    if error_img:
        write_csv(os.path.join(root, error_fn), error_img, header + ['error_heading'])
        print(f"共 {len(error_img)} 条下载错误记录已保存到 {error_fn}")
    else:
        print("所有街景图片下载完成，无错误记录")