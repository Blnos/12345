#!/usr/bin/env python3
"""
从当前目录下的所有 '_images' 结尾的子文件夹中随机选取10个文件夹，
每个文件夹随机复制200张图片到 image_example 目录，保持目录结构。
"""

import os
import sys
import random
import shutil
import argparse

# 设置标准输出编码为 UTF-8，避免中文乱码
if sys.version_info >= (3, 7):
    sys.stdout.reconfigure(encoding='utf-8')
else:
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# 支持的图片扩展名
IMAGE_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff', '.webp', '.svg'}

def get_image_files(folder_path):
    """返回文件夹中所有图片文件的列表"""
    image_files = []
    for filename in os.listdir(folder_path):
        filepath = os.path.join(folder_path, filename)
        if os.path.isfile(filepath):
            ext = os.path.splitext(filename)[1].lower()
            if ext in IMAGE_EXTENSIONS:
                image_files.append(filename)
    return image_files

def has_image_files(folder_path):
    """检查文件夹中是否至少有一张图片文件"""
    for filename in os.listdir(folder_path):
        filepath = os.path.join(folder_path, filename)
        if os.path.isfile(filepath):
            ext = os.path.splitext(filename)[1].lower()
            if ext in IMAGE_EXTENSIONS:
                return True
    return False

def main():
    parser = argparse.ArgumentParser(description='创建 image_example 测试数据集')
    parser.add_argument('--source', default='.', help='源目录，默认为当前目录')
    parser.add_argument('--dest', default='image_example', help='目标目录，默认为 image_example')
    parser.add_argument('--folder-count', type=int, default=10, help='随机选择的文件夹数量，默认为10')
    parser.add_argument('--image-count', type=int, default=200, help='每个文件夹随机选择的图片数量，默认为200')
    parser.add_argument('--seed', type=int, default=None, help='随机种子，用于可重复性')
    args = parser.parse_args()

    source_dir = os.path.abspath(args.source)
    dest_dir = os.path.abspath(args.dest)

    print(f"源目录: {source_dir}")
    print(f"目标目录: {dest_dir}")
    print(f"随机选择 {args.folder_count} 个文件夹，每个文件夹随机选择 {args.image_count} 张图片")

    if args.seed is not None:
        random.seed(args.seed)
        print(f"使用随机种子: {args.seed}")

    # 1. 获取所有以 '_images' 结尾且包含图片的子文件夹
    folders_with_images = []
    for item in os.listdir(source_dir):
        item_path = os.path.join(source_dir, item)
        if os.path.isdir(item_path) and item.endswith('_images'):
            # 检查文件夹中是否有图片文件
            if has_image_files(item_path):
                folders_with_images.append(item)
            else:
                print(f"跳过空文件夹: {item}")

    if not folders_with_images:
        print("错误: 未找到包含图片的文件夹。")
        sys.exit(1)

    print(f"找到 {len(folders_with_images)} 个包含图片的文件夹。")

    # 2. 随机选择文件夹
    if len(folders_with_images) < args.folder_count:
        print(f"警告: 只有 {len(folders_with_images)} 个包含图片的文件夹，将选择所有文件夹。")
        selected_folders = folders_with_images
    else:
        selected_folders = random.sample(folders_with_images, args.folder_count)

    selected_folders.sort()  # 排序以便输出更清晰
    print(f"\n随机选择的文件夹 ({len(selected_folders)} 个):")
    for folder in selected_folders:
        print(f"  - {folder}")

    # 3. 确保目标目录存在，如果已存在则报错
    if os.path.exists(dest_dir):
        print(f"\n错误: 目标目录 '{dest_dir}' 已存在。请删除或重命名该目录后再运行。")
        sys.exit(1)
    os.makedirs(dest_dir)

    # 4. 遍历每个选中的文件夹，复制图片
    summary = []
    for folder in selected_folders:
        folder_path = os.path.join(source_dir, folder)
        dest_folder_path = os.path.join(dest_dir, folder)
        os.makedirs(dest_folder_path)

        # 获取所有图片文件
        image_files = get_image_files(folder_path)
        total_images = len(image_files)

        if total_images == 0:
            print(f"\n警告: 文件夹 '{folder}' 中没有图片文件，跳过。")
            shutil.rmtree(dest_folder_path)  # 删除空的目标文件夹
            continue

        # 随机选择图片
        if total_images < args.image_count:
            print(f"\n提示: 文件夹 '{folder}' 中只有 {total_images} 张图片，将复制全部图片。")
            selected_images = image_files
        else:
            selected_images = random.sample(image_files, args.image_count)

        # 复制文件
        copied_count = 0
        for filename in selected_images:
            src_path = os.path.join(folder_path, filename)
            dst_path = os.path.join(dest_folder_path, filename)
            try:
                shutil.copy2(src_path, dst_path)
                copied_count += 1
            except Exception as e:
                print(f"错误: 复制文件 '{filename}' 失败: {e}")

        summary.append((folder, copied_count, total_images))
        print(f"文件夹 '{folder}': 复制了 {copied_count} 张图片（共 {total_images} 张）")

    # 5. 输出汇总信息
    print("\n" + "="*60)
    print("任务完成！汇总信息:")
    print("="*60)
    total_copied = 0
    for folder, copied, total in summary:
        print(f"{folder}: 复制 {copied} 张图片（文件夹中共 {total} 张）")
        total_copied += copied
    print(f"\n总计复制图片: {total_copied} 张")
    print(f"目标目录: {dest_dir}")

if __name__ == '__main__':
    main()

#
# 自定义参数
# python create_image_example.py --folder-count 5 --image-count 100 --seed 123