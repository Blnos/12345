import geopandas as gpd
import osmnx as ox
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import re
from shapely.geometry import Polygon, MultiPolygon, Point, LineString, MultiLineString
from centerline.geometry import Centerline
import networkx as nx
from shapely.ops import linemerge, unary_union, polygonize
from concurrent.futures import ThreadPoolExecutor
from collections import defaultdict
import warnings
import traceback

# -------------------------- 全局配置 --------------------------
BASE_DIR = Path(r"/beijing-road/辅助数据_路网提取_北京各街道范围/beijing_geojson_all")
NETWORK_TYPE = "all"
STEP_METERS = 50  # 采样点间隔（米）
OUTPUT_DIR = Path("output_road_network")
MIN_EDGE_LENGTH = 50  # 短边过滤阈值（米）
INTERSECTION_TOLERANCE = 10  # 交点合并阈值（米，暂未使用）
BUFFER_DISTANCE = 50  # 路网缓冲区半径（米）
CENTERLINE_INTERP_DIST = 10  # 中心线插值精度（米）
DANGLING_LINE_MIN_LENGTH = 50  # 新增：仅删除<50米的极短悬挂线（避免删长路网）

# -------------------------------------------------------------

# 忽略无关警告
warnings.filterwarnings('ignore')

# 创建输出目录
OUTPUT_DIR.mkdir(exist_ok=True)
POINT_SAVE_DIR = OUTPUT_DIR / "road_points"
BUFFER_SAVE_DIR = OUTPUT_DIR / "road_buffers"
CENTERLINE_SAVE_DIR = OUTPUT_DIR / "centerlines"
TOPOLOGY_SAVE_DIR = OUTPUT_DIR / "cleaned_topology"
POLYGON_SAVE_DIR = OUTPUT_DIR / "polygonized_areas"

for dir_path in [POINT_SAVE_DIR, BUFFER_SAVE_DIR, CENTERLINE_SAVE_DIR, TOPOLOGY_SAVE_DIR, POLYGON_SAVE_DIR]:
    dir_path.mkdir(exist_ok=True)


def sanitize_filename(filename):
    """清理文件名中的非法字符"""
    return re.sub(r'[\\/:*?"<>|]', '_', filename)


def process_line_chunk(lines_chunk):
    """辅助函数：处理线要素分块，提取节点和边"""
    nodes = []
    edges = []
    for line in lines_chunk:
        if isinstance(line, LineString):
            coords = list(line.coords)
            # 提取节点
            for coord in coords:
                nodes.append(Point(coord))
            # 提取边（相邻坐标对）
            for i in range(len(coords) - 1):
                edge = (Point(coords[i]), Point(coords[i + 1]))
                edges.append(edge)
    return nodes, edges


def extract_centerlines(gdf_merged):
    """提取缓冲区中心线"""
    center_lines = []
    interpolation_distance = CENTERLINE_INTERP_DIST  # 控制中心线的精度

    # 对每个多边形提取中心线
    for geom in gdf_merged.geometry:
        # 确保几何有效
        geom = geom.make_valid() if not geom.is_valid else geom

        if isinstance(geom, (Polygon, MultiPolygon)):
            # 对MultiPolygon，分别处理每个Polygon
            if isinstance(geom, MultiPolygon):
                for poly in geom.geoms:
                    try:
                        center_line = Centerline(poly, interpolation_distance=interpolation_distance)
                        if center_line is not None:
                            center_lines.append(center_line.geometry)
                    except Exception as e:
                        print(f"提取子多边形中心线失败: {e}")
                        continue
            else:
                try:
                    center_line = Centerline(geom, interpolation_distance=interpolation_distance)
                    if center_line is not None:
                        center_lines.append(center_line.geometry)
                except Exception as e:
                    print(f"提取多边形中心线失败: {e}")
                    continue

    # 返回中心线GeoDataFrame
    return gpd.GeoDataFrame(geometry=center_lines, crs=gdf_merged.crs) if center_lines else gpd.GeoDataFrame(
        geometry=[], crs=gdf_merged.crs)


def clean_network_topology_step1(gdf):
    """轻量化清理：仅删除伪节点，保留悬挂点（避免过度删线）"""
    if gdf.empty:
        return gpd.GeoDataFrame(geometry=[], crs=gdf.crs)

    # 合并所有线要素
    all_lines = unary_union(gdf.geometry.tolist())
    if isinstance(all_lines, LineString):
        all_lines = MultiLineString([all_lines])
    elif not isinstance(all_lines, MultiLineString):
        return gpd.GeoDataFrame(geometry=[], crs=gdf.crs)

    # 仅合并伪节点（度=2的节点），保留悬挂点（度=1）
    cleaned_lines = []
    for line in all_lines.geoms:
        coords = list(line.coords)
        if len(coords) < 2:
            continue
        # 保留所有非空线段（不再过滤悬挂点）
        cleaned_lines.append(line)

    # 合并线段（仅合并连续无伪节点的线段）
    merged_lines = linemerge(cleaned_lines) if cleaned_lines else MultiLineString()
    cleaned_gdf = gpd.GeoDataFrame(geometry=[merged_lines], crs=gdf.crs)
    exploded_gdf = cleaned_gdf.explode(index_parts=True).reset_index(drop=True)
    return exploded_gdf


def remove_dangling_lines(gdf):
    """轻量化清理：仅删除<50米的极短悬挂线，保留长悬挂线"""
    if gdf.empty:
        return gdf

    while True:
        nodes = []
        line_lengths = {}  # 记录每条线的长度
        for idx, line in enumerate(gdf.geometry):
            if line.geom_type == 'LineString':
                coords = list(line.coords)
                if len(coords) >= 2:
                    start_node = Point(coords[0])
                    end_node = Point(coords[-1])
                    nodes.extend([start_node, end_node])
                    line_lengths[idx] = line.length  # 记录线长度

        # 统计端点连接数
        node_connections = defaultdict(int)
        for line in gdf.geometry:
            if line.geom_type == 'LineString':
                coords = list(line.coords)
                if len(coords) >= 2:
                    start_node = Point(coords[0])
                    end_node = Point(coords[-1])
                    node_connections[start_node] += 1
                    node_connections[end_node] += 1

        dangling_nodes = {node for node, count in node_connections.items() if count == 1}
        if not dangling_nodes:
            break

        # 仅删除「端点是悬挂点 + 长度<DANGLING_LINE_MIN_LENGTH」的线
        cleaned_lines = []
        for idx, line in enumerate(gdf.geometry):
            if line.geom_type != 'LineString':
                continue
            coords = list(line.coords)
            if len(coords) < 2:
                continue
            start_node = Point(coords[0])
            end_node = Point(coords[-1])
            line_len = line_lengths.get(idx, 0)

            # 仅删除极短悬挂线，保留长悬挂线
            if (start_node in dangling_nodes or end_node in dangling_nodes) and line_len < DANGLING_LINE_MIN_LENGTH:
                continue  # 跳过（删除）极短悬挂线
            cleaned_lines.append(line)

        if len(cleaned_lines) == len(gdf):
            break
        gdf = gpd.GeoDataFrame(geometry=cleaned_lines, crs=gdf.crs)

    return gdf


def process_single_area(file_path):
    """处理单个区域的完整流程：下载路网→缓冲区→中心线→拓扑清理→采样点"""
    area_name = file_path.stem
    print(f"\n===== 开始处理：{area_name} =====")

    # 1. 读取区域边界JSON
    try:
        gdf_bound = gpd.read_file(file_path)
    except Exception as e:
        print(f"[x] 读取 {area_name} 边界失败: {e}")
        return None

    # 确保边界坐标系为WGS84（EPSG:4326）
    if gdf_bound.crs is None:
        gdf_bound.crs = "EPSG:4326"
    elif gdf_bound.crs != "EPSG:4326":
        gdf_bound = gdf_bound.to_crs("EPSG:4326")

    # 提取所有有效多边形（处理MultiPolygon）
    valid_polygons = []
    for geom in gdf_bound.geometry:
        if geom is not None and not geom.is_empty:
            if isinstance(geom, MultiPolygon):
                valid_polygons.extend([p for p in geom.geoms if not p.is_empty])
            else:
                valid_polygons.append(geom)

    if not valid_polygons:
        print(f"[!] {area_name} 无有效边界多边形，跳过")
        return None

    print(f"[-] 读取到 {area_name} 共{len(valid_polygons)}个有效子多边形，准备下载路网")
    bound = unary_union(valid_polygons)  # 合并为整体边界

    # 2. 批量下载每个子多边形的路网并合并
    G_proj_list = []  # 存储投影后的路网图
    for idx, sub_polygon in enumerate(valid_polygons):
        print(f"\n[-] 正在下载第{idx + 1}/{len(valid_polygons)}个子多边形的路网...")
        try:
            # 下载原始路网（地理坐标系）
            G_geo = ox.graph_from_polygon(sub_polygon, network_type=NETWORK_TYPE, simplify=False)
            if G_geo.number_of_edges() == 0:
                print(f"[!] 第{idx + 1}个子多边形无路网，跳过")
                continue

            # 转换为投影坐标系（米为单位）
            G_proj = ox.project_graph(G_geo)
            G_proj_list.append(G_proj)
            print(f"[+] 第{idx + 1}个子多边形路网下载完成，边数：{G_proj.number_of_edges()}")
        except Exception as e:
            print(f"[x] 第{idx + 1}个子多边形路网下载失败: {e}")
            continue

    # 检查是否下载到有效路网
    if not G_proj_list:
        print(f"[!] {area_name} 所有子多边形均未下载到路网，跳过")
        return None

    # 合并所有子多边形的路网
    print(f"\n[-] 合并{len(G_proj_list)}个子多边形的路网...")
    G_proj = G_proj_list[0]
    for g in G_proj_list[1:]:
        G_proj.add_nodes_from(g.nodes(data=True))
        G_proj.add_edges_from(g.edges(keys=True, data=True))

    proj_crs = G_proj.graph['crs']
    # 转换为GeoDataFrame并过滤短边
    edges_proj = ox.graph_to_gdfs(G_proj, nodes=False, edges=True)
    filter_gdf = edges_proj[edges_proj['length'] >= MIN_EDGE_LENGTH]
    print(f"[-] {area_name} 路网合并完成，有效边数：{len(filter_gdf)}，投影坐标系：{proj_crs}")

    try:
        # 3. 构建路网缓冲区
        print(f"\n===== 步骤1：构建{BUFFER_DISTANCE}米缓冲区 =====")
        buffered = filter_gdf['geometry'].buffer(BUFFER_DISTANCE)
        # 合并缓冲区（兼容geopandas新旧版本）
        try:
            merged_buffer = buffered.union_all()
        except AttributeError:
            merged_buffer = buffered.unary_union
        gdf_merged = gpd.GeoDataFrame(geometry=[merged_buffer], crs=proj_crs)
        # 保存缓冲区
        buffer_path = BUFFER_SAVE_DIR / f"{sanitize_filename(area_name)}_buffer.shp"
        gdf_merged.to_file(buffer_path, encoding='utf-8')
        print(f"[+] 缓冲区已保存：{buffer_path}")

        # 4. 提取缓冲区中心线
        print(f"\n===== 步骤2：提取缓冲区中心线 =====")
        centerlines = extract_centerlines(gdf_merged)
        if centerlines.empty:
            print(f"[!] {area_name} 未提取到中心线，跳过后续步骤")
            return None

        # 转换区域边界为投影坐标系的线
        bound_proj = gpd.GeoDataFrame(geometry=[bound], crs="EPSG:4326").to_crs(proj_crs).geometry.iloc[0]
        bound_line = bound_proj.boundary
        # 合并中心线和区域边界线
        bound_gdf = gpd.GeoDataFrame(geometry=[bound_line], crs=proj_crs)
        merged_lines = pd.concat([centerlines, bound_gdf], ignore_index=True)
        # 保存中心线
        centerline_path = CENTERLINE_SAVE_DIR / f"{sanitize_filename(area_name)}_centerline.shp"
        merged_lines.to_file(centerline_path, encoding='utf-8')
        print(f"[+] 中心线+边界线已保存：{centerline_path}")

        # 5. 清理网络拓扑
        print(f"\n===== 步骤3：清理网络拓扑 =====")
        # 第一步：处理悬挂点和伪节点
        cleaned_network = clean_network_topology_step1(merged_lines)
        # 第二步：递归删除悬挂线
        cleaned_network = remove_dangling_lines(cleaned_network)
        if cleaned_network.empty:
            print(f"[!] {area_name} 拓扑清理后路网为空，跳过")
            return None
        # 保存清理后的拓扑
        topology_path = TOPOLOGY_SAVE_DIR / f"{sanitize_filename(area_name)}_cleaned_topology.shp"
        cleaned_network.to_file(topology_path, encoding='utf-8')
        print(f"[+] 清理后的拓扑已保存：{topology_path}")

        # 6. 线要素转面要素
        print(f"[-] 开始线转面处理...")
        # 合并所有线
        try:
            merged_lines_geom = cleaned_network.union_all()
        except AttributeError:
            merged_lines_geom = cleaned_network.unary_union
        # 构建多边形并过滤无效几何
        polygons = list(polygonize(merged_lines_geom)) if merged_lines_geom else []
        polygon_gdf = gpd.GeoDataFrame(geometry=polygons, crs=proj_crs)
        polygon_gdf = polygon_gdf[polygon_gdf.geometry.is_valid]
        # 保存多边形
        polygon_path = POLYGON_SAVE_DIR / f"{sanitize_filename(area_name)}_polygons.shp"
        polygon_gdf.to_file(polygon_path, encoding='utf-8')
        print(f"[+] 线转面完成，保存 {len(polygon_gdf)} 个有效多边形：{polygon_path}")

        # 7. 基于清理后的路网生成等距采样点
        print(f"\n===== 步骤4：生成{STEP_METERS}米间隔采样点 =====")
        points_utm = []
        for _, row in tqdm(cleaned_network.iterrows(), total=len(cleaned_network), desc=f"{area_name} 采样"):
            line = row['geometry']
            if line.geom_type not in ['LineString', 'MultiLineString']:
                continue
            length = line.length
            # 沿线等距插值生成点
            for dist in range(0, int(length), STEP_METERS):
                if dist < length:
                    point_utm = line.interpolate(dist)
                    points_utm.append(point_utm)

        if not points_utm:
            print(f"[!] {area_name} 未生成采样点")
            return None

        # 转换为经纬度并保存
        gdf_points_proj = gpd.GeoDataFrame(geometry=points_utm, crs=proj_crs)
        gdf_points_geo = gdf_points_proj.to_crs("EPSG:4326")
        # 构建采样点数据
        data_list = []
        for idx, pt in enumerate(gdf_points_geo.geometry):
            data_list.append({
                'ID': idx,
                'Area': area_name,
                'Longitude': round(pt.x, 6),
                'Latitude': round(pt.y, 6)
            })
        # 保存为CSV
        csv_path = POINT_SAVE_DIR / f"point_{sanitize_filename(area_name)}.csv"
        df = pd.DataFrame(data_list)
        df.to_csv(csv_path, index=False, encoding='utf-8-sig')
        print(f"[+] 采样点保存完成：{csv_path}（共{len(df)}个点）")

        # 返回处理结果
        return {
            "buffer": gdf_merged,
            "centerline": merged_lines,
            "cleaned_topology": cleaned_network,
            "polygons": polygon_gdf,
            "points": df
        }

    except Exception as e:
        print(f"[x] {area_name} 处理失败: {e}")
        traceback.print_exc()
        return None


# -------------------------- 批量执行入口 --------------------------
if __name__ == "__main__":
    print(f"当前osmnx版本：{ox.__version__}")
    # 查找BASE_DIR下所有JSON文件
    json_files = list(BASE_DIR.glob("*.json"))

    if not json_files:
        print(f"[x] 在目录 {BASE_DIR} 中未找到任何JSON文件！")
    else:
        print(f"[+] 找到 {len(json_files)} 个区域文件，开始批量处理...")
        for file in json_files:
            process_single_area(file)

    print("\n===== 所有区域处理完成 =====")