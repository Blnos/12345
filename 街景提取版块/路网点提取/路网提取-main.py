import geopandas as gpd
import numpy as np
from shapely.geometry import Point, LineString, MultiLineString
import pyproj
from shapely.ops import transform


def create_points_on_lines(polyline_path, distance, output_shp_path, output_csv_path):
    """
    功能：根据路网线要素，按固定距离生成采集点（含经纬度输出）
    参数说明：
    polyline_path     输入路网shp文件路径（线要素）
    distance          打点间隔距离（单位：米）
    output_shp_path   输出点要素shp文件路径
    output_csv_path   输出经纬度CSV文件路径（WGS84坐标系）
    """
    # 读取路网数据
    gdf = gpd.read_file(polyline_path)

    # 确保输入数据为WGS84地理坐标系（EPSG:4326）
    if gdf.crs != "EPSG:4326":
        gdf = gdf.to_crs("EPSG:4326")

    # 定义投影转换（WGS84 → Web Mercator，用于精确距离计算）
    wgs84 = pyproj.CRS('EPSG:4326')
    utm = pyproj.CRS('EPSG:3857')  # Web Mercator投影（米为单位）
    project = pyproj.Transformer.from_crs(wgs84, utm, always_xy=True).transform
    project_inverse = pyproj.Transformer.from_crs(utm, wgs84, always_xy=True).transform

    points_data = []
    point_id = 1  # 全局点ID

    # 遍历每条路网线
    for idx, row in gdf.iterrows():
        geometry = row.geometry
        oid = row.name  # 原始路网线ID
        points_in_road_id = 1  # 单条路内的点序号

        # 处理单条线和多条线（MultiLineString）
        if isinstance(geometry, MultiLineString):
            lines = list(geometry.geoms)
        elif isinstance(geometry, LineString):
            lines = [geometry]
        else:
            print(f"不支持的几何类型（OID: {oid}）：{type(geometry)}")
            continue

        # 对每条子线生成间隔点
        for line in lines:
            # 投影到Web Mercator计算距离
            line_utm = transform(project, line)
            line_length = line_utm.length

            # 生成间隔距离序列（从distance开始，到线长结束，步长distance）
            distances = np.arange(distance, line_length + distance, distance)
            for d in distances:
                # 在线上插值生成点（Web Mercator坐标系）
                point_utm = line_utm.interpolate(d)
                # 转换回WGS84经纬度坐标系
                point_wgs84 = transform(project_inverse, point_utm)
                # 存储点数据（含关联路网ID、距离等属性）
                points_data.append({
                    'geometry': point_wgs84,
                    'LineOID': oid,  # 关联原始路网ID
                    'Distance': d,  # 点到线起点的距离（米）
                    'Global_PointID': point_id,  # 全局唯一点ID
                    'Road_PointID': points_in_road_id  # 单条路内点序号
                })
                point_id += 1
                points_in_road_id += 1

    # 创建输出点要素GeoDataFrame
    points_gdf = gpd.GeoDataFrame(points_data, crs="EPSG:4326")
    # 关联原始路网的属性字段（除几何体外）
    points_gdf = points_gdf.merge(
        gdf.drop(columns='geometry'),
        left_on='LineOID',
        right_index=True,
        how='left'
    )
    points_gdf = points_gdf.set_geometry('geometry')  # 确认几何字段

    # 提取经纬度到属性表
    points_gdf['Longitude'] = points_gdf.geometry.x  # 经度（WGS84）
    points_gdf['Latitude'] = points_gdf.geometry.y  # 纬度（WGS84）

    # 保存结果文件
    points_gdf.to_file(output_shp_path)  # 保存为Shapefile
    # 保存CSV（含核心属性）
    csv_cols = ['Global_PointID', 'Road_PointID', 'LineOID', 'Longitude', 'Latitude', 'Distance']
    points_gdf[csv_cols].to_csv(output_csv_path, index=False, encoding='utf-8')

    # 输出运行信息
    print(f"生成完成！")
    print(f"输出坐标系：WGS84（EPSG:4326）")
    print(f"总生成点数：{len(points_gdf)}")
    print(f"Shapefile路径：{output_shp_path}")
    print(f"CSV路径：{output_csv_path}")


# -------------------------- 运行示例 --------------------------
if __name__ == "__main__":
    # 请根据实际路径修改以下参数
    polyline_path = r"D:\街景数据文件\北京道路\test_东城区_路网数据.shp" # 输入路网SHP路径
    output_shp_path = r"output_points.shp"  # 输出点SHP路径
    output_csv_path = r"output_points.csv"  # 输出经纬度CSV路径
    distance = 100  # 间隔距离（米）

    # 调用函数生成点
    create_points_on_lines(polyline_path, distance, output_shp_path, output_csv_path)