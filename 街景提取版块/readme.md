# 百度街景爬虫

　　该项目用于根据指定的wgs84经纬度坐标获取对应位置的百度地图的街景图像。

## 内容
### 街景提取的主程序('beijing-road'文件夹)：

该文件夹包含街景提取全工作文件。
街景提取的思路是：

 `` 确认各个街道范围-->
 提取各个街道osm路网-->
 整理osm路网[合并双向路、删除冗余路网等]-->
 路网打点-->
 由路网点提取街景 ``

+ [cache](beijing-road/cache)文件夹：所有工作过程中产生的缓存历史数据。
+ [路网提取](beijing-road/%E8%B7%AF%E7%BD%91%E6%8F%90%E5%8F%96)文件夹：
  + [路网提取.py](beijing-road/%E8%B7%AF%E7%BD%91%E6%8F%90%E5%8F%96/%E8%B7%AF%E7%BD%91%E6%8F%90%E5%8F%96.py) 可实现：
    1. 输入区域范围的json格式文件
    2. 提取范围内osm路网
    3. 整理osm路网--路网上打点
    4. 输出路网点
    5. 同时输出一些辅助验证的数据（例如：道路简化过程中生成的缓冲带、中心线等，可以用来根据实际情况验证并调节程序中的参数大小）
  +  [output_road_network](beijing-road/%E8%B7%AF%E7%BD%91%E6%8F%90%E5%8F%96/output_road_network)文件夹：路网提取.py文件的输出
      + [road_points](beijing-road/%E8%B7%AF%E7%BD%91%E6%8F%90%E5%8F%96/output_road_network/road_points) 下一步街景提取时要用到的路网点
      + [辅助验证数据](beijing-road/%E8%B7%AF%E7%BD%91%E6%8F%90%E5%8F%96/output_road_network/%E8%BE%85%E5%8A%A9%E9%AA%8C%E8%AF%81%E6%95%B0%E6%8D%AE) 道路中心线，缓冲区等过程数据
  
+ [街景提取](beijing-road/%E8%A1%97%E6%99%AF%E6%8F%90%E5%8F%96)文件夹：
  + [街景提取_noAK.py](beijing-road/%E8%A1%97%E6%99%AF%E6%8F%90%E5%8F%96/%E8%A1%97%E6%99%AF%E6%8F%90%E5%8F%96_noAK.py) 无需AK的提取脚本
  + [街景提取_test.py](beijing-road/%E8%A1%97%E6%99%AF%E6%8F%90%E5%8F%96/%E8%A1%97%E6%99%AF%E6%8F%90%E5%8F%96_test.py) 需要AK的提取脚本，由于限额问题暂不适用
  + [image_dir](beijing-road/%E8%A1%97%E6%99%AF%E6%8F%90%E5%8F%96/image_dir) 街景提取_noAK.py的工作结果
    + xxx_街道images：每个街道的街景图像，命名规则为：路网点序号_街道名称_经度_纬度_俯仰角
    + error_points：未能成功提取的街道路网点
      + 第 1 至 N 列	原始数据	也就是 point_xxx.csv 里原来的所有列（ID、经纬度等），完全原样保留。
      + 最后一列	error_info	这是程序自动添加的新列，记录具体的错误信息。
      + 
+ 另外几个辅助数据文件夹都是过程中验证代码效果的，没有实际调用；仅供参考

  
## 环境依赖(待完善)
   当前我的python版本Python 3.10.19
   完整运行全部项目需要安装[requirements.txt](requirements.txt)中的依赖（Anaconda 中直接导出的）
   仅运行[街景提取_noAK.py](beijing-road/%E8%A1%97%E6%99%AF%E6%8F%90%E5%8F%96/%E8%A1%97%E6%99%AF%E6%8F%90%E5%8F%96_noAK.py)应该不用额外装什么？
　　

