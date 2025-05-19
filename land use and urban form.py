#%%
'''IMPORTS'''
import torch
import folium
import osmnx as ox # OSM 데이터
import numpy as np
import pandas as pd
import torch.nn as nn
import networkx as nx
import geopandas as gpd
import contextily as ctx
import matplotlib.pyplot as plt

from pyproj import CRS
from shapely.ops import nearest_points
from shapely.geometry import Point, Polygon, box

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

from matplotlib.path import Path
from matplotlib.spines import Spine
from matplotlib.projections.polar import PolarAxes
from matplotlib.projections import register_projection

from xgboost import XGBClassifier

# %%
'''SETTINGS'''
plt.rcParams['figure.figsize'] = (12, 10) # matplotlib 설정
ox.settings.log_console = True
ox.settings.use_cache = True

#%%
'''DATA'''
city_name = "Seoul, South Korea" # 분석할 도시

city_boundary = ox.geocode_to_gdf(city_name) # 도시 경계 데이터

buildings = ox.features_from_place(city_name, tags={'building': True}) # 건물 데이터

road_network = ox.graph_from_place(city_name, network_type='drive') # 도로망 데이터

landuse = ox.features_from_place(city_name, tags={'landuse': True}) # 토지 이용 데이터

parks = ox.features_from_place(city_name, tags={'leisure': 'park'}) # 공원 데이터

#%%
# 도시에 적합한 UTM 좌표계 결정, 서울의 경우 EPSG:32652(북반구의 52존)
utm_crs = CRS.from_epsg(32652)

# 도시 경계를 UTM 좌표계로 변환
city_boundary_utm = city_boundary.to_crs(utm_crs)
city_geometry = city_boundary_utm.geometry.unary_union

if not buildings.empty: # 동일한 좌표계로 변환
    buildings = buildings.to_crs(utm_crs)

if not landuse.empty:
    landuse = landuse.to_crs(utm_crs)

if not parks.empty:
    parks = parks.to_crs(utm_crs)

# %%
'''EDA'''
buildings.columns
buildings.head()

#%%
road_network.number_of_nodes()
road_network.number_of_edges()
nodes, edges = ox.graph_to_gdfs(road_network) # 노드 및 노드 속성, 엣지 및 엣지 속성
print(nodes.shape)  # 노드 수와 속성 개수
print(edges.shape)  # 엣지 수와 속성 개수

#%%
landuse.head()
parks.head()

# %%
# 도시 경계, 건물 데이터(검정) 위치, 토지 이용 시각화
fig, ax = plt.subplots()
city_boundary_utm.plot(ax=ax, alpha=0.4, color='lightcoral')
landuse.plot(ax=ax, column='landuse', legend=True, categorical=True, alpha=0.7)
buildings.plot(ax=ax, color='black', alpha=0.5)
ctx.add_basemap(ax, crs=city_boundary_utm.crs.to_string())
ctx.add_basemap(ax, crs=landuse.crs.to_string())
ctx.add_basemap(ax, crs=buildings.crs.to_string())
plt.title(f'Boundaries & Building location & Land use: {city_name}')
plt.tight_layout()
plt.show()

#%%
# 도로망 시각화
fig, ax = ox.plot_graph(road_network, node_size=0, edge_linewidth=0.5)
ax.set_title("Road network: Seoul, South Korea")

# %%
# 도시 경계 지역 나누기(서울특별시, 경기도)
bounds = city_boundary_utm.total_bounds
minx, miny, maxx, maxy = bounds

print(f"Bounds: {minx}, {miny}, {maxx}, {maxy}")

cell_size = 500

# 격자 생성
rows = int((maxy - miny) / cell_size) + 1
cols = int((maxx - minx) / cell_size) + 1

print(f"Creating {rows}x{cols} grid cells") 

geometries = []
cell_ids = []

cell_id = 0 # 격자 생성
for i in range(rows):
    for j in range(cols):
        x0 = minx + j * cell_size
        y0 = miny + i * cell_size
        x1 = x0 + cell_size
        y1 = y0 + cell_size
        
        cell = box(x0, y0, x1, y1)
        
        if cell.intersects(city_geometry): # 도시 경계와 교차하는지 확인
            geometries.append(cell)
            cell_ids.append(cell_id)
            cell_id += 1

print(f"Found {len(geometries)} cells that intersect with the city boundary")

if len(geometries) > 0: # 격자를 GeoDataFrame으로 변환
    grid = gpd.GeoDataFrame({'cell_id': cell_ids, 'geometry': geometries}, crs=utm_crs)
    
    grid_3857 = grid.to_crs(epsg=3857) # 시각화를 위한 좌표계 변환
    city_boundary_3857 = city_boundary.to_crs(epsg=3857)
    
    fig, ax = plt.subplots(figsize=(12, 10))   
    grid_3857.plot(ax=ax, facecolor='none', edgecolor='black')
    city_boundary_3857.plot(ax=ax, facecolor='none', edgecolor='red')
    ctx.add_basemap(ax, crs='EPSG:3857')
    plt.title(f'Grid: {city_name}')
    plt.tight_layout()
    plt.show()
    
    print(f"Total grid cells: {len(grid)}")
else:
    print("No grid cells were created. Check coordinate systems and boundary data.")

grid.head()

#%%
'''Analyze city behavior metrics''' 
total_cells = len(grid)
grid_metrics = []
print(f"총 격자: {total_cells}개")

for idx, cell in grid.iterrows():
    print(f'idx:{idx}')
    print(f'cell:{cell}')
    print(f"Processing cell {idx}")
    if idx % 10 == 0:  # 10개 격자마다 진행상황 출력
        print(f"격자 분석 중: {idx}/{total_cells} ({idx/total_cells*100:.1f}%)")
        
    try:
        if not buildings.empty: # 해당 격자에 포함된 건물 찾기
            cell_buildings = buildings[buildings.intersects(cell.geometry)]
        else:
            cell_buildings = gpd.GeoDataFrame(geometry=[], crs=utm_crs)
            
        building_density = len(cell_buildings) / (cell.geometry.area / 10**6) # 건물 밀도 (건물 수 / 면적 km²)
            
        if len(cell_buildings) > 0: # 건물 면적 비율 (건물 면적 합 / 격자 면적)
            building_area_ratio = cell_buildings.geometry.area.sum() / cell.geometry.area
        else:
            building_area_ratio = 0
            
        try: # 도로망 관련 지표 계산
            cell_geom_wgs84 = gpd.GeoDataFrame(geometry=[cell.geometry], crs=utm_crs).to_crs("EPSG:4326").geometry[0] # UTM 좌표를 WGS84로 변환하여 도로망 가져오기
                    
            cell_graph = ox.graph_from_polygon(cell_geom_wgs84, network_type='drive', simplify=True) # 도로망 가져오기 (속도 향상을 위해 간소화된 방법 사용)
                    
            if len(list(cell_graph.edges)) > 0: # 도로 길이 계산 (미터 단위) - 수정된 부분
                edges = ox.graph_to_gdfs(cell_graph, nodes=False, edges=True) # 최신 OSMNX 버전에서는 다르게 에지 길이를 가져옴
                if 'length' in edges.columns:
                    road_length = edges['length'].sum()
                    road_density = road_length / (cell.geometry.area / 10**6)  # km²당 도로 길이
                else:
                    road_density = 0
            else:
                road_density = 0
                    
            node_density = len(cell_graph.nodes) / (cell.geometry.area / 10**6) # 교차로 밀도 (교차로 수 / 면적 km²)
                    
        except Exception as e:
            print(f"격자 {idx}의 도로망 분석 중 오류 발생: {e}")
            road_density = 0
            node_density = 0
            
        if not landuse.empty: # 토지 이용 다양성 (Shannon 지수)
            cell_landuse = landuse[landuse.intersects(cell.geometry)]
            if len(cell_landuse) > 0 and 'landuse' in cell_landuse.columns:
                landuse_counts = cell_landuse['landuse'].value_counts(normalize=True)
                landuse_diversity = -sum(p * np.log(p) for p in landuse_counts if p > 0)
            else:
                landuse_diversity = 0
        else:
            landuse_diversity = 0
            
        if not parks.empty: # 공원 비율 (공원 면적 / 격자 면적)
            cell_parks = parks[parks.intersects(cell.geometry)]
            if len(cell_parks) > 0:
                park_ratio = cell_parks.geometry.area.sum() / cell.geometry.area
            else:
                park_ratio = 0
        else:
            park_ratio = 0
                
        grid_metrics.append({
            'cell_id': cell['cell_id'],
            'building_density': building_density,
            'building_area_ratio': building_area_ratio,
            'road_density': road_density,
            'node_density': node_density,
            'landuse_diversity': landuse_diversity,
            'park_ratio': park_ratio
        })
        
    except Exception as e:
        print(f"격자 {idx} 분석 중 오류 발생: {e}")
        grid_metrics.append({ # 기본값 처리
            'cell_id': cell['cell_id'],
            'building_density': 0,
            'building_area_ratio': 0,
            'road_density': 0,
            'node_density': 0,
            'landuse_diversity': 0,
            'park_ratio': 0
        })

metrics_df = pd.DataFrame(grid_metrics)

grid = grid.merge(metrics_df, on='cell_id') # 격자와 지표 병합

grid.to_file("seoul_urban_form_analysis.gpkg", driver="GPKG")
print("분석 결과가 'seoul_urban_form_analysis.gpkg' 파일로 저장되었습니다.")

try: # 지표 시각화 예시 (건물 밀도)
    fig, ax = plt.subplots(figsize=(12, 10))
    grid.plot(column='building_density', cmap='viridis', ax=ax, legend=True)
    grid_wgs84 = grid.to_crs("EPSG:4326") # 위성 이미지 배경 추가 (UTM -> WGS84 변환 필요)
    ctx.add_basemap(ax, crs=grid_wgs84.crs.to_string())
    plt.title(f'{city_name} 건물 밀도')
    plt.tight_layout()
    plt.savefig("seoul_building_density.png", dpi=300)
    plt.close()
    print("건물 밀도 시각화 이미지가 'seoul_building_density.png' 파일로 저장되었습니다.")
except Exception as e:
    print(f"시각화 중 오류 발생: {e}")

