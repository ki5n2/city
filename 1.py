# 도시 경계 지역 나누기(서울특별시, 경기도)
bounds = city_boundary.total_bounds
minx, miny, maxx, maxy = bounds

cell_size = 1000   

grid_cells = []
for x0 in np.arange(minx, maxx, cell_size):
    for y0 in np.arange(miny, maxy, cell_size):
        x1 = x0 + cell_size
        y1 = y0 + cell_size
        grid_cells.append(box(x0, y0, x1, y1))

grid = gpd.GeoDataFrame(geometry=grid_cells, crs=utm_crs)

grid = grid[grid.intersects(city_boundary_utm.unary_union)] # 도시 경계 내의 격자만 선택

grid_3857 = grid.to_crs(epsg=3857)
city_boundary_3857 = city_boundary.to_crs(epsg=3857)

grid['cell_id'] = range(len(grid))
grid_metrics = []

grid_3857['cell_id'] = range(len(grid_3857))
grid_3857_metrics = []
