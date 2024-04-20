#!/usr/bin/env python3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Voronoi, voronoi_plot_2d
from sklearn.cluster import KMeans
import geopandas as gpd
import mpld3



starbucks_data_path = 'Data/directory.csv'
starbucks_df = pd.read_csv(starbucks_data_path)
cleaned_data = starbucks_df.dropna(subset=['Latitude', 'Longitude'])
us_df = cleaned_data[cleaned_data['Country'] == 'US']
us_df_filtered = us_df[
    (us_df['Latitude'] >= 24) & 
    (us_df['Latitude'] <= 50) &
    (us_df['Longitude'] >= -125) &
    (us_df['Longitude'] <= -66)
]

points = us_df_filtered[['Longitude', 'Latitude']].to_numpy()

kmeans = KMeans(n_clusters=50)  # Especifica el número deseado de clusters
kmeans.fit(points)
centroids = kmeans.cluster_centers_
vor = Voronoi(centroids)

usa_map = gpd.read_file('Data/cb_2021_us_state_20m/cb_2021_us_state_20m.shp')
usa_map = usa_map.to_crs("EPSG:4326")

fig, ax = plt.subplots(figsize=(15, 15))  
usa_map.plot(ax=ax, color='white', edgecolor='black', alpha=0.5)  
ax.scatter(points[:, 0], points[:, 1], color='red', s=1)  
voronoi_plot_2d(vor, ax=ax, show_vertices=False, line_colors='blue',line_alpha=0.5)  
plt.title('Distribución de Tiendas Starbucks en EE.UU.')
plt.xlabel('Longitud')
plt.ylabel('Latitud')
mpld3.save_html(fig, 'starbucks_distribution.html')
plt.show()
