import umap
import seaborn as sns
import pacmap
import trimap
from sklearn.manifold import TSNE
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN, OPTICS
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np

def preprocess_energy_data(file_path='smard15-24.csv', frac=0.1):
    #file-path'smard18-24.csv'
    df = pd.read_csv(file_path)
    # Replace '-' with '0' in the dataset
    df.replace('-', '0', inplace=True)

    # Extract year, month, and hour from 'Start date' column
    df['year'] = pd.to_datetime(df['Start date'], format='%Y-%m-%d %H:%M:%S').dt.year
    df['month'] = pd.to_datetime(df['Start date'], format='%Y-%m-%d %H:%M:%S').dt.month
    df['hour'] = pd.to_datetime(df['Start date'], format='%Y-%m-%d %H:%M:%S').dt.hour

    # Remove commas from numeric columns and convert to numeric
    df = df.replace({',': ''}, regex=True)
    df = df.apply(pd.to_numeric, errors='coerce')

    # Remove columns containing 'date' (case-insensitive) and exclude datetime columns
    df = df.loc[:, ~df.columns.str.contains('date', case=False)]
    # Extract generation-related columns (columns with '[MWh]' in their names)
    generation_columns = df.filter(like='[MWh]', axis=1)
    generation_columns.columns = generation_columns.columns.str[:-27]  # Clean column names
    generation_columns['year'], generation_columns['month'], generation_columns['hour'] = df['year'], df['month'], df['hour']

    price_columns = df.loc[:, df.columns.str.contains(r'\[€/MWh\]', case=False)]
    price_columns.columns = price_columns.columns.str[:-23]  # Clean column names
    price_columns['year'], price_columns['month'], price_columns['hour'] = df['year'], df['month'], df['hour']

    return generation_columns, price_columns, df.sample(frac=frac, random_state=42)

def plot_heatmap_time_of_day_to_month(df, title, hour_column='hour', month_column='month',value_column='Total (grid load) [MWh] Original resolutions', figsize=(10, 8)):
    """
    Function to plot a heatmap for Time-of-Day to Month.

    Parameters:
    - df (pd.DataFrame): DataFrame containing the data.
    - title (str): Title for the heatmap.
    - hour_column (str): The name of the column representing the hour of the day. Default is 'hour'.
    - month_column (str): The name of the column representing the month. Default is 'month'.
    - value_column (str): The name of the column to aggregate (e.g., grid load). Default is 'Total (grid load) [MWh] Original resolutions'.
    - figsize (tuple): Size of the plot. Default is (10, 8).
    """
    heatmap_data = df.pivot_table(
        values='Total (grid load) [MWh] Original resolutions',
        index=hour_column,
        columns=month_column,
        aggfunc='mean'
    )
    heatmap_data = heatmap_data.iloc[::-1]  # Reverse the rows to match typical heatmap format
    plt.figure(figsize=figsize)
    sns.heatmap(heatmap_data, annot=False, cmap='YlOrRd', cbar_kws={"label": value_column})
    plt.xlabel('Month')
    plt.ylabel('Hour of the Day')
    plt.title(title)
    plt.show()
def plot_dimensionality_reduction(data, colorby, method='PCA2D', cmap='viridis', drop_colorby=False, pointsize=1.0):
    """
    Parameters:
    - data (pd.DataFrame): DataFrame containing the data to be reduced.
    - colorby (str): Column to color data points by (e.g., 'year', 'month', 'hour').
    - method (str): Dimensionality reduction method ('PCA2D', 'PCA3D', 'UMAP', 'PaCMAP', 'TriMap', 't-SNE').
    - cmap (str): Colormap to use for visualizing colorby variable.
    - frac (float): Fraction of data to sample for visualization.
    - drop_colorby (bool): Whether to drop 'year', 'month', 'hour' columns before scaling.
    - pointsize (int): Size of scatter plot points.
    """
    # Sample the data

    scaled_data = StandardScaler().fit_transform(data)
    if drop_colorby:
        scaled_data = StandardScaler().fit_transform(data.drop(columns=["year", "month", "hour", "Cluster"]))
    # Initialize the plot
    if method == 'PCA2D':
        plt.figure(figsize=(10, 8))
        pca = PCA(n_components=2)
        components = pca.fit_transform(scaled_data)
        component_df = pd.DataFrame(data=components, columns=['PC1', 'PC2'])
        sc = plt.scatter(component_df['PC1'], component_df['PC2'], c=data[colorby], cmap=cmap, s=pointsize)
        plt.xlabel('Principal Component 1')
        plt.ylabel('Principal Component 2')
        plt.title('2D PCA Visualization')

    elif method == 'PCA3D':
        pca = PCA(n_components=3)
        components = pca.fit_transform(scaled_data)
        component_df = pd.DataFrame(data=components, columns=['PC1', 'PC2', 'PC3'])
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection='3d')
        sc = ax.scatter(
            component_df['PC1'],
            component_df['PC2'],
            component_df['PC3'],
            c=data[colorby],
            cmap=cmap,
            s=pointsize
        )
        ax.set_xlabel('Principal Component 1')
        ax.set_ylabel('Principal Component 2')
        ax.set_zlabel('Principal Component 3')
        plt.title('3D PCA Visualization')

    elif method == 'UMAP':
        plt.figure(figsize=(10, 8))
        umap_model = umap.UMAP(n_components=2, n_neighbors=15, min_dist=0.3, random_state=42)
        umap_components = umap_model.fit_transform(scaled_data)
        umap_df = pd.DataFrame(data=umap_components, columns=['UMAP1', 'UMAP2'])
        sc = plt.scatter(umap_df['UMAP1'], umap_df['UMAP2'], c=data[colorby], cmap=cmap, s=pointsize)
        plt.xlabel('UMAP Component 1')
        plt.ylabel('UMAP Component 2')
        plt.title('2D UMAP Visualization')
    # PaCMAP
    elif method == 'PaCMAP':
        pacmap_model = pacmap.PaCMAP(n_components=2, n_neighbors=10, MN_ratio=0.5, FP_ratio=2.0, random_state=42)
        pacmap_components = pacmap_model.fit_transform(scaled_data)
        pacmap_df = pd.DataFrame(data=pacmap_components, columns=['PaCMAP1', 'PaCMAP2'])
        sc = plt.scatter(pacmap_df['PaCMAP1'], pacmap_df['PaCMAP2'], c=data[colorby], cmap=cmap, s=pointsize)
        plt.xlabel('PaCMAP Component 1')
        plt.ylabel('PaCMAP Component 2')
        plt.title('2D PaCMAP Visualization')

    # TriMap
    elif method == 'TriMap':
        trimap_model = trimap.TRIMAP(n_inliers=10, n_outliers=5, n_random=5, n_dims=2)
        trimap_components = trimap_model.fit_transform(scaled_data)
        trimap_df = pd.DataFrame(data=trimap_components, columns=['TriMap1', 'TriMap2'])
        sc = plt.scatter(trimap_df['TriMap1'], trimap_df['TriMap2'], c=data[colorby], cmap=cmap, s=pointsize)
        plt.xlabel('TriMap Component 1')
        plt.ylabel('TriMap Component 2')
        plt.title('2D TriMap Visualization')

    # t-SNE
    elif method == 't-SNE':
        tsne_model = TSNE(n_components=2, perplexity=30, n_iter=1000, learning_rate=200, random_state=42)
        tsne_components = tsne_model.fit_transform(scaled_data)
        tsne_df = pd.DataFrame(data=tsne_components, columns=['t-SNE1', 't-SNE2'])
        sc = plt.scatter(tsne_df['t-SNE1'], tsne_df['t-SNE2'], c=data[colorby], cmap=cmap, s=pointsize)
        plt.xlabel('t-SNE Component 1')
        plt.ylabel('t-SNE Component 2')
        plt.title('2D t-SNE Visualization')

    # Add other methods as needed...

    # Add the colorbar linked to the scatter plot
    cbar = plt.colorbar(sc)
    cbar.set_label(colorby)

    # Show the plot
    plt.show()
def plot_correlation_matrix_all(df, title, figsize=(14, 12), vmin=-0.5, vmax=1):
    """
    Parameters:
    - df (pd.DataFrame): DataFrame containing the data to be analyzed.
    - title (str): Title for the heatmap.
    - figsize (tuple): Size of the plot. Default is (14, 12).
    - vmin, vmax (float): Min and max values for color scaling. Default is -0.5 and 1.
    """
    numeric_columns = df.select_dtypes(include='number')
    corr_matrix_all = numeric_columns.corr()
    plt.figure(figsize=figsize)
    sns.heatmap(corr_matrix_all, annot=True, cmap='YlOrRd', fmt='.2f', linewidths=0.5, vmin=vmin, vmax=vmax,cbar_kws={"label": "Correlation"})
    plt.title(title)
    plt.show()
from sklearn.cluster import KMeans, DBSCAN, OPTICS, HDBSCAN, AgglomerativeClustering,AffinityPropagation, MeanShift, SpectralClustering
from sklearn.preprocessing import StandardScaler
import pandas as pd

def add_clusters_to_data(data, method='kmeans', n_clusters=4, eps=0.2, min_samples=2, max_eps=10.0):
    """
    Adds a cluster column to the input dataset using various clustering methods.

    Parameters:
    - data (pd.DataFrame): Input dataset (features only).
    - method (str): Clustering method ('kmeans', 'dbscan', or 'optics').
    - n_clusters (int): Number of clusters for K-Means (ignored for DBSCAN and OPTICS).
    - eps (float): Maximum distance between two samples for DBSCAN (ignored for K-Means and OPTICS).
    - min_samples (int): Minimum samples for DBSCAN and OPTICS (ignored for K-Means).
    - max_eps (float): Maximum distance for OPTICS (ignored for K-Means and DBSCAN).
    - random_state (int): Random seed for reproducibility in K-Means.

    Returns:
    - pd.DataFrame: Original data with an additional column for cluster labels.
    """
    # Standardize the data
    scaled_data = StandardScaler().fit_transform(data)

    # Initialize the clustering algorithm
    if method == 'kmeans':
        model = KMeans(n_clusters=n_clusters, random_state=42)
    elif method == 'agglomerative':
        model = AgglomerativeClustering(n_clusters=n_clusters)
    elif method == 'meanshift':
        model = MeanShift()
    elif method == 'dbscan':
        model = DBSCAN(eps=eps, min_samples=min_samples)
    elif method == 'hdbscan':
        model = HDBSCAN(min_samples=min_samples)
    elif method == 'optics':
        model = OPTICS(min_samples=30, max_eps=0.3, xi=0.05, min_cluster_size=50)
    elif method == 'spectral':
        model = SpectralClustering(n_clusters=n_clusters)
    else:
        raise ValueError(f"Unsupported clustering method: {method}")

    # Fit the model and get cluster labels
    model.fit(scaled_data)
    cluster_labels = model.labels_

    # Add the cluster labels as a new column
    data_with_clusters = data.copy()
    data_with_clusters['Cluster'] = cluster_labels


    # Evaluation metrics
    metrics = {}
    metrics['Davies-Bouldin'] = davies_bouldin_score(scaled_data, cluster_labels)

    metrics['Calinski-Harabasz'] = calinski_harabasz_score(scaled_data, cluster_labels)
    metrics['Hartigan'] = model.inertia_ if hasattr(model, 'inertia_') else np.nan
    if len(set(cluster_labels)) > 1:
        centers = model.cluster_centers_ if hasattr(model, 'cluster_centers_') else np.array([scaled_data[cluster_labels == label].mean(axis=0) for label in set(cluster_labels)])
        nearest_neighbor = np.min([np.linalg.norm(center - other_center) for i, center in enumerate(centers) for j, other_center in enumerate(centers) if i != j])
        metrics['VCN'] = nearest_neighbor
        metrics['Silhouette'] = silhouette_score(scaled_data, cluster_labels)


    # Print the metrics to the console
    print("Clustering Evaluation Metrics:")
    for metric, value in metrics.items():
        print(f"{metric}: {value}")
    return data_with_clusters



if __name__ == '__main__':
    generation_data, price_data, all_data = preprocess_energy_data('smard15-24.csv',frac=0.1)
    clustered_data = add_clusters_to_data(all_data, method='spectral')
    plot_dimensionality_reduction(clustered_data, "Cluster", method="UMAP", cmap='viridis', drop_colorby=True,pointsize=1)
    plot_dimensionality_reduction(clustered_data, "Cluster", method="PaCMAP", cmap='viridis', drop_colorby=True,pointsize=1)
    plot_dimensionality_reduction(clustered_data, "Cluster", method="TriMap", cmap='viridis', drop_colorby=True,pointsize=1)
    plot_dimensionality_reduction(clustered_data, "Cluster", method="t-SNE", cmap='viridis', drop_colorby=True,pointsize=1)


    #Past plots
    plot_dimensionality_reduction(all_data, "year", method="PCA2D", cmap='twilight', drop_colorby=True,pointsize=1)
    plot_correlation_matrix_all(generation_data, 'Correlation Matrix of Energy Generation')
    plot_correlation_matrix_all(price_data, 'Correlation Matrix of Energy Prices')
    plot_heatmap_time_of_day_to_month(all_data, 'Heatmap of Time-of-Day to Month')
