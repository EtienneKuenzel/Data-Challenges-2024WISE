import umap
from sklearn.decomposition import PCA
import seaborn as sns
import pacmap
import trimap
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import pandas as pd


def preprocess_energy_data(file_path='smard15-24.csv'):
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

    return generation_columns, price_columns, df



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

def plot_dimensionality_reduction(data, colorby, method='PCA2D', cmap='viridis', frac=0.1, drop_colorby=False, pointsize=1):
    """
    Parameters:
    - generation_data (pd.DataFrame): DataFrame containing the data to be reduced.
    - colorby (str): 'year', 'month', 'hour'
    - method (str): 'PCA', 'UMAP', 'PaCMAP', 'TriMap', 't-SNE'
    - cmap (str): 'viridis', 'twilight' for looping colorby variable like months, hours
    - frac (float): fraction of data to sample for visualization with 0.1 all plots taken around or under 1min to render
    - drop_colorby: True to drop column by which datapoints are colored
    """
    # Sample the data
    data = data.sample(frac=frac, random_state=42)
    scaled_data = StandardScaler().fit_transform(data)
    if drop_colorby: scaled_data = StandardScaler().fit_transform(data.drop(columns=[colorby]))

    # Initialize the plot
    plt.figure(figsize=(10, 8))

    # PCA
    if method == 'PCA2D':
        pca = PCA(n_components=2)
        components = pca.fit_transform(scaled_data)
        component_df = pd.DataFrame(data=components, columns=['PC1', 'PC2'])
        plt.scatter(component_df['PC1'], component_df['PC2'], c=data[colorby], cmap=cmap, s=pointsize)
        plt.xlabel('Principal Component 1')
        plt.ylabel('Principal Component 2')
        plt.title('2D PCA Visualization')
    if method == 'PCA3D':
        pca = PCA(n_components=3)
        components = pca.fit_transform(scaled_data)
        component_df = pd.DataFrame(data=components, columns=['PC1', 'PC2', 'PC3'])
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(component_df['PC1'], component_df['PC2'], component_df['PC3'], c=generation_data[colorby],cmap=cmap, s=pointsize)
        ax.set_xlabel('Principal Component 1')
        ax.set_ylabel('Principal Component 2')
        ax.set_zlabel('Principal Component 3')
        plt.title('3D PCA Visualization')

    # UMAP
    elif method == 'UMAP':
        umap_model = umap.UMAP(n_components=2, n_neighbors=15, min_dist=0.3, n_epochs=50, metric='euclidean', low_memory=True, random_state=42)
        umap_components = umap_model.fit_transform(scaled_data)
        umap_df = pd.DataFrame(data=umap_components, columns=['UMAP1', 'UMAP2'])
        plt.scatter(umap_df['UMAP1'], umap_df['UMAP2'], c=data[colorby], cmap=cmap, s=10)
        plt.xlabel('UMAP Component 1')
        plt.ylabel('UMAP Component 2')
        plt.title('2D UMAP Visualization')

    # PaCMAP
    elif method == 'PaCMAP':
        pacmap_model = pacmap.PaCMAP(n_components=2, n_neighbors=10, MN_ratio=0.5, FP_ratio=2.0, random_state=42)
        pacmap_components = pacmap_model.fit_transform(scaled_data)
        pacmap_df = pd.DataFrame(data=pacmap_components, columns=['PaCMAP1', 'PaCMAP2'])
        plt.scatter(pacmap_df['PaCMAP1'], pacmap_df['PaCMAP2'], c=data[colorby], cmap=cmap, s=pointsize)
        plt.xlabel('PaCMAP Component 1')
        plt.ylabel('PaCMAP Component 2')
        plt.title('2D PaCMAP Visualization')

    # TriMap
    elif method == 'TriMap':
        trimap_model = trimap.TRIMAP(n_inliers=10, n_outliers=5, n_random=5, n_dims=2)
        trimap_components = trimap_model.fit_transform(scaled_data)
        trimap_df = pd.DataFrame(data=trimap_components, columns=['TriMap1', 'TriMap2'])
        plt.scatter(trimap_df['TriMap1'], trimap_df['TriMap2'], c=data[colorby], cmap=cmap, s=pointsize)
        plt.xlabel('TriMap Component 1')
        plt.ylabel('TriMap Component 2')
        plt.title('2D TriMap Visualization')

    # t-SNE
    elif method == 't-SNE':
        tsne_model = TSNE(n_components=2, perplexity=30, n_iter=1000, learning_rate=200, random_state=42)
        tsne_components = tsne_model.fit_transform(scaled_data)
        tsne_df = pd.DataFrame(data=tsne_components, columns=['t-SNE1', 't-SNE2'])
        plt.scatter(tsne_df['t-SNE1'], tsne_df['t-SNE2'], c=data[colorby], cmap=cmap, s=pointsize)
        plt.xlabel('t-SNE Component 1')
        plt.ylabel('t-SNE Component 2')
        plt.title('2D t-SNE Visualization')

    # Show colorbar
    cbar = plt.colorbar()
    cbar.set_label(colorby)

    # Show the plot
    plt.show()


if __name__ == '__main__':

    generation_data, price_data, all_data = preprocess_energy_data('smard15-24.csv')
    plot_dimensionality_reduction(all_data, 'month', method='TriMap', cmap='twilight', frac=0.1, drop_colorby=False, pointsize=1)

    #die plots müssen nicht adapted werden
    plot_correlation_matrix_all(generation_data, 'Correlation Matrix of Energy Generation')
    plot_correlation_matrix_all(price_data, 'Correlation Matrix of Energy Prices')
    plot_heatmap_time_of_day_to_month(all_data, 'Heatmap of Time-of-Day to Month')
