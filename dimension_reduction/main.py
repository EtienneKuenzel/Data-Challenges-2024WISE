import umap
import seaborn as sns
import pacmap
import trimap
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.cluster import KMeans, DBSCAN, OPTICS, HDBSCAN, AgglomerativeClustering,AffinityPropagation, MeanShift, SpectralClustering, Birch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
import matplotlib.dates as mdates
import imageio
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler


def LSTm():
    df = pd.read_csv("smard18-24.csv", delimiter=",", parse_dates=["Start date"], decimal=".", thousands=",")
    df.rename(columns={"Germany/Luxembourg [€/MWh] Calculated resolutions": "DE_Price"}, inplace=True)
    df = df.dropna(subset=["DE_Price"])

    # Apply rolling mean smoothing
    window_size = 24 * 4 * 7  # 1 week rolling average
    df["DE_Price"] = df["DE_Price"].rolling(window=window_size, min_periods=1).mean()

    # Normalize Data
    scaler = MinMaxScaler(feature_range=(-1, 1))
    df["DE_Price"] = scaler.fit_transform(df["DE_Price"].values.reshape(-1, 1))

    # Split Data
    train_size = int(len(df) * 0.75)
    train_data = df["DE_Price"].values[:train_size]
    test_data = df["DE_Price"].values[train_size:]


    # Convert to sequences
    def create_sequences(data, seq_length=24):
        sequences, labels = [], []
        for i in range(len(data) - seq_length):
            sequences.append(data[i:i + seq_length])
            labels.append(data[i + seq_length])
        return torch.tensor(sequences, dtype=torch.float32), torch.tensor(labels, dtype=torch.float32)


    seq_length = 24
    X_train, y_train = create_sequences(train_data, seq_length)
    X_test, y_test = create_sequences(test_data, seq_length)


    # Define LSTM Model
    class LSTMModel(nn.Module):
        def __init__(self):
            super(LSTMModel, self).__init__()
            self.lstm = nn.LSTM(input_size=1, hidden_size=50, num_layers=2, batch_first=True)
            self.fc = nn.Linear(50, 1)
        def forward(self, x):
            x, _ = self.lstm(x.unsqueeze(-1))
            x = self.fc(x[:, -1])
            return x


    # Training
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LSTMModel().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    X_train, y_train = X_train.to(device), y_train.to(device)
    X_test, y_test = X_test.to(device), y_test.to(device)

    epochs = 50
    filenames = []
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        output = model(X_train)
        loss = criterion(output.squeeze(), y_train)
        loss.backward()
        optimizer.step()
        print(f"Epoch {epoch + 1}, Loss: {loss.item():.4f}")

        # Generate Predictions at Each Epoch
        model.eval()
        with torch.no_grad():
            predictions = model(X_test).cpu().numpy()
        predictions = scaler.inverse_transform(predictions)
        y_test_np = scaler.inverse_transform(y_test.cpu().numpy().reshape(-1, 1))

        # Plot results
        plt.figure(figsize=(12, 6))
        plt.plot(df["Start date"].values[train_size + seq_length:], y_test_np, label="Actual Price", color="blue")
        plt.plot(df["Start date"].values[train_size + seq_length:], predictions, label="Predicted Price", color="red")
        plt.xlabel("Time")
        plt.ylabel("Price (€/MWh)")
        plt.title(f"Epoch {epoch + 1}: Electricity Price Prediction")
        plt.legend()
        plt.grid(True)

        # Save frame
        filename = f"frame_{epoch + 1}.png"
        plt.savefig(filename)
        filenames.append(filename)
        plt.close()

    # Create GIF
    gif_filename = "forecast_evolution.gif"
    with imageio.get_writer(gif_filename, mode='I', duration=0.5) as writer:
        for filename in filenames:
            image = imageio.imread(filename)
            writer.append_data(image)

    print(f"GIF saved as {gif_filename}")

def detrend_series(x, y):
    # Implement detrending here (e.g., linear detrending)
    return y - np.polyval(np.polyfit(x.astype(int), y, 1), x.astype(int))
# Function for piecewise approximation (linear regression)
def piecewise_linear_approximation(x, y, segment_length):
    """Divide data into segments and apply linear regression for each segment."""
    segments_x = [x[i:i + segment_length] for i in range(0, len(x), segment_length)]
    segments_y = [y[i:i + segment_length] for i in range(0, len(y), segment_length)]

    linear_segments_x = []
    linear_segments_y = []

    for seg_x, seg_y in zip(segments_x, segments_y):
        model = LinearRegression()
        model.fit(seg_x.reshape(-1, 1), seg_y)
        predicted_y = model.predict(seg_x.reshape(-1, 1))
        linear_segments_x.append(seg_x)
        linear_segments_y.append(predicted_y)

    return linear_segments_x, linear_segments_y

def linear_approximations(file_1="smard20.csv",file_2="smard23.csv", seg_size=200):
    df1 = pd.read_csv(file_1, delimiter=",", parse_dates=["Start date"], decimal=".", thousands=",")
    df2 = pd.read_csv(file_2, delimiter=",", parse_dates=["Start date"], decimal=".", thousands=",")

    # Rename columns
    df1.rename(columns={"Germany/Luxembourg [€/MWh] Calculated resolutions": "DE_Price_1"}, inplace=True)
    df2.rename(columns={"Germany/Luxembourg [€/MWh] Calculated resolutions": "DE_Price_2"}, inplace=True)

    # Drop missing values
    df1 = df1.dropna(subset=["DE_Price_1"])
    df2 = df2.dropna(subset=["DE_Price_2"])

    # Adjust start date to the same reference year
    df1["Start date"] = df1["Start date"].apply(lambda x: x.replace(year=2000))
    df2["Start date"] = df2["Start date"].apply(lambda x: x.replace(year=2000))

    # Apply a moving average
    window_size = 24*4*7
    df1["DE_Price_1"] = df1["DE_Price_1"].rolling(window=window_size, min_periods=1).mean()
    df2["DE_Price_2"] = df2["DE_Price_2"].rolling(window=window_size, min_periods=1).mean()

    # Standardize the data
    """df1["DE_Price_1"] = df1["DE_Price_1"] - df1["DE_Price_1"].mean()
    df2["DE_Price_2"] = df2["DE_Price_2"] - df2["DE_Price_2"].mean()
    
    df1["DE_Price_1"] = df1["DE_Price_1"]/df1["DE_Price_1"].std()
    df2["DE_Price_2"] = df2["DE_Price_2"]/df2["DE_Price_2"].std()
    
    # Detrend the data
    df1["DE_Price_1"] = detrend_series(df1["Start date"], df1["DE_Price_1"])
    df2["DE_Price_2"] = detrend_series(df2["Start date"], df2["DE_Price_2"])"""

    # Merge the data
    df_merged = df1.merge(df2, on="Start date", how="inner")

    # Convert dates to numeric values for linear regression
    dates_as_numeric = mdates.date2num(df_merged["Start date"])

    # Apply piecewise linear approximation
    segment_length = seg_size  # Choose an appropriate segment length
    linear_segments_x_1, linear_segments_y_1 = piecewise_linear_approximation(dates_as_numeric,
                                                                              df_merged["DE_Price_1"].values,
                                                                              segment_length)
    linear_segments_x_2, linear_segments_y_2 = piecewise_linear_approximation(dates_as_numeric,
                                                                              df_merged["DE_Price_2"].values,
                                                                              segment_length)

    # Plot original data and linear piecewise approximations
    plt.figure(figsize=(12, 6))
    #plt.plot(df1["Start date"], df1["DE_Price_1"], label="2020 Price of Electricity (€/MWh)", color="blue", linewidth=1)
    #plt.plot(df2["Start date"], df2["DE_Price_2"], label="2023 Price of Electricity (€/MWh)", color="red", linewidth=1)

    # Plot each piecewise linear segment
    for seg_x_1, seg_y_1, seg_x_2, seg_y_2 in zip(linear_segments_x_1, linear_segments_y_1, linear_segments_x_2,
                                                  linear_segments_y_2):
        # Convert numeric dates back to datetime for proper plotting
        seg_x_1_dates = mdates.num2date(seg_x_1)
        seg_x_2_dates = mdates.num2date(seg_x_2)

        # Plot the linear regression lines for each segment
        plt.plot(seg_x_1_dates, seg_y_1, color="blue", alpha=1, linewidth=1)
        plt.plot(seg_x_2_dates, seg_y_2, color="red", alpha=1, linewidth=1)

    # Formatting the plot
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%m"))  # Show only Month
    plt.xlabel("Time")
    plt.ylabel("Price (€/MWh)")
    plt.ylim(-200, 200)
    plt.title("Electricity Price in Germany (Piecewise Linear Approximation)")
    plt.legend()
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.show()

def time_series(file_path1 = "smard20.csv", file_path2 = "smard23.csv"):

    df1 = pd.read_csv(file_path1, delimiter=",", parse_dates=["Start date"], decimal=".", thousands=",")
    df2 = pd.read_csv(file_path2, delimiter=",", parse_dates=["Start date"], decimal=".", thousands=",")

    # Rename columns
    df1.rename(columns={"Germany/Luxembourg [€/MWh] Calculated resolutions": "DE_Price_1"}, inplace=True)
    df2.rename(columns={"Germany/Luxembourg [€/MWh] Calculated resolutions": "DE_Price_2"}, inplace=True)

    # Drop missing values
    df1 = df1.dropna(subset=["DE_Price_1"])
    df2 = df2.dropna(subset=["DE_Price_2"])

    # Adjust start date to the same reference year
    df1["Start date"] = df1["Start date"].apply(lambda x: x.replace(year=2000))
    df2["Start date"] = df2["Start date"].apply(lambda x: x.replace(year=2000))
    # Apply a moving average
    window_size = 24*4*7
    df1["DE_Price_1"] = df1["DE_Price_1"].rolling(window=window_size, min_periods=1).mean()
    df2["DE_Price_2"] = df2["DE_Price_2"].rolling(window=window_size, min_periods=1).mean()

    df1["DE_Price_1"] = df1["DE_Price_1"] - df1["DE_Price_1"].mean()
    df2["DE_Price_2"] = df2["DE_Price_2"] - df2["DE_Price_2"].mean()

    df1["DE_Price_1"] = df1["DE_Price_1"]/df1["DE_Price_1"].std()
    df2["DE_Price_2"] = df2["DE_Price_2"]/df2["DE_Price_2"].std()

    df1["DE_Price_1"] = detrend_series(df1["Start date"], df1["DE_Price_1"])
    df2["DE_Price_2"] = detrend_series(df2["Start date"], df2["DE_Price_2"])
    df_merged = df1.merge(df2, on="Start date", how="inner")
    euclidean_distance = np.sqrt(np.sum((df_merged["DE_Price_1"] - df_merged["DE_Price_2"]) ** 2))
    print(f"Euclidean Distance: {euclidean_distance}")
    df_merged = df1.merge(df2, on="Start date", how="inner")
    distance, path = fastdtw([(x,) for x in df_merged["DE_Price_1"].values], [(x,) for x in df_merged["DE_Price_2"].values], dist=euclidean)
    print(f"DTW Distance: {distance}")
    indices_1, indices_2 = zip(*path)
    indices_1 = indices_1[::100]
    indices_2 = indices_2[::100]


    # Plot original and smoothed data
    plt.figure(figsize=(12, 6))
    plt.plot(df1["Start date"], df1["DE_Price_1"], label="2020 Price of Electricity (€/MWh)", color="blue", linewidth=1)
    plt.plot(df2["Start date"], df2["DE_Price_2"], label="2023 Price of Electricity (€/MWh)", color="red", linewidth=1)
    for i, j in zip(indices_1, indices_2):
        plt.plot([df_merged["Start date"].iloc[i], df_merged["Start date"].iloc[j]],
                 [df_merged["DE_Price_1"].iloc[i], df_merged["DE_Price_2"].iloc[j]],
                 color="black", linestyle="--", linewidth=0.2)

    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%m"))  # Show only Month

    plt.xlabel("Time")
    plt.ylabel("Price (€/MWh)")
    plt.ylim(-200,200)
    plt.title("Electricity Price in Germany")
    plt.legend()
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.show()



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
        scaled_data = StandardScaler().fit_transform(data.drop(columns=["Cluster"])) #"year", "month", "hour"
    # Initialize the plot
    if method == 'PCA2D':
        plt.figure(figsize=(10, 8))
        pca = PCA(n_components=2)
        components = pca.fit_transform(scaled_data)
        df = pd.DataFrame(data=components, columns=['PC1', 'PC2'])
        sc = plt.scatter(df['PC1'], df['PC2'], c=data[colorby], cmap=cmap, s=pointsize)
        plt.xlabel('Principal Component 1')
        plt.ylabel('Principal Component 2')
        plt.title('2D PCA Visualization')

    elif method == 'PCA3D':
        pca = PCA(n_components=3)
        components = pca.fit_transform(scaled_data)
        df = pd.DataFrame(data=components, columns=['PC1', 'PC2', 'PC3'])
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection='3d')
        sc = ax.scatter(
            df['PC1'],
            df['PC2'],
            df['PC3'],
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
        df = pd.DataFrame(data=umap_components, columns=['UMAP1', 'UMAP2'])
        sc = plt.scatter(df['UMAP1'], df['UMAP2'], c=data[colorby], cmap=cmap, s=pointsize)
        plt.xlabel('UMAP Component 1')
        plt.ylabel('UMAP Component 2')
        plt.title('2D UMAP Visualization')
    # PaCMAP
    elif method == 'PaCMAP':
        pacmap_model = pacmap.PaCMAP(n_components=2, n_neighbors=10, MN_ratio=0.5, FP_ratio=2.0, random_state=42)
        pacmap_components = pacmap_model.fit_transform(scaled_data)
        df = pd.DataFrame(data=pacmap_components, columns=['PaCMAP1', 'PaCMAP2'])
        sc = plt.scatter(df['PaCMAP1'], df['PaCMAP2'], c=data[colorby], cmap=cmap, s=pointsize)
        plt.xlabel('PaCMAP Component 1')
        plt.ylabel('PaCMAP Component 2')
        plt.title('2D PaCMAP Visualization')

    # TriMap
    elif method == 'TriMap':
        trimap_model = trimap.TRIMAP(n_inliers=10, n_outliers=5, n_random=5, n_dims=2)
        trimap_components = trimap_model.fit_transform(scaled_data)
        df = pd.DataFrame(data=trimap_components, columns=['TriMap1', 'TriMap2'])
        sc = plt.scatter(df['TriMap1'], df['TriMap2'], c=data[colorby], cmap=cmap, s=pointsize)
        plt.xlabel('TriMap Component 1')
        plt.ylabel('TriMap Component 2')
        plt.title('2D TriMap Visualization')

    # t-SNE
    elif method == 't-SNE':
        tsne_model = TSNE(n_components=2, perplexity=30, n_iter=1000, learning_rate=200, random_state=42)
        tsne_components = tsne_model.fit_transform(scaled_data)
        df = pd.DataFrame(data=tsne_components, columns=['t-SNE1', 't-SNE2'])
        sc = plt.scatter(df['t-SNE1'], df['t-SNE2'], c=data[colorby], cmap=cmap, s=pointsize)
        plt.xlabel('t-SNE Component 1')
        plt.ylabel('t-SNE Component 2')
        plt.title('2D t-SNE Visualization')

    # Add other methods as needed...

    # Add the colorbar linked to the scatter plot
    cbar = plt.colorbar(sc)
    cbar.set_label(colorby)
    plt.show()
    return df
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
    elif method == 'affinity':
        model = AffinityPropagation()
    elif method == 'birch':
        model = Birch(n_clusters=n_clusters)
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
    #metrics['Davies-Bouldin'] = davies_bouldin_score(scaled_data, cluster_labels)

    #metrics['Calinski-Harabasz'] = calinski_harabasz_score(scaled_data, cluster_labels)
    #metrics['Hartigan'] = model.inertia_ if hasattr(model, 'inertia_') else np.nan
    if len(set(cluster_labels)) > 1:
        centers = model.cluster_centers_ if hasattr(model, 'cluster_centers_') else np.array([scaled_data[cluster_labels == label].mean(axis=0) for label in set(cluster_labels)])
        nearest_neighbor = np.min([np.linalg.norm(center - other_center) for i, center in enumerate(centers) for j, other_center in enumerate(centers) if i != j])
        #metrics['VCN'] = nearest_neighbor
        metrics['Silhouette'] = silhouette_score(scaled_data, cluster_labels)


    # Print the metrics to the console
    print("Clustering Evaluation Metrics:")
    for metric, value in metrics.items():
        print(f"{metric}: {value}")
    return data_with_clusters
def plot_3d_to_2d(data, colormap='viridis', title='', color_label='Clusters'):
    """
    Plots 3D data in 2D, using the third column to color the points.

    Parameters:
        data (np.ndarray): 3D data as a NumPy array of shape (n, 3),
                           where columns represent x, y, and z respectively.
        colormap (str): Matplotlib colormap to use for coloring the points.
        title (str): Title of the plot.
        color_label (str): Label for the colorbar.
    """
    if data.shape[1] != 3:
        raise ValueError("Input data must have exactly three columns (x, y, z).")

    x = data["PaCMAP1"]
    y = data["PaCMAP2"]
    z = data["Cluster"]

    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(x, y, c=z, cmap=colormap, s=2)
    plt.colorbar(scatter, label=color_label)
    plt.title(title)
    plt.xlabel("X-axis")
    plt.ylabel("Y-axis")
    plt.tight_layout()
    plt.show()
import pandas as pd

def add_change_bins_to_csv(input_csv, output_csv, bins, labels):
    """
    Adds a new column to the CSV file indicating the change in
    'Total (grid load) [MWh] Original resolutions' relative to the previous row,
    classified into bins. Prints the percentage of data falling into each bin.

    Args:
        input_csv (str): Path to the input CSV file.
        output_csv (str): Path to save the modified CSV file.
        bins (list): List of bin edges for classification.
        labels (list): Labels for the bins.

    Returns:
        None
    """
    # Load the CSV file into a DataFrame
    df = pd.read_csv(input_csv)
    df = df.replace({',': ''}, regex=True)
    df = df.apply(pd.to_numeric, errors='coerce')
    # Ensure the column exists in the DataFrame
    column_name = "Total (grid load) [MWh] Original resolutions"
    if column_name not in df.columns:
        raise ValueError(f"Column '{column_name}' not found in the CSV file.")

    # Calculate the change compared to the previous row
    df['Grid Load Change'] = df[column_name].diff()

    # Classify the changes into bins
    df['Grid Load Change Bin'] = pd.cut(df['Grid Load Change'], bins=bins, labels=labels, include_lowest=True)

    # Calculate and print the percentage of data in each bin
    bin_counts = df['Grid Load Change Bin'].value_counts(normalize=True) * 100
    print("Percentage of data in each bin:")
    for label, percentage in bin_counts.items():
        print(f"{label}: {percentage:.2f}%")

    # Save the modified DataFrame to a new CSV file
    df.to_csv(output_csv, index=False)
    print(f"Modified CSV saved to {output_csv}")
def load_and_preprocess_data(file_path, target_column, sample_fraction=0.1, test_size=0.8, random_state=42):
    df = pd.read_csv(file_path)
    label_encoder = LabelEncoder()
    df = df.sample(frac=sample_fraction, random_state=random_state)
    df[target_column] = label_encoder.fit_transform(df[target_column])

    numerical_features = df.select_dtypes(include=[np.number])
    df[numerical_features.columns] = numerical_features.fillna(numerical_features.mean())
    df['Start date'] = df['Start date'].fillna('Unknown')
    df['End date'] = df['End date'].fillna('Unknown')
    df['DE/AT/LU [€/MWh] Calculated resolutions'] = df['DE/AT/LU [€/MWh] Calculated resolutions'].fillna(0)

    features = df.drop(columns=["Start date", "End date", target_column])
    X = features.select_dtypes(include=[np.number])
    y = df[target_column]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test


def evaluate_model(model, X_train, y_train, X_test, y_test, description):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted')
    accuracy = accuracy_score(y_test, y_pred)
    print(f"{description}:")
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall: {recall:.4f}")
    print(f"  F1 Score: {f1:.4f}")


def train_and_evaluate_models(X_train_scaled, X_test_scaled, y_train, y_test):
    classifiers = {
        "Support Vector Machine": SVC(kernel="rbf", probability=True, random_state=42),
        "Random Forest": RandomForestClassifier(random_state=42),
        "K-Nearest Neighbors": KNeighborsClassifier(),
    }

    print("Original Dataset:")
    for name, model in classifiers.items():
        evaluate_model(model, X_train_scaled, y_train, X_test_scaled, y_test, name)

    samplers = {
        "Undersampling": RandomUnderSampler(random_state=42),
        "Oversampling (SMOTE)": SMOTE(random_state=42),
    }

    for sampler_name, sampler in samplers.items():
        print(f"\nAfter {sampler_name}:")
        X_res, y_res = sampler.fit_resample(X_train_scaled, y_train)
        for name, model in classifiers.items():
            evaluate_model(model, X_res, y_res, X_test_scaled, y_test, f"{sampler_name} - {name}")





if __name__ == '__main__':
    pass
    # Function Call
    #file_path = "smard18-24+bin.csv"
    #target_column = "Grid Load Change Bin"
    #X_train_scaled, X_test_scaled, y_train, y_test = load_and_preprocess_data(file_path, target_column)
    #train_and_evaluate_models(X_train_scaled, X_test_scaled, y_train, y_test)

    # Example usage:
    # Define the bins and labels for classification
    #example_bins = [-float('inf'), -100, -10, 10, 100, float('inf')]
    #example_labels = ['Large Decrease', 'Small Decrease', 'No Change', 'Small Increase', 'Large Increase']

    #add_change_bins_to_csv('smard18-24.csv', 'output.csv', example_bins, example_labels)
    #generation_data, price_data, all_data = preprocess_energy_data('smard18-24.csv',frac=0.1)
    #a = plot_dimensionality_reduction(all_data, "year", method="PaCMAP", cmap='viridis', drop_colorby=False,pointsize=2)


    #clustered_data = add_clusters_to_data(a, method='birch', n_clusters=6)
    #plot_3d_to_2d(clustered_data)
    #Past plots
    #plot_dimensionality_reduction(all_data, "year", method="PCA2D", cmap='twilight', drop_colorby=True,pointsize=1)
    #plot_correlation_matrix_all(generation_data, 'Correlation Matrix of Energy Generation')
    #plot_correlation_matrix_all(price_data, 'Correlation Matrix of Energy Prices')
    #plot_heatmap_time_of_day_to_month(all_data, 'Heatmap of Time-of-Day to Month')
