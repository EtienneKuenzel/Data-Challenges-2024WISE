import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import pandas as pd
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D
import umap
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

import seaborn as sns
import matplotlib.pyplot as plt




def plot_correlation_matrix_all(df, title, figsize=(14, 12), vmin=-0.5, vmax=1):
    """
    Function to plot a correlation matrix heatmap for all numeric data in the given DataFrame.

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

def plot_heatmap_time_of_day_to_month(df, title, hour_column='hour', month_column='month',
                                      value_column='Total (grid load) [MWh] Original resolutions', figsize=(10, 8)):
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
        values=value_column,
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

def plot_2d_pca(df, colorby, cmap='viridis'):
    """
    Function to perform PCA on given data and plot the 2D scatter plot.

    Parameters:
    - df (pd.DataFrame): DataFrame containing the data to be analyzed. Should have a 'year' column for coloring.
    - n_components (int): Number of PCA components to keep. Default is 2.
    - figsize (tuple): Size of the plot. Default is (10, 8).
    - cmap (str): Colormap for scatter plot. Default is 'viridis'.
    """
    # 2D PCA
    pca = PCA(n_components=2)
    principal_components = pca.fit_transform(StandardScaler().fit_transform(df))

    # Create a DataFrame for the principal components
    pca_df = pd.DataFrame(data=principal_components, columns=[f'PC{i + 1}' for i in range(2)])

    # Create 2D scatter plot
    plt.figure(figsize=(10,10))
    scatter = plt.scatter(pca_df['PC1'], pca_df['PC2'], c=df[colorby], cmap=cmap, s=0.1)

    # Set labels
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')

    # Colorbar
    cbar = plt.colorbar(scatter)
    cbar.set_label('Year')
    plt.title('2D PCA of Energy Data')
    plt.show()
def plot_3d_pca(generation_columns, colorby, cmap='viridis'):
    """
    Function to perform PCA on given data and plot the 3D scatter plot.

    Parameters:
    - generation_columns (pd.DataFrame): DataFrame containing the data to be analyzed. Should have a 'year' column for coloring.
    - n_components (int): Number of PCA components to keep. Default is 3.
    - figsize (tuple): Size of the plot. Default is (10, 8).
    - cmap (str): Colormap for scatter plot. Default is 'viridis'.
    """
    principal_components = PCA(n_components=3).fit_transform(StandardScaler().fit_transform(generation_columns))
    pca_df = pd.DataFrame(data=principal_components, columns=[f'PC{i + 1}' for i in range(3)])
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    scatter = ax.scatter(pca_df['PC1'], pca_df['PC2'], pca_df['PC3'], c=generation_columns[colorby], cmap=cmap, s=0.2)

    ax.set_xlabel('Principal Component 1')
    ax.set_ylabel('Principal Component 2')
    ax.set_zlabel('Principal Component 3')
    cbar = plt.colorbar(scatter)
    cbar.set_label('Year')
    plt.title('3D PCA of Energy Data')
    plt.show()


def plot_umap(generation_data, colorby, cmap, frac):
    # Sample the data and ensure it's reassigned
    generation_data = generation_data.sample(frac=frac, random_state=42)
    print("A")

    # UMAP model with adjusted parameters
    umap_model = umap.UMAP(
        n_components=2,
        n_neighbors=15,  # Adjusted for better performance
        min_dist=0.3,  # Larger distance for better separation
        n_epochs=50,  # Reduced epochs
        metric='euclidean',  # Simpler metric
        low_memory=True,  # Use low memory approximation
        random_state=42
    )
    print("A")
    # Apply UMAP transformation
    umap_components = umap_model.fit_transform(StandardScaler().fit_transform(generation_data))
    print("A")
    # Prepare UMAP results into a DataFrame
    umap_df = pd.DataFrame(data=umap_components, columns=['UMAP1', 'UMAP2'])
    print("A")
    # 2D Visualization of UMAP results
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(umap_df['UMAP1'], umap_df['UMAP2'], c=generation_data[colorby], cmap=cmap, s=10)
    plt.xlabel('UMAP Component 1')
    plt.ylabel('UMAP Component 2')

    # Add colorbar for the color mapping
    cbar = plt.colorbar(scatter)
    cbar.set_label(colorby)

    # Title and show the plot
    plt.title('2D UMAP of Energy Data')
    plt.show()

if __name__ == '__main__':
    # Load the combined CSV file
    df = pd.read_csv('price+gen+con.csv')
    df.replace('-', '0', inplace=True)

    df['year'] = pd.to_datetime(df['Start date'], format='%b %d, %Y %I:%M %p').dt.year
    df['month'] = pd.to_datetime(df['Start date'], format='%b %d, %Y %I:%M %p').dt.month
    df['hour'] = pd.to_datetime(df['Start date'], format='%b %d, %Y %I:%M %p').dt.hour
    # Clean up and convert columns as necessary
    df = df.replace({',': ''}, regex=True)  # Remove commas from numeric columns
    df = df.apply(pd.to_numeric, errors='coerce')  # Convert columns to numeric
    df = df.loc[:, ~df.columns.str.contains('date', case=False)]  # Remove date columns
    df = df.drop(['Error'], axis=1).select_dtypes(exclude=['datetime'])    # Extract relevant generation and consumption columns (adjust as needed)
    generation_columns = df.filter(like='[MWh]', axis=1)
    generation_columns.columns = generation_columns.columns.str[:-27]  # Clean column names
    generation_columns['year'], generation_columns['month'], generation_columns['hour'] = df['year'], df['month'], df['hour']

    price_columns = df.loc[:, df.columns.str.contains(r'\[€/MWh\]', case=False)]
    price_columns.columns = price_columns.columns.str[:-23]  # Clean column names
    price_columns['year'], price_columns['month'], price_columns['hour'] = df['year'], df['month'], df['hour']


    #plot_2d_pca(generation_columns, 'Total (grid load) [MWh] Original resolutions'[:-27])
    plot_umap(df, "month", 'twilight', 0.01 )
    #plot_3d_pca()
    #plot_correlation_matrix_all(generation_columns, 'Correlation Matrix of Energy Generation')
    #plot_correlation_matrix_all(price_columns, 'Correlation Matrix of Energy Prices')
    #plot_correlation_matrix_all(df, 'Correlation Matrix of All Data')
    #plot_heatmap_time_of_day_to_month(df, 'Heatmap of Time-of-Day to Month')
