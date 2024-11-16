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
if __name__ == '__main__':
    # Load the CSV file
    consumption_file = 'Actual_consumption_201501010000_202411130000_Quarterhour.csv'
    consumption_data = pd.read_csv(consumption_file, delimiter=';')
    consumption_data.replace('-', '0', inplace=True)
    # Convert the numerical columns to float
    consumption_data['Total Grid Load[MWh]'] = consumption_data['Total (grid load) [MWh] Original resolutions'].str.replace(',', '').astype(float)
    consumption_data['Residual load [MWh] Original resolutions'] = consumption_data['Residual load [MWh] Original resolutions'].str.replace(',','').astype(float)
    consumption_data['Hydro pumped storage [MWh] Original resolutions'] = consumption_data['Hydro pumped storage [MWh] Original resolutions'].str.replace(',', '').astype(float)
    consumption_data['Start date'] = pd.to_datetime(consumption_data['Start date'], format='%b %d, %Y %I:%M %p')
    consumption_data['month'] = consumption_data['Start date'].dt.month
    consumption_data['day'] = consumption_data['Start date'].dt.day
    consumption_data['hour'] = consumption_data['Start date'].dt.hour
    generation_file = 'Actual_generation_201501010000_202411130000_Quarterhour.csv'
    generation_data = pd.read_csv(generation_file, delimiter=';')
    generation_data.replace('-', '0', inplace=True)
    # Columns to process in the DataFrame
    columns_to_convert = [
        'Biomass [MWh] Original resolutions',
        'Hydropower [MWh] Original resolutions',
        'Wind offshore [MWh] Original resolutions',
        'Wind onshore [MWh] Original resolutions',
        'Photovoltaics [MWh] Original resolutions',
        'Nuclear [MWh] Original resolutions',
        'Lignite [MWh] Original resolutions',
        'Hard coal [MWh] Original resolutions',
        'Fossil gas [MWh] Original resolutions', 'Other renewable [MWh] Original resolutions','Hydro pumped storage [MWh] Original resolutions','Other conventional [MWh] Original resolutions']



    every_nth_datapoint = 100
    colorby = "hour" #"quarter" "year" "hour"


    generation_data['year'] = consumption_data['Start date'].dt.year[::every_nth_datapoint]
    generation_data['quarter'] = consumption_data['Start date'].dt.quarter[::every_nth_datapoint]
    generation_data['hour'] = consumption_data['Start date'].dt.hour[::every_nth_datapoint]
    for column in columns_to_convert:
        generation_data[column] = pd.to_numeric(generation_data[column].str.replace(',', '', regex=False),errors='coerce')
        generation_data[column] = generation_data[column].iloc[::every_nth_datapoint]
    # Fill NaN values with the mean
    generation_data[columns_to_convert] = generation_data[columns_to_convert].fillna(0)#type of data filling


    #3D-PCA
    pca = PCA(n_components=3)
    principal_components = pca.fit_transform(StandardScaler().fit_transform(generation_data[columns_to_convert]))
    # Create a DataFrame with the principal components
    pca_df = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2', 'PC3'])
    # 3D Visualization
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    # Use color mapping based on the quarter and set marker size
    scatter = ax.scatter(pca_df['PC1'], pca_df['PC2'], pca_df['PC3'],c=generation_data[colorby], cmap='viridis', s=10)
    ax.set_xlabel('Principal Component 1')
    ax.set_ylabel('Principal Component 2')
    ax.set_zlabel('Principal Component 3')
    cbar = plt.colorbar(scatter)
    cbar.set_label(colorby)
    plt.title('3D PCA of Energy Data')
    plt.show()

    #2DPCA
    pca = PCA(n_components=2)
    principal_components = pca.fit_transform(StandardScaler().fit_transform(generation_data[columns_to_convert]))
    pca_df = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2'])
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(pca_df['PC1'], pca_df['PC2'],c=generation_data[colorby], cmap='viridis', s=10)
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    cbar = plt.colorbar(scatter)
    cbar.set_label(colorby)
    plt.title('2D PCA of Energy Data')
    plt.show()

    """# UMAP
    umap_model = umap.UMAP(
        n_components=2,
        n_neighbors=5,  # Lower n_neighbors
        min_dist=0.1,
        n_epochs=100,  # Reduce epochs
        metric='euclidean',  # Simpler distance metric
        low_memory=True,  # Approximate NN search
        random_state=42)
    umap_components = umap_model.fit_transform(StandardScaler().fit_transform(generation_data[columns_to_convert]))
    umap_df = pd.DataFrame(data=umap_components, columns=['UMAP1', 'UMAP2'])
    # 2D Visualization of UMAP results
    plt.figure(figsize=(10, 8))
    # Use color mapping based on the 'colorby' column (e.g., year, quarter, etc.)
    scatter = plt.scatter(umap_df['UMAP1'], umap_df['UMAP2'],c=generation_data[colorby], cmap='viridis', s=10)
    plt.xlabel('UMAP Component 1')
    plt.ylabel('UMAP Component 2')
    cbar = plt.colorbar(scatter)
    cbar.set_label(colorby)
    plt.title('2D UMAP of Energy Data')
    plt.show()"""
    """#t-SNE
    tsne_model = TSNE(
        n_components=2,
        perplexity=30,
        n_iter=1000,
        learning_rate=200,
        random_state=42,
        n_jobs=-1
    )
    tsne_components = tsne_model.fit_transform(StandardScaler().fit_transform(generation_data[columns_to_convert]))
    tsne_df = pd.DataFrame(data=tsne_components, columns=['t-SNE1', 't-SNE2'])
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(tsne_df['t-SNE1'], tsne_df['t-SNE2'],c=generation_data[colorby], cmap='viridis', s=10)
    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')
    cbar = plt.colorbar(scatter)
    cbar.set_label(colorby)
    plt.title('2D t-SNE of Energy Generation Data')
    plt.show()"""
    """# PaCMAP 
    pacmap_model = pacmap.PaCMAP(n_components=2, n_neighbors=10, random_state=42)
    pacmap_components = pacmap_model.fit_transform(StandardScaler().fit_transform(generation_data[columns_to_convert]))
    pacmap_df = pd.DataFrame(data=pacmap_components, columns=['PaCMAP1', 'PaCMAP2'])
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(pacmap_df['PaCMAP1'], pacmap_df['PaCMAP2'],c=generation_data[colorby], cmap='viridis', s=10)
    plt.xlabel('PaCMAP Component 1')
    plt.ylabel('PaCMAP Component 2')
    cbar = plt.colorbar(scatter)
    cbar.set_label(colorby)
    plt.title('2D PaCMAP of Energy Generation Data')
    plt.show()"""
    """#TriMAP
    trimap_model = trimap.TRIMAP(n_components=2, n_inliers=10, n_outliers=5, random_state=42)
    trimap_components = trimap_model.fit_transform(StandardScaler().fit_transform(generation_data[columns_to_convert]))
    trimap_df = pd.DataFrame(data=trimap_components, columns=['TriMAP1', 'TriMAP2'])
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(trimap_df['TriMAP1'], trimap_df['TriMAP2'],c=generation_data[colorby], cmap='viridis', s=10)
    plt.xlabel('TriMAP Component 1')
    plt.ylabel('TriMAP Component 2')
    cbar = plt.colorbar(scatter)
    cbar.set_label(colorby)
    plt.title('2D TriMAP of Energy Generation Data')
    plt.show()"""


    #Heatmap of TimeofDay to Month
    heatmap_data = consumption_data.pivot_table(
        values='Total Grid Load[MWh]',
        index='hour',
        columns='month',
        aggfunc='mean')
    heatmap_data = heatmap_data.iloc[::-1]
    sns.heatmap(heatmap_data, annot=False, cmap='YlOrRd',  cbar_kws = {"label": "Total Grid Load [MWh]"})  # annot=True to show values inside the cells
    plt.xlabel('Month')
    plt.ylabel('Hour of the Day')
    plt.title('Heatmap of TimeofDay to Month')
    plt.show()

    #Correlation Matrix of Generation
    df = pd.read_csv('price+gen+con.csv')
    df = df.replace({',': ''}, regex=True)
    df = df.apply(pd.to_numeric, errors='coerce')
    df = df.loc[:, ~df.columns.str.contains('date', case=False)]
    df = df.select_dtypes(exclude=['datetime'])
    df1 = df.loc[:, df.columns.str.contains(r'\[MWh]', case=False)]
    df1.columns = df1.columns.str[:-27]
    numeric_columns = df1.select_dtypes(include='number')
    corr_matrix = numeric_columns.corr()
    sns.heatmap(corr_matrix, annot=True, cmap='YlOrRd', fmt='.2f', linewidths=0.5, vmin=-0.5, vmax=1,  cbar_kws = {"label": "Amount of Correlation"})
    plt.title('Correlation Matrix of Energy Generation')
    plt.show()

    #Correlation Matrix of Prices
    df1 = df.loc[:, df.columns.str.contains(r'\[€/MWh\]', case=False)]
    df1.columns = df1.columns.str[:-23]
    numeric_columns = df1.select_dtypes(include='number')
    corr_matrix = numeric_columns.corr()
    plt.figure(figsize=(14, 12))
    sns.heatmap(corr_matrix, annot=True, cmap='YlOrRd', fmt='.2f', linewidths=0.5, vmin=-0.5, vmax=1, cbar_kws = {"label": "Amount of Correlation"})
    plt.title('Correlation Matrix of Energy Prices')
    plt.show()

    #Correlation Matrix of all Datapoints
    df1.columns = df1.columns.str[:-23]#missing string code
    numeric_columns = df.select_dtypes(include='number')
    corr_matrix = numeric_columns.corr()
    plt.figure(figsize=(14, 12))
    sns.heatmap(corr_matrix, annot=True, cmap='YlOrRd', fmt='.2f', linewidths=0.5, vmin=-0.5, vmax=1, cbar_kws = {"label": "Amount of Correlation"})
    plt.title('Correlation Matrix of all Datapoints')
    plt.show()

