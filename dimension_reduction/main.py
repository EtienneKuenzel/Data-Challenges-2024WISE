import pandas as pd
import numpy as np
import umap.umap_ as umap  # Correct import
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import seaborn as sns

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
    datapoints = 345840/12*2
    # Columns to process in the DataFrame
    columns_to_convert = [
        'Biomass [MWh] Original resolutions',
        'Hydropower [MWh] Original resolutions',
        'Wind offshore [MWh] Original resolutions',
        'Wind onshore [MWh] Original resolutions',
        'Photovoltaics [MWh] Original resolutions',
        'Other renewable [MWh] Original resolutions',
        'Nuclear [MWh] Original resolutions',
        'Lignite [MWh] Original resolutions',
        'Hard coal [MWh] Original resolutions',
        'Fossil gas [MWh] Original resolutions',
        'Hydro pumped storage [MWh] Original resolutions',
        'Other conventional [MWh] Original resolutions'
    ]
    for column in columns_to_convert:
        print(len(generation_data[column]))
        generation_data[column] = generation_data[column].str.replace(',', '').astype(float)
    generation_data['Start date'] = pd.to_datetime(generation_data['Start date'], format='%b %d, %Y %I:%M %p')
    generation_data['month'] = generation_data['Start date'].dt.month
    generation_data['day'] = generation_data['Start date'].dt.dayofyear
    generation_data['hour'] = generation_data['Start date'].dt.hour

    generation_data_long = generation_data.melt(id_vars=['day'],
                                                value_vars=columns_to_convert,
                                                var_name='generation_type',
                                                value_name='generation_amount')

    # Plotting
    plt.figure(figsize=(12, 6))
    sns.kdeplot(data=generation_data_long, x='day', y='generation_amount', hue='generation_type',
                fill=True, cmap='tab20', alpha=0.6)

    plt.title('Density of Electricity Generation by Type and Day of the Year')
    plt.xlabel('Day of the Year')
    plt.ylabel('Generation Amount (MWh)')
    plt.legend(title='Generation Type', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()


    #HeatMap
    heatmap_data = consumption_data.pivot_table(
        values='Total Grid Load[MWh]',
        index='hour',
        columns='month',
        aggfunc='mean'
    )
    heatmap_data = heatmap_data.iloc[::-1]
    sns.heatmap(heatmap_data, annot=False, cmap='YlOrRd',  cbar_kws = {"label": "Total Grid Load [MWh]"})  # annot=True to show values inside the cells
    plt.xlabel('Month')
    plt.ylabel('Hour of the Day')
    plt.show()

