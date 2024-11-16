
import pandas as pd
# Load the CSV files
df1 = pd.read_csv('Day-ahead_prices_201501010000_202411140000_Quarterhour.csv', delimiter=';')
df2 = pd.read_csv('Actual_generation_201501010000_202411130000_Quarterhour.csv', delimiter=';')
df3 = pd.read_csv('Actual_consumption_201501010000_202411130000_Quarterhour.csv', delimiter=';')

# Merge the DataFrames on the common column(s)
# You can specify 'on' to merge on the overlapping columns
merged_df = pd.merge(df1, df2, on=['Start date'], how='outer')
merged_df = pd.merge(merged_df, df3, on=['Start date'], how='outer')

# Save the merged DataFrame to a new CSV file
merged_df.to_csv('price+gen+con.csv', index=False)


