# Step 1: Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Step 2: Load Dataset
# ⚠️ Update the file path if your file is stored elsewhere
df = pd.read_excel(r'C:\Users\prajj\OneDrive\Desktop\Book1.xlsx')

# Step 3: Basic Exploration
print("\n=== Basic Exploration ===")
print(df.head())
print(df.tail())
print(df.info())
print(df.describe())
print("Shape:", df.shape)
print("Columns:", df.columns.tolist())
print("Data Types:\n", df.dtypes)

# Step 4: Data Selection
print("\n=== Data Selection ===")
print(df.loc[0])
print(df.loc[0, 'Name'] if 'Name' in df.columns else "Column 'Name' not found.")
print(df.iloc[0, 0])

# Conditional filtering
if 'Age' in df.columns:
    filtered_df = df.query('Age > 30')
    print("\nFiltered rows where Age > 30:\n", filtered_df)

# Step 5: Data Manipulation
print("\n=== Data Manipulation ===")

if 'Age' in df.columns:
    df_dropped = df.drop(columns=['Age'])
    print("Dropped 'Age' column (temporary):\n", df_dropped.head())

if 'Name' in df.columns:
    df_renamed = df.rename(columns={'Name': 'Full Name'})
    print("Renamed column 'Name' → 'Full Name':\n", df_renamed.head())

if 'Age' in df.columns:
    df_sorted = df.sort_values(by='Age')
    print("Sorted by Age:\n", df_sorted.head())

df_filled = df.fillna(0)
print("Missing values filled with 0.")

df_unique = df.drop_duplicates()
print("Removed duplicates. New shape:", df_unique.shape)

df_replaced = df.replace({'David': 'Davidoff'})
print("Replaced 'David' with 'Davidoff'.")

# Step 6: Grouping & Aggregation
print("\n=== Grouping & Aggregation ===")

if 'City' in df.columns and 'Salary' in df.columns:
    grouped_df = df.groupby('City')['Salary'].sum()
    print("Total Salary by City:\n", grouped_df)

if 'Name' in df.columns:
    agg_df = df.groupby('Name').agg({'Age': 'mean'}) if 'Age' in df.columns else None
    print("\nMean Age by Name:\n", agg_df)

    agg_df2 = df.groupby('Name').agg({
        'Age': ['mean', 'sum'],
        'Salary': ['min', 'max']
    }) if {'Age', 'Salary'}.issubset(df.columns) else None
    print("\nAggregated Statistics by Name:\n", agg_df2)

# Step 7: Data Cleaning
print("\n=== Data Cleaning ===")
df_cleaned = df.dropna()
print("Dropped missing values. Shape:", df_cleaned.shape)
df_filled = df.fillna(0)
print("Filled missing values with 0 again.")

# Step 8: String Operations
print("\n=== String Operations ===")
if 'Name' in df.columns:
    df['Contains_D'] = df['Name'].astype(str).str.contains('D')
    df['Name'] = df['Name'].astype(str).str.strip()
    print(df.head())

# Step 9: Statistical Analysis
print("\n=== Statistical Analysis ===")
if 'Name' in df.columns:
    print(df['Name'].value_counts())
print(df.corr(numeric_only=True))

# Step 10: Add Random Year Column
print("\n=== Adding Random 'Year' Column ===")
df['Year'] = np.random.choice(range(2020, 2026), size=len(df))
print(df.head())

# Step 11: Data Visualization
print("\n=== Data Visualization ===")

if {'Name', 'Salary'}.issubset(df.columns):
    df.plot(x='Name', y='Salary', kind='line', marker='o', figsize=(15, 6))
    plt.title('Line Plot: Name vs Salary')
    plt.show()

if {'Year', 'Salary'}.issubset(df.columns):
    df.plot(x='Year', y='Salary', kind='bar', color='skyblue', figsize=(10, 6))
    plt.title('Bar Plot: Year vs Salary')
    plt.show()

if 'Salary' in df.columns:
    df['Salary'].plot(kind='hist', color='green', edgecolor='black')
    plt.title('Histogram of Salary')
    plt.show()

if {'Salary', 'Year'}.issubset(df.columns):
    df.plot(x='Salary', y='Year', kind='scatter', color='red')
    plt.title('Scatter Plot: Salary vs Year')
    plt.show()

# Step 12: Save Updated Dataset
df.to_excel('Book1_updated.xlsx', index=False)
print("\n✅ All operations completed! Updated file saved as 'Book1_updated.xlsx'")