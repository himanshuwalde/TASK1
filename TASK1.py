# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
# Note: Replace 'retail_sales_data.csv' with your actual dataset file
try:
    df = pd.read_csv(r"C:\Users\VICTUS\Downloads\menu.csv")
    print("Data loaded successfully!")
except FileNotFoundError:
    print("Dataset file not found. Please check the file path.")
    
# Display basic information about the dataset
print("\nDataset Info:")
print(df.info())

# Display first 5 rows
print("\nFirst 5 rows:")
print(df.head())

# Check for missing values
print("\nMissing values per column:")
print(df.isnull().sum())

# Handle missing values (example approaches)
# For numerical columns, fill with mean/median
# For categorical columns, fill with mode or 'Unknown'

# Check for duplicates
print("\nNumber of duplicates:", df.duplicated().sum())

# Remove duplicates if any
df = df.drop_duplicates()

# Convert date column to datetime if exists
if 'Date' in df.columns:
    df['Date'] = pd.to_datetime(df['Date'])
    
# Check data types and convert if necessary
print("\nData types:")
print(df.dtypes)

# Generate descriptive statistics for numerical columns
print("\nDescriptive Statistics:")
print(df.describe())

# For categorical columns
categorical_cols = df.select_dtypes(include=['object']).columns
for col in categorical_cols:
    print(f"\nValue counts for {col}:")
    print(df[col].value_counts())

# Time series analysis (if date column exists)
if 'Date' in df.columns:
    # Set date as index
    df_time = df.set_index('Date')
    
    # Resample by month and calculate total sales
    monthly_sales = df_time['Sales'].resample('M').sum()
    
    # Plot monthly sales
    plt.figure(figsize=(12, 6))
    monthly_sales.plot(title='Monthly Sales Trend')
    plt.xlabel('Date')
    plt.ylabel('Total Sales')
    plt.grid(True)
    plt.show()
    
    # Seasonal decomposition (optional)
    from statsmodels.tsa.seasonal import seasonal_decompose
    result = seasonal_decompose(monthly_sales, model='additive')
    result.plot()
    plt.show()

    # Customer analysis (example)
if 'CustomerID' in df.columns:
    # Top customers by sales
    top_customers = df.groupby('CustomerID')['Sales'].sum().sort_values(ascending=False).head(10)
    
    plt.figure(figsize=(10, 5))
    top_customers.plot(kind='bar')
    plt.title('Top 10 Customers by Sales')
    plt.ylabel('Total Sales')
    plt.show()

# Product analysis (example)
if 'Product' in df.columns:
    # Top selling products
    top_products = df['Product'].value_counts().head(10)
    
    plt.figure(figsize=(10, 5))
    top_products.plot(kind='bar')
    plt.title('Top 10 Selling Products')
    plt.ylabel('Number of Sales')
    plt.show()

 # Correlation heatmap (for numerical columns)
numerical_cols = df.select_dtypes(include=[np.number]).columns
if len(numerical_cols) > 1:
    plt.figure(figsize=(10, 8))
    sns.heatmap(df[numerical_cols].corr(), annot=True, cmap='coolwarm')
    plt.title('Correlation Heatmap')
    plt.show()

# Sales distribution
if 'Sales' in df.columns:
    plt.figure(figsize=(10, 6))
    sns.histplot(df['Sales'], bins=30, kde=True)
    plt.title('Sales Distribution')
    plt.show()