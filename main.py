# Import required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_wine

# Task 1: Load and Explore the Dataset
try:
    # Attempt to load dataset
    wine = load_wine()
    df = pd.DataFrame(wine.data, columns=wine.feature_names)
    df['Class'] = wine.target
    
except Exception as e:
    print(f"Error loading dataset: {e}")
    exit()

# Data Cleaning Step
try:
    # Check for missing values
    missing_values = df.isnull().sum().sum()
    if missing_values > 0:
        print(f"\nFound {missing_values} missing values. Cleaning dataset...")
        df = df.dropna()
    else:
        print("\nNo missing values found. Dataset is clean.")
        
except Exception as e:
    print(f"Error during data cleaning: {e}")
    exit()

# Display basic information
print("\nFirst few rows of the dataset:")
print(df.head())

print("\nData Types:")
print(df.dtypes)

# Task 2: Basic Data Analysis
try:
    print("\nBasic Statistics:")
    print(df.describe())

    # Grouping and analysis
    grouped_by_class = df.groupby('Class').mean()
    print("\nGrouped by Class and Mean of Each Feature:")
    print(grouped_by_class)

    print("\nInteresting Patterns:")
    print("Class 0 Average Alcohol Content:", grouped_by_class.loc[0, 'alcohol'])
    print("Class 1 Average Alcohol Content:", grouped_by_class.loc[1, 'alcohol'])
    print("Class 2 Average Alcohol Content:", grouped_by_class.loc[2, 'alcohol'])

except Exception as e:
    print(f"Error during data analysis: {e}")

# Task 3: Data Visualization
sns.set(style="whitegrid")

try:
    # Line Chart (Adapted for non-temporal data)
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=grouped_by_class['alcohol'], marker='o')
    plt.title('Average Alcohol Content Across Wine Classes')
    plt.xlabel('Wine Class')
    plt.ylabel('Average Alcohol Content')
    plt.grid(True)
    plt.show()

    # Bar Chart
    plt.figure(figsize=(10, 6))
    sns.barplot(x=grouped_by_class.index, y=grouped_by_class['alcohol'])
    plt.title('Comparison of Alcohol Content Across Wine Classes')
    plt.xlabel('Wine Class')
    plt.ylabel('Average Alcohol Content')
    plt.show()

    # Histogram
    plt.figure(figsize=(10, 6))
    sns.histplot(df['alcohol'], bins=20, kde=True)
    plt.title('Distribution of Alcohol Content')
    plt.xlabel('Alcohol Content')
    plt.ylabel('Frequency')
    plt.show()

    # Scatter Plot
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=df['alcohol'], y=df['flavanoids'], hue=df['Class'], palette="Set1")
    plt.title('Alcohol Content vs. Flavonoids')
    plt.xlabel('Alcohol Content')
    plt.ylabel('Flavonoids')
    plt.legend(title='Wine Class')
    plt.show()

except Exception as e:
    print(f"Error during visualization: {e}")
