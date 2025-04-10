# AI Assignment 1 - House Price India Dataset

# Importing Libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Load Dataset
df = pd.read_csv("House Price India.csv")
df.columns = df.columns.str.strip()
print(df.columns)
df.head()

# -----------------------------
# Univariate Analysis
# -----------------------------

# Distribution of Price
plt.figure(figsize=(10, 6))
sns.histplot(df['Price'], kde=True)
plt.title('Distribution of House Prices')
plt.xlabel('Price (Lakhs)')
plt.ylabel('Frequency')
plt.show()

# Countplot of Area Type
plt.figure(figsize=(8, 5))
sns.countplot(x='area_type', data=df)
plt.title('Count of Area Types')
plt.xticks(rotation=45)
plt.show()

# -----------------------------
# Bivariate Analysis
# -----------------------------

# Price vs Area Type
plt.figure(figsize=(10, 6))
sns.boxplot(x='area_type', y='price', data=df)
plt.title('Price vs Area Type')
plt.xticks(rotation=45)
plt.show()

# Price vs Number of Bathrooms
plt.figure(figsize=(10, 6))
sns.boxplot(x='bath', y='price', data=df)
plt.title('Price vs Number of Bathrooms')
plt.xticks(rotation=0)
plt.show()

# -----------------------------
# Multivariate Analysis
# -----------------------------

# Correlation Heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()

# Pairplot (Optional - for subset only due to size)
# sns.pairplot(df[['price', 'bath', 'balcony']])
# plt.show()

# -----------------------------
# Descriptive Statistics
# -----------------------------

# Descriptive Stats
print("Descriptive Statistics:")
print(df.describe())

# -----------------------------
# Handle Missing Values
# -----------------------------

# Check Missing Values
print("\nMissing Values:")
print(df.isnull().sum())

# Drop rows with missing values
df_cleaned = df.dropna()

print("\nShape before cleaning:", df.shape)
print("Shape after cleaning:", df_cleaned.shape)

# OR Fill missing values (alternative approach shown below)
# df['balcony'].fillna(df['balcony'].median(), inplace=True)
# df['bath'].fillna(df['bath'].median(), inplace=True)
# df['size'].fillna(df['size'].mode()[0], inplace=True)
# df['location'].fillna('Unknown', inplace=True)

# Save cleaned dataset (optional)
df_cleaned.to_csv("Cleaned_Bengaluru_House_Data.csv", index=False)
