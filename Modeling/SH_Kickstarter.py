#!/usr/bin/env python
# coding: utf-8

# # Kickstarter

# ## Data Processing

# In[1]:


import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', None)


# In[2]:


# Load the data from the Excel file
df = pd.read_excel('Kickstarter.xlsx')

# keep just 'failed' and 'successful' for 'state'
df = df[df['state'].isin(['failed','successful'])]

df.head()


# In[3]:


summary_stats = df.describe(include='all')
summary_stats


# In[4]:


# Category Counts
categorical_columns = ['state', 'currency']
for col in categorical_columns:
    print(f'\nFrequency of categories in {col}:\n', df[col].value_counts())


# In[5]:


# Check for null values in each column
null_values = df.isnull().sum()

print(null_values)


# In[6]:


# Handling missing values in 'category'
df['category'] = df['category'].fillna('unknown')


# In[7]:


# Drop rows with any null values
df_cleaned = df.dropna()
null_values = df_cleaned.isnull().sum()
print(null_values)


# In[8]:


# goal amount to USD
if 'goal' in df_cleaned.columns and 'static_usd_rate' in df_cleaned.columns:
    # Create the 'goal_usd' feature
    df_cleaned['goal_usd'] = df_cleaned['goal'] * df_cleaned['static_usd_rate']
else:
    print("Columns 'goal' and/or 'static_usd_rate' do not exist in DataFrame")

df_cleaned.head()


# In[9]:


from sklearn.preprocessing import MinMaxScaler

# Normalizing numerical variables
scaler = MinMaxScaler()
numerical_cols = ['goal', 'pledged', 'backers_count', 'usd_pledged']
df_cleaned[numerical_cols] = scaler.fit_transform(df_cleaned[numerical_cols])


# In[10]:


# Handling date and time columns
#datetime_cols = ['deadline', 'state_changed_at', 'created_at', 'launched_at']
#for col in datetime_cols:
#    df_cleaned[col + '_year'] = df_cleaned[col].dt.year
#    df_cleaned[col + '_month'] = df_cleaned[col].dt.month
#    df_cleaned[col + '_day'] = df_cleaned[col].dt.day
#    df_cleaned.drop(col, axis=1, inplace=True)


# In[11]:


df_cleaned.head()


# #### feature engineering

# In[12]:


df_cleaned.columns


# In[13]:


# Convert goal amount to USD
df_cleaned['goal_usd'] = df_cleaned['goal'] * df_cleaned['static_usd_rate']

# Grouping and aggregating based on year and category
goal_category_yr = df_cleaned.groupby(['created_at_yr', 'category'])['goal_usd'].agg(['mean', 'max', 'min', np.std]).reset_index()
goal_category_yr = goal_category_yr.rename(columns={'mean': 'goal_usd_cat_mean', 'max': 'goal_usd_cat_max', 'min': 'goal_usd_cat_min', 'std': 'goal_usd_cat_std'})

# Merging the aggregated data back
df_cleaned = df_cleaned.merge(goal_category_yr, on=['created_at_yr', 'category'], how='left')

# Check if the new columns are added
print("Columns after merging:", df_cleaned.columns)

# Filling missing values in aggregated columns
try:
    df_cleaned['goal_usd_cat_mean'] = df_cleaned['goal_usd_cat_mean'].transform(lambda x: x.fillna(x.mean()))
    df_cleaned['goal_usd_cat_max'] = df_cleaned['goal_usd_cat_max'].transform(lambda x: x.fillna(x.max()))
    df_cleaned['goal_usd_cat_min'] = df_cleaned['goal_usd_cat_min'].transform(lambda x: x.fillna(x.min()))
    df_cleaned['goal_usd_cat_std'] = df_cleaned['goal_usd_cat_std'].transform(lambda x: x.fillna(x.std()))
except KeyError as e:
    print(f"Error: {e}. This column does not exist in df_cleaned.")

df_cleaned.head()


# ## Supervised learning - classification model

# In[33]:


# non-numeric columns
non_numeric_columns = df_cleaned.select_dtypes(include=['object']).columns.tolist()
print(non_numeric_columns)


# In[34]:


df_cleaned.columns


# ### Logistic Regression

# In[35]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer

# Encode the target variable 'state'
label_encoder = LabelEncoder()
df_cleaned['state'] = label_encoder.fit_transform(df_cleaned['state'])

# Define a list of variables to be used
features_to_use = ['pledged', 'goal_usd','backers_count', 'country', 'currency', 'category', 
                   'staff_pick', 'created_at_weekday', 'launched_at_weekday', 'create_to_launch_days', 
                   'launch_to_deadline_days', 'goal_usd_cat_mean', 'goal_usd_cat_max', 
                   'goal_usd_cat_min', 'goal_usd_cat_std']

# Identify non-numeric features from features_to_use for one-hot encoding
non_numeric_features = ['country', 'currency', 'category', 'staff_pick', 'created_at_weekday', 'launched_at_weekday']

# One-hot encoding for non-numeric features
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), non_numeric_features)], remainder='passthrough')

# Select only the features in 'features_to_use' for X
X = ct.fit_transform(df_cleaned[features_to_use])
y = df_cleaned['state']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

# Logistic Regression model
logreg = LogisticRegression(max_iter=1000)
logreg.fit(X_train, y_train)

# Predict and evaluate the model
y_pred = logreg.predict(X_test)
print("Classification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Accuracy Score:", accuracy_score(y_test, y_pred))


# ### RandomForest

# In[36]:


from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer

# Encode the target variable 'state'
label_encoder = LabelEncoder()
df_cleaned['state'] = label_encoder.fit_transform(df_cleaned['state'])


features_to_use = ['pledged', 'goal_usd', 'backers_count', 'country', 'currency', 'category', 
                   'staff_pick', 'created_at_weekday', 'launched_at_weekday', 'create_to_launch_days', 
                   'launch_to_deadline_days', 'goal_usd_cat_mean', 'goal_usd_cat_max', 
                   'goal_usd_cat_min', 'goal_usd_cat_std']

# Identify non-numeric features from features_to_use for one-hot encoding
non_numeric_features = ['country', 'currency', 'category', 'staff_pick', 'created_at_weekday', 'launched_at_weekday']

# One-hot encoding for non-numeric features
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), non_numeric_features)], remainder='passthrough')

# Select only the features in 'features_to_use' for X
X = ct.fit_transform(df_cleaned[features_to_use])
y = df_cleaned['state']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

# Random Forest model
rf_model = RandomForestClassifier(n_estimators=100, random_state=0)
rf_model.fit(X_train, y_train)

# Predict and evaluate the model
y_pred = rf_model.predict(X_test)
print("Classification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Accuracy Score:", accuracy_score(y_test, y_pred))


# ### Gradient Boosting

# In[37]:


from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer

# Encode the target variable 'state'
label_encoder = LabelEncoder()
df_cleaned['state'] = label_encoder.fit_transform(df_cleaned['state'])

# Define a list of variables to be used
features_to_use = ['pledged', 'goal_usd', 'backers_count', 'country', 'currency', 'category', 
                   'staff_pick', 'created_at_weekday', 'launched_at_weekday', 'create_to_launch_days', 
                   'launch_to_deadline_days', 'goal_usd_cat_mean', 'goal_usd_cat_max', 
                   'goal_usd_cat_min', 'goal_usd_cat_std']

# Identify non-numeric features from features_to_use for one-hot encoding
non_numeric_features = ['country', 'currency', 'category', 'staff_pick', 'created_at_weekday', 'launched_at_weekday']

# One-hot encoding for non-numeric features
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), non_numeric_features)], remainder='passthrough')

# Select only the features in 'features_to_use' for X
X = ct.fit_transform(df_cleaned[features_to_use])
y = df_cleaned['state']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

# Gradient Boosting model
gb_model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=0)
gb_model.fit(X_train, y_train)

# Predict and evaluate the model
y_pred = gb_model.predict(X_test)
print("Classification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Accuracy Score:", accuracy_score(y_test, y_pred))


# ### Feature Importance

# In[38]:


from sklearn.ensemble import RandomForestClassifier


# Fit Random Forest model to the entire dataset
forest = RandomForestClassifier(n_estimators=100, random_state=0)
forest.fit(X, y)

# Get feature importances
importances = forest.feature_importances_

# Match feature names to their importances
feature_names = ct.get_feature_names_out()
feature_importances = pd.DataFrame({'Feature': feature_names, 'Importance': importances})

# Sort features by importance
feature_importances = feature_importances.sort_values(by='Importance', ascending=False)

# Print each feature and its importance
for index, row in feature_importances.iterrows():
    print(f"{row['Feature']}: {row['Importance']}")


# In[39]:


import matplotlib.pyplot as plt
import seaborn as sns

# Sort features by importance for better visualization
feature_importances_sorted = feature_importances.sort_values(by='Importance', ascending=True)

# Create bar plot
plt.figure(figsize=(10, 8))
sns.barplot(x='Importance', y='Feature', data=feature_importances_sorted)

plt.title('Feature Importances from Random Forest')
plt.xlabel('Importance')
plt.ylabel('Features')
plt.show()


# In[40]:


from sklearn.ensemble import GradientBoostingClassifier

# Train Gradient Boosting Classifier
gb_model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=0)
gb_model.fit(X_train, y_train)

# Get feature importances
feature_importances_raw = gb_model.feature_importances_


# Get feature names after one-hot encoding
transformed_feature_names = ct.get_feature_names_out()

# Map each transformed feature name to the original feature name
original_feature_names_mapping = {}
for orig_feature in features_to_use:
    if orig_feature in non_numeric_features:
        for transformed_feature in transformed_feature_names:
            if transformed_feature.startswith(f'encoder__{orig_feature}'):
                original_feature_names_mapping[transformed_feature] = orig_feature
    else:
        original_feature_names_mapping[orig_feature] = orig_feature

# Initialize a Series to store aggregated importances
original_feature_importances = pd.Series(0, index=features_to_use)

# Iterate over all transformed feature names and their importances
for transformed_name, importance in zip(transformed_feature_names, feature_importances_raw):
    # Check if the transformed name starts with one of the non-numeric features
    for non_num_feature in non_numeric_features:
        if transformed_name.startswith(f'encoder__{non_num_feature}_'):
            original_feature_importances[non_num_feature] += importance
            break
    else:
        if transformed_name in features_to_use:
            original_feature_importances[transformed_name] += importance

# Ensure the importances are normalized to sum to 1
original_feature_importances /= original_feature_importances.sum()

# Sort features by their aggregated importance
feature_importances_sorted = original_feature_importances.sort_values(ascending=True)

# Bar plot
plt.figure(figsize=(12, 10))
sns.barplot(x=feature_importances_sorted.values, y=feature_importances_sorted.index)

plt.title('Aggregated Feature Importances from Gradient Boosting')
plt.xlabel('Importance')
plt.ylabel('Features')
plt.show()


# ## Unsupervised learning - Clustering

# ### Hierarchical clustering

# In[14]:


from sklearn.preprocessing import StandardScaler

# unique values in the 'state' column
print("Unique values in the 'state' column:", df_cleaned['state'].unique())


# In[15]:


# Replace 'successful' and 'failed' with actual labels
df_filtered = df_cleaned[df_cleaned['state'].isin(['successful', 'failed'])]

# Verify if df_filtered has data
if len(df_filtered) == 0:
    print("No data available for the specified 'state' labels.")
    # Exit if no data is available
    exit()
df_filtered['state'].value_counts()


# #### standardizing the features and hierarrchical clustering

# In[16]:


from scipy.cluster.hierarchy import dendrogram, linkage
import matplotlib.pyplot as plt

# Select features for clustering (excluding 'state' and non-numeric features)
features_for_clustering = ['pledged', 'goal_usd', 'backers_count', 'create_to_launch_days', 
                           'launch_to_deadline_days', 'goal_usd_cat_mean', 'goal_usd_cat_max', 
                           'goal_usd_cat_min', 'goal_usd_cat_std']


scaler = StandardScaler()
X_scaled = scaler.fit_transform(df_filtered[features_for_clustering])

# hierarchical clustering
Z = linkage(X_scaled, method='ward')

# Plot the dendrogram
plt.figure(figsize=(10, 7))
plt.title('Hierarchical Clustering Dendrogram')
dendrogram(Z, labels=df_filtered.index)
plt.xlabel('Observations')
plt.ylabel('Distance')
plt.show()


# #### clustering and analyzing

# In[17]:


from scipy.cluster.hierarchy import fcluster

n_clusters = 5

clusters = fcluster(Z, n_clusters, criterion='maxclust')

# Add cluster labels to the DataFrame
df_filtered['Cluster'] = clusters

# Analyze clusters
for i in range(1, n_clusters + 1):
    cluster = df_filtered[df_filtered['Cluster'] == i]
    print(f"Cluster {i}:")
    print(cluster[features_for_clustering].mean())
    print('-' * 50)


# ### PCA

# In[18]:


from sklearn.decomposition import PCA

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)


# In[19]:


import matplotlib.pyplot as plt

plt.figure(figsize=(10, 8))

# Scatter plot of the clusters
for i in range(1, n_clusters + 1):
    plt.scatter(X_pca[clusters == i, 0], X_pca[clusters == i, 1], label=f'Cluster {i}')

plt.title('Clusters with Hierarchical Clustering (PCA-reduced)')
plt.xlabel('PCA Feature 1')
plt.ylabel('PCA Feature 2')
plt.legend()
plt.show()


# ### K-means

# In[20]:


from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

print("Unique values in the 'state' column:", df_cleaned['state'].unique())
df_filtered = df_cleaned[df_cleaned['state'].isin(['successful', 'failed'])]

if len(df_filtered) == 0:
    print("No data available for the specified 'state' labels.")
    exit()

# Select features for clustering
features_for_clustering = ['pledged', 'goal_usd', 'backers_count', 'create_to_launch_days', 
                           'launch_to_deadline_days', 'goal_usd_cat_mean', 'goal_usd_cat_max', 
                           'goal_usd_cat_min', 'goal_usd_cat_std']

# Standardizing the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df_filtered[features_for_clustering])


# In[21]:


# Use the Elbow Method to find the optimal number of clusters
inertia = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, random_state=0)
    kmeans.fit(X_scaled)
    inertia.append(kmeans.inertia_)

# Plotting the Elbow Method graph
plt.figure(figsize=(10, 6))
plt.plot(range(1, 11), inertia, marker='o')
plt.title('Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('Inertia')
plt.show()


# In[22]:


# Define the number of clusters based on the Elbow graph
n_clusters = 10

# Apply K-Means Clustering
kmeans = KMeans(n_clusters=n_clusters, random_state=0)
df_filtered['Cluster'] = kmeans.fit_predict(X_scaled)

# Analyze clusters
for i in range(n_clusters):
    cluster = df_filtered[df_filtered['Cluster'] == i]
    print(f"Cluster {i}:")
    print(cluster[features_for_clustering].mean())
    print('-' * 50)


# #### PCA

# In[23]:


optimal_k = 10 
kmeans = KMeans(n_clusters=optimal_k, random_state=0)
clusters = kmeans.fit_predict(X_scaled)


from sklearn.decomposition import PCA

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)


# In[24]:


import matplotlib.pyplot as plt

plt.figure(figsize=(10, 8))

# Scatter plot of the clusters
for i in range(optimal_k):
    plt.scatter(X_pca[clusters == i, 0], X_pca[clusters == i, 1], label=f'Cluster {i}')

# Mark the centroids
centroids = pca.transform(kmeans.cluster_centers_)
plt.scatter(centroids[:, 0], centroids[:, 1], s=100, c='black', marker='X', label='Centroids')

plt.title('Clusters with K-Means')
plt.xlabel('PCA Feature 1')
plt.ylabel('PCA Feature 2')
plt.legend()
plt.show()


# ### DBSCAN

# In[25]:


from sklearn.preprocessing import StandardScaler

# Filter the dataset for 'successful' and 'failed' states only
df_filtered = df_cleaned[df_cleaned['state'].isin(['successful', 'failed'])]

# Select features for clustering
features_for_clustering = ['pledged', 'goal_usd', 'backers_count', 'create_to_launch_days', 
                           'launch_to_deadline_days', 'goal_usd_cat_mean', 'goal_usd_cat_max', 
                           'goal_usd_cat_min', 'goal_usd_cat_std']

# Standardizing the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df_filtered[features_for_clustering])


# In[26]:


from sklearn.cluster import DBSCAN

# Apply DBSCAN clustering
dbscan = DBSCAN(eps=0.5, min_samples=5)
clusters = dbscan.fit_predict(X_scaled)

# Add cluster labels to the DataFrame
df_filtered['Cluster'] = clusters


# In[27]:


# Number of clusters in labels, ignoring noise if present.
n_clusters_ = len(set(clusters)) - (1 if -1 in clusters else 0)
n_noise_ = list(clusters).count(-1)

print('Estimated number of clusters: %d' % n_clusters_)
print('Estimated number of noise points: %d' % n_noise_)

# Analyze clusters (excluding noise points)
for i in range(n_clusters_):
    cluster = df_filtered[df_filtered['Cluster'] == i]
    print(f"Cluster {i}:")
    print(cluster[features_for_clustering].mean())
    print('-' * 50)


# #### Silhouette score

# In[28]:


from sklearn.metrics import silhouette_score

# Calculate Silhouette Score for K-means
silhouette_kmeans = silhouette_score(X_scaled, kmeans.labels_)

# Calculate Silhouette Score for Hierarchical Clustering
silhouette_hierarchical = silhouette_score(X_scaled, fcluster(Z, n_clusters, criterion='maxclust'))

print("Silhouette Score for K-means:", silhouette_kmeans)
print("Silhouette Score for Hierarchical Clustering:", silhouette_hierarchical)


# In[29]:


# Silhouette for DBSCAN

# Only calculate the score for non-noise labels (label != -1)
non_noise_indices = df_filtered['Cluster'] != -1

if sum(non_noise_indices) == 0:
    print("No clusters found (excluding noise), unable to calculate Silhouette Score.")
else:
    silhouette_dbscan = silhouette_score(X_scaled[non_noise_indices], df_filtered['Cluster'][non_noise_indices])
    print(f"Silhouette Score for DBSCAN: {silhouette_dbscan}")


# #### Pseudo F-statistic

# In[30]:


def pseudo_f_statistic(X, labels, centroids):
    n_clusters = len(set(labels))
    n_samples = len(X)

    # Between-group sum of squares
    between_group_ss = sum([len(X[labels == k]) * sum((centroid - X[labels == k]) ** 2) for k, centroid in enumerate(centroids)])
    
    # Within-group sum of squares
    within_group_ss = sum([sum((x - centroids[label]) ** 2) for x, label in zip(X, labels)])
    
    f_stat = (between_group_ss / (n_clusters - 1)) / (within_group_ss / (n_samples - n_clusters))
    return f_stat


# In[31]:


# Hierarchical clustering

optimal_clusters = 5
hierarchical_labels = fcluster(Z, optimal_clusters, criterion='maxclust')
# Adjust labels to be zero-based
hierarchical_labels_zero_based = hierarchical_labels - 1

hierarchical_centroids = [X_scaled[hierarchical_labels_zero_based == k].mean(axis=0) for k in range(optimal_clusters)]

def pseudo_f_statistic(X, labels, centroids):
    n_clusters = len(set(labels))
    n_samples = len(X)

    # Between-group sum of squares
    between_group_ss = sum([len(X[labels == k]) * sum((centroid - X[labels == k]) ** 2) for k, centroid in enumerate(centroids)])
    
    # Within-group sum of squares
    within_group_ss = sum([sum((x - centroids[label]) ** 2) for x, label in zip(X, labels)])
    
    f_stat = (between_group_ss / (n_clusters - 1)) / (within_group_ss / (n_samples - n_clusters))
    return f_stat

f_stat_hierarchical = pseudo_f_statistic(X_scaled, hierarchical_labels_zero_based, hierarchical_centroids)
print(f"Pseudo F-Statistic for Hierarchical Clustering: {f_stat_hierarchical}")


# In[32]:


# K-means

from sklearn.cluster import KMeans

optimal_k = 10
kmeans = KMeans(n_clusters=optimal_k, random_state=0)
kmeans_labels = kmeans.fit_predict(X_scaled)
kmeans_centroids = kmeans.cluster_centers_

f_stat_kmeans = pseudo_f_statistic(X_scaled, kmeans_labels, kmeans_centroids)
print(f"Pseudo F-Statistic for K-Means: {f_stat_kmeans}")


# In[ ]:




