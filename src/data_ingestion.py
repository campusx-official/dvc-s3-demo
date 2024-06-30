import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import os

df = pd.read_csv('./data/raw/student_performance.csv')

# Separating features and target variable
X = df.drop(columns=['Placed'])
y = df['Placed']

# Scaling the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Applying PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# Creating a DataFrame with PCA results
df_pca = pd.DataFrame(data=X_pca, columns=['PC1', 'PC2'])
df_pca['Placed'] = y.values

df_pca.to_csv(os.path.join('data','processed','student_performance_pca.csv'), index=False)
