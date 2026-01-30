import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

data = pd.read_csv("students_data.csv")

X = data[['StudyHours', 'Attendance', 'Score']]

X = X.fillna(X.mean())

print("Original Data:")
print(X)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

pca_df = pd.DataFrame(X_pca, columns=['PC1', 'PC2'])

print("\nPCA Output:")
print(pca_df)

print("\nExplained Variance Ratio:", pca.explained_variance_ratio_)
print("Total Variance:", sum(pca.explained_variance_ratio_))

plt.scatter(pca_df['PC1'], pca_df['PC2'])
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("PCA Graph")
plt.show()