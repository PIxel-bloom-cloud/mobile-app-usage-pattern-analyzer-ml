import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

# Load dataset
data = pd.read_csv("mobile_usage.csv")

X = data[['social','gaming','productivity','entertainment']]
y = data['user_type']

# ---------------- CLUSTERING ----------------
kmeans = KMeans(n_clusters=3, random_state=0)
clusters = kmeans.fit_predict(X)

plt.figure()
plt.scatter(X['social'], X['gaming'], c=clusters)
plt.xlabel("Social Usage")
plt.ylabel("Gaming Usage")
plt.title("Clustering (K-Means)")
plt.show()

# ---------------- BAR GRAPH ----------------
data[['social','gaming','productivity','entertainment']].mean().plot(kind='bar')
plt.title("Average App Usage")
plt.ylabel("Hours")
plt.show()

# ---------------- CLASSIFICATION ----------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = KNeighborsClassifier()
model.fit(X_train, y_train)

pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, pred))

# ---------------- CONFUSION MATRIX ----------------
cm = confusion_matrix(y_test, pred)

plt.figure()
plt.imshow(cm)
plt.title("Confusion Matrix")
plt.colorbar()
plt.show()