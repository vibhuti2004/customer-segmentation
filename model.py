import pandas as pd
import matplotlib.pyplot as plt 
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

#data load
df=pd.read_csv("customer_data.csv")

#feature selection 
x=df[["Income","Score"]]

#scaling
scaler=StandardScaler()
x_scale=scaler.fit_transform(x)


#model 
model=KMeans(n_clusters=2,random_state=42)
df["Cluster"]= model.fit_predict(x_scale)

print(df)
print("Cluster Centers:",model.cluster_centers_)
print("Labels:",model.labels_)

# visualization
plt.scatter(df["Income"],df["Score"],c=df["Cluster"],cmap="rainbow",s=100)
plt.title("Customer Segmentation")
plt.xlabel("Income")
plt.ylabel("Score")
plt.show()
