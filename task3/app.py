import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import folium
from streamlit_folium import st_folium
from sklearn.cluster import KMeans
from folium.plugins import HeatMap
st.set_page_config(page_title="Restaurant Location Analysis",layout="wide")
st.title("Restaurant Location-Based Analysis")
try:
    
    df=pd.read_csv('Dataset.csv')
    df.columns=df.columns.str.strip()
    
    st.subheader("Dataset Preview")
    st.dataframe(df.head())
    st.subheader("Restaurant Locations Map")
    map_center=[df['Latitude'].mean(),df['Longitude'].mean()]
    m=folium.Map(location=map_center,zoom_start=5)
    for _,row in df.iterrows():
        folium.CircleMarker(location=[row['Latitude'],row['Longitude']],radius=2,color='blue',fill=True).add_to(m)
    st_folium(m,width=700,height=500)
    st.subheader("Density HeatMap")
    heat_map=folium.Map(location=map_center,zoom_start=5)
    HeatMap(df[['Latitude','Longitude']].dropna()).add_to(heat_map)
    st_folium(heat_map,width=700,height=500)
    st.subheader("Restaurant by City")
    city_counts=df['City'].value_counts()
    fig,ax=plt.subplots()
    city_counts.head(10).plot(kind='bar',ax=ax)
    ax.set_title("Top 10 cities by restaurant count")
    st.pyplot(fig)
    st.subheader("City-wise Statistics")
    city_stats=df.groupby('City').agg({'Restaurant ID':'count','Aggregate rating':'mean','Price range':'mean'}).rename(columns={'Restaurant ID':'Restaurant Count'})
    st.dataframe(city_stats.sort_values(by='Restaurant Count',ascending=False).head(10))
    st.subheader("Cuisine Distribution")
    df['Cuisines']=df['Cuisines'].str.split(', ')
    cuisine_df=df.explode('Cuisines')
    top_cuisines=cuisine_df['Cuisines'].value_counts().head(10)
    fig2,ax2=plt.subplots()
    top_cuisines.plot(kind='bar',ax=ax2)
    ax2.set_title("Top 10 Cuisines")
    st.pyplot(fig2)
    st.subheader("Price Range Distribution")
    fig3,ax3=plt.subplots()
    sns.boxplot(x='Price range',data=df,ax=ax3)
    st.pyplot(fig3)
    st.subheader("Rating Distribution")
    fig4,ax4=plt.subplots()
    sns.histplot(df['Aggregate rating'], bins=20, ax=ax4)
    st.pyplot(fig4)
    locality_counts = df['Locality'].value_counts().head(10)
    fig5, ax5 = plt.subplots()
    locality_counts.plot(kind='bar', ax=ax5)
    ax5.set_title("Top 10 Localities by Restaurant Count")
    st.subheader("Top Localities")
    st.pyplot(fig5)
    
   

   
    df_cluster = df.dropna(subset=['Latitude', 'Longitude']).copy()

    kmeans = KMeans(n_clusters=5, random_state=42)
    df_cluster['Cluster'] = kmeans.fit_predict(df_cluster[['Latitude', 'Longitude']])
    st.subheader("🧠 Restaurant Clusters (K-Means)")
    cluster_map = folium.Map(
        location=[df_cluster['Latitude'].mean(), df_cluster['Longitude'].mean()],
        zoom_start=5
    )

    colors = ['red', 'blue', 'green', 'purple', 'orange']

    for _, row in df_cluster.iterrows():
        folium.CircleMarker(
            location=[row['Latitude'], row['Longitude']],
            radius=3,
            color=colors[row['Cluster']],
            fill=True,
            fill_color=colors[row['Cluster']],
            popup=f"Cluster: {row['Cluster']}"
        ).add_to(cluster_map)

    st_folium(cluster_map, width=700, height=500)
    from sklearn.linear_model import LinearRegression

    X = df[['Price range', 'Votes']]
    y = df['Aggregate rating']

    model = LinearRegression()
    model.fit(X, y)
    st.subheader("Predict Restaurant Rating")

    price = st.slider("Select Price Range", 1, 4, 2)
    votes = st.number_input("Enter Votes", min_value=0, value=100)
    prediction = model.predict([[price, votes]])

    st.success(f"Predicted Rating: {prediction[0]:.2f}")
    

    y_pred = model.predict(X)

    fig, ax = plt.subplots()
    ax.scatter(y, y_pred)
    ax.set_xlabel("Actual Rating")
    ax.set_ylabel("Predicted Rating")
    ax.set_title("Actual vs Predicted Ratings")

    st.pyplot(fig)
    st.subheader("Key insights")
    st.markdown(""" - Restaurants are highly concentrated in major cities (especially NCR region).
    - Higher density areas tend to have lower average ratings due to competition.
    - Most restaurants fall into affordable price ranges (1–2).
    - North Indian and Fast Food cuisines dominate.
    - Restaurants cluster around high-footfall areas like malls and business hubs.
    """)
    
except:
     st.warning("⚠️ Dataset not found. Please place Dataset.csv in the project folder.")
     st.stop()
    
