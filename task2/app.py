from flask import Flask,render_template,request
import requests
import pandas as pd 
import numpy as np

import random
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics.pairwise import cosine_similarity
app=Flask(__name__)
df=pd.read_csv('Dataset.csv')
df['Cuisines']=df['Cuisines'].fillna('Unknown')
df['Cuisine List']=df['Cuisines'].apply(lambda x:[i.strip() for i in x.split(',')])
df=df.drop_duplicates(subset=['Restaurant Name','Cuisines'])
mlb=MultiLabelBinarizer()
cuisine_encoded=pd.DataFrame(mlb.fit_transform(df['Cuisine List']),columns=mlb.classes_,index=df.index)
features=pd.concat([cuisine_encoded,df[['Price range']]],axis=1)
def recommend_restaurants(cuisine_pref,price_pref,min_rating=0,top_n=5):
    cuisine_encoded_user=mlb.transform([cuisine_pref])
    user_vector=np.concatenate([cuisine_encoded_user[0],[price_pref]])
    sims=cosine_similarity([user_vector],features)[0]
    df['Similarity']=sims 
    filtered_df = df[
        (df['Aggregate rating'] >= min_rating) &
        (df['Cuisines'].str.contains(cuisine_pref[0], case=False))
    ]
    if filtered_df.empty:
        return pd.DataFrame()
      
    recommendations=filtered_df.sort_values(by=['Similarity','Aggregate rating'],ascending=False)
    return recommendations[['Restaurant Name','Cuisines','Price range','Aggregate rating']].head(top_n)
@app.route('/',methods=['GET','POST'])
def home():
    results=None 
    if request.method=='POST':
        cuisine=request.form['cuisine']
        price=int(request.form['price'])
        rating=float(request.form['rating'])
        results=recommend_restaurants(cuisine_pref=[cuisine],price_pref=price,min_rating=rating).to_dict(orient='records')
        for r in results:
            r['image']=get_food_image(r['Cuisines'])
    return render_template('index.html',results=results)
def get_food_image(cuisine):
    
    try:
        base=cuisine.split(',')[0].strip().lower()
        if "indian" in base:
            category = random.choice(["biryani", "dosa", "idly"])
        elif "japanese" in base:
            category = "noodles"   # closest match
        elif "italian" in base:
            category = "pasta"
       
        elif "american" in base:
            category = "burger"
        elif "dessert" in base:
            category = "dessert"
        else:
            category = "pizza"
        url = f"https://foodish-api.com/api/images/{category}"
        res=requests.get(url)
        data=res.json()
        if isinstance(data,dict):
            return data.get("image")
            
        elif isinstance(data,list):
            return data[0]
    except Exception as e:
        print("Foodish Error:",e)
    return "https://picsum.photos/300/200"
if __name__=='__main__':
    app.run(debug=True)
    
