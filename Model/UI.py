import streamlit as st
import pickle
import requests
import pandas as pd
from PIL import Image
from io import BytesIO

# -------------------------------
# Image Repo
# -------------------------------

CROP_IMAGES = {
    "rice": "https://upload.wikimedia.org/wikipedia/commons/6/6f/Rice_field_in_Thailand.jpg",
    "maize": "https://upload.wikimedia.org/wikipedia/commons/2/23/Corncobs.jpg",
    "chickpea": "https://upload.wikimedia.org/wikipedia/commons/6/6a/Chickpeas_on_a_tree.jpg",
    "kidneybeans": "https://upload.wikimedia.org/wikipedia/commons/0/01/Kidney_beans.jpg",
    "pigeonpeas": "https://upload.wikimedia.org/wikipedia/commons/9/9a/Cajanus_cajan_flowers.jpg",
    "mothbeans": "https://upload.wikimedia.org/wikipedia/commons/6/6f/Matki.JPG",
    "mungbean": "https://upload.wikimedia.org/wikipedia/commons/5/51/Mung_bean_plant_2.jpg",
    "blackgram": "https://upload.wikimedia.org/wikipedia/commons/1/1b/Plant_of_Urad_dal_or_black_gram.JPG",
    "lentil": "https://upload.wikimedia.org/wikipedia/commons/4/44/Lentils_in_bowl.jpg",
    "pomegranate": "https://upload.wikimedia.org/wikipedia/commons/8/85/Pomegranate_arils.jpg",
    "banana": "https://upload.wikimedia.org/wikipedia/commons/8/8a/Banana-Single.jpg",
    "mango": "https://upload.wikimedia.org/wikipedia/commons/9/90/Hapus_Mango.jpg",
    "grapes": "https://upload.wikimedia.org/wikipedia/commons/3/36/Table_grapes_on_white.jpg",
    "watermelon": "https://upload.wikimedia.org/wikipedia/commons/e/e4/Watermelon.jpg",
    "muskmelon": "https://upload.wikimedia.org/wikipedia/commons/4/4d/Cantaloupe_cross_section.jpg",
    "apple": "https://upload.wikimedia.org/wikipedia/commons/1/15/Red_Apple.jpg",
    "orange": "https://upload.wikimedia.org/wikipedia/commons/c/c4/Orange-Fruit-Pieces.jpg",
    "papaya": "https://upload.wikimedia.org/wikipedia/commons/9/9f/Papaya_cross_section_BNC.jpg",
    "coconut": "https://upload.wikimedia.org/wikipedia/commons/c/cb/Coconut_cross_section.jpg",
    "cotton": "https://upload.wikimedia.org/wikipedia/commons/a/a0/Cotton_bolls.JPG",
    "jute": "https://upload.wikimedia.org/wikipedia/commons/7/77/Jute_bale.jpg",
    "coffee": "https://upload.wikimedia.org/wikipedia/commons/5/56/Coffea_arabica_in_flower.jpg"
}


data =pd.read_csv(r"C:\Users\Adhi Ganapathy\Documents\Python_ws\Recommendation Engine 13092025\Crop recommendation system\Crop_recommendation.csv")



# -------------------------------
# Load preprocessed data
# -------------------------------
with open(r"C:\Users\Adhi Ganapathy\Documents\Python_ws\Recommendation Engine 13092025\Crop recommendation system\Tree.pkl", "rb") as file:
     Label =pickle.load(file)
     XB =pickle.load(file)
     column_name =pickle.load(file)
     XGBClassifier=  pickle.load(file)
     BaggingClassifier= pickle.load(file)
     RandomForestClassifier = pickle.load(file)
     XGBoostAcc  =pickle.load(file)
     RFBoostAcc  =pickle.load(file)
     DTBoostAcc  =pickle.load(file)
    
st.sidebar.title("🌱 Crop Recommendation System")








def predictions():
     

     # User inputs
     N = st.number_input("Nitrogen (N)", min_value=0, max_value=200, value=50)
     P = st.number_input("Phosphorus (P)", min_value=0, max_value=200, value=50)
     K = st.number_input("Potassium (K)", min_value=0, max_value=200, value=50)
     temperature = st.number_input("Temperature (°C)", min_value=-10.0, max_value=60.0, value=25.0)
     humidity = st.number_input("Humidity (%)", min_value=0.0, max_value=100.0, value=60.0)
     ph = st.number_input("Soil pH", min_value=0.0, max_value=14.0, value=7.0)
     rainfall = st.number_input("Rainfall (mm)", min_value=0.0, max_value=500.0, value=100.0)


     if st.button("Recommend"):
        # Collect inputs into an array
        Features = pd.DataFrame([[N, P, K, temperature, humidity, ph, rainfall]],columns=column_name)
        prediction =XB.predict(Features)
        crop = Label.inverse_transform([prediction[0]])[0]
        if not crop:
            st.error("No recommendations found! Please try another crop.")
        else:
            st.success(f"🌾 Recommended Crop: **{crop}**")
         


page =st.sidebar.radio("Go To",["Overview","Model Evaluation","Prediction"])

def main():

    if page == "Overview":
        st.title("📊 Dataset Overview")
 
        st.write(data.head())

    elif page == "Model Evaluation":
        st.title("📈 Model Evaluation")
    

    # Display in Streamlit
 
        st.write("XG Boost Accuracy      :", f'{XGBoostAcc:.2%}')
        st.write("Random Forest Accuracy :", f'{RFBoostAcc:.2%}')
        st.write("Decision Tree Accuracy :", f'{DTBoostAcc:.2%}')
    

    elif page=="Prediction":
    
        st.title( "🌱 Prediction")
        predictions()






if __name__ == "__main__":
    main()         