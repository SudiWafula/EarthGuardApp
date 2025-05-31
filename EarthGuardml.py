#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Streamlit app with soil prediction and recommendation based on spatial features
import streamlit as st
import joblib
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point

# Load trained model
model = joblib.load("rf_binned_model.pkl")

# Define bin size (must match training)
bin_size = 100

# Define scoring functions
def score_n(x):
    if pd.isna(x): return None
    elif x >= 0.2: return 100
    elif x >= 0.1: return 75
    elif x >= 0.05: return 50
    else: return 25

def score_p(x):
    if pd.isna(x): return None
    elif x >= 15: return 100
    elif x >= 10: return 75
    elif x >= 5: return 50
    else: return 25

def score_k(x):
    if pd.isna(x): return None
    elif x >= 150: return 100
    elif x >= 100: return 75
    elif x >= 60: return 50
    else: return 25

def score_toc(x):
    if pd.isna(x): return None
    elif x >= 2: return 100
    elif x >= 1: return 75
    elif x >= 0.5: return 50
    else: return 25

def score_ph(x):
    if pd.isna(x): return None
    elif 6.0 <= x <= 7.5: return 100
    elif 5.5 <= x < 6.0 or 7.6 <= x <= 8.0: return 75
    elif 5.0 <= x < 5.5 or 8.1 <= x <= 8.5: return 50
    else: return 25

# Compute SMHI score
def compute_smhi(df):
    df["Score_N"] = df["total_Nitrogen_percent_"].apply(score_n)
    df["Score_P"] = df["phosphorus_Olsen_ppm"].apply(score_p)
    df["Score_K"] = df["potassium_meq_percent_"].apply(score_k)
    df["Score_TOC"] = df["total_Org_Carbon_percent_"].apply(score_toc)
    df["Score_pH"] = df["soil_pH"].apply(score_ph)

    weights = {"Score_N": 0.25, "Score_P": 0.20, "Score_K": 0.20, "Score_TOC": 0.20, "Score_pH": 0.15}
    df["SMHI_Score"] = round(
        df["Score_N"] * weights["Score_N"] +
        df["Score_P"] * weights["Score_P"] +
        df["Score_K"] * weights["Score_K"] +
        df["Score_TOC"] * weights["Score_TOC"] +
        df["Score_pH"] * weights["Score_pH"], 2
    )
    return df

# Function to transform lat/lon to feature vector
def latlon_to_bin_features(lat, lon):
    pt = gpd.GeoDataFrame([[Point(lon, lat)]], columns=['geometry'], crs='EPSG:4326')
    pt = pt.to_crs(pt.estimate_utm_crs())
    pt['x'] = pt.geometry.x
    pt['y'] = pt.geometry.y
    pt['x_bin'] = (pt['x'] // bin_size).astype(int)
    pt['y_bin'] = (pt['y'] // bin_size).astype(int)
    pt['x_bin'] = pt['x_bin'].astype(str)
    pt['y_bin'] = pt['y_bin'].astype(str)
    one_hot = pd.get_dummies(pd.DataFrame({
        'x_bin': pt['x_bin'],
        'y_bin': pt['y_bin']
    }).astype(str))
    base_features = pd.DataFrame({'x': pt['x'], 'y': pt['y']})
    X_new = pd.concat([base_features.reset_index(drop=True), one_hot.reset_index(drop=True)], axis=1)
    return X_new

# Streamlit UI
st.title("🌱 Soil Property Estimator (with Spatial Indexing)")

lat = st.number_input("Latitude", format="%.6f", value=-0.5)
lon = st.number_input("Longitude", format="%.6f", value=37.5)

if st.button("Predict Soil Values"):
    X_new = latlon_to_bin_features(lat, lon)
    model_columns = model.feature_names_in_ if hasattr(model, 'feature_names_in_') else model.get_booster().feature_names
    for col in model_columns:
        if col not in X_new.columns:
            X_new[col] = 0  # fill missing one-hot columns with zero
    X_new = X_new[model_columns]  # reorder to match model
    prediction = model.predict(X_new)
    output = pd.DataFrame(prediction, columns=[
        'soil_pH',
        'total_Nitrogen_percent_',
        'total_Org_Carbon_percent_',
        'phosphorus_Olsen_ppm',
        'potassium_meq_percent_'
    ])

    # Compute scores and SMHI
    scored_output = compute_smhi(output.copy())

    st.write("### Predicted Soil Properties:")
    st.dataframe(scored_output.round(4))

    st.write("### SMHI Score:")
    st.metric(label="Soil Management Health Index", value=scored_output['SMHI_Score'].iloc[0])


# In[2]:


#!pip install geopandas


# In[ ]:




