import pandas as pd
import streamlit as st
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier

# App Title
st.markdown("""
    <div style="background-color:#4CAF50;padding:10px;border-radius:10px">
        <h1 style="color:white;text-align:center;">Iris Flower Prediction App</h1>
    </div>
""", unsafe_allow_html=True)

st.write("""
### Predict the Iris Flower Type
This app uses **Random Forest Classifier** to predict the type of Iris flower based on your input parameters.
""")

# Sidebar header
st.sidebar.header("User Input Parameters")


# Function for User Input
def user_input_features():
    sepal_length = st.sidebar.slider("Sepal length (cm)", 4.3, 7.9, 5.4)
    sepal_width = st.sidebar.slider("Sepal width (cm)", 2.0, 4.4, 3.4)
    petal_length = st.sidebar.slider("Petal length (cm)", 1.0, 6.9, 1.3)
    petal_width = st.sidebar.slider("Petal width (cm)", 0.1, 2.5, 0.2)

    data = {
        "Sepal Length": sepal_length,
        "Sepal Width": sepal_width,
        "Petal Length": petal_length,
        "Petal Width": petal_width
    }

    features = pd.DataFrame(data, index=[0])
    return features


# Get user input
df = user_input_features()

# Display user input parameters
st.subheader("User Input Parameters")
st.write(df)

# Load the Iris dataset and train the model
iris = datasets.load_iris()
X = iris.data
y = iris.target

clf = RandomForestClassifier()
clf.fit(X, y)

# Predictions
prediction = clf.predict(df)
prediction_proba = clf.predict_proba(df)



# Display Prediction
st.subheader("Prediction")
st.write(f"**Predicted Iris type:** {iris.target_names[prediction][0]}")

# Display Prediction Probability
st.subheader("Prediction Probability")
st.write(f"The prediction probabilities for the three types of Iris flowers are as follows:")

proba_df = pd.DataFrame(prediction_proba, columns=iris.target_names)
st.write(proba_df)

# Adding custom styles to the prediction results
st.markdown("""
    <style>
    .stSubheader, .stMarkdown {
        color: #0066cc;
    }
    </style>
""", unsafe_allow_html=True)

# Footer with an image/logo
st.markdown("""
    <hr>
    <footer>
        <p style='text-align: center;'>Built with ❤️ by John Evans Okyere.</p>
    </footer>
""", unsafe_allow_html=True)
