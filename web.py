import streamlit as st
import tensorflow as tf
from tensorflow import keras
from sklearn import preprocessing
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd
from matplotlib import pyplot as plt
import math
from tensorflow.keras.utils import to_categorical
import joblib


classes = ['None', 'Mild', "Moderate", "Moderately Severe", "Severe"]
CHUNK_SIZE = 15000
data = pd.read_csv('depressionDataset.csv', chunksize=CHUNK_SIZE)
df = data.get_chunk(CHUNK_SIZE)
df = df.drop(['id', 'time', 'period.name', 'start.time', 'score'], axis=1)
y = df.pop('class')

features_names = ['q1', 'q2', 'q3', 'q4', 'q5', 'q6', 'q7', 'q8', 'q9', 'q10']
features = df[features_names]

y_encoded = to_categorical(y)

X_train, X_test, y_train, y_test = train_test_split(
    features, y_encoded, test_size=0.3, random_state=42)

# TRAIN_SIZE = math.floor(y.size * 80 / 100)
# TEST_SIZE = y.size - TRAIN_SIZE
# X_train, X_test = features[:TRAIN_SIZE], features[TRAIN_SIZE:]
# y_train, y_test = y_encoded[:TRAIN_SIZE], y_encoded[TRAIN_SIZE:]

normal_scaler = preprocessing.StandardScaler()

train_features = normal_scaler.fit_transform(X_train)
val_features = normal_scaler.transform(X_test)

X_train = np.clip(train_features, -5, 5)
X_test = np.clip(val_features, -5, 5)


tf.convert_to_tensor(X_train)
tf.convert_to_tensor(X_test)
tf.convert_to_tensor(y_train)
tf.convert_to_tensor(y_test)


def classifier(classifier_name):
    params = dict()
    if classifier_name == "KNN":
        K = st.sidebar.slider("K", 1, 15)
        params['K'] = K

    elif classifier_name == "Neural Network":
        params['model'] = 'nn_model.h5'

    else:
        max_depth = st.sidebar.slider("max_depth", 2, 15)
        n_estimators = st.sidebar.slider("n_estimators", 1, 100)
        params['max_depth'] = max_depth
        params['n_estimators'] = n_estimators

    return params


def user_inputs():
    choices = [
        'Not at all', 'Several Days', "More than half the days", "Nearly every day"]

    second_choices = ['Not difficult at all', 'Somewhat difficult',
                      'Very difficult', 'Extremely difficult']

    questions = [
        'Little interest or pleasure in doing things',
        'Feeling down, depressed, or hopeless',
        'Trouble falling or staying asleep, or sleeping too much',
        'Feeling tired or having little energy',
        'Poor appetite or overeating',
        'Feeling bad about yourself - or that you are a failure or have let yourself or your family down',
        'Trouble concentrating on things, such as reading the newspaper or watching television',
        'Moving or speaking so slowly that other people could have noticed',
        'Thoughts that you would be better off dead, or of hurting yourself',
        "If you've had any days with issues above, how difficult have these problems made it for you at work, home, school, or with other people?",
    ]

    X = []

    for question in questions:
        ans = choices.index(st.radio(question, choices))
        if question == "If you've had any days with issues above, how difficult have these problems made it for you at work, home, school, or with other people?":
            ans = second_choices.index(st.radio(question, second_choices))
        X.append(ans)

    return X


def model(clf, params):
    if clf == "Random Forest":
        model = RandomForestClassifier()
        # model = RandomForestClassifier(n_estimators=params['n_estimators'], max_depth=params['max_depth'], random_state=1234)
    elif clf == "KNN":
        model = joblib.load('knn_model.pkl')
        # model = KNeighborsClassifier(n_neighbors=params['K'])
    else:
        model = keras.models.load_model(params['model'])

    return model


st.title("Depression Prediction App")
st.write("""
# This app predicts the depression level of a person
""")

st.sidebar.header("Classifier")
classifier_name = st.sidebar.selectbox(
    "Select Classifier", ("KNN", "Random Forest", "Neural Network"))

params = classifier(classifier_name)
X = user_inputs()

model = model(classifier_name, params)

st.write("Classifier used: ", classifier_name)

if classifier_name != 'Neural Network':
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    st.write("Prediction accuracy: ", acc)
else:
    acc = model.evaluate(X_test, y_test)
    st.write("Prediction accuracy: ", acc[1])

buttonPressed = st.button('See result!')

if buttonPressed:
    prediction = model.predict([X])
    st.info(classes[np.argmax(prediction)])
