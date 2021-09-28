import streamlit as st
from sklearn.datasets import load_iris 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import pickle
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns

# data = load_iris()
# X = data.data
# Y = data.target
# X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.2, random_state=1234)
# model = LogisticRegression(C=0.01)
# model.fit(X_train, Y_train)
# pickle_out = open('final_model.sav','wb')
# pickle.dump(model,pickle_out)


def predict(arr):
    with open('final_model.sav','rb') as f:
        model = pickle.load(f)
    #classes = {0:'Iris Setosa',1:'Iris Versicolor',2:'Iris Virginica'}
    percent = model.predict_proba([arr])[0]
    return (classes[np.argmax(percent)],percent)

classes = {0:'Iris Setosa',1:'Iris Versicolor',2:'Iris Virginica'}
class_labels = list(classes.values())
st.title('Classification')
st.markdown('**Objective**')
def predict_class():
    data = list(map(float,[sepal_length,sepal_width,petal_length, petal_width]))
    result, percent = predict(data)
    st.write('The prediction is ',result)
    percents = [np.round(x,2) for x in percent]
    fig, ax = plt.subplots()
    ax = sns.barplot(percents,class_labels, palette='winter',orient='h')
    ax.set_yticklabels(class_labels, rotation =0)
    plt.title('Show Prediction')
    for index, value in enumerate(percents):
        plt.text(value, index, str(value))
    st.pyplot(fig)


sepal_length = st.text_input('Enter sepal length','')
sepal_width = st.text_input('Enter sepal_width', '')
petal_length = st.text_input('Enter petal_length', '')
petal_width = st.text_input('Enter petal_width', '')
if st.button('Predict'):
    data = load_iris()
    X = data.data
    Y = data.target
    X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.2, random_state=1234)
    model = LogisticRegression(C=0.01)
    model.fit(X_train, Y_train)
    pickle_out = open('final_model.sav','wb')
    pickle.dump(model,pickle_out)
    predict_class()