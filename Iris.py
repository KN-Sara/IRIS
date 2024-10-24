import streamlit as st
import pickle
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

# Step 1: Train the Model and Save it using Pickle (This will run when the app is started)
def train_model():
    # Load the Iris dataset
    iris = load_iris()

    # Features (X) and labels (y)
    X = iris.data
    y = iris.target

    # Split the dataset into training and testing
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Initialize and train the classifier
    clf = DecisionTreeClassifier()
    clf.fit(X_train, y_train)

    # Save the model using pickle
    with open('iris_model.pkl', 'wb') as f:
        pickle.dump(clf, f)

# Step 2: Load the model from the Pickle file
def load_model():
    with open('iris_model.pkl', 'rb') as f:
        return pickle.load(f)

# Step 3: Streamlit Interface
def main():
    st.title("Iris Flower Classification")

    # Input fields for flower features
    sepal_length = st.slider('Sepal Length (cm)', 4.0, 8.0, 5.0)
    sepal_width = st.slider('Sepal Width (cm)', 2.0, 4.5, 3.0)
    petal_length = st.slider('Petal Length (cm)', 1.0, 7.0, 4.0)
    petal_width = st.slider('Petal Width (cm)', 0.1, 2.5, 1.0)

    # Load the trained model
    model = load_model()

    # When the user clicks the button
    if st.button('Classify'):
        # Prepare the input for the model
        input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])

        # Perform prediction
        prediction = model.predict(input_data)

        # Map the prediction to the class name
        species = ['Setosa', 'Versicolour', 'Virginica']
        st.write(f"The predicted species is: {species[prediction[0]]}")

# Train the model (This should be done only once, and you can comment this out after the first run)
train_model()

# Start the Streamlit app
if __name__ == '__main__':
    main()