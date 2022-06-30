"""

    Simple Streamlit webserver application for serving developed classification
	models.

    Author: Explore Data Science Academy.

    Note:
    ---------------------------------------------------------------------
    Please follow the instructions provided within the README.md file
    located within this directory for guidance on how to use this script
    correctly.
    ---------------------------------------------------------------------

    Description: This file is used to launch a minimal streamlit web
	application. You are expected to extend the functionality of this script
	as part of your predict project.

	For further help with the Streamlit framework, see:

	https://docs.streamlit.io/en/latest/

"""
# Streamlit dependencies
import streamlit as st
import joblib,os

# Data dependencies
import pandas as pd

# Vectorizer
news_vectorizer = open("resources/tfidf_vector","rb")
tweet_cv = joblib.load(news_vectorizer) # loading your vectorizer from the pkl file

# Load your raw data
raw = pd.read_csv("resources/train.csv")

# The main function where we will build the actual app
def main():
	"""Tweet Classifier App with Streamlit """

# Creates a main title and subheader on your page -
	# these are static across all pages
	st.title("CLASSIFY IT")
	st.subheader("Classification of Tweet species")

	# Creating sidebar with selection box -
	# you can create multiple pages this way
	options = ["Home","Information","Model Accuracy", "Data","Contact Us"]
	selection = st.sidebar.selectbox("Navigate", options)
    
    #Creating The Imprtant part for Predictions
	tweet_text = st.sidebar.text_area("Classify Tweet","Tweet Here")
	if st.sidebar.button("Classify"):
		# Transforming user input with vectorizer
		vect_text = tweet_cv.transform([tweet_text]).toarray()
		# Load your .pkl file with the model of your choice + make predictions
		# Try loading in multiple models to give the user a choice
		predictor = joblib.load(open(os.path.join("resources/lsvc.pkl"),"rb"))
		prediction = predictor.predict(vect_text)
		# When model has successfully run, will print prediction
		# You can use a dictionary or similar structure to make this output
		# more human interpretable.
		st.sidebar.success("Tweet Categorized as: {}".format(prediction)) 
        st.sidebar.image("./Keys.jpg")
      
        
	# Building out the "Home" page
	if selection == "Home":
		st.info("Welcome to **_Earth_** and its possible future")
		st.image("./Earth.jpg")
		# You can read a markdown file from supporting resources folder
		st.markdown("## Our Story")
        
		st.markdown("Humans and wild animals face new challenges for survival because of climate change. More frequent and intense drought, storms, heat waves, rising sea levels, melting glaciers and warming oceans can directly harm animals, destroy the places they live, and wreak havoc on people’s livelihoods and communities.")
        
		st.markdown("**CBB1 D.S Discovery** We discovered that that we could make a solution to a existing problem, what if we could create a machine learning model that is able to classidfy whether or not a a person believes in climate change based on their tweets? It would give us the advantage as to know which people we could potentially market for. Thus inspiring Classify It.")
        
		st.markdown("Here at  D.S Discovery, Data Science Company. We are on the path to find some great breakthroughs  as we are proud supporters in combating Climate Change. Wouldn’t you want an app that could Predict whether a person or entity is for or against climate change. Imagine wanting to know if someone would invest in renewable energy  or more likely to spend more money on non-renewable  energy.") 

		st.markdown("We have the solution for you! This Web app has been designed to take any raw data tweet and Classify It into either, Pro, Anti, Neutral or News.")       
        

	# Building out the "Information" page
	if selection == "Information":
		st.info(" *About us*")
		# You can read a markdown file from supporting resources folder
		st.markdown("## CBB1 D.S Discovery consists of 5 members ##")
        
		st.markdown("Riaan James-Verwey - Coordinator/Data Scientist")
		st.markdown("Samuel Mnisi - Coordinator/Data Scientist")
		st.markdown("Zanele Myeni - Data Scientist")
		st.markdown("Thato Lethetsa - Data Scientist")
		st.markdown("Dineo Makwala - Data Scientist")
        
		st.markdown("")
        
		st.markdown("Trello board : https://trello.com/b/HCP0k4mh/classification-regression")
		st.markdown("Github Repo  here : https://github.com/SoulR95/Class-Regression-CBB1")
        
	# Building out the "Data" page
	if selection == "Model Accuracy":        
		st.subheader("Different Models and their Score")
        
	# Building out the "Data" page
	if selection == "Data":        
		st.subheader("Raw Twitter data and label")
		if st.checkbox('Show raw data'): # data is hidden if box is unchecked
			st.write(raw[['sentiment', 'message']]) # will write the df to the page

# Required to let Streamlit instantiate our web app.  
if __name__ == '__main__':
	main()
