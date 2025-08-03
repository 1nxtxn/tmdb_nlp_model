# Project Title: NLP Movie Recommendation Model

Briefly describe the purpose/result(s) of your project, the skills you applied, and the AI4ALL Ignite program.

Purpose: To make searching for movies more efficient by prompting the user for a few important variables to help make the top 3 recommendations for their preferences.


## Problem Statement <!--- do not change this line -->

The Solution for our problem is that people can save time from the hassle of searching on platforms like Netflix and instead can identify what movie fits best for their preferences and to ensure a seamless experience for our users.

## Key Results <!--- do not change this line -->

1. *Identified three biases in ChatGPT's responses*
   - *When prompted about this world event*
   - *When prompted about this field of science*
   - *When prompted about this political party*
  
2. *Taking a database of the top 5,000 global movies and sorting through 48 genre clusters to deliver the most optimal recommendation*
3. *Developed a 3-D Principal Component Analysis Graph (PCA) to plot each individual cluster and also listed the corresponding movies within each cluster and visualized the graph using plotly*
4. *# Implemented Word Cloud and Plotly to develop a Visualization of the distribution of the Top 15 most popular genres in our database which included: Drama, Comedy, Thriller, Action, Adventure, etc.*


## Methodologies <!--- do not change this line -->

*To accomplish our goal, we developed a content-based movie recommendation system using natural language processing (NLP), clustering, and sentiment analysis techniques. Our methodology centered on interpreting a user's natural language input and filtering relevant films based on genre, release year, and emotional tone.*

- We applied **TF-IDF vectorization** to the cleaned movie overviews in the TMDB 5000 Movies dataset, converting plot descriptions into numerical feature vectors.
- **KMeans clustering** was used to group similar movies based on textual features, allowing us to match the user's intent within thematically similar clusters.
- We implemented **sentiment analysis** using the VADER model to compute a compound sentiment score for each movie overview. Each score was categorized into a `positive`, `neutral`, or `negative` tone to match the emotional vibe of the user's query.
- A custom **recommendation function** was created to accept user input (text description, optional genre, year range, and tone), calculate similarity scores using **cosine similarity**, and return the top 3 movie matches.
- The user experience was designed entirely within **Google Colab**, using `input()` prompts to simulate an interactive recommendation engine without the need for a front-end interface.


## Data Sources <!--- do not change this line -->

We included our relevant data sources that were used in our project below.

*Kaggle TMDB Top 5,000 Movies Dataset: [Link to Kaggle Dataset]([https://www.kaggle.com/datasets](https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata?utm_source=chatgpt.com&select=tmdb_5000_movies.csv))*

## Technologies Used <!--- do not change this line -->

Below listed are the technologies, libraries, and frameworks used in your project.

- *python*
- *pandas*
- *plotly*
- *scikit-learn*
- *seaborn*
- *nltk*
- *numpy*
- *qgrid*
- *tf-idf*
- *vaderSentiment*


## Authors <!--- do not change this line -->

Below are listed the names of the contributors of the project.

*This project was completed in collaboration with:*
- *Nathan Seife*
- *Simon Plotkin*
- *Shatoya Gardner*
- *Syeda Bushra*
