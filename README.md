# Movie genres prediction

Movies are almost always categorized with more than one genre and it is not an easy task to identify all the genres correctly. This type of task is called multi-label classification and I wanted to be able to build a Machine Learning model capable of predicting the genres of a movie given its plot summary.

This was also a good opportunity to practice one way of deploying a machine learning model into production.

To this end I implemented a Python package which exposes a Flask REST API with the following endpoints:

1. A training endpoint at `localhost:5000/genres/train` to which we POST a CSV file with the header `movie_id,synopsis,genres`, where `genres` is a space-separated list of movie genres.
2. A prediction endpoint at `localhost:5000/genres/predict` to which we POST a CSV file with the header `movie_id,synopsis` and returns a CSV file with header `movie_id,predicted_genres`, where `predicted_genres` is a space-separated list of the top 5 predicted movie genres.

The csv files are in the `Data` folder along with the GloVe 6B 100d word embeddings from [Stanford university](https://nlp.stanford.edu/projects/glove/) used to train the model.

I used Postman to send the csv files as binary data to the flask web service in order to test it.