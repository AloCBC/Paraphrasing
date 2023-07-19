# Paraphrasing

## Welcome to the Paraphrasing Identification Project!

Here we will try different ways to define and group texts based on their topic.
We would like to classify:
- When two texts are the paraphrase of each other
- When they talk about the same thing and have the same opinion
- When they talk about the same thing and have diverging opinion
- When they don't talk about the same thing

## Existing Models and goal of this project

Very good models exist already. To mention only famous companies on this activity: Quillbot and Grammarly.
Despite Deep Learning being the ideal solution today ofr this task, the goal of this project is to find as many diverse ways to get close to the result, Deep Learning included.

An example of non Deep Learning project would be:
- Find how much texts look like each other
- Cluster them by similarity with unlimited number of clusters
- Check if those groups of texts are indeed similar or not.

It will be done in the Text Similarity Manager project.
As of now, the following tools are being put to disposal:
- Similarity Manager

# Text Similarity Manager
The idea behind this project is to be able to compare a large amount of texts (>10k) and give a similarity score to all of them in less than 10 min.
For this, I developped a package based on the MinHash Algorithm.
To know more about minhashing and LSH, I found this article very clear: https://www.pinecone.io/learn/series/faiss/locality-sensitive-hashing/
