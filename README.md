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

# Text Similarity Manager - LSH MinHash Algorithm
If you want to compare a large amount of texts (>10k) in a fast enough time, the LSH algorithms can be the way.
To know more about LSH, I found this article very clear: https://www.pinecone.io/learn/series/faiss/locality-sensitive-hashing/

The principle is always the following: We would like a hash which, instead of avoiding collision like ShA-2, has a chance to collide only texts in a close distance to each other.

<img src="https://github.com/SamGuercho/Paraphrasing/assets/57171996/b178522d-4291-4d9f-b45b-e57993e7b910" width="600" height="300">

Among the alorithm, I decided to develop the MinHash algorithm in a way that would allow to do the following:
- For a certain number of hash functions, always keep the same seeds for hashing
- Flexible enough to add new use cases. ex: do we want to apply on all the texts as standalone, or to compare to older texts?
- Enrich little by little the inputs to hash: triplets of characters, tokens, ascii characters...

You can find the package in the file text_sim_manager.

## How to use the MinHash package

Import your texts as a list of texts:

    from nltk import tokenize
    with open("path/to/your/text", 'r') as f:
        texts = tokenize.sent_tokenize(f.read())

Then run the following:

    from text_sim_manager.minhash import MinHash
    import time
    
    start = time.time()
    
    minhash = MinHash()
    signatures, indexed_texts = minhash.fit_transform(texts)
    minhash.compare_hashes(signatures, indexed_texts)
    
    timedelta = time.time() - start
    print(f'Estimated time to compare {len(texts)}: {time.strftime("%H:%M:%S", time.gmtime(timedelta))}.

Your output is a dataframe having this format:
<img width="846" alt="image" src="https://github.com/SamGuercho/Paraphrasing/assets/57171996/ae296ce1-c275-41fd-a441-256c1827afba">

Where the indexes correspond to the apparition of each text in the list of texts.
