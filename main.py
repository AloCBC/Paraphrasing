from text_sim_manager.minhash import MinHash
import time
from nltk import tokenize

def run():
    with open("/Users/samguercho/Projects/Paraphrasing/data/test/test.src", 'r') as f:
        texts = tokenize.sent_tokenize(f.read())
        for i in range(6):
            texts.extend(texts[:(len(texts)//2)])
        start = time.time()
        minhash = MinHash()
        sig, indexed_texts= minhash.fit_transform(texts)
        minhash.compare_hashes(sig, indexed_texts)
        timedelta = time.time() - start
        print(f'The total time to produce the df is {time.strftime("%H:%M:%S", time.gmtime(timedelta))} seconds')


if __name__ == "__main__":
    run()

