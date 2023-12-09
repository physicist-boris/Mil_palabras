import gensim
from modelling.lda_method.preprocessing.preprocess import preprocess_text
import os
import docx2txt
import json


def classify_episode(path_to_registered_model, path_to_registered_tokenizer, path_data_to_classify, path_to_saved_results):
    lda_llm_model = gensim.models.LdaMulticore.load(path_to_registered_model,)
    filenames = os.listdir(path_data_to_classify)
    processed_docs = []
    for filename in filenames: 
        doc = docx2txt.process(os.path.join(path_data_to_classify, filename))
        processed_docs.append(preprocess_text(doc))
    with open(path_to_registered_tokenizer, "r") as f:
        dictionary = json.load(f)
    bow_corpus = [dictionary.doc2bow(doc) for doc in processed_docs]
    tfidf = gensim.models.TfidfModel(bow_corpus)
    corpus_tfidf = tfidf[bow_corpus]
    dict_topics_by_corpus ={}
    for i in range(0, len(corpus_tfidf)):
        dict_topics_by_corpus[i] = []
        compteur_topic = 0
        for index, score in sorted(lda_llm_model[corpus_tfidf[i]], key=lambda tup: -1*tup[1]):
             dict_topics_by_corpus[i].append(lda_llm_model.print_topic(index, 10))
             compteur_topic += 1
             if compteur_topic == 4:
                 break
        print(i)
    dict_corpus_by_topics = {}
    for k,v in dict_topics_by_corpus.items():
        for x in v:
            dict_corpus_by_topics.setdefault(x, []).append(k)
    with open(os.path.join(path_to_saved_results,"classification_result.json"), "w") as f:
        json.dump(dict_corpus_by_topics, f)

if __name__ == "__main__":
    import argparse


    parser = argparse.ArgumentParser(prog="classifier_episode", 
                                     description="Cette application classifie les episodes d'un podcast en se basant sur la description")
    parser.add_argument('path_to_registered_model', type=str)
    parser.add_argument('path_data_to_classify', type=str)
    parser.add_argument('path_to_registered_tokenizer', type=str)
    parser.add_argument('path_to_saved_results', type=str)
    args = parser.parse_args()
    classify_episode(args.path_to_registered_model,
                     args.path_to_registered_tokenizer,
                     args.path_data_to_classify,
                     args.path_to_saved_results)