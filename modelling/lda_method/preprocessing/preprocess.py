import gensim
from nltk.stem import SnowballStemmer
import spacy
import os
import json


def preprocess_text(text):
    # Lemmatize and stemming text
    result = []
    stemmer = SnowballStemmer("spanish")
    nlp = spacy.load("es_core_news_sm")
    for token in gensim.utils.simple_preprocess(text):
        if token not in gensim.parsing.preprocessing.STOPWORDS and len(token) > 3:
            token_lemmatized = nlp(token)[0].lemma_
            #token_lemmatized_stemmed = stemmer.stem(token_lemmatized)
            if len(token_lemmatized) > 4:
                result.append(token_lemmatized)
    return result

def preprocess_doc(path_to_original_data, path_to_preprocessed_data, 
                        add_new_preprocessed_data_only):
    filenames = os.listdir(path_to_original_data)
    dict_of_preprocessed_doc= {}
    if add_new_preprocessed_data_only:
        with open(os.path.join(path_to_preprocessed_data, "preprocessed_description_data.json"), "r") as f:
            dict_of_preprocessed_doc = json.load(f)
        filenames = [filename for filename in filenames if filename[1:4] not in dict_of_preprocessed_doc.keys()]
    # for each doc loop
    for filename in filenames:
        doc_id = filename[0:3]
        with open(os.path.join(path_to_original_data, filename), "r") as f:
            doc = f.read()
        # the output will be a list 
        doc_preprocessed = preprocess_text(doc)
        dict_of_preprocessed_doc[doc_id] = doc_preprocessed
        print(doc_id)
    # save to json
    with open(os.path.join(path_to_preprocessed_data, "preprocessed_description_data_lemma_only_resume.json"), "w") as f:
        json.dump(dict_of_preprocessed_doc, f)


if __name__ == "__main__":
    import argparse


    parser = argparse.ArgumentParser(prog="preprocessing_phase", 
                                     description="Cette application prepare le texte pour le topic modelling")
    parser.add_argument('input_path_preprocessing', type=str)
    parser.add_argument('--output_path_preprocessing', type=str, default = os.getcwd())
    parser.add_argument('--add_new_preprocessed_data_only', type=bool, default = False)
    args = parser.parse_args()
    preprocess_doc(path_to_original_data= args.input_path_preprocessing,
                    path_to_preprocessed_data = args.output_path_preprocessing,
                add_new_preprocessed_data_only = args.add_new_preprocessed_data_only)

