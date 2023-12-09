import json
import tensorflow_hub as hub
import numpy as np
from numpy.linalg import norm
import os 


def generate_contents_table(path_to_saved_results):
    # Load USE model from tensorflow hub
    embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")
    # Load data
    with open(os.path.join(path_to_saved_results, r"results_topics_by_text.json"), "r") as f:
        dict_results_topics_by_text = json.load(f)

    subjects = []
    for key in dict_results_topics_by_text.keys():
        for subject in dict_results_topics_by_text[key]:
            subjects.append(subject)


    # Generate embeddings for the sentences
    embeddings = embed(subjects)
    array_embeddings = embeddings.numpy()
    
    # Aggregate similar topics into one topic using cosine similarity
    while array_embeddings.shape[0] >= 1:
        A = array_embeddings[0:, :]
        B = array_embeddings[0, :]
        # Compute cosine similarity and generate cluster of subjects
        cosine = np.dot(A,B)/(norm(A, axis =1)*norm(B))
        index_subjects_chosen = [index_cos for index_cos, cos in enumerate(cosine) if cos >=0.60]
        cluster_subjects = [subjects[i] for i in index_subjects_chosen]
        print(cluster_subjects)
        # Replace subjects of the same cluster by the first subject of the cluster. Similar to choosing a title for the cluster
        for key in dict_results_topics_by_text.keys():
            for subject in cluster_subjects:
                if subject in dict_results_topics_by_text[key]:
                    index_element = dict_results_topics_by_text[key].index(subject)
                    dict_results_topics_by_text[key][index_element] = cluster_subjects[0]
        # Delete subjects which have been clusterised and replaced
        for index in sorted(index_subjects_chosen, reverse=True):
            del subjects[index]
        array_embeddings = np.delete(array_embeddings, index_subjects_chosen, axis=0)
        

    # Reverse key value dict to generate a dict of subjects as key and episode ids as values
    dict_text_by_topics = {}
    for k,v in dict_results_topics_by_text.items():
        for x in v:
            dict_text_by_topics.setdefault(x, []).append(k)
    with open(os.path.join(path_to_saved_results, r"classification_result.json"), "w") as f:
        json.dump(dict_text_by_topics, f)
