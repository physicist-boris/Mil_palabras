import gensim
import json
import matplotlib.pyplot as plt
import numpy as np
import mlflow


def lda_model(path_to_preprocessed_data, runame):
    with open(path_to_preprocessed_data, "r") as f:
        dict_preprocessed_data = json.load(f)
    processed_docs = [dict_preprocessed_data[doc_id] for doc_id in dict_preprocessed_data.keys()]
    dictionary = gensim.corpora.Dictionary(processed_docs)
    bow_corpus = [dictionary.doc2bow(doc) for doc in processed_docs]
    #bow_corpus[4310]


    tfidf = gensim.models.TfidfModel(bow_corpus)
    corpus_tfidf = tfidf[bow_corpus]
    list_of_coherence_score = []
    list_of_perplexity_score = []
    with mlflow.start_run(run_name = runame) as run:
        for numb_topics in range(15, 16):
            with mlflow.start_run(nested=True, run_name = f"model_with_{numb_topics}_topics") as child_run:
                mlflow.log_param('number of topics', numb_topics)
                mlflow.set_tag('model_numb_topics', f'{ numb_topics }')
                lda_model_tfidf = gensim.models.LdaMulticore(corpus_tfidf, num_topics=numb_topics, id2word=dictionary, passes=2, workers=4)
                #calculate coherence and change corpus evaluation
                cm = gensim.models.coherencemodel.CoherenceModel(model= lda_model_tfidf, corpus = bow_corpus, texts = processed_docs, coherence= "c_v")
                coherence_score = cm.get_coherence()
                #calculate perplexity and change corpus evaluation
                perplexity_score = lda_model_tfidf.log_perplexity(corpus_tfidf)
                mlflow.log_metric('c_v_coherence', coherence_score) 
                mlflow.log_metric("perplexity", perplexity_score)
                list_of_coherence_score.append(coherence_score)
                list_of_perplexity_score.append(perplexity_score)
                save_info_topics = ""
                for idx, topic in lda_model_tfidf.print_topics(-1):
                    save_info_topics += 'Topic: {} Word: {}'.format(idx, topic) + "\n"
                #verfifier si mflow marche sauvegarde en memoire
                with open("output/files/saved_topics.txt", "w") as f:
                    f.write(save_info_topics)
                mlflow.log_artifact('output/files/saved_topics.txt', artifact_path = "output/files")
                lda_model_tfidf.save(f"output/saved_models/model_with_{numb_topics}_topics.model")
                mlflow.log_artifact(f"output/saved_models/model_with_{numb_topics}_topics.model", artifact_path = "output/models" )
                mlflow.log_artifact(f"output/saved_models/model_with_{numb_topics}_topics.model.expElogbeta.npy", artifact_path = "output/models" )
                mlflow.log_artifact(f"output/saved_models/model_with_{numb_topics}_topics.model.id2word", artifact_path = "output/models" )
                mlflow.log_artifact(f"output/saved_models/model_with_{numb_topics}_topics.model.state", artifact_path = "output/models" )
            print(numb_topics)
        plt.plot(np.arange(1,35,1), np.asarray(list_of_coherence_score), 'r-', label = "coherence c_v")
        plt.xlabel("num_topics")
        plt.ylabel("Coherence score")
        plt.legend()
        plt.savefig("output/figures/coherence_score.png")
        mlflow.log_artifact('output/figures/coherence_score.png', artifact_path = "output/figures")
        plt.clf()
        plt.plot(np.arange(1,35,1), np.asarray(list_of_perplexity_score), 'b-', label = "perplexity")
        plt.xlabel("num_topics")
        plt.ylabel("Perplexity score")
        plt.legend()
        plt.savefig("output/figures/perplexity_score.png")
        mlflow.log_artifact('output/figures/perplexity_score.png', artifact_path = "output/figures")


if __name__ == "__main__":
    import argparse


    MLFLOW_TRACKING_URI = "sqlite:///mlflow.db"
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment("topic_modelling_lda_LLM")  
    

    parser = argparse.ArgumentParser(prog="lda_training_evaluation_phase", 
                                     description="Cette application utilise la méthode LDA pour trouver le topic modelling")
    parser.add_argument('path_to_preprocessed_data', type=str)
    parser.add_argument('runame', type=str)
    args = parser.parse_args()
    lda_model(args.path_to_preprocessed_data, runame=args.runame)
