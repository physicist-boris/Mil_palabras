if __name__ == "__main__":
    import argparse
    import os
    from preprocessing.extract_subjects import extract_by_regex, extract_by_llm
    from organize_subjects_by_text import organize_subjects
    from generate_tables import generate_contents_table


    parser = argparse.ArgumentParser(prog= "Module de Preprocessing", description="Phase de preprocessing pour le clustering des textes de description audio de Mil Palabras")
    parser.add_argument('path_to_original_data', type=str)
    parser.add_argument('--path_to_preprocessed_data', type=str, default = os.getcwd())
    parser.add_argument('--path_to_saved_results', type=str, default = os.getcwd())
    args = parser.parse_args()


    extract_by_regex(path_to_original_data = args.path_to_original_data,
                           path_to_preprocessed_data = args.path_to_preprocessed_data)
    extract_by_llm(path_to_original_data = args.path_to_original_data,
                           path_to_preprocessed_data = args.path_to_preprocessed_data)
    organize_subjects(path_to_preprocessed_data=args.path_to_preprocessed_data, 
                      path_to_saved_results=args.path_to_saved_results)
    generate_contents_table(path_to_saved_results=args.path_to_saved_results)

