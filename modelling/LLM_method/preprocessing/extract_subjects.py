import re 
import os
import docx2txt
import gc
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoTokenizer, GenerationConfig


def extract_by_regex(path_to_original_data, path_to_preprocessed_data):
    pattern_listed_subjects = r"(?<=[*•-])[ ¿]*[a-zA-Z]{1,}([^\n]+)[\?]*"
    pattern_inside_question = r"(?<=¿)[a-zA-Z 0-9,]*([^\n]+)(?=\?)"
    pattern_detect_etiquetas = r"(?<=etiquetas:)[a-zA-Z 0-9,]*([^\n]+)"

    filenames = os.listdir(path_to_original_data)

    dict_subjects_by_text = {}
    for filename in filenames:
        text = docx2txt.process(os.path.join(path_to_original_data, filename))
        dict_subjects_by_text[filename[1:4]] = []
        # match listed subject theme in text description
        for match in re.finditer(pattern_listed_subjects, text):
            dict_subjects_by_text[filename[1:4]].append(match.group())
        # replace spanish question by only matching the sentence inside the question
        for i in range(0, len(dict_subjects_by_text[filename[1:4]])):
            match = re.search(pattern_inside_question, dict_subjects_by_text[filename[1:4]][i])
            if match:
                dict_subjects_by_text[filename[1:4]][i] = match.group()
        # match subject element listed in text etiquetas
        text = text.lower()
        match = re.search(pattern_detect_etiquetas, text)
        if match:
            etiquetas_matches = match.group().split(",")
            for etiquetas in etiquetas_matches:
                dict_subjects_by_text[filename[1:4]].append(etiquetas)
        # save the matched subjects for the text in a text file with the same id
        if len(dict_subjects_by_text[filename[1:4]]) != 0:
            subjects_text = "\n".join(dict_subjects_by_text[filename[1:4]])
            with open(os.path.join(path_to_preprocessed_data, f"{filename[1:4]}.txt"), "w", encoding="utf-8") as f:
                f.write(subjects_text)


def generate_subject(
        instruction,
        max_new_tokens=290,
        temperature=0.01,
        top_p=0.75,
        top_k=40,
        **kwargs
):
    # Load hf llama model and tokenizer with transformers lib
    model_id = "meta-llama/Llama-2-13b-chat-hf"
    model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True, device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    # Tokenize data and prepare config for inference with the model
    inputs = tokenizer(instruction, return_tensors="pt")
    input_ids = inputs["input_ids"].to("cuda")
    input_size = input_ids.shape[1]
    attention_mask = inputs["attention_mask"].to("cuda")
    generation_config = GenerationConfig(
        temperature=temperature,
        repetition_penalty=1.0,
        top_p=top_p,
        top_k=top_k,
        do_sample=True,
        **kwargs,
    )
    # Model generation
    with torch.no_grad():
        generation_output = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            generation_config=generation_config,
            return_dict_in_generate=True,
            output_scores=True,
            max_new_tokens=max_new_tokens
        )
    # Re-structure model output
    s = generation_output.sequences[0][input_size:]
    try:
        output = tokenizer.decode(s).split("### Response:")[1]
    except IndexError:
        output = tokenizer.decode(s)
    return output

def extract_by_llm(path_to_original_data, path_to_preprocessed_data):
    # free memory
    gc.collect()
    torch.cuda.empty_cache()

    filenames_to_process = os.listdir(path_to_original_data)
    # extract filenames of already preprocessed files
    already_processed_files = os.listdir(path_to_preprocessed_data)
    already_processed_filenames = []
    for filename in already_processed_files:
        already_processed_filenames.append(filename[0:3])


    for filename in filenames_to_process:
        if filename[1:4] not in already_processed_filenames:
            text = docx2txt.process(os.path.join(path_to_original_data, filename))
            # free memory
            gc.collect()
            torch.cuda.empty_cache()
            # Generate subject extraction by llm and save in a text file
            instruction = "With the following text, provide a summarized list of main topics in spanish.\n\n" + text
            response = generate_subject(instruction=instruction)
            with open(os.path.join(path_to_preprocessed_data, f"{filename[1:4]}.txt"), "w", encoding="utf-8") as f:
                f.write(response)
            