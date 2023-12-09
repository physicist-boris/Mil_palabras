import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoTokenizer, GenerationConfig
import os
import gc
import re
import json


def generate_list_subjects(
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
    s = generation_output.sequences[0][input_size:]
    # Re-structure model output
    try:
        output = tokenizer.decode(s).split("### Response:")[1]
        pattern_inside_question = r"(?<=[0-9][.-])[ ¿]*[*a-zA-Z]{1,}([^\n]+)[\?]*"
        matches = []
        for match in re.finditer(pattern_inside_question, output):
            matches.append(match.group())
        output = matches
    except IndexError:
        output = tokenizer.decode(s)
        pattern_inside_question = r"(?<=[0-9][.-])[ ¿]*[*a-zA-Z]{1,}([^\n]+)[\?]*"
        matches = []
        for match in re.finditer(pattern_inside_question, output):
            matches.append(match.group())
        output = matches
    return output

def organize_subjects(path_to_preprocessed_data, path_to_saved_results):
    # Free memory
    gc.collect()
    torch.cuda.empty_cache()
    
    already_processed_files = os.listdir(path_to_preprocessed_data)

    dict_topics_by_text = {}
    for filename in already_processed_files:
        with open(os.path.join(path_to_preprocessed_data, filename), "r", encoding="utf-8") as f:
            text = f.read()
        text = text.strip(' \n ')
        # Free memory
        gc.collect()
        torch.cuda.empty_cache()
        # Reorganize text topics in list with llama
        instruction = "With the following text, provide a summarized list of main topics in spanish using 2 or 3 words each. The response must have this format: \n ### Response:  \n\n 1. \n 2. \n\n" + text
        response = generate_list_subjects(instruction=instruction)
        dict_topics_by_text[filename[0:3]] = response
    # Save results in json file
    with open(os.path.join(path_to_saved_results,"results_topics_by_text.json"), "w") as f:
        json.dump(dict_topics_by_text, f)
