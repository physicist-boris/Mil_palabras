import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoTokenizer, GenerationConfig
import os
import gensim
import json
from modelling.lda_method.preprocessing.preprocess import preprocess_text


def create_instruction(instruction, input_data=None, context=None):
    sections = {
        "Instrucción": instruction,
        "Entrada": input_data,
        "Contexto": context,
    }

    system_prompt = "A continuación hay una instrucción que describe una tarea, junto con una entrada que proporciona más contexto. Escriba una respuesta que complete adecuadamente la solicitud.\n\n"
    prompt = system_prompt

    for title, content in sections.items():
        if content is not None:
            prompt += f"### {title}:\n{content}\n\n"

    prompt += "### Respuesta:\n"

    return prompt


def generate(
        instruction,
        input=None,
        context=None,
        max_new_tokens=160,
        temperature=0.1,
        top_p=0.75,
        top_k=40,
        num_beams=4,
        **kwargs
):
    
    model_id = "clibrain/Llama-2-7b-ft-instruct-es"

    model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True, device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    
    prompt = create_instruction(instruction, input, context)
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"].to("cuda")
    attention_mask = inputs["attention_mask"].to("cuda")
    generation_config = GenerationConfig(
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        num_beams=num_beams,
        do_sample=True,
        **kwargs,
    )
    with torch.no_grad():
        generation_output = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            generation_config=generation_config,
            return_dict_in_generate=True,
            output_scores=True,
            max_new_tokens=max_new_tokens,
            early_stopping=True
        )
    s = generation_output.sequences[0]
    output = tokenizer.decode(s)
    return output.split("### Respuesta:")[1].lstrip("\n")

def generate_for_all(path_to_original_data, path_to_sample_prompt, register_tokenizer,
                     path_to_registered_tokenizer):
    filenames = os.listdir(path_to_original_data)
    outputs = []
    for filename in filenames:
        input_data = docx2txt.process(os.path.join(path_to_original_data, filename))
        instruction = "resumen este texto y los tópicos o asuntos clave que aborda sin repetición."
        output = generate(instruction, input = input_data)
        outputs.append(output)
        """
        instruction = "Enumera los temas en este texto en forma de lista.\n\n\n Por ejemplo,\n\n"
        instruction += docx2txt.process(path_to_sample_prompt)
        final_output= generate(instruction, input = output, max_new_tokens=100)
        """
        with open(f"output/resume_generated/{filename[1:4]}_resume.txt", "w") as f:
            f.write(output)

        print("### Respuesta:\n")
        print(output)
    if register_tokenizer:
        processed_docs_dictionary = [preprocess_text(doc) for doc in outputs]
        dictionary = gensim.corpora.Dictionary(processed_docs_dictionary)
        with open(os.path.join(path_to_registered_tokenizer, "registered_tokenizer.json"), "w") as f:
            json.dump(dictionary, f)
    return outputs


if __name__ == "__main__":
    import gc
    from tqdm import tqdm
    import docx2txt
    import argparse


    parser = argparse.ArgumentParser(prog="train_llama2-7b_phase", 
                                     description="Cette application entraine llama2-7b a extraire les sujets importants dans un texte en espagnol")
    parser.add_argument('path_to_original_data', type=str)
    parser.add_argument('--path_to_sample_prompt', type=str, default = os.getcwd())
    parser.add_argument('--register_tokenizer', type=bool, default = False)
    parser.add_argument('--path_to_registered_tokenizer', type=str, default = os.getcwd())
    args = parser.parse_args()

    tqdm.pandas()
    #free memory
    gc.collect()
    torch.cuda.empty_cache()

    path_to_original_data = args.path_to_original_data
    path_to_sample_prompt = args.path_to_sample_prompt
    generate_for_all(path_to_original_data = args.path_to_original_data, path_to_sample_prompt = args.path_to_sample_prompt,
                     register_tokenizer=args.register_tokenizer,
                     path_to_registered_tokenizer=args.path_to_registered_tokenizer)