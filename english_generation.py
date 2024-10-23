from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from datetime import datetime

device = torch.device("cuda")


def setup_model():
    model_id = "openai-community/gpt2"
    tokenizer = AutoTokenizer.from_pretrained(model_id, add_prefix_space=False)
    tokenizerTEMP = AutoTokenizer.from_pretrained(
        model_id, add_prefix_space=True
    )
    model_gpt = AutoModelForCausalLM.from_pretrained(
        model_id, pad_token_id=tokenizer.eos_token_id, device_map=device
    ).to(device)
    return model_gpt, tokenizer, tokenizerTEMP


def define_excluded_and_punc_tokens(tokenizer, tokenizerTEMP):
    excluded_tokens = tokenizerTEMP(
        ["\n", "http", "www.", "UESPWiki", "\n\n", "\n\n\n", "50256"]
    ).input_ids
    excluded_tokens += tokenizer(
        ["\n", "http", "www.", "UESPWiki", "\n\n", "\n\n\n", "50256"]
    ).input_ids
    punc_tokens = np.arange(0, 250, 1)
    return excluded_tokens, punc_tokens


def get_prompt_text(prompt):
    prompts = {
        1: "I have a religion because it is crazy how I like religion and what they have to offer to society",
        2: "I can't handle this for any longer",
        3: "We all know earth is",
        4: "No way! How did you ",
        5: "I had a terrifying dream last night in which",
    }
    return prompts.get(prompt, "")


def prepare_df_para_data(
    sampling_methods, ps, temperature_range, span_values, nSim_each
):
    df_para_data = []
    sim_count = 0
    for sampling_method in sampling_methods:
        for p in ps[sampling_method]:
            for temperature in temperature_range:
                for span_value in span_values:
                    for rep in range(nSim_each):
                        backward_attention_length = span_value
                        df_para_data.append(
                            {
                                "sim": sim_count,
                                "repetition": rep,
                                "sampling_method": sampling_method,
                                "sampling_parameter": p,
                                "temperature": temperature,
                                "backward_attention_length": backward_attention_length,
                                "forward_attention_length": 3,
                            }
                        )
                        sim_count += 1
    return df_para_data


def metrics_calculations(logits_tensor, output_only):
    probs = torch.nn.functional.softmax(logits_tensor, dim=-1)
    max_probs, _ = torch.max(probs, dim=-1)
    log_probs = torch.log(probs + 1e-9)
    actual_probs = torch.gather(probs, 1, output_only.t()).t().squeeze()

    probability_differences_tensor_single_generation = max_probs - actual_probs
    entropy_tensor_single_generation = (-probs * log_probs).sum(dim=-1)
    information_content_tensor_single_generation = -torch.log(
        actual_probs + 1e-9
    )
    entropy_deviations_tensor_single_generation = (
        entropy_tensor_single_generation
        - information_content_tensor_single_generation
    )

    return (
        probability_differences_tensor_single_generation,
        entropy_tensor_single_generation,
        information_content_tensor_single_generation,
        entropy_deviations_tensor_single_generation,
    )


def generate_single_step(
    model_gpt, input_ids_temp, gen_args, excluded_tokens, row, num_beams, device
):
    """
    Generate a single step of text based on the current input_ids and generation arguments.
    """
    gen_args["input_ids"] = input_ids_temp
    gen_args["attention_mask"] = torch.ones_like(input_ids_temp).to(device)
    gen_args["max_length"] = row["forward_attention_length"] + input_ids_temp.shape[1]
    gen_args["min_length"] = row["forward_attention_length"] + input_ids_temp.shape[1]
    
    temp_output = model_gpt.generate(**gen_args) #NB ALWAYS LENGTH OF 6 but then output_all evolves normally
    logits_tensor = torch.cat(
        [
            temp_output.logits[i][temp_output.beam_indices[0, i], :].unsqueeze(0)
            for i in range(len(temp_output.logits))
        ],
        dim=0,
    )

    return temp_output, logits_tensor


def generate_samples_and_metrics(
    df_para,
    tokenizer,
    input_ids,
    model_gpt,
    excluded_tokens,
    punc_tokens,
    num_beams,
    target_length,
    prompt,
):
    results_list_single_prompt = []
    
    for index, row in tqdm(df_para.iterrows(), total=df_para.shape[0]):
        
        torch.cuda.empty_cache()
        output_all = input_ids
        input_ids_temp = input_ids[
            :,
            max(
                0,
                input_ids.shape[1] - df_para.backward_attention_length[index],
            ) : input_ids.shape[1],
        ]
        cont = True

        # initialization of metrics'logits
        probability_differences_tensor = torch.tensor([]).to(device)
        entropy_tensor = torch.tensor([]).to(device)
        information_content_tensor = torch.tensor([]).to(device)
        entropy_deviations_tensor = torch.tensor([]).to(device)

        sampling_method = row["sampling_method"]
        input_length = row["backward_attention_length"] 
        
        gen_args = {
            "do_sample": True,
            "bad_words_ids": excluded_tokens,
            "num_return_sequences": 1,
            "temperature": row["temperature"],
            "num_beams": num_beams,
            "no_repeat_ngram_size": 2,
            "pad_token_id": model_gpt.config.pad_token_id,
            "return_dict_in_generate": True,
            "output_logits": True,
            "output_scores": True 
        }

        if sampling_method == "top_p":
            gen_args["top_p"] = row["sampling_parameter"]
        elif sampling_method == "top_k":
            gen_args["top_k"] = row["sampling_parameter"]
        elif sampling_method == "typical_p":
            gen_args["typical_p"] = row["sampling_parameter"]


        first_iteration = True

        while cont:
            if first_iteration : 
                input_ids_temp=input_ids
                first_iteration=False

            temp_output, logits_tensor = generate_single_step(
            model_gpt, input_ids_temp, gen_args, excluded_tokens, row, num_beams, device)

            output_only = temp_output.sequences[:, input_ids_temp.shape[1]:]
            output_all = torch.cat((output_all, output_only), 1)
            if output_all.shape[1] >= target_length:
                cont = False
            
            # Use only last N tokens as input for next iteration, where N is a function of retrospective span (backward_attention_length), and punctuation tokens are ignored
            nPunct = sum(
                sum(
                    output_all[0, -input_length:] == i for i in punc_tokens
                ).bool()
            )
            it_input_length = input_length + nPunct.item()
            it_input_length = min(it_input_length, output_all.shape[1])
            input_ids_temp = output_all[:, -it_input_length:]
            
            (
                probability_differences_tensor_single_generation,
                entropy_tensor_single_generation,
                information_content_tensor_single_generation,
                entropy_deviations_tensor_single_generation,
            ) = metrics_calculations(logits_tensor, output_only)
            probability_differences_tensor = torch.cat(
                (
                    probability_differences_tensor,
                    probability_differences_tensor_single_generation,
                ),
                dim=0,
            )
            entropy_tensor = torch.cat(
                (entropy_tensor, entropy_tensor_single_generation), dim=0
            )
            information_content_tensor = torch.cat(
                (
                    information_content_tensor,
                    information_content_tensor_single_generation,
                ),
                dim=0,
            )
            entropy_deviations_tensor = torch.cat(
                (
                    entropy_deviations_tensor,
                    entropy_deviations_tensor_single_generation,
                ),
                dim=0,
            )

        results = tokenizer.decode(output_all[0], skip_special_tokens=False)
        print(len(results))

        single_sample = {
            "repetition": row["repetition"],
            "prompt": prompt,
            "sampling_method": sampling_method,
            "p": row["sampling_parameter"],
            "temperature": row["temperature"],
            "backward_attention_length": row["backward_attention_length"],
            "result": results,
            "probability_differences_tensor": probability_differences_tensor,
            "entropy_tensor": entropy_tensor,
            "information_content_tensor": information_content_tensor,
            "entropy_deviations_tensor": entropy_deviations_tensor,
        }

        results_list_single_prompt.append(single_sample)

    return results_list_single_prompt


def generate_samples_for_prompts(
    nSim_each,
    num_beams,
    target_length,
    prompts,
    sampling_methods,
    ps,
    temps,
    contexts,
):
    model_gpt, tokenizer, tokenizerTEMP = setup_model()

    # Define excluded tokens and punctuation tokens
    excluded_tokens, punc_tokens = define_excluded_and_punc_tokens(
        tokenizer, tokenizerTEMP
    )

    results_list = []
    for prompt in prompts:
        start = get_prompt_text(prompt)
        input_ids = tokenizer.encode_plus(
            start, return_tensors="pt", add_special_tokens=False
        ).to(device)
        input_ids = input_ids.input_ids

        df_para_data = prepare_df_para_data(
            sampling_methods, ps, temps, contexts, nSim_each
        )
        df_para = pd.DataFrame(df_para_data)
        print(df_para)
       
        results_list_single_prompt = generate_samples_and_metrics(
            df_para,
            tokenizer,
            input_ids,
            model_gpt,
            excluded_tokens,
            punc_tokens,
            num_beams,
            target_length,
            prompt,
        )
        results_list.extend(results_list_single_prompt)

    results_df = pd.DataFrame(results_list)
    save_dataframe(results_df, 'results') # Save the DataFrame with a unique filename

    return results_df


def save_dataframe(results_df, filename_base):
    now = datetime.now()
    date_str = now.strftime("%d.%m")
    timestamp = now.strftime("%H:%M")
    filename = f"/home/ubuntu/helene/{filename_base}_{date_str}_{timestamp}.csv"
    results_df.to_csv(filename, index=False)
    print(f"DataFrame saved to {filename}")

    return results_df


def main_generate():
    
    # nSim_each = 5
    # num_beams = 3
    # target_length = 200
    # sampling_methods = ["top_p", "top_k", "typical_p"]
    # ps = {
    #     "top_p": [0.2, 0.4, 0.6, 0.8, 1.0],
    #     "top_k": [4, 8, 16, 32, 64],
    #     "typical_p": [0.2, 0.4, 0.6, 0.8, 1.0],
    # }

    # prompts = [1, 2, 3, 4, 5, 6]
    # temperatures = [0.01, 0.25, 0.5, 0.75, 
    #                 1.0, 1.25, 1.5, 1.75, 2.0]
    # retroactive_span_values = [2, 4, 8, 16, 32, 64, 128, 256]

    nSim_each = 1
    num_beams = 3
    target_length = 100
    sampling_methods = [ "typical_p", "top_p"]
    ps = {
        "top_p": [ 0.5, 1.0],
        "top_k": [4, 8, 16, 32, 64],
        "typical_p": [ 0.5, 1.0],
    }

    prompts = [1, 2]
    temperatures = [0.01, 1.0, 2.0]
    retroactive_span_values = [4, 16, 128, 256]


    final = generate_samples_for_prompts(
        nSim_each,
        num_beams,
        target_length,
        prompts,
        sampling_methods,
        ps,
        temperatures,
        retroactive_span_values,
    )
    return final

if __name__ == "__main__":
    
    main_generate()
