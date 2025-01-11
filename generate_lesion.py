import os
import torch
import argparse
import numpy as np
import warnings
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import logging

# Suppress warnings
warnings.filterwarnings("ignore")
logging.set_verbosity_error()

from models.modeling_gpt2 import GPT2LMHeadModel
from models.modeling_llama import LlamaForCausalLM
from models.modeling_phi3 import Phi3ForCausalLM
from models.modeling_gemma import GemmaForCausalLM
from models.modeling_falcon import FalconForCausalLM
from models.modeling_mistral import MistralForCausalLM

CACHE_DIR = os.environ.get("LOC_CACHE", f"cache")

if __name__ == "__main__":

    argparser = argparse.ArgumentParser()
    argparser.add_argument("--model-name", type=str, required=True)
    argparser.add_argument("--prompt", type=str, required=True)
    argparser.add_argument("--percentage", type=float, required=True)
    argparser.add_argument("--network", type=str, default="language", choices=["language", "moral", "random", "none"])   
    argparser.add_argument("--device", type=str, default=None)
    argparser.add_argument("--seed", type=int, default=42)
    argparser.add_argument("--pooling", type=str, default="last-token", choices=["last-token", "mean"])
    argparser.add_argument("--localize-range", type=str, default="100-100")

    args = argparser.parse_args()

    seed = args.seed
    percentage = args.percentage
    device = args.device if args.device is not None else "cuda" if torch.cuda.is_available() else "cpu"
    model_name = args.model_name
    network = args.network
    prompt = args.prompt
    pooling = args.pooling
    loc_range = args.localize_range

    print(f"--------------------------------------------------------")
    print(f"> Running with model {model_name}")

    if "gpt2" in model_name.lower():
        model = GPT2LMHeadModel.from_pretrained(model_name, trust_remote_code=True)
        tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side='left')
        tokenizer.pad_token = tokenizer.eos_token
    elif "llama" in model_name.lower():
        model = LlamaForCausalLM.from_pretrained(model_name)
    elif "phi" in model_name.lower():
        if "phi-3" in model_name.lower():
            model = Phi3ForCausalLM.from_pretrained(model_name)
        else:  # phi-4
            model = Phi3ForCausalLM.from_pretrained(model_name, trust_remote_code=True)
    elif "gemma" in model_name.lower():
        model = GemmaForCausalLM.from_pretrained(model_name)
    elif "falcon" in model_name.lower():
        model = FalconForCausalLM.from_pretrained(model_name)
    elif "mistral" in model_name.lower():
        model = MistralForCausalLM.from_pretrained(model_name)
    else:
        raise ValueError(f"Model {model_name} not supported")

    if "gpt2" not in model_name.lower():
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.pad_token = tokenizer.eos_token

    model.to(device)
    model.eval()
    
    # Only print a simplified version of the model architecture
    print(f"> Model loaded successfully with {sum(p.numel() for p in model.parameters())} parameters")

    model_name = os.path.basename(model_name)

    print(f"> Running with {network} mask")

    if network in ["moral", "random"]:
        # mask_path = f"{model_name}_network=language_pooling={pooling}_range={loc_range}_perc={percentage}_nunits=None_pretrained=True.npy"
        mask_path = f"{model_name}_network=moral_pooling={pooling}_range={loc_range}_perc={percentage}_nunits=None_pretrained=True.npy"
    else:
        mask_path = None

    if mask_path is not None:
        language_mask = np.load(f"{CACHE_DIR}/{mask_path}")
        num_active_units = int(language_mask.sum())

        if network == "random":
            num_layers, hidden_dim = language_mask.shape
            total_num_units = np.prod(language_mask.shape)
            invlang_mask_indices = np.arange(total_num_units)[(1 - language_mask).flatten().astype(bool)]
            np.random.seed(seed)
            rand_indices = np.random.choice(invlang_mask_indices, size=num_active_units, replace=False)
            lang_mask_rand = np.full(total_num_units, 0)
            lang_mask_rand[rand_indices] = 1
            assert np.sum(lang_mask_rand) == num_active_units
            language_mask = lang_mask_rand.reshape((num_layers, hidden_dim))


        model.set_language_selective_mask(torch.tensor(language_mask).to(device))
        print("Loaded network mask with", num_active_units, "units, with shape", language_mask.shape)
    else:
        model.set_language_selective_mask(None)

    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    outputs = model.generate(
        **inputs,
        max_new_tokens=20,
        do_sample=True,
        temperature=0.7,
        num_return_sequences=1,
    )

    print(tokenizer.decode(outputs[0], skip_special_tokens=True))