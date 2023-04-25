import argparse
import sys
import functools

import torch
import numpy as np

from typing import List, Callable

from transformers import GPT2TokenizerFast

from bindings.pytorch import torch_binding


torch_binding_cpp_path = "./bindings/pytorch/torchbind.cpp"
rwkv_cuda_path = "./include/rwkv/cuda/rwkv.cu"
headers_include_path = "./include/"

def typical_sampler(logits, temp=1.0, tau=0.95, **kwargs):
    
    probs = torch.nn.functional.softmax(logits.float(), dim=-1)
    logits = -torch.log(probs)
    
    ent = torch.nansum(logits * probs, dim=-1, keepdim=True)
    shifted_logits = torch.abs(logits - ent)
    
    sorted_ids = torch.argsort(shifted_logits)
    sorted_logits = shifted_logits[sorted_ids]
    
    sorted_probs = probs[sorted_ids]
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1).cpu().numpy()
    
    cutoff = np.sum(cumulative_probs < tau)
    probs[shifted_logits > sorted_logits[cutoff]] = 0
    
    if temp != 1.0:
        probs = probs ** (1.0 / temp)
    
    out = torch.multinomial(probs, num_samples=1)[0]
    return int(out)


def init_state(*, model, tokens):
    state = model.emptyState
    for token in tokens:
        output, state = model.forward(token, state=state)
    return state
    
def generate_tokens(
    *, 
    model,
    input_tokens: List[int],
    sampler_func: Callable, 
    tokens_to_generate: int,
    ):
    
    state = init_state(model=model, tokens=input_tokens)
    last_token = input_tokens[-1]
    for _ in range(tokens_to_generate):
    
        output, state = model.forward(last_token, state)
        last_token = sampler_func(output)
        yield last_token
        

def chat(
    *, 
    model,
    reverse_sequence="Bob:",
    sampler_func=functools.partial(typical_sampler, temp=1, tau=0.5),
    tokenizer=GPT2TokenizerFast.from_pretrained("EleutherAI/gpt-neox-20b"),
    ):
    """A chat between Alice and Bob"""
    
    reverse_sequence_window = len(reverse_sequence) * -1
    
    chat_history = ""
    chat_history += "Bob: {0}\n\nAlice:".format(input("Bob: "))
    print("==============")
    print(chat_history, end='')
    while True:
        input_tokens = tokenizer.encode(chat_history)

        output_tokens = generate_tokens(
            model=model, 
            input_tokens=tokenizer.encode(chat_history),
            sampler_func=sampler_func,
            tokens_to_generate=100000
        )

        for output_token in output_tokens:
            decoded_token = tokenizer.decode(output_token)
            chat_history += decoded_token
            print(decoded_token, end='')
            
            # if stop sequence occured(a substring of chat history containing reverse sequence)
            # then we need to reprompt here

            if chat_history[reverse_sequence_window:] == reverse_sequence:
                # need to reprompt here
                chat_history += " {0}\n\nAlice:".format(input(" "))
                break
        
        


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
            description='Simple chat with rwkv cpp cuda'
            )

    parser.add_argument(
            '--model_path', 
            help='Path to a quantized model.bin', 
            required=True,
            )


    parser.add_argument(
            '--torch_binding_cpp_path', 
            help='Path to torchbind.cpp file containing bindings for torch', 
            required=False,
            default=torch_binding_cpp_path,
            )

    parser.add_argument(
            '--rwkv_cuda_path', 
            help='Path to rwkv.cu containg cuda implementations of common funcs.', 
            required=False,
            default=rwkv_cuda_path,
            )

    parser.add_argument(
            '--include_path', 
            help='Path to include folder containing headers of NumCpp, rwkv.h, etc.', 
            required=False,
            default=headers_include_path,
            )


    args = vars(parser.parse_args())

    model_path = args['model_path']
    torch_binding_cpp_path = args['torch_binding_cpp_path']
    rwkv_cuda_path = args['rwkv_cuda_path']
    header_include_path = args['include_path']

    print("Loading CPP bindings...", end='')

    # load cpp bindings into torch.ops namespace, so that underlying
    # RwkvCppWrapper class could use the binding of torchbind.cpp and rwkv.cu 
    # via torch.ops.rwkv.<func_name>
    torch_binding.load_cpp_bindings(
        torch_binding_cpp_path=torch_binding_cpp_path,
        rwkv_cuda_path=rwkv_cuda_path,
        headers_include_path=headers_include_path,
        )

    print(" Loaded!")


    print(f"Loading {model_path}...", end='')
    # Load the model itself here
    model = torch_binding.load_cpp_rwkv(path=model_path)
    print(" Loaded!")

    print("Chat is opening! To exit press Ctrl+C at any point.")
    chat(model=model)


