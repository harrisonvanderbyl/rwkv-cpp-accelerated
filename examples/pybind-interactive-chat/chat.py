import argparse
import time

from typing import List, Callable

from bindings.bindings import binding


def generate_tokens(
    *, 
    model: binding.ModelWrapper,
    input_tokens: List[int],
    tokens_to_generate: int,
    ):
    
    model.init_state()
    model.load_context(input_tokens)
    last_token = input_tokens[-1]
    for _ in range(tokens_to_generate):
        model.forward(last_token)
        last_token = model.sample()
        yield last_token
        

def chat(
    *, 
    model: binding.ModelWrapper,
    tokenizer: binding.TokenizerWrapper,
    reverse_sequence="Bob:",
    ):
    """A chat between Alice and Bob"""
    
    reverse_sequence_window = len(reverse_sequence) * -1
    
    chat_history = ""
    chat_history += "Bob: {0}\n\nAlice:".format(input("Bob: "))
    print("==============")
    print(chat_history, end='')
    while True:

        output_tokens = generate_tokens(
            model=model, 
            input_tokens=tokenizer.encode(chat_history),
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
            '--vocab_path', 
            help='Path to a vocab.json for a tokenizer', 
            required=True,
            )

    parser.add_argument(
            '--merges_path', 
            help='Path to a merges.txt for a tokenizer', 
            required=True,
            )


    args = vars(parser.parse_args())

    model_path = args['model_path']
    vocab_path = args['vocab_path']
    merges_path = args['merges_path']

    print(f"Loading {model_path}...", end='')
    t1 = time.time()
    model = binding.ModelWrapper(model_path=model_path)
    print(f" Loaded in {time.time() - t1}!")

    print(f"Loading tokenizer...", end='')
    t1 = time.time()
    tokenizer = binding.TokenizerWrapper(
            vocab_path=vocab_path,
            merges_path=merges_path,
            )
    print(f" Loaded in {time.time() - t1}!")


    print("Chat is opening! To exit press Ctrl+C at any point.")
    chat(model=model, tokenizer=tokenizer)


