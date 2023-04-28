import os
import json

from typing import List
from dataclasses import dataclass

from flask import Flask
from flask import request

app = Flask(__name__)

MODEL_PATH = os.environ["MODEL_PATH"]
VOCAB_PATH = os.environ["VOCAB_PATH"]
MERGES_PATH = os.environ["MERGES_PATH"]

# don't forget about SO_LIB_PATH env that is being used in the following import
from bindings.pybind import binding

MODEL = binding.ModelWrapper(model_path=MODEL_PATH)
TOKENIZER = binding.TokenizerWrapper(
        vocab_path=VOCAB_PATH, 
        merges_path=MERGES_PATH
        )


@dataclass
class CompleteRequestDataclass:
    body: str
    tokens: int
    stop_sequence: str = None
    with_body: bool = False


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


def complete_string(complete: CompleteRequestDataclass):
    input_tokens = TOKENIZER.encode(complete.body)
    output_tokens = generate_tokens(
            model=MODEL, 
            input_tokens=input_tokens,
            tokens_to_generate=complete.tokens
            )
    stop_sequence_window = len(complete.stop_sequence) * -1 if complete.stop_sequence else None

    return_string = complete.body if complete.with_body else ""
    for output_token in output_tokens:
        decoded_token = TOKENIZER.decode(output_token)
        return_string += decoded_token
        
        # if stop sequence occured(a substring of chat history containing reverse sequence)
        # then we need to reprompt here

        if complete.stop_sequence and return_string[stop_sequence_window:] == complete.stop_sequence:
            # need to reprompt here
            break

    return return_string
 
 
@app.route('/complete', methods=['POST'])
def add_message():
    complete = CompleteRequestDataclass(**request.json)
    completed = complete_string(complete)
    return json.dumps({"completion": completed})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
