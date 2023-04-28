import os

from dataclasses import dataclass

from flask import Flask

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
class CompleteDataclass:
    body: str
    tokens: int

@app.route('/complete', methods=['POST'])
def add_message():
    complete = CompleteDataclass(**request.json)
    print(complete)
    return "received"


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=80)
