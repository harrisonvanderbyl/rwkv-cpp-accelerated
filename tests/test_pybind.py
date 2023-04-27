import os
import time

os.environ["SO_LIB_PATH"] = "bindings.pybind.rwkv"

from bindings.pybind import binding


t1 = time.time()
model = binding.ModelWrapper(model_path="model-3b.bin")
print(f"Loading time {time.time() - t1}s")

t1 = time.time()
tokenizer = binding.TokenizerWrapper(
        vocab_path="./include/rwkv/tokenizer/vocab/vocab.json", 
        merges_path="./include/rwkv/tokenizer/vocab/merges.txt"
        )

print(f"Loading time {time.time() - t1}s")

tokenized_prompt = tokenizer.encode("To see the world in a grain of")
model.init_state(tokenized_prompt)

output_tokens_decoded = [tokenizer.decode(token) for token in tokenized_prompt]
print(output_tokens_decoded)


last_token = tokenized_prompt[-1]


print(model.forward(last_token))


new_token = model.sample()
last_token_decoded = tokenizer.decode(new_token)
print(last_token_decoded)
