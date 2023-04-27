import os
import time

os.environ["SO_LIB_PATH"] = "bindings.pybind.rwkv"

from bindings.pybind import binding



t1 = time.time()
model = binding.ModelWrapper("model-3b.bin")
print(f"Loading time {time.time() - t1}s")

t1 = time.time()
tokenizer = binding.TokenizerWrapper(
        "./include/rwkv/tokenizer/vocab/vocab.json", 
        "./include/rwkv/tokenizer/vocab/merges.txt"
        )

print(f"Loading time {time.time() - t1}s")

binding.CPP_LIB.initState(model.cpp_instance)
tokenized_prompt = tokenizer.encode("To see the world in a grain of")
for token in tokenized_prompt:
    print(token)
    output, state = model.forward(token)


output_tokens_decoded = [tokenizer.decode(token) for token in tokenized_prompt]
print(output_tokens_decoded)


last_token = tokenized_prompt[-1]


print(model.forward(last_token))


new_token = model.sample()
last_token_decoded = tokenizer.decode(new_token)
print(last_token_decoded)
