import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from litecache.models.adapter import create_adapter

model = AutoModelForCausalLM.from_pretrained("gpt2", torch_dtype=torch.float16)
tokenizer = AutoTokenizer.from_pretrained("gpt2")

adapter = create_adapter(model)

prompt = "Once upon a time, there was a magical forest filled with"
input_ids = tokenizer.encode(prompt, return_tensors="pt")

print(f"Prompt: {prompt}")

# Generate using our paged cache!
output_ids = adapter.generate(
    input_ids,
    max_length=50,
    do_sample=True
)

print(adapter.cache.get_num_free_blocks())
generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
print(f"\nGenerated: {generated_text}")
