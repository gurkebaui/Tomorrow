import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "Qwen/Qwen2.5-0.5B"  # Model

tokenizer = AutoTokenizer.from_pretrained(model_name)  # load tokenizer

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto",  # load model with auto dtype and device map
)

text = "System: you are young man from japan that speaks english \n User: Hi, how are you? \n Assistant: "  # input
inputs = tokenizer(text, return_tensors="pt").to(
    model.device
)  # tokenize and move to device


# Run inference without gradient calculation
with torch.no_grad():
    outputs = model(**inputs)

# Print the shape of the logits tensor
print(outputs.logits.shape)

# generate Text
generated = model.generate(**inputs, max_new_tokens=100)
print(tokenizer.decode(generated[0], skip_special_tokens=True))
