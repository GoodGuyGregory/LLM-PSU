from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

llama_32_1B = "meta-llama/Llama-3.2-1B"

# creates a Tokenizer specifically for the llama_model requested
tokenizer = AutoTokenizer.from_pretrained(llama_32_1B)

# Set the padding token ID to be the same as the EOS token ID
tokenizer.pad_token_id = tokenizer.eos_token_id

model = AutoModelForCausalLM.from_pretrained(llama_32_1B, torch_dtype=torch.float16)

model = model.to('cpu')

# Your special prompt


story_prompt = "Once upon a time "
    
# Encode the prompt into token IDs
prompt_ids = tokenizer.encode(story_prompt, return_tensors="pt")

# Create an attention mask
attention_mask = prompt_ids.ne(tokenizer.pad_token_id)

# Generate a response from llama_3.2-1B
outputs = model.generate(prompt_ids,
                        attention_mask=attention_mask,
                        max_length=200,
                        do_sample=True,
                        num_return_sequences=1,
                        pad_token_id=tokenizer.eos_token_id,
                        temperature=0.93,
                        top_k=30,
                        top_p=0.90,
                        repetition_penalty=1.2
                    )

# Decode the generated response
generated_tokens = outputs[0]

generated_story = tokenizer.decode(generated_tokens, skip_special_tokens=True)
print(generated_story)
