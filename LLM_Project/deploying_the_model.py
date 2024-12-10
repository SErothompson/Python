from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Load the fine-tuned model and tokenizer
tokenizer = GPT2Tokenizer.from_pretrained('./finetuned_gpt2')
model = GPT2LMHeadModel.from_pretrained('./finetuned_gpt2')

# Function to generate a response
def generate_response(prompt):
    inputs = tokenizer(prompt, return_tensors='pt', padding=True, truncation=True)
    input_ids = inputs['input_ids']
    attention_mask = inputs['attention_mask']
    
    outputs = model.generate(input_ids, attention_mask=attention_mask, max_length=150, num_return_sequences=1, pad_token_id=tokenizer.eos_token_id)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

# Interactive conversation
while True:
    user_input = input("You: ")
    if user_input.lower() in ['exit', 'quit']:
        break
    response = generate_response(user_input)
    print("AI:", response)