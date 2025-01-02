from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Load the fine-tuned model and tokenizer
tokenizer = GPT2Tokenizer.from_pretrained('./finetuned_gpt2')
model = GPT2LMHeadModel.from_pretrained('./finetuned_gpt2')

# Function to generate a response
def generate_response(prompt):
    inputs = tokenizer(prompt, return_tensors='pt', padding=True, truncation=True)
    
    outputs = model.generate(
        inputs['input_ids'],
        attention_mask=inputs['attention_mask'],
        max_length=150,
        num_return_sequences=1,
        pad_token_id=tokenizer.eos_token_id,
        # Add these parameters to reduce repetition
        no_repeat_ngram_size=2,        # Prevents repetition of n-grams
        repetition_penalty=1.2,        # Penalizes repetition
        temperature=0.7,               # Adds randomness (0.7 is a good balance)
        top_k=50,                      # Limits vocabulary to top k tokens
        top_p=0.95,                    # Nucleus sampling
        early_stopping=True            # Stops when EOS token is generated
    )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response


# Interactive conversation
while True:
    user_input = input("You: ")
    if user_input.lower() in ['exit', 'quit']:
        break
    response = generate_response(user_input)
    print("AI:", response)