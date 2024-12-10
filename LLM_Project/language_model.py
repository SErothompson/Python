import pandas as pd
from transformers import GPT2Tokenizer, GPT2LMHeadModel, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from torch.utils.data import Dataset
import torch

# Load pre-trained model and tokenizer
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

# Specify the padding token
tokenizer.pad_token = tokenizer.eos_token

model = GPT2LMHeadModel.from_pretrained('gpt2')

# Tokenize input text for generation
input_text = "Once upon a time"
inputs = tokenizer(input_text, return_tensors='pt', padding=True, truncation=True, max_length=50)
input_ids = inputs['input_ids']
attention_mask = inputs['attention_mask']

# Generate text
output = model.generate(input_ids, attention_mask=attention_mask, max_length=50, num_return_sequences=1)
generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
print("Generated Text:", generated_text)

# Define a custom dataset class
class CustomDataset(Dataset):
    def __init__(self, csv_file, tokenizer, block_size=128):
        self.dataset = pd.read_csv(csv_file)
        self.tokenizer = tokenizer
        self.block_size = block_size
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        text = self.dataset.iloc[idx, 1]  # Assuming the text data is in the second column
        tokens = self.tokenizer(text, return_tensors='pt', padding='max_length', truncation=True, max_length=self.block_size)
        input_ids = tokens['input_ids'].squeeze()  # Remove batch dimension
        attention_mask = tokens['attention_mask'].squeeze()  # Remove batch dimension
        return input_ids, attention_mask

# Load the training dataset
train_dataset = CustomDataset('training_data.csv', tokenizer)

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,
)

# Fine-tune the model
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=1,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
)

def custom_data_collator(features):
    input_ids = torch.stack([f[0] for f in features])
    attention_mask = torch.stack([f[1] for f in features])
    labels = input_ids.clone()
    return {'input_ids': input_ids, 'attention_mask': attention_mask, 'labels': labels}

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=custom_data_collator,
    train_dataset=train_dataset,
)

trainer.train()

# Save the model
model.save_pretrained('./gpt2-finetuned')
tokenizer.save_pretrained('./gpt2-finetuned')

# Load the model
model = GPT2LMHeadModel.from_pretrained('./gpt2-finetuned')
tokenizer = GPT2Tokenizer.from_pretrained('./gpt2-finetuned')

# Evaluate the model with keyboard input
print("Please enter text for evaluation:")
eval_text = input()
eval_inputs = tokenizer(eval_text, return_tensors='pt', padding=True, truncation=True, max_length=50)
eval_input_ids = eval_inputs['input_ids']
eval_attention_mask = eval_inputs['attention_mask']
eval_output = model.generate(eval_input_ids, attention_mask=eval_attention_mask, max_length=50, num_return_sequences=1)
generated_eval_text = tokenizer.decode(eval_output[0], skip_special_tokens=True)
print("Generated Text After Fine-Tuning:", generated_eval_text)

# Manual evaluation
# Calculate perplexity (simple example, for illustrative purposes)
with torch.no_grad():
    outputs = model(eval_input_ids, attention_mask=eval_attention_mask, labels=eval_input_ids)
    loss = outputs.loss
    perplexity = torch.exp(loss)
print(f"Perplexity: {perplexity.item()}")