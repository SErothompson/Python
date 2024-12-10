import torch
import pandas as pd
from transformers import GPT2Tokenizer, GPT2LMHeadModel, Trainer, TrainingArguments, DataCollatorForLanguageModeling

# Check if GPU is available
device = ""
if torch.cuda.is_available():
    device = torch.device("cuda")
    print("Using CUDA")
else:
    device = torch.device("cpu")
    print("Using CPU")

# Load pre-trained model and tokenizer
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

# Add a padding token if it doesn't exist
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})

model = GPT2LMHeadModel.from_pretrained('gpt2')
model.resize_token_embeddings(len(tokenizer))
model.to(device)  # Move model to the GPU if available

# Read the CSV file
def load_dataset_from_csv(filepath):
    df = pd.read_csv(filepath)
    return df['text'].tolist()

# Fine-tune the model
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=2,
    save_steps=10_000,
    save_total_limit=2,
    logging_dir='./logs',
)

class ConversationDataset(torch.utils.data.Dataset):
    def __init__(self, tokenizer, lines, block_size=1024):
        self.examples = []
        for line in lines:
            tokenized_text = tokenizer.encode(line, add_special_tokens=True)
            if len(tokenized_text) > block_size:
                tokenized_text = tokenized_text[:block_size]  # Truncate sequences longer than block_size
            attention_mask = [1] * len(tokenized_text)
            
            # Pad the sequences
            padding_length = block_size - len(tokenized_text)
            if padding_length > 0:
                tokenized_text += [tokenizer.pad_token_id] * padding_length
                attention_mask += [0] * padding_length
            
            self.examples.append((torch.tensor(tokenized_text, dtype=torch.long), torch.tensor(attention_mask, dtype=torch.long)))
    
    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        return self.examples[i]

# Load the dataset from CSV
dataset = ConversationDataset(tokenizer, load_dataset_from_csv('training_data.csv'))

def custom_data_collator(features):
    input_ids = torch.stack([f[0] for f in features])
    attention_mask = torch.stack([f[1] for f in features])
    labels = input_ids.clone()
    return {'input_ids': input_ids, 'attention_mask': attention_mask, 'labels': labels}

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=custom_data_collator,
    train_dataset=dataset,
)

trainer.train()

# Save the fine-tuned model
model.save_pretrained('./finetuned_gpt2')
tokenizer.save_pretrained('./finetuned_gpt2')