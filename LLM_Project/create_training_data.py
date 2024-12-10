import os
import PyPDF2
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Ensure you have the necessary NLTK data files
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')

# Directory containing PDFs
pdf_dir = 'pdfs'

# Function to extract text from a PDF file
def extract_text_from_pdf(pdf_path):
    with open(pdf_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        text = ''
        for page_num in range(len(reader.pages)):
            page = reader.pages[page_num]
            text += page.extract_text()
    return text

# Process all PDFs in the directory
data = []
for filename in os.listdir(pdf_dir):
    if filename.endswith('.pdf'):
        pdf_path = os.path.join(pdf_dir, filename)
        text = extract_text_from_pdf(pdf_path)
        
        # Preprocess text
        tokens = word_tokenize(text)
        tokens = [word for word in tokens if word.isalnum()]
        tokens = [word for word in tokens if word.lower() not in stopwords.words('english')]
        processed_text = ' '.join(tokens)
        
        # Append to the dataset
        data.append({'filename': filename, 'text': processed_text})

# Create a DataFrame
df = pd.DataFrame(data)

# Save the DataFrame to a CSV file
output_csv = 'training_data.csv'
df.to_csv(output_csv, index=False)

print(f'Training data saved to {output_csv}')