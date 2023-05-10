import torch
from transformers import AutoTokenizer
from components.TransformerClassifier import TransformerClassifier

# Load the saved model
PATH_TO_MODEL = 'TrainedClassifier'
model = torch.load(PATH_TO_MODEL, map_location=torch.device('cpu'))
model.eval()

# Set up the tokenizer
tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')

while(True):
    # Input text
    text = input('Please type an input')
    # text = "This movie is poor and shit and bad!"

    # Tokenize the input text
    inputs = tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        max_length=512,
        padding='max_length',
        return_attention_mask=True,
        return_tensors='pt'
    )

    # Extract the input IDs and attention mask
    input_ids = inputs['input_ids']
    attention_mask = inputs['attention_mask']

    # Run the model on the input text
    with torch.no_grad():
        outputs = model(input_ids)

    # Get the predicted sentiment
    prediction = torch.argmax(outputs, dim=1)
    print(f"Predicted sentiment: {prediction.item()}")
