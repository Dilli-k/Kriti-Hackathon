import torch
from transformers import AutoTokenizer, AutoModel
import re

# Load the tokenizer and model
TOKENIZER_SCIBERT_PATH = r"C:\Users\91824\OneDrive\Documents\New folder\ResearchArticle\tokenizerscibert"
MODEL_SCIBERT_PATH = r"C:\Users\91824\OneDrive\Documents\New folder\ResearchArticle\modelscibert"
CUSTOM_MODEL_PATH = r"C:\Users\91824\OneDrive\Documents\New folder\ResearchArticle\model.bin"

tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_SCIBERT_PATH)

# Column titles
column_titles = [
    'cs.AI', 'cs.AR', 'cs.CE', 'cs.CL', 'cs.CR', 'cs.CV', 'cs.DB', 'cs.DC', 
    'cs.DM', 'cs.GT', 'cs.IR', 'cs.IT', 'cs.LG', 'cs.LO', 'cs.NI', 'cs.OS', 
    'cs.PL', 'cs.RO', 'cs.SD', 'cs.SE', 'econ.EM', 'econ.GN', 'econ.TH', 
    'eess.AS', 'eess.IV', 'eess.SP', 'math.AC', 'math.AP', 'math.AT', 
    'math.CO', 'math.CV', 'math.GR', 'math.IT', 'math.LO', 'math.NT', 
    'math.PR', 'math.QA', 'math.ST', 'q-bio.BM', 'q-bio.CB', 'q-bio.GN', 
    'q-bio.MN', 'q-bio.NC', 'q-bio.TO', 'q-fin.CP', 'q-fin.EC', 'q-fin.GN', 
    'q-fin.MF', 'q-fin.PM', 'q-fin.PR', 'q-fin.RM', 'q-fin.TR', 'stat.AP', 
    'stat.CO', 'stat.ME', 'stat.ML', 'stat.TH'
]

class BERTClass(torch.nn.Module):
    def __init__(self):
        super(BERTClass, self).__init__()
        self.roberta = AutoModel.from_pretrained(MODEL_SCIBERT_PATH)
        self.fc = torch.nn.Linear(768, 57)
    
    def forward(self, ids, mask, token_type_ids):
        _, features = self.roberta(ids, attention_mask=mask, token_type_ids=token_type_ids, return_dict=False)
        output = self.fc(features)
        return output

# Load the model and move to the correct device
def load_model():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = BERTClass()
    model_state_dict = torch.load(CUSTOM_MODEL_PATH, map_location=device)
    model.load_state_dict(model_state_dict)
    model.to(device)
    return model, tokenizer, device

# Text cleaning function
def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z?.!,¿]+", " ", text)
    text = re.sub(r"http\S+", "", text)
    html = re.compile(r'<.*?>')
    text = html.sub(r'', text)
    punctuations = '@#!?+&*[]-%.:/();$=><|{}^' + "'`" + '_'
    for p in punctuations:
        text = text.replace(p, '' )
    text = " ".join([word.lower() for word in text.split()])
    return text

# Prediction function
def predict(text, model, tokenizer, device):
    # Clean the input text
    cleaned_text = clean_text(text)
    
    # Tokenize the text
    inputs = tokenizer.encode_plus(
        cleaned_text,
        truncation=True,
        add_special_tokens=True,
        max_length=512,
        padding='max_length',
        return_token_type_ids=True
    )
    
    # Convert to PyTorch tensors
    ids = torch.tensor(inputs['input_ids']).unsqueeze(0).to(device)
    mask = torch.tensor(inputs['attention_mask']).unsqueeze(0).to(device)
    token_type_ids = torch.tensor(inputs['token_type_ids']).unsqueeze(0).to(device)
    
    # Get model predictions
    with torch.no_grad():
        outputs = model(ids, mask=mask, token_type_ids=token_type_ids)
    
    # Convert model output to a human-readable format (assuming binary classification)
    prediction = outputs.squeeze().tolist()

    # Convert model output to binary values
    prediction = [0 if i < -1.05 else 1 for i in outputs.squeeze().tolist()]
    
    return prediction

# Predict and get column titles with value 1
def get_relevant_columns(text, model, tokenizer, device):
    # Get prediction list
    prediction = predict(text, model, tokenizer, device)
    
    # Filter and get corresponding column titles
    relevant_columns = [column_titles[i] for i, val in enumerate(prediction) if val == 1]
    
    return relevant_columns






