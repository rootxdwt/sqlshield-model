from functools import wraps
from flask import request,jsonify
import torch
import torch.nn.functional as F
from models.base import CharCNN
from utils.tokenizer import ALPHABET

MAX_LEN = 100
NUM_CLASSES = 2

BATCH_SIZE = 64
LEARNING_RATE = 0.001
EPOCHS = 10
EMBEDDING_DIM = 64
NUM_CHARS = len(ALPHABET) + 1  # +1 for padding


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = CharCNN(NUM_CHARS, EMBEDDING_DIM, num_classes=NUM_CLASSES)

model_url = "https://cdn.xdcs.me/sqlshield.pth"# pretrained weight stored on my personal CDN
state_dict = torch.hub.load_state_dict_from_url(model_url, map_location=device)

def preprocess_input(text):
    """
    Converts a raw SQL string into a tensor for the model.
    1. Lowercase
    2. Map chars to indices
    3. Pad or Truncate to MAX_LEN
    """
    char_to_ix = {ch: i+1 for i, ch in enumerate(ALPHABET)}
    
    # clean and tokenize
    text = str(text).lower()
    idx = [char_to_ix[c] for c in text if c in char_to_ix]
    
    # pad or truncate
    if len(idx) < MAX_LEN:
        idx += [0] * (MAX_LEN - len(idx)) # 0 is padding
    else:
        idx = idx[:MAX_LEN]
        
    # convert to tensor and add batch dimension [1, MAX_LEN]
    return torch.tensor(idx, dtype=torch.long).unsqueeze(0)

def waf(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        id = request.args.get('id', '')
        pw = request.args.get('pw', '')
        try:
            model.load_state_dict(state_dict)
        except Exception as e:
            return jsonify({'error': 'model load failed'}), 500

        model.to(device)
        model.eval() # turns off dropout layers

        # prepare input
        id_tensor = preprocess_input(id).to(device)
        pw_tensor = preprocess_input(pw).to(device)

        # prediction
        with torch.no_grad():
            id_log = model(id_tensor)
            pw_log = model(pw_tensor)
            id_prob = F.softmax(id_log, dim=1)
            pw_prob =F.softmax(pw_log,dim=1)
            
            # get result
            _, id_pred = torch.max(id_prob, 1)
            _, pw_pred = torch.max(pw_prob, 1)

        # report
        if id_pred.item() == 1 or pw_pred.item()==1:
            return jsonify({'error': 'Invalid input detected'}), 400

        return f(*args, **kwargs)
    return decorated_function