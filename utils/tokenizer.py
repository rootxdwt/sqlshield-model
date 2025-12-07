import torch

# vocabulary definition
ALPHABET = "abcdefghijklmnopqrstuvwxyz0123456789-,;.!?:'\"/\\|_@#$%^&*~`+ =<>()[]{}"

def str_to_tensor(text_str, max_len=200, alphabet=ALPHABET):
    """
    Converts a single string into a Tensor of indices.
    Args:
        text_str (str): input payload.
        max_len (int): pad or truncate the string to this length.
    Returns:
        torch.LongTensor: The vectorized string.
    """
    # Create a mapping from char -> index
    char_to_index = {char: i + 1 for i, char in enumerate(alphabet)} # 0 is reserved for padding
    
    # Initialize a vector of zeros (Padding by default)
    indices = [0] * max_len
    
    # Fill the vector with actual indices
    for i, char in enumerate(text_str[:max_len]):
        if char.lower() in char_to_index:
            indices[i] = char_to_index[char.lower()]
            
    return torch.tensor(indices, dtype=torch.long)