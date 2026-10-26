"""
Text Encoder using DistilBERT.
"""

import torch
import torch.nn as nn
from transformers import DistilBertModel, DistilBertTokenizer


class TextEncoder(nn.Module):
    """
    Text encoder using DistilBERT.
    """
    
    def __init__(
        self,
        model_name="distilbert-base-uncased",
        hidden_size=768,
        max_length=512,
        dropout=0.1,
        freeze_base=False
    ):
        """
        Initialize the TextEncoder.
        Input:
            model_name - HuggingFace text model name
            hidden_size - output feature dimension
            max_length - maximum number of input tokens
            dropout - dropout probability
            freeze_base - freeze the pretrained text model weights when True
        """
        super().__init__()
        
        # Load pre-trained DistilBERT
        self.bert = DistilBertModel.from_pretrained(model_name)
        self.tokenizer = DistilBertTokenizer.from_pretrained(model_name)
        self.hidden_size = hidden_size
        self.max_length = max_length
        
        # Freeze base model if specified
        if freeze_base:
            for param in self.bert.parameters():
                param.requires_grad = False
        
        # Projection layer if needed
        if self.bert.config.dim != hidden_size:
            self.projection = nn.Linear(self.bert.config.dim, hidden_size)
        else:
            self.projection = nn.Identity()
            
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, input_ids, attention_mask=None):
        """
        Encode text.
        Input:
            input_ids - Token IDs tensor of shape (batch, seq_len)
            attention_mask - Optional attention mask

        Output:
            Encoded features of shape (batch, seq_len, hidden_size)
        """
        # Get BERT outputs
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        # Get hidden states (batch, seq_len, hidden_size)
        hidden_states = outputs.last_hidden_state
        
        # Project to desired hidden size
        hidden_states = self.projection(hidden_states)
        hidden_states = self.dropout(hidden_states)
        
        return hidden_states
    
    def tokenize(self, texts, padding=True, truncation=True):
        """
        Tokenize texts.
        Input:
            texts - List of text strings
            padding - Whether to pad sequences
            truncation - Whether to truncate sequences

        Output:
            Dictionary with input_ids and attention_mask
        """
        return self.tokenizer(
            texts,
            padding=padding,
            truncation=truncation,
            max_length=self.max_length,
            return_tensors="pt"
        )
    
    def get_output_dim(self):
        """
        Get output dimension.
        Input:
            None

        Output:
            Output feature dimension (int)
        """
        return self.hidden_size

