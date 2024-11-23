import torch
import torch.nn as nn
import math


class PFCPositionalEncoding(nn.Module):
    def __init__(self, dim_event: int = 768, size_episode:int = 5, dropout: float=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(size_episode).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim_event, 2).float() * (-math.log(10000.0) / dim_event))
        pe = torch.zeros(size_episode, 1, dim_event)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """forward

        Args:
            x (torch.Tensor): Takes a tensor with dimension [seq_len, batch_size, embedding_dim]

        Returns:
            torch.Tensor: Returns Positional-encoded Tensor
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

class PFC(nn.Module):
    """Prefrontal Cortex (前頭前野)
    Integration of Knowledge and Intelligence (知の統合)
    1. NeoCortex (新皮質): x (characteristics2) -> 512 knowledge
    2. DLPFC (背外側前頭前野): 512 knowledge -> 8 intelligence
    """

    def __init__(
        self,
        dim_event:int = 768,
        size_episode: int = 5,
        dim_out: int = 8,
        d_model: int = 512,
        nhead: int = 8,
        dim_feedforward: int = 2048,
        num_encoder_layers: int = 6 
    ):
        """PFC

        Args:
            dim_event (int, optional): Number of dimension of an event. Defaults to 768.
            size_episode (int, optional): Number of events in an episode. Defaults to 5.
            dim_out (int, optional): Output dimension of emotion. Defaults to 8.
            nhead (int, optional): Number of heads. Defaults to 8.
            dim_feedforward (int, optional): Dimension to feed forward. Defaults to 2048.
            num_encoder_layers (int, optional): Number of encoded layer. Defaults to 6.
        """
        self.size_episode = size_episode
        self.dim_event = dim_event
        super(PFC, self).__init__()
        
        self.conv = nn.Linear(dim_event, d_model)
        
        self.positional_encoding = PFCPositionalEncoding(d_model, size_episode)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_encoder_layers)
        
        self.classifier =nn.Sequential(
            nn.Linear(d_model, dim_out),
            nn.Sigmoid()
        )
    
    def forward(self, src: torch.Tensor) -> torch.Tensor:
        """forward

        Args:
            src (torch.Tensor): Input episode batch with shape [size_episode, batch_size, dim_event]

        Returns:
            torch.Tensor: Output tensor [dim_out]
        """
        if not (src.size(0) == self.size_episode and src.size(2) == self.dim_event):
            raise ValueError(f"Input shape must be (size_episode, batch_size, dim_event) = [{self.size_episode}, *, {self.dim_event}], got {src.shape}")
        src = self.conv(src)
        src = self.positional_encoding(src)
        # Pass through the Transformer Encoder
        encoded_output: torch.Tensor = self.encoder(src)
        # Pool the output
        pooled_output = encoded_output.mean(dim=0)
        
        #Pass through classification head
        return self.classifier(pooled_output)
    

class PFC_nn(nn.Module):
    def __init__(self, dim_event: int = 768, size_episode: int = 5, dim_out: int = 8):
        super(PFC_nn, self).__init__()
        self.flatten = nn.Flatten()
        self.size_episode = size_episode
        self.dim_event = dim_event
        self.dim_out = dim_out
        self.fc1 = nn.Linear(size_episode * dim_event, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 256)
        self.fc4 = nn.Linear(256, 128)
        self.fc5 = nn.Linear(128, 64)
        self.fc6 = nn.Linear(64, dim_out)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
    
    def forward(self, src: torch.Tensor) -> torch.Tensor:
        if not (src.size(0) == self.size_episode and src.size(2) == self.dim_event):
            raise ValueError(f"Input shape must be (size_episode, batch_size, dim_event) = [{self.size_episode}, *, {self.dim_event}], got {src.shape}")
        src = self.flatten(src)
        x = self.fc1(src)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc3(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc4(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc5(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc6(x)
        x = torch.sigmoid(x) 
        return x