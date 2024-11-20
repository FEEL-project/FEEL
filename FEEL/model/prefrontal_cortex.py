import torch
import torch.nn as nn
import math

class PFCPositionalEncoding(nn.Module):
    def __init__(self, dim_event: int = 768, num_events:int = 5, dropout: float=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(num_events).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim_event, 2).float() * (-math.log(10000.0) / dim_event))
        pe = torch.zeros(num_events, 1, dim_event)
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
        num_events: int = 5,
        dim_out: int = 8,
        d_model: int = 512,
        nhead: int = 8,
        dim_feedforward: int = 2048,
        num_encoder_layers: int = 6 
    ):
        """PFC

        Args:
            dim_event (int, optional): Number of dimension in an episode. Defaults to 768.
            num_events (int, optional): Number of episodes replayed. Defaults to 5.
            dim_out (int, optional): Output dimension of emotion. Defaults to 8.
            nhead (int, optional): Number of heads. Defaults to 8.
            dim_feedforward (int, optional): Dimension to feed forward. Defaults to 2048.
            num_encoder_layers (int, optional): Number of encoded layer. Defaults to 6.
        """
        super(PFC, self).__init__()
        
        self.conv = nn.Linear(dim_event, d_model)
        
        self.positional_encoding = PFCPositionalEncoding(d_model, num_events)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_encoder_layers)
        
        self.classifier = nn.Linear(d_model, dim_out)
    
    def forward(self, src: torch.Tensor) -> torch.Tensor:
        """forward

        Args:
            src (torch.Tensor): Flattened / squeezed version of Characteristic2 ([dim_event])

        Returns:
            torch.Tensor: Output tensor ([1, dim_out])
        """
        src = self.conv(src)
        print(81, src.shape)
        src = self.positional_encoding(src)
        print(83, src.shape)
        # Pass through the Transformer Encoder
        encoded_output: torch.Tensor = self.encoder(src)
        print(86, src.shape)
        # Pool the output
        pooled_output = encoded_output.mean(dim=0)
        
        #Pass through classification head
        return self.classifier(pooled_output)