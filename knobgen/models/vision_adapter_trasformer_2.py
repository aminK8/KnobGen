import torch
import torch.nn as nn

class VisionAdapterTransformer2(nn.Module):
    def __init__(self,
                 in_channels_image=1024,
                 in_channels_text=768,
                 max_position_embeddings_text=77,
                 hidden_channels=1024,
                 output_channels=768,
                 number_heads=8,
                 num_transformer_layers=6,
                 num_image_token=256):
        super(VisionAdapterTransformer2, self).__init__()
        self.max_position_embeddings_text = max_position_embeddings_text
        
        # 1D Convolutional layers for both tensors
        self.conv2 = nn.Conv1d(in_channels=in_channels_image, out_channels=hidden_channels, kernel_size=1)
        self.conv1 = nn.Conv1d(in_channels=in_channels_text, out_channels=hidden_channels, kernel_size=1)
        
        # Transformer blocks for cross-attention
        self.transformer_layers = nn.ModuleList([
            nn.TransformerDecoderLayer(d_model=hidden_channels, nhead=number_heads)
            for _ in range(num_transformer_layers)
        ])
        
        # Linear layer to reduce sequence length to 77
        # self.reduce_seq_length = nn.Linear(num_image_token, self.max_position_embeddings_text)  # 256 -> 77
        
        # Fully connected layers to process the Transformer output
        self.fc1 = nn.Linear(hidden_channels, hidden_channels)
        self.gelu = nn.GELU()
        self.fc2 = nn.Linear(hidden_channels, output_channels)
        
    def forward(self, encoded_condition_image, encoder_hidden_states):
        # encoded_condition_image shape: [batch, 256, 1024]
        # encoder_hidden_states shape: [batch, 77, 768]
        
        txt_tensor = encoder_hidden_states.permute(0, 2, 1)  # Shape: [batch, 77, 768] -> [batch, 768, 77]
        img_tensor = encoded_condition_image.permute(0, 2, 1)  # Shape: [batch, 256, 1024] -> [batch, 1024, 256]
        
        # Apply 1D convolutions
        txt_tensor = self.conv1(txt_tensor)  # Shape: [batch, 768, 77] -> [batch, hidden_channels, 77]
        img_tensor = self.conv2(img_tensor)  # Shape: [batch, 1024, 256] -> [batch, hidden_channels, 256]
        # print(f"before before attention x2 shape is {x2.shape}")

        # Interpolate x1 to match x2's sequence length (256)
        # x1 = nn.functional.interpolate(x1, size=x2.shape[2], mode='linear', align_corners=False)  # Shape: [batch, hidden_channels, 256]
        
        # Prepare for transformer layers
        txt_tensor = txt_tensor.permute(2, 0, 1)  # Shape: [77, batch, hidden_channels] hidden_channels -> number of tokens
        img_tensor = img_tensor.permute(2, 0, 1)  # Shape: [256, batch, hidden_channels]
        # print(f"before attention x2 shape is {x2.shape}")
        # Pass through the Transformer layers
        first_txt_tensor = txt_tensor.clone()
        for layer in self.transformer_layers:
            txt_tensor = layer(tgt=txt_tensor, memory=img_tensor) + first_txt_tensor  # Cross-Attention: x2 attends to x1
        
        # print(f"after attention x2 shape is {x2.shape}")
        # Reduce the sequence length back to 77
        txt_tensor = txt_tensor.permute(1, 2, 0)  # Shape: [batch, nn.hidden_channels, 77]
        # txt_tensor = self.reduce_seq_length(txt_tensor)  # Shape: [batch, hidden_channels, 77]
        
        txt_tensor = txt_tensor.permute(0, 2, 1) # Shape: [batch, 77, hidden_channels]
        # Fully connected layers
        out = self.fc1(txt_tensor) # Shape: [batch, 77, hidden_channels]
        out = self.gelu(out)
        out = self.fc2(out) # Shape: [batch, 77, output_channels]
        
        return out


# Example usage
# if __name__ == "__main__":
#     batch_size = 1
#     tensor1 = torch.randn(batch_size, 256, 1024)
#     tensor2 = torch.randn(batch_size, 77, 768)
    
#     model = VisionAdapterTransformer2()
#     output = model(tensor1, tensor2)
    
#     print("Output shape:", output.shape)  # Should be [batch_size, 768, 77]
#     num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
#     print(f"Number of trainable parameters: {num_params}")
#     num_params = sum(p.numel() for p in model.transformer_layers.parameters() if p.requires_grad)
#     print(f"Number of trainable parameters for attention: {num_params}")
