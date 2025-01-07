import torch
import torch.nn as nn
from resnet import resnet18, resnet34, resnet50, resnet101
import numpy as np


class Model(nn.Module):
    def __init__(self, resnet_type = 50, embedding_size = 8):
        super(Model, self).__init__()

        '''
        ENCODER FOR THE MRI
        '''
        print("Using 3D resnet ", resnet_type)
        if resnet_type == 18:
            self.encoder = resnet18()
            output_channels = 512
        elif resnet_type == 34:
            self.encoder = resnet34()
            output_channels = 512
        elif resnet_type == 50:
            self.encoder = resnet50()
            output_channels = 2048
        elif resnet_type == 101:
            self.encoder = resnet101()
            output_channels = 2048


        '''
        FCL FOR PROCESSING OUTPUT OF ENCODER
        '''
        self.latent_reducer = nn.Sequential(
            nn.Linear(output_channels, 500),
            nn.LeakyReLU(inplace=True),
            nn.Linear(500, 100),
            nn.LeakyReLU(inplace=True),
            nn.Linear(100, 30),
            nn.LeakyReLU(inplace=True),
            nn.Linear(30, embedding_size),
            nn.LeakyReLU(inplace=True),
        )

        '''
        EMBEDDING LAYERS FOR RACE AND MENOPAUSE
        '''
        self.race_embedding = nn.Embedding(6, 3)
        self.menopause_embedding = nn.Embedding(3, 2)
        
        # Final fully connected layer to combine scalar + embedded categorical features
        self.final_fc = nn.Sequential(
            nn.Linear(3 + 3 + 2, embedding_size), 
            nn.LeakyReLU(),
            nn.Dropout(0.1),
            nn.BatchNorm1d(embedding_size)
        )

        '''
        MULTI-HEAD SELF-ATTENTION BLOCK
        '''
        self.multihead_attn = nn.MultiheadAttention(embed_dim = embedding_size, num_heads = 2)
        self.layer_norm1 = nn.LayerNorm(embedding_size)
        self.ffn = nn.Sequential(
            nn.Linear(embedding_size, embedding_size * 3),
            nn.LeakyReLU(),
            nn.Linear(embedding_size * 3, embedding_size),
            nn.Dropout(0.1)
        )
        self.layer_norm2 = nn.LayerNorm(embedding_size)

        '''
        FINAL CLASSIFICATION LAYER FOR OUTPUT OF ATTENTION
        '''
        self.classify_attention = nn.Sequential(
            nn.Linear(embedding_size * 3, 1)
        )



    def forward(self, first_img, second_img, non_mri_data):
        first_encoded = self.encoder1(first_img).squeeze()
        second_encoded = self.encoder1(second_img).squeeze()
        batch_size = len(non_mri_data)
        
        if batch_size == 1:
            first_encoded = first_encoded.unsqueeze(0)
            second_encoded = second_encoded.unsqueeze(0)

        first_reduced = self.latent_reducer(first_encoded)
        second_reduced = self.latent_reducer(second_encoded)

        flat_non_mri_data = np.array([item[:3] + item[3] + item[4] for item in non_mri_data])

        scalar_features = torch.tensor(flat_non_mri_data[:, :3], dtype=torch.float32).to(first_img.device)  
        race_ohe = torch.tensor(flat_non_mri_data[:, 3:9], dtype=torch.long).to(first_img.device)
        metapause_ohe = torch.tensor(flat_non_mri_data[:, 9:], dtype=torch.long).to(first_img.device)

        race_indices = torch.argmax(race_ohe, dim=1)  
        metapause_indices = torch.argmax(metapause_ohe, dim=1)
        race_embedding = self.race_embedding(race_indices)
        menopause_embedding = self.menopause_embedding(metapause_indices)

        combined_non_mri = torch.cat([scalar_features, race_embedding, menopause_embedding], dim=1) 
        combined_non_mri = self.final_fc(combined_non_mri)


        '''
        MULTI-HEAD ATTENTION TRANSFORMER

        1. Stack MRI and clinical latent features into a sequence for attention 
         - (shape: (3, batch_size, 6))
        2.  Apply Multihead Attention
         - attn_output: [3, batch, 6]
        3. Aggregate the attention outputs over sequence dimension
        4. Turn [3, batch, embed] to [batch, embed * 3] where for every entry, we have a concatenated values
        '''
        combined_features = torch.stack([first_reduced, second_reduced, combined_non_mri], dim=0)
        attn_output, attn_weights = self.multihead_attn(combined_features, combined_features, combined_features)
       
        x = self.layer_norm1(combined_features + attn_output)
        ffn_output = self.ffn(x)
        x = self.layer_norm2(x + ffn_output)
        aggregated_features = x.permute(1, 0, 2).reshape(batch_size, -1)  # [batch, embed * 3]
        logits = self.classify_attention(aggregated_features)

        return logits 
