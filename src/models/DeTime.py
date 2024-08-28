import torch
import torch.nn as nn

# Define the models
class DeTime(nn.Module):
    def __init__(
        self,
        model_name,  # linear / mlp
        input_size,
        output_size,
        hidden_size=None,
        dropout_rate=0,
    ):
        super().__init__()
        
        # build decoder
        if model_name == 'mlp':
            decoder_list = []
            decoder_list.append(nn.Linear(input_size, hidden_size))
            decoder_list.append(nn.ReLU())
            decoder_list.append(nn.Dropout(dropout_rate))
            decoder_list.append(nn.Linear(hidden_size, output_size))
            self.decoder = nn.Sequential(*decoder_list)
        else:
            self.decoder = nn.Linear(input_size, output_size)

        # build loss function
        self.loss_fn = nn.BCELoss(reduction="none")  

    def forward(
        self,
        spikes,  # (bs, input_size)
        target=None,  # (bs, output_size)
    ):
        
        logits = torch.sigmoid(self.decoder(spikes))
        if target is not None:
            loss = self.loss_fn(logits, target).sum()
        else:
            loss = None
        n_examples = spikes.shape[0]

        return {
            "logits": logits,
            "loss": loss,
            "n_examples": n_examples,
        }
        