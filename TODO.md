NDT-GPT is currently doing autoregressive prediction on Rates, so there's no random noise here. (e.g. use a Poisson process to generate spikes) In what case I will need this random process?

Evaluation under the situation that attn_mask lengths are different within one batch needs to be changed. (the model part is correct, only the evaluation code needs to be changed a bit.)