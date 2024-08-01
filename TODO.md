NDT-GPT is currently doing autoregressive prediction on Rates, so there's no random noise here. (e.g. use a Poisson process to generate spikes) In what case I will need this random process?

Evaluation under the situation that attn_mask lengths are different within one batch needs to be changed. (the model part is correct, only the evaluation code needs to be changed a bit.)

# Session that has Start times:
51e53aff-1d5d-4182-a684-aba783d50ae5
ff96bfe1-d925-4553-94b5-bf8297adf259
72cb5550-43b4-4ef0-add5-e4adfdfb5e02
824cf03d-4012-4ab1-b499-c83a92c5589e
746d1902-fa59-4cab-b0aa-013be36060d5