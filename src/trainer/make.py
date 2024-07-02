from trainer.base import Trainer
from trainer.gpt import GPTTrainer

def make_trainer(
    model,
    train_dataloader,
    eval_dataloader,
    optimizer,
    **kwargs
):
    if model.config.model_class == 'NDT-GPT':
        return GPTTrainer(
            model=model,
            train_dataloader=train_dataloader,
            eval_dataloader=eval_dataloader,
            optimizer=optimizer,
            **kwargs
        )
    else:
        return Trainer(
            model=model,
            train_dataloader=train_dataloader,
            eval_dataloader=eval_dataloader,
            optimizer=optimizer,
            **kwargs
        )
