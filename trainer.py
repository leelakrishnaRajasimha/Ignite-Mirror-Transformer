from ignite.engine import Engine, Events
from ignite.metrics import RunningAverage
import torch.nn as nn
import torch

def create_trainer(model, optimizer, criterion, device, PAD_TOKEN):
    def train_step(engine, batch):
        model.train()
        optimizer.zero_grad()

        batch = batch.to(device)

        # Shift for next token prediction.
        tgt_input = batch[:,:-1]
        targets = batch[:, 1:]

        mask = torch.ones_like(targets,dtype=torch.float)

        # seq_len is SOS + original + EOS + reversed.
        # We only want to predict the second half.
        half_len = batch.size(1) // 2
        mask[:, :half_len] = 0.0 # Zero out the input portion.

        # Model internally builds causal mask.
        outputs = model(tgt_input)

        # Calculating loss per element.
        loss_fn = nn.CrossEntropyLoss(reduction='none', ignore_index=PAD_TOKEN)
        
        loss = loss_fn(outputs.transpose(1,2), targets) #[Batch, seq_len].

        masked_loss = (loss * mask).sum() / mask.sum()

        masked_loss.backward()
        optimizer.step()

        return masked_loss.item()
    trainer = Engine(train_step)
    RunningAverage(output_transform=lambda x: x).attach(trainer, 'loss')

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_results(engine):
        print(f"Epoch [{engine.state.epoch}] Loss: {engine.state.metrics['loss']:.4f}")

    @trainer.on(Events.COMPLETED)
    def save_model(engine):
        torch.save(model.state_dict(),"mirror_transformer.pth")
        print("Training completed. Model saved as mirror_transformer.pth")

    return trainer

