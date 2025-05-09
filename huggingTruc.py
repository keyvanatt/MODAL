import torch
import hydra
import wandb

from transformers import (
    Trainer, 
    TrainingArguments, 
    DefaultDataCollator,
    EarlyStoppingCallback
)
from transformers import get_scheduler
from torch.optim import AdamW
from utils.sanity import show_images

# Pour la data augmentation
from torchvision import transforms

@hydra.main(config_path="configs", config_name="train2")
def main(cfg):
    model = train(cfg)


def train(cfg):
    # --- WandB
    if cfg.log:
        wandb.init(project="challenge_CSC_43M04_EP", name=cfg.experiment_name)

    # --- Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Instanciation directe, sans manipulation de cfg
    model = hydra.utils.instantiate(cfg.model.instance).to(device)
    
    # --- DataModule et data augmentation
    datamodule = hydra.utils.instantiate(cfg.datamodule)
    # on wrappe les transforms pour le train
    train_ds = datamodule.train_dataloader().dataset
    train_ds.transform = transforms.Compose([
        transforms.RandomResizedCrop(cfg.img_size),
        transforms.RandomHorizontalFlip(p=cfg.aug_flip),
        transforms.ColorJitter(**cfg.color_jitter),
        transforms.ToTensor(),
    ])
    eval_ds = datamodule.val_dataloader().dataset

    # --- Optimizer manuel pour pouvoir ajouter weight decay visé
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": cfg.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = AdamW(optimizer_grouped, lr=cfg.learning_rate)

    # --- Scheduler (warmup + cos)
    num_training_steps = len(train_ds) // cfg.batch_size * cfg.epochs
    scheduler = get_scheduler(
        name=cfg.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=cfg.warmup_steps,
        num_training_steps=num_training_steps,
    )

    # --- TrainingArguments avec gradient clipping
    training_args = TrainingArguments(
        output_dir=cfg.checkpoint_dir,
        num_train_epochs=cfg.epochs,
        per_device_train_batch_size=cfg.batch_size,
        per_device_eval_batch_size=cfg.batch_size,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="steps",
        logging_steps=cfg.logging_steps,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        fp16=cfg.fp16,
        gradient_accumulation_steps=cfg.grad_accum_steps,
        gradient_checkpointing=False,
        # Clip gradients
        max_grad_norm=cfg.max_grad_norm,
        # Callbacks
        callbacks=[EarlyStoppingCallback(early_stopping_patience=cfg.early_stopping_patience)],
        report_to="wandb" if cfg.log else None,
        run_name=cfg.experiment_name if cfg.log else None,
    )

    data_collator = DefaultDataCollator()

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=data_collator,
        optimizers=(optimizer, scheduler),
    )

    # --- Sanity checks
    if cfg.log:
        train_sanity = show_images(datamodule.train_dataloader(), name="assets/sanity/train_images")
        wandb.log({"sanity_checks/train_images": wandb.Image(train_sanity)})
        val_sanity = show_images(datamodule.val_dataloader(),   name="assets/sanity/val_images")
        wandb.log({"sanity_checks/val_images": wandb.Image(val_sanity)})

    # --- Lancement
    trainer.train()
    trainer.save_model(cfg.checkpoint_dir)

    return model


if __name__ == "__main__":
    main()
