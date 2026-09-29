import os
import yaml
import torch
from torch import optim
from src.model import KlyroConfig, KlyroForCausalLM
from src.tokenizer import KlyroTokenizer
from src.dataset import get_language_datasets, mix_datasets, prepare_dataloader
from src.checkpoint import save_checkpoint, load_latest_checkpoint
from src.hf_utils import upload_file_to_hf, download_file_from_hf, ensure_repo_exists
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_config(config_path="config/train_config.yaml"):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config

def setup_model_and_tokenizer(config):
    # Model config
    model_cfg = KlyroConfig(
        vocab_size=config["model"]["vocab_size"],
        hidden_size=config["model"]["hidden_size"],
        num_hidden_layers=config["model"]["num_hidden_layers"],
        num_attention_heads=config["model"]["num_attention_heads"],
        intermediate_size=config["model"]["intermediate_size"],
        max_position_embeddings=config["model"]["max_position_embeddings"],
        hidden_act=config["model"]["hidden_act"],
        layer_norm_eps=config["model"]["layer_norm_eps"],
        initializer_range=config["model"]["initializer_range"],
        pad_token_id=3,  # will be overridden after tokenizer loads
    )
    model = KlyroForCausalLM(model_cfg)
    logger.info(f"Model initialized with {model.num_parameters():,} trainable parameters")

    # Tokenizer (will be loaded or trained later)
    tokenizer = KlyroTokenizer(vocab_size=config["model"]["vocab_size"])
    return model, tokenizer, model_cfg

def get_optimizer_scheduler(model, config, num_training_steps):
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config["training"]["learning_rate"],
        weight_decay=config["training"]["weight_decay"],
    )
    # Linear warmup then linear decay
    from transformers import get_linear_schedule_with_warmup
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=config["training"]["warmup_steps"],
        num_training_steps=num_training_steps,
    )
    return optimizer, scheduler

def train_loop(model, tokenizer, optimizer, scheduler, train_dataloader, eval_dataloader, config, device):
    model.to(device)
    model.train()
    global_step = 0
    best_eval_loss = float("inf")
    scaler = torch.cuda.amp.GradScaler() if config["training"]["fp16"] else None

    for epoch in range(100):  # large number, controlled by max_steps
        for batch in train_dataloader:
            optimizer.zero_grad()
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = input_ids.clone()

            with torch.cuda.amp.autocast(enabled=config["training"]["fp16"]):
                outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs["loss"]
                loss = loss / config["training"]["gradient_accumulation_steps"]

            if scaler:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            if (global_step + 1) % config["training"]["gradient_accumulation_steps"] == 0:
                if scaler:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config["training"]["gradient_clip_val"])
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config["training"]["gradient_clip_val"])
                    optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            global_step += 1

            if global_step % config["training"]["logging_steps"] == 0:
                logger.info(f"Step {global_step} - loss: {loss.item():.4f}")

            if global_step % config["training"]["eval_steps"] == 0 and eval_dataloader is not None:
                eval_loss = evaluate(model, eval_dataloader, device)
                logger.info(f"Step {global_step} - eval loss: {eval_loss:.4f}")
                if eval_loss < best_eval_loss:
                    best_eval_loss = eval_loss
                    # Optionally save best model
                    pass

            if global_step % config["training"]["save_steps"] == 0:
                ckpt_path = save_checkpoint(
                    model, optimizer, scheduler, global_step, config,
                    checkpoint_dir=config["checkpoint"]["local_dir"]
                )
                if config["checkpoint"]["push_to_hub"]:
                    upload_file_to_hf(ckpt_path, config["huggingface"]["repo_id"], token=os.getenv("HF_TOKEN"))
                logger.info(f"Checkpoint saved at step {global_step}")

            if global_step >= config["training"]["max_steps"]:
                logger.info("Reached max training steps.")
                return

def evaluate(model, dataloader, device):
    model.eval()
    total_loss = 0.0
    total_steps = 0
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = input_ids.clone()
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs["loss"]
            total_loss += loss.item()
            total_steps += 1
    model.eval()
    return total_loss / max(total_steps, 1)

def main():
    config = load_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Ensure Hugging Face repo exists
    if config["huggingface"]["repo_id"]:
        ensure_repo_exists(config["huggingface"]["repo_id"], token=os.getenv("HF_TOKEN"))

    model, tokenizer, model_cfg = setup_model_and_tokenizer(config)

    # TODO: Load or train tokenizer based on config and data
    # For now assume tokenizer already available locally or will be trained in notebook

    # Prepare datasets
    train_datasets = get_language_datasets(config, split="train")
    # Mix according to russian_ratio
    eng_prob = 1.0 - config["data"]["russian_ratio"]
    rus_prob = config["data"]["russian_ratio"]
    mixed_train = mix_datasets(train_datasets, probabilities=[eng_prob, rus_prob], seed=config["training"]["seed"])
    train_loader = prepare_dataloader(
        mixed_train,
        tokenizer,
        batch_size=config["training"]["batch_size"],
        max_length=config["model"]["max_position_embeddings"],
        shuffle=True,
    )

    # Evaluation dataset (could be a small validation split)
    eval_datasets = get_language_datasets(config, split="validation")
    mixed_eval = mix_datasets(eval_datasets, probabilities=[eng_prob, rus_prob], seed=config["training"]["seed"])
    eval_loader = prepare_dataloader(
        mixed_eval,
        tokenizer,
        batch_size=config["training"]["batch_size"],
        max_length=config["model"]["max_position_embeddings"],
        shuffle=False,
    ) if eval_datasets else None

    # Optimizer and scheduler
    optimizer, scheduler = get_optimizer_scheduler(
        model,
        config,
        num_training_steps=config["training"]["max_steps"],
    )

    # Resume logic
    start_epoch = 0
    global_step = 0
    if config["checkpoint"]["resume_from_checkpoint"]:
        latest = load_latest_checkpoint(config["checkpoint"]["local_dir"])
        if latest:
            logger.info(f"Resuming from checkpoint {latest}")
            model, optimizer, scheduler, global_step = load_latest_checkpoint(
                config["checkpoint"]["local_dir"], model, optimizer, scheduler
            )
            start_epoch = global_step // (len(train_loader) // config["training"]["gradient_accumulation_steps"] + 1)

    train_loop(model, tokenizer, optimizer, scheduler, train_loader, eval_loader, config, device)

    # Final save
    final_path = os.path.join(config["checkpoint"]["local_dir"], "final_model.pt")
    torch.save({
        "model_state_dict": model.state_dict(),
        "config": config,
    }, final_path)
    if config["checkpoint"]["push_to_hub"]:
        upload_file_to_hf(final_path, config["huggingface"]["repo_id"], token=os.getenv("HF_TOKEN"))
    logger.info("Training finished.")

if __name__ == "__main__":
    main()