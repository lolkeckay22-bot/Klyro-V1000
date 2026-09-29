import os
import torch
from src.model import KlyroConfig, KlyroForCausalLM
from src.tokenizer import KlyroTokenizer
from src.hf_utils import download_file_from_hf

def load_model_and_tokenizer(model_dir=None, repo_id=None, token=None, device="cpu"):
    """
    Load model and tokenizer either from local directory or from Hugging Face Hub.
    If repo_id is provided, will download from HF.
    """
    if model_dir and os.path.isdir(model_dir):
        # Load tokenizer
        tokenizer = KlyroTokenizer()
        tokenizer.load(model_dir)
        # Load model checkpoint
        checkpoint_path = os.path.join(model_dir, "pytorch_model.bin")
        if not os.path.exists(checkpoint_path):
            # maybe final_model.pt
            checkpoint_path = os.path.join(model_dir, "final_model.pt")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        config_dict = checkpoint.get("config", {})
        # Build config
        cfg = KlyroConfig(
            vocab_size=config_dict.get("model", {}).get("vocab_size", 16000),
            hidden_size=config_dict.get("model", {}).get("hidden_size", 512),
            num_hidden_layers=config_dict.get("model", {}).get("num_hidden_layers", 8),
            num_attention_heads=config_dict.get("model", {}).get("num_attention_heads", 8),
            intermediate_size=config_dict.get("model", {}).get("intermediate_size", 2048),
            max_position_embeddings=config_dict.get("model", {}).get("max_position_embeddings", 1024),
            hidden_act=config_dict.get("model", {}).get("hidden_act", "gelu"),
            layer_norm_eps=config_dict.get("model", {}).get("layer_norm_eps", 1e-5),
            initializer_range=config_dict.get("model", {}).get("initializer_range", 0.02),
            pad_token_id=3,
        )
        model = KlyroForCausalLM(cfg)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(device)
        model.eval()
        return model, tokenizer
    elif repo_id:
        # Download from HF
        # We assume the repo contains tokenizer.json and pytorch_model.bin
        tokenizer_path = download_file_from_hf(repo_id, "tokenizer.json", token=token)
        model_path = download_file_from_hf(repo_id, "pytorch_model.bin", token=token)
        tokenizer = KlyroTokenizer()
        tokenizer.load(os.path.dirname(tokenizer_path))
        checkpoint = torch.load(model_path, map_location=device)
        config_dict = checkpoint.get("config", {})
        cfg = KlyroConfig(
            vocab_size=config_dict.get("model", {}).get("vocab_size", 16000),
            hidden_size=config_dict.get("model", {}).get("hidden_size", 512),
            num_hidden_layers=config_dict.get("model", {}).get("num_hidden_layers", 8),
            num_attention_heads=config_dict.get("model", {}).get("num_attention_heads", 8),
            intermediate_size=config_dict.get("model", {}).get("intermediate_size", 2048),
            max_position_embeddings=config_dict.get("model", {}).get("max_position_embeddings", 1024),
            hidden_act=config_dict.get("model", {}).get("hidden_act", "gelu"),
            layer_norm_eps=config_dict.get("model", {}).get("layer_norm_eps", 1e-5),
            initializer_range=config_dict.get("model", {}).get("initializer_range", 0.02),
            pad_token_id=3,
        )
        model = KlyroForCausalLM(cfg)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(device)
        model.eval()
        return model, tokenizer
    else:
        raise ValueError("Either model_dir or repo_id must be provided")

def generate(model, tokenizer, prompt, max_length=100, temperature=1.0, top_k=50, device="cpu"):
    model.eval()
    input_ids = tokenizer.encode(prompt)
    input_ids = torch.tensor([input_ids], dtype=torch.long).to(device)
    generated = input_ids.clone()

    for _ in range(max_length):
        with torch.no_grad():
            outputs = model(generated)
            logits = outputs["logits"][:, -1, :] / temperature
            # Top-k filtering
            top_k_logits, top_k_indices = torch.topk(logits, top_k)
            probs = torch.nn.functional.softmax(top_k_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            next_token_id = top_k_indices.gather(-1, next_token)
            generated = torch.cat([generated, next_token_id], dim=-1)
            if next_token_id.item() == tokenizer.token_to_id("[EOS]"):
                break
    output_ids = generated[0].tolist()
    return tokenizer.decode(output_ids)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo_id", type=str, help="HF repo id, e.g., username/Klyro-V1000")
    parser.add_argument("--prompt", type=str, default="Hello world")
    parser.add_argument("--max_length", type=int, default=50)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    token = os.getenv("HF_TOKEN")
    model, tokenizer = load_model_and_tokenizer(repo_id=args.repo_id, token=token, device=args.device)
    output = generate(
        model,
        tokenizer,
        args.prompt,
        max_length=args.max_length,
        temperature=args.temperature,
        top_k=args.top_k,
        device=args.device,
    )
    print("Prompt:", args.prompt)
    print("Generation:", output)