import argparse
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# Wrapper module to extract the final hidden_states
class Wrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model.eval()

    def forward(self, inputs_embeds, attention_mask, position_ids):
        outputs = self.model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            use_cache=False,
            output_hidden_states=True
        )
        return outputs.hidden_states[-1]  # Return final hidden_states

def main():
    parser = argparse.ArgumentParser(description="Export LoRA merged model to ONNX")
    parser.add_argument("--model_path", type=str, required=True, help="Path to base model")
    parser.add_argument("--lora_path", type=str, required=True, help="Path to LoRA weights")
    parser.add_argument("--onnx_path", type=str, required=True, help="Output path for ONNX model")
    args = parser.parse_args()

    device = torch.device("cpu")

    # Load tokenizer and base model
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_fast=False)
    base_model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.float32,
        device_map=None,
        low_cpu_mem_usage=True
    )

    # Apply LoRA weights and merge into base model
    model = PeftModel.from_pretrained(
        base_model,
        args.lora_path,
        torch_dtype=torch.float32,
        device_map=None
    )
    model = model.merge_and_unload().to(device).eval()

    # Example input for tracing
    sample_text = "i"
    toks = tokenizer(sample_text, return_tensors="pt")
    inputs_embeds = model.get_input_embeddings()(toks.input_ids)
    attention_mask = toks.attention_mask
    batch, seq_len = toks.input_ids.shape
    position_ids = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch, -1)

    # Wrap the model
    wrapper = Wrapper(model).to(device)

    # Export to ONNX
    torch.onnx.export(
        wrapper,
        (inputs_embeds, attention_mask, position_ids),
        args.onnx_path,
        opset_version=17,
        do_constant_folding=True,
        input_names=["inputs_embeds", "attention_mask", "position_ids"],
        output_names=["hidden_states"],
        dynamic_axes={
            "inputs_embeds": {0: "batch", 1: "seq_len"},
            "attention_mask": {0: "batch", 1: "seq_len"},
            "position_ids": {0: "batch", 1: "seq_len"},
            "hidden_states": {0: "batch", 1: "token_seq"}
        }
    )

    print(f"✅ ONNX model exported successfully to {args.onnx_path}")

if __name__ == "__main__":
    main()
