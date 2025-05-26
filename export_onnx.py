import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

model_path = "/path/to/your/model/path"
onnx_path  = "/path/to/onnx/model.onnx"
lora_model_path = "/path/to/lora/model/path"

device = torch.device("cpu")

tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
base_model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.float32,
    device_map=None,
    low_cpu_mem_usage=True
)

model = PeftModel.from_pretrained(
    base_model,
    lora_model_path,
    torch_dtype=torch.float32,
    device_map=None
)
model = model.merge_and_unload().to(device).eval()

text = "i"
toks = tokenizer(text, return_tensors="pt")
inputs_embeds = model.get_input_embeddings()(toks.input_ids)
attention_mask = toks.attention_mask
batch, seq = toks.input_ids.shape
position_ids = torch.arange(seq, device=device).unsqueeze(0).expand(batch, -1)

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
        
        return outputs.hidden_states[-1]

wrapper = Wrapper(model).to(device)

torch.onnx.export(
    wrapper,
    (inputs_embeds, attention_mask, position_ids),
    onnx_path,
    opset_version=17,
    do_constant_folding=True,
    input_names=["inputs_embeds", "attention_mask", "position_ids"],
    output_names=["hidden_states"],
    dynamic_axes={
        "inputs_embeds":   {0: "batch", 1: "seq_len"},
        "attention_mask":  {0: "batch", 1: "seq_len"},
        "position_ids":    {0: "batch", 1: "seq_len"},
        "hidden_states":   {0: "batch", 1: "token_seq"}
    }
)

print("✅ ONNX saved at", onnx_path)
