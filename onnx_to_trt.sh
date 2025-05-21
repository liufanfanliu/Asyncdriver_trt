/usr/src/tensorrt/bin/trtexec --onnx=./onnx32_lora_hidden/model.onnx \
  --minShapes=inputs_embeds:1x1x2048,attention_mask:1x1,position_ids:1x1 \
  --optShapes=inputs_embeds:1x64x2048,attention_mask:1x64,position_ids:1x64 \
  --maxShapes=inputs_embeds:1x512x2048,attention_mask:1x512,position_ids:1x512 \
  --saveEngine=model_dynamic_hiddden_lora_32.engine
  --fp16