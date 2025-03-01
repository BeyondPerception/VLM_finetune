import torch
from transformers import AutoModelForCausalLM, AutoProcessor
import ffmpeg

device = "cuda:0"
model_path = "/volume/VideoLLaMA3-7B"
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    trust_remote_code=True,
    device_map={"": device},
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
)
processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)

conversation = [
    {"role": "system", "content": "You are a helpful assistant."},
    {
        "role": "user",
        "content": [
            {"type": "video", "video": {
                "video_path": "./volume/dataset/Subject 4/01.mp4", "fps": 30, "max_frames": 180}},
            {"type": "text", "text": "What is happening in the video?"},
        ]
    },
]

try:
    inputs = processor(
        conversation=conversation,
        add_system_prompt=True,
        add_generation_prompt=True,
        return_tensors="pt"
    )
except ffmpeg.Error as e:
    print(e.stderr)
    import sys
    sys.exit(1)
inputs = {k: v.to(device) if isinstance(v, torch.Tensor)
          else v for k, v in inputs.items()}
if "pixel_values" in inputs:
    inputs["pixel_values"] = inputs["pixel_values"].to(torch.bfloat16)
output_ids = model.generate(**inputs, max_new_tokens=1024)
response = processor.batch_decode(
    output_ids, skip_special_tokens=True)[0].strip()
print(response)
