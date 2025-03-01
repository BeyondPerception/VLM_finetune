import asyncio
import base64
import json
import sys
from io import BytesIO

import ffmpeg
import imageio.v3 as iio
import torch
import websockets
import numpy as np
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor

sys.path.append('./')

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

instruct = """You are monitoring a camera feed for an old persons home. Watch the video,
and determine if someone is falling or has fallen.

Identify if there is a person in the frame, and if someone falls down.

PLEASE RESPOND IN JSON FORMAT and do not include any other text.

Example: Someone falls down
Output: "{"fall": true, "person": true}"

Example: Room is empty
Output: "{"fall": false, "person": false}"

Example: People in video, but no one falls
Output: "{"fall": false, "person": true}"
"""


async def process_image(images):
    # Convert PIL images to a video and save to /tmp/output.mp4
    video_path = "/tmp/output.mp4"
    frames = [np.array(img) for img in images]
    fps = 30
    iio.imwrite(video_path, frames, fps=fps)

    conversation = [
        {"role": "system", "content": "You are a helpful assistnt."},
        {
            "role": "user",
            "content": [
                {"type": "video", "video": {
                    "video_path": "/tmp/output.mp4", "fps": 30, "max_frames": 180}},
                {"type": "text", "text": instruct},
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

    try:
        data = json.loads(response)
    except json.JSONDecodeError:
        data = {"fall": False, "person": False}

    return data


async def handler(websocket):
    async for message in websocket:
        try:
            if isinstance(message, str):
                message = message
                data = json.loads(message)

                images = []
                for image_str in data:
                    image_bytes = base64.b64decode(image_str)
                    image = Image.open(BytesIO(image_bytes))
                    images.append(image)

                response = await process_image(images)
                await websocket.send(json.dumps(response))
            else:
                await websocket.send(json.dumps({"error": "Invalid message type"}))
        except Exception as e:
            print(e)
            await websocket.send(json.dumps({"error": str(e)}))


async def main():
    server = await websockets.serve(handler, "0.0.0.0", 8765, max_size=1e9)
    print("WebSocket server started on ws://localhost:8765")
    await server.wait_closed()

if __name__ == "__main__":
    asyncio.run(main())
