import asyncio
import base64
import json
import sys
from io import BytesIO

import ffmpeg
import torch
import websockets
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

Example: Video is empty
Output: "{"fall": false, "person": false}"

Example: People in video, but no one falls
Output: "{"fall": false, "person": true}"
"""

instruct = "Please describe what is happening in the video. If it seems like nothing is happening, or the frame is empty, please indicate that as well."

conversation = [
    {"role": "system", "content": "You are a helpful assistant."},
    {
        "role": "user",
        "content": [
            {"type": "video", "video": {
                "video_path": "/volume/VLM_finetune/dataset/Subject 4/Fall/01.mp4", "fps": 30, "max_frames": 180}},
            {"type": "text", "text": "What is happening in the video?"},
        ]
    },
]


model_path = '/volume/VideoLLaMA2.1-7B-16F'
model, processor, tokenizer = model_init(model_path)


async def process_image(images):
    output = mm_infer(
        processor[modal](images), instruct, model=model, tokenizer=tokenizer, do_sample=False, modal=modal)

    # try:
    #     data = json.loads(output)
    # except json.JSONDecodeError:
    #     data = {"fall": False, "person": False}

    return output


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
            await websocket.send(json.dumps({"error": str(e)}))


async def main():
    server = await websockets.serve(handler, "0.0.0.0", 8765, max_size=1e9)
    print("WebSocket server started on ws://localhost:8765")
    await server.wait_closed()

if __name__ == "__main__":
    asyncio.run(main())
