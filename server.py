import asyncio
import base64
import json
import sys
from io import BytesIO

import websockets
from PIL import Image
from videollama2 import mm_infer, model_init
from videollama2.utils import disable_torch_init

sys.path.append('./')

disable_torch_init()

# Model setup
modal = 'video'
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

# instruct = "Please describe what is happening in the video."

model_path = '/volume/VideoLLaMA2.1-7B-16F'
model, processor, tokenizer = model_init(model_path)


async def process_image(images):
    output = mm_infer(
        processor[modal](images), instruct, model=model, tokenizer=tokenizer, do_sample=False, modal=modal)

    try:
        data = json.loads(output)
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
            await websocket.send(json.dumps({"error": str(e)}))


async def main():
    server = await websockets.serve(handler, "0.0.0.0", 8765, max_size=1e9)
    print("WebSocket server started on ws://localhost:8765")
    await server.wait_closed()

if __name__ == "__main__":
    asyncio.run(main())
