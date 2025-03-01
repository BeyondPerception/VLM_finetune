from videollama2.utils import disable_torch_init
from videollama2 import model_init, mm_infer
import sys
sys.path.append('./')


def inference():
    disable_torch_init()

    # Video Inference
    modal = 'video'
    modal_path = '/volume/VLM_finetune/dataset/Subject 4/ADL/01.mp4'
    instruct = """What is happening in the video? Does someone fall down? PLEASE RESPOND IN JSON FORMAT and do not inclde any other text.
    
    Example: Someone falls down
    Output: "{"fall": true}"
    """

    model_path = '/volume/VideoLLaMA2.1-7B-16F'
    model, processor, tokenizer = model_init(model_path)
    output = mm_infer(processor[modal](
        modal_path), instruct, model=model, tokenizer=tokenizer, do_sample=False, modal=modal)

    print(output)


if __name__ == "__main__":
    inference()
