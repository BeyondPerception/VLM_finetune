# VLM_finetune

This is the inference server deployed to northflank and using the coreweave GPUs.

It is a websocket server that receives a stream of images and detects if there is a person who has fallen.

The model we used is the Video-LLama2 model finetuned on the dataset in this directory (GMDCSA-24) using qlora.
