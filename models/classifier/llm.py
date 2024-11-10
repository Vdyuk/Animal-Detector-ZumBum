import requests
from PIL import Image

import torch
from transformers import AutoProcessor, AutoModelForPreTraining


model_id = "OpenFace-CQUPT/Human_LLaVA"
cuda = 0
model = AutoModelForPreTraining.from_pretrained("OpenFace-CQUPT/Human_LLaVA", torch_dtype=torch.float16).to(cuda)

processor = AutoProcessor.from_pretrained(model_id,trust_remote_code=True)


text = "Does animal have head?"
prompt = "USER: <image>\n" + text + "\nASSISTANT:"
image_file = "bbox0.jpg"
raw_image = Image.open(image_file)
# raw_image = Image.open(requests.get(image_file, stream=True).raw)
inputs = processor(images=raw_image, text=prompt, return_tensors='pt').to(cuda, torch.float16)

output = model.generate(**inputs, max_new_tokens=400, do_sample=False)
predict = processor.decode(output[0][:], skip_special_tokens=True)
print(predict)