# ☁️ 🦜

Instantly deploy your local LoRA-tuned Llama in the cloud and scale to as much throughput as you want.

```python
from typing import List
from cloud_lora.main import CloudLora, GenerationRequest
import peft
from peft import PeftModel

PROMPTS: List[str] = ...

peft_model = ... # create your Llama model, then apply your LoRA adapters

cloud_model = CloudLora.create(peft_model)

from concurrent import futures

with futures.ThreadPoolExecutor(128) as pool:
    generation_requests = map(lambda prompt: GenerationRequest(prompt=prompt), PROMPTS)
    for result in pool.map(cloud_model.remote().get_completion, generation_requests):
        print(result)
```
