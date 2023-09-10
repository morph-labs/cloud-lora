import os
import tempfile
from dataclasses import dataclass
from peft import PeftModel
from tqdm import tqdm
from typing import Any
import requests

CLOUD_LORA_URL: str = os.environ.get("CLOUD_LORA_URL")
assert CLOUD_LORA_URL, f"Error: CLOUD_LORA_URL not set."

@dataclass
class GenerationRequest:
    prompt: str

@dataclass
class CloudLoraRemote:
    uuid: str

    def get_completion(self, req: GenerationRequest):
        resp = requests.post(os.path.join(CLOUD_LORA_URL, "generate", self.uuid), json=req)
        return resp.json()["output"]

class CloudLoraCreateParams:
    adapter_config_path: Any
    adapter_model_bin_path: Any

    @classmethod
    def from_peft_model(cls, peft_model: PeftModel):
        tmpdir = tempfile.TemporaryDirectory().__enter__()
        peft_model.save_pretrained(tmpdir)
        # with open(os.path.join(tmpdir, "adapter_config.json"), "r") as f:
        #     adapter_config = json.load(f)
        return cls(
            adapter_config=os.path.join(tmpdir, "adapter_config.json"),
            adapter_model_bin_path=os.path.join(tmpdir, "adapter_model.bin"),
        )


class CloudLoraCreateResult:
    uuid: str


def _create(create_params: CloudLoraCreateParams) -> CloudLoraCreateResult:
    with open(create_params.adapter_config_path, "rb") as f1:
        with open(create_params.adapter_model_bin_path, "rb") as f2:
            result = requests.post(
                os.path.join(CLOUD_LORA_URL, "upload-files"),
                files=dict(
                    adapter_config=f1,
                    adapter_model=f2,
                )
            )
    uuid = result.json()["uuid"]
    return CloudLoraCreateResult(uuid=uuid)


class CloudLora:
    @staticmethod
    def create(peft_model: PeftModel):
        cloud_lora_create_result = _create(CloudLoraCreateParams.from_peft_model(peft_model))
        # method for registering the model and uploading weights
        uuid = (
            cloud_lora_create_result.uuid
        )  # <- contains a UUID to the model which resolves server-side to the GCS blob

        def remote(self):
            return CloudLoraRemote(uuid=uuid)

        return type("CloudLora", (PeftModel,), dict(remote=remote))(peft_model)
