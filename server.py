import os
from modal import Stub, Secret, Image, asgi_app, method


### Container Definition ###

# Used to download the model within the modal container
def download_model_to_folder():
    from huggingface_hub import snapshot_download

    snapshot_download(
        "meta-llama/Llama-2-7b-chat-hf",
        local_dir="/model",
        token=os.environ["HUGGINGFACE_TOKEN"],
    )

MODEL_DIR = "/model"

# Container image definition
llama_image = (
    Image.from_dockerhub("nvcr.io/nvidia/pytorch:22.12-py3")
    .pip_install(
        "torch==2.0.1", index_url="https://download.pytorch.org/whl/cu118"
    )
    # Pin vLLM to 07/19/2023
    .pip_install(
        "vllm @ git+https://github.com/vllm-project/vllm.git@bda41c70ddb124134935a90a0d51304d2ac035e8"
    )
    # Use the barebones hf-transfer package for maximum download speeds. No progress bar, but expect 700MB/s.
    .pip_install("hf-transfer~=0.1")
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
    .run_function(
        download_model_to_folder, secret=Secret.from_name("a2a-huggingface-secret")
    )
)

stub = Stub("example-vllm-inference", image=llama_image)


### LLM Class ###
@stub.cls(
    gpu="A100", 
    container_idle_timeout=60 * 5, 
    secret=Secret.from_name("a2a-huggingface-secret")
)
class Llamma7B:
    def __enter__(self):
        from vllm import LLM

        # Load the model. Tip: MPT models may require `trust_remote_code=true`.
        self.llm = LLM(MODEL_DIR)

    @method()
    def generate(self, input):
        from vllm import SamplingParams

        sampling_params = SamplingParams(
            temperature=0.75,
            top_p=1,
            max_tokens=100,
            presence_penalty=1.15,
        )
        result = self.llm.generate(input, sampling_params)
        return result
    

### FastAPI Server ###
@stub.function(
    container_idle_timeout=300,
    timeout=600,
)
@asgi_app()
def server():
    from fastapi import FastAPI
    from fastapi.responses import StreamingResponse
    from pydantic import BaseModel
    import json

    app = FastAPI()
    llama7B = Llamma7B()

    class PredictionRequest(BaseModel):
        input: str

    @app.post("/predict")
    async def predict(prediction_request: PredictionRequest):
        print(f"Input: {prediction_request.input}")
        result = llama7B.generate.call(prediction_request.input)
        print(f"Result: {json.dumps(result)}")
        return result

    return app