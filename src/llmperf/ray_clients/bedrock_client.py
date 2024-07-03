import io
import json
import os
import time
from typing import Any, Dict

import boto3
from boto3 import Session
from botocore.config import Config
import ray
from transformers import LlamaTokenizerFast

from llmperf.ray_llm_client import LLMClient
from llmperf.models import RequestConfig
from llmperf import common_metrics

ACCEPT = "application/json"
CONTENT_TYPE = "application/json"

BEDROCK_ENDPOINT_MAP = {
    "us-west-2": "https://bedrock-runtime.us-west-2.amazonaws.com",
    "us-east-1": "https://bedrock-runtime.us-east-1.amazonaws.com"
}


@ray.remote
class BedrockClient(LLMClient):
    """Client for Bedrock API."""

    def __init__(self):
        # Sagemaker doesn't return the number of tokens that are generated so we approximate it by
        # using the llama tokenizer.
        self.tokenizer = LlamaTokenizerFast.from_pretrained(
            "hf-internal-testing/llama-tokenizer"
        )
    
    def _get_bedrock_client(self):
        access_key = os.environ.get("AWS_ACCESS_KEY_ID")
        secret_key = os.environ.get("AWS_SECRET_ACCESS_KEY")

        if access_key is not None and secret_key is not None:
            # if the access credentials are explicitly provided
            session = Session(aws_access_key_id=access_key, aws_secret_access_key=secret_key)
        else:
            # if credentials are not provided, use the node group role's credentials
            session = Session(profile_name="bedrock")
        
        boto_config = Config(
            read_timeout=900, connect_timeout=900, retries={"max_attempts": 3}
        )

        region = os.environ.get("AWS_REGION_NAME")
        return session.client(
            service_name="bedrock-runtime",
            region_name=region,
            endpoint_url=BEDROCK_ENDPOINT_MAP.get(region),
            config=boto_config,
        )

    def llm_request(self, request_config: RequestConfig) -> Dict[str, Any]:

        if not os.environ.get("AWS_REGION_NAME"):
            raise ValueError("AWS_REGION_NAME must be set.")

        prompt = request_config.prompt
        prompt, _ = prompt

        model = request_config.model
        br_runtime = self._get_bedrock_client()

        sampling_params = request_config.sampling_params

        sampling_params["temperature"] = 0.1
        sampling_params["top_p"] = 0.9
        sampling_params["top_k"] = 50
        sampling_params["max_tokens"] = 512

        print(f'Sampling params: {sampling_params}')

        request = {
            "anthropic_version": "bedrock-2023-05-31",
            "messages": [
                {
                    "role": "user", 
                    "content": [
                        {"type": "text", "text": prompt},
                    ],
                },
            ],
            **request_config.sampling_params,
            
        }

        time_to_next_token = []
        tokens_requested = 0
        tokens_received = 0
        ttft = 0
        error_response_code = None
        generated_text = ""
        error_msg = ""
        output_throughput = 0
        total_request_time = 0
        metrics = {}

        try:
            response = br_runtime.invoke_model_with_response_stream(
                modelId=model,
                body=json.dumps(request),
                accept=ACCEPT,
                contentType=CONTENT_TYPE,
            )

            # Extract and print the response text in real-time using example from https://docs.aws.amazon.com/code-library/latest/ug/python_3_bedrock-runtime_code_examples.html
            for event in response["body"]:
                chunk = json.loads(event["chunk"]["bytes"])
                if chunk["type"] == "content_block_delta":
                    generated_text += chunk["delta"].get("text", "") 
                if chunk["type"] == "message_stop":
                    ttft = chunk["amazon-bedrock-invocationMetrics"].get("firstByteLatency", 0)
                    total_request_time = chunk["amazon-bedrock-invocationMetrics"].get("invocationLatency", 0)
                    tokens_requested = chunk["amazon-bedrock-invocationMetrics"].get("inputTokenCount", 0)
                    tokens_received = chunk["amazon-bedrock-invocationMetrics"].get("outputTokenCount", 0)

            print(f'Response: {generated_text}')
            output_throughput = tokens_received / total_request_time

        except Exception as e:
            print(f"Warning Or Error: {e}")
            print(error_response_code)
            error_msg = str(e)
            error_response_code = 500

        metrics[common_metrics.ERROR_MSG] = error_msg
        metrics[common_metrics.ERROR_CODE] = error_response_code
        metrics[common_metrics.INTER_TOKEN_LAT] = time_to_next_token
        metrics[common_metrics.TTFT] = ttft
        metrics[common_metrics.E2E_LAT] = total_request_time
        metrics[common_metrics.REQ_OUTPUT_THROUGHPUT] = output_throughput
        metrics[common_metrics.NUM_TOTAL_TOKENS] = tokens_received + tokens_requested
        metrics[common_metrics.NUM_OUTPUT_TOKENS] = tokens_received
        metrics[common_metrics.NUM_INPUT_TOKENS] = tokens_requested

        return metrics, generated_text, request_config
