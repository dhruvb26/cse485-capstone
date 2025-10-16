import json
import logging
from threading import Thread
from typing import Dict, List, Tuple, cast

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    TextIteratorStreamer,
)
from transformers.tokenization_utils_base import BatchEncoding

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)


def _check_gpu() -> None:
    """Check whether a GPU is available, exit if not."""
    if torch.cuda.is_available():
        logger.info(
            "GPU is available and running on GPU %s", torch.cuda.get_device_name(0)
        )
    else:
        logger.info("GPU is not available, exiting program")
        exit(1)


def _load_model(
    model_name: str = "Qwen/Qwen2.5-7B",
) -> Tuple[PreTrainedModel, PreTrainedTokenizerBase]:
    """Load a Hugging Face model and tokenizer from cache or hub."""
    try:
        cache_dir: str = "/scratch/dbansa11/hf_models"

        tokenizer: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(
            model_name, cache_dir=cache_dir, use_fast=True
        )
        model: PreTrainedModel = AutoModelForCausalLM.from_pretrained(
            model_name, dtype="auto", device_map="auto", cache_dir=cache_dir
        )

        return model, tokenizer
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        exit(1)


def run_model(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    messages: List[Dict[str, str]],
    stream: bool = False,
) -> str:
    """Run the model with optional streaming output.

    Args:
        model: The loaded language model
        tokenizer: The model tokenizer
        messages: List of conversation messages (system, user, assistant)
        stream: If True, prints tokens as generated; if False, returns complete text

    Returns:
        Complete generated text as string
    """
    try:
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        if not isinstance(prompt, str):
            raise ValueError(f"Expected string prompt, got {type(prompt)}")

        inputs: BatchEncoding = tokenizer(prompt, return_tensors="pt").to(model.device)

        if stream:
            return _generate_streaming_with_print(model, tokenizer, inputs)
        else:
            # Non-streaming mode: return complete text
            outputs = model.generate(**inputs, max_new_tokens=1024)  # type: ignore
            return tokenizer.decode(outputs[0], skip_special_tokens=True)

    except Exception as e:
        logger.error(f"Error running model: {e}")
        return f"Error: {e}"


def _generate_streaming_with_print(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    inputs: BatchEncoding,
) -> str:
    """Helper function for streaming generation with real-time printing."""
    try:
        # Create a streamer that will yield tokens as they're generated
        streamer = TextIteratorStreamer(
            cast(AutoTokenizer, tokenizer), skip_special_tokens=True, skip_prompt=True
        )

        # Generation parameters
        generation_kwargs = {
            **inputs,
            "max_new_tokens": 1024,
            "do_sample": True,
            "temperature": 0.7,
            "streamer": streamer,
        }

        # Start generation in a separate thread
        def generate() -> None:
            model.generate(**generation_kwargs)  # type: ignore

        thread = Thread(target=generate)
        thread.start()

        full_response = ""
        for new_text in streamer:
            print(new_text, end="", flush=True)
            full_response += new_text
        print("\n")

        thread.join()

        return full_response

    except Exception as e:
        logger.error(f"Error in streaming generation: {e}")
        error_msg = f"Error: {e}"
        print(error_msg, end="", flush=True)
        return error_msg


def main() -> None:
    """Main entrypoint: interactive conversation with the model."""
    _check_gpu()
    model, tokenizer = _load_model(model_name="Qwen/Qwen2.5-7B-Instruct")

    with open("prompt.json", "r") as f:
        prompt_data = json.load(f)

    messages = [{"role": "system", "content": prompt_data["system"]}]
    logger.info("Type your messages and press Enter to chat. Ctrl+C to exit.")

    try:
        while True:
            try:
                user_input = input("\n[User] - ").strip()
            except EOFError:
                logger.info("Goodbye!")
                break

            if not user_input:
                logger.info("Please enter a message.")
                continue

            messages.append({"role": "user", "content": user_input})

            print("[Assistant] - ", end="", flush=True)

            response = run_model(
                model,
                tokenizer,
                messages,
                stream=True,
            )

            messages.append({"role": "assistant", "content": response})

    except KeyboardInterrupt:
        logger.info("Goodbye!")
        exit(0)


if __name__ == "__main__":
    main()
