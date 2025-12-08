"""
Demo inference server for aligned models.

This module provides a simple Flask server for interactive model inference.
"""

import argparse
import os
from typing import Dict, Optional

import torch


def create_app(
    checkpoint_path: str,
    device: str = "auto",
):
    """
    Create Flask application for model serving.
    
    Args:
        checkpoint_path: Path to model checkpoint.
        device: Device to run inference on.
    
    Returns:
        Flask application instance.
    """
    try:
        from flask import Flask, jsonify, request
    except ImportError:
        raise ImportError("Flask not installed. Run: pip install flask")
    
    from src.models.gemma_wrapper import GemmaWrapper
    from src.tokenization.tokenizer_wrapper import TokenizerWrapper
    from src.core.utils import get_device
    
    app = Flask(__name__)
    
    # Load model and tokenizer lazily
    model = None
    tokenizer = None
    
    def load_model():
        nonlocal model, tokenizer
        if model is None:
            device_obj = get_device(device)
            print(f"Loading model from {checkpoint_path}...")
            
            # Determine if checkpoint is a directory or base model name
            if os.path.isdir(checkpoint_path):
                # Load from local checkpoint
                model_path = os.path.join(checkpoint_path, "model")
                if os.path.exists(model_path):
                    # Custom checkpoint format
                    base_model = "google/gemma-3-270m-it"  # Default
                    model = GemmaWrapper(base_model, device=str(device_obj))
                    model.load(model_path)
                else:
                    # HF format checkpoint
                    model = GemmaWrapper(checkpoint_path, device=str(device_obj))
            else:
                # Load from HF hub
                model = GemmaWrapper(checkpoint_path, device=str(device_obj))
            
            tokenizer = TokenizerWrapper.from_pretrained(
                checkpoint_path if not os.path.isdir(checkpoint_path)
                else "google/gemma-3-270m-it"
            )
            
            model.eval()
            print("Model loaded successfully!")
    
    @app.route("/health", methods=["GET"])
    def health():
        """Health check endpoint."""
        return jsonify({"status": "healthy"})
    
    @app.route("/generate", methods=["POST"])
    def generate():
        """
        Generate text from a prompt.
        
        Request body:
            {
                "prompt": "Your prompt here",
                "max_tokens": 128,
                "temperature": 0.7,
                "top_p": 0.9
            }
        
        Response:
            {
                "generated_text": "Model response",
                "prompt": "Original prompt"
            }
        """
        load_model()
        
        data = request.get_json()
        
        if not data or "prompt" not in data:
            return jsonify({"error": "Missing prompt"}), 400
        
        prompt = data["prompt"]
        max_tokens = data.get("max_tokens", 128)
        temperature = data.get("temperature", 0.7)
        top_p = data.get("top_p", 0.9)
        
        # Format as chat if needed
        formatted = f"<start_of_turn>user\n{prompt}<end_of_turn>\n<start_of_turn>model\n"
        
        # Encode
        encoded = tokenizer.encode(formatted)
        input_ids = encoded["input_ids"].unsqueeze(0).to(model.device)
        attention_mask = encoded["attention_mask"].unsqueeze(0).to(model.device)
        
        # Generate
        with torch.no_grad():
            output_ids = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=temperature > 0,
            )
        
        # Decode
        generated = tokenizer.decode(
            output_ids[0, input_ids.size(1):].tolist(),
            skip_special_tokens=True,
        )
        
        # Clean up response
        if "<end_of_turn>" in generated:
            generated = generated.split("<end_of_turn>")[0]
        
        return jsonify({
            "generated_text": generated.strip(),
            "prompt": prompt,
        })
    
    @app.route("/batch_generate", methods=["POST"])
    def batch_generate():
        """
        Generate text for multiple prompts.
        
        Request body:
            {
                "prompts": ["prompt1", "prompt2"],
                "max_tokens": 128
            }
        """
        load_model()
        
        data = request.get_json()
        
        if not data or "prompts" not in data:
            return jsonify({"error": "Missing prompts"}), 400
        
        prompts = data["prompts"]
        max_tokens = data.get("max_tokens", 128)
        
        results = []
        for prompt in prompts:
            formatted = f"<start_of_turn>user\n{prompt}<end_of_turn>\n<start_of_turn>model\n"
            encoded = tokenizer.encode(formatted)
            input_ids = encoded["input_ids"].unsqueeze(0).to(model.device)
            attention_mask = encoded["attention_mask"].unsqueeze(0).to(model.device)
            
            with torch.no_grad():
                output_ids = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=max_tokens,
                    do_sample=False,
                )
            
            generated = tokenizer.decode(
                output_ids[0, input_ids.size(1):].tolist(),
                skip_special_tokens=True,
            )
            
            if "<end_of_turn>" in generated:
                generated = generated.split("<end_of_turn>")[0]
            
            results.append({
                "prompt": prompt,
                "generated_text": generated.strip(),
            })
        
        return jsonify({"results": results})
    
    return app


def main():
    """Run the demo server."""
    parser = argparse.ArgumentParser(description="Gemma Alignment Demo Server")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint or HF model name",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Host to bind to",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=5000,
        help="Port to bind to",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device to run on (auto, cuda, cpu)",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Run in debug mode",
    )
    
    args = parser.parse_args()
    
    app = create_app(args.checkpoint, device=args.device)
    
    print(f"Starting server on {args.host}:{args.port}")
    print(f"Model checkpoint: {args.checkpoint}")
    print(f"Device: {args.device}")
    print("\nEndpoints:")
    print("  GET  /health - Health check")
    print("  POST /generate - Generate text from prompt")
    print("  POST /batch_generate - Generate from multiple prompts")
    
    app.run(
        host=args.host,
        port=args.port,
        debug=args.debug,
    )


if __name__ == "__main__":
    main()
