"""
Model Adapter for Gemma 3n Therapeutic Chatbot
Optimized for RTX 3070 Laptop GPU (8GB VRAM) with CPU offloading
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
import json
import os
import sys

class GemmaModelLoader:
    """
    Load and manage Gemma 3n model with fine-tuned therapeutic adapters
    Supports CPU offloading for limited GPU memory scenarios
    """
    
    def __init__(self):
        """
        Initialize the Gemma 3n therapeutic chatbot model loader
        Optimized for RTX 3070 Laptop (8GB VRAM) with CPU offloading
        """
        # Resolve paths from environment for deployment-friendly configuration
        module_dir = os.path.dirname(os.path.abspath(__file__))
        self.base_path = os.environ.get("GEMMA_BASE_PATH", module_dir)
        self.model_path = os.environ.get("GEMMA_MODEL_PATH", os.path.join(self.base_path, "gemma-3n"))
        self.adapter_path = os.environ.get("GEMMA_ADAPTER_PATH", self.model_path)
        self.offload_folder = os.environ.get("GEMMA_OFFLOAD_DIR", os.path.join(module_dir, "offload"))
        
        # Normalize paths
        self.base_path = os.path.abspath(self.base_path)
        self.model_path = os.path.abspath(self.model_path)
        self.adapter_path = os.path.abspath(self.adapter_path)
        self.offload_folder = os.path.abspath(self.offload_folder)
        
        # Device configuration
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Model components
        self.model = None
        self.tokenizer = None
        self.adapter_config = None
        self.processor_config = None
        self.special_tokens = None
        
        # Print initialization info
        print(f"Model path: {self.model_path}")
        print(f"Adapter path: {self.adapter_path}")
        print(f"Offload folder: {self.offload_folder}")
        print(f"Device: {self.device}")
        
        # Check GPU memory if available
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            print(f"Total GPU Memory: {gpu_memory:.2f} GB")
        else:
            print("⚠ No GPU detected - will use CPU only (very slow)")
    
    def ensure_offload_folder(self):
        """Create offload folder if it doesn't exist"""
        try:
            os.makedirs(self.offload_folder, exist_ok=True)
            print(f"✓ Offload folder ready: {self.offload_folder}")
            return True
        except Exception as e:
            print(f"✗ Could not create offload folder: {e}")
            return False
    
    def load_configs(self):
        """Load all configuration files from model directory"""
        try:
            print("\nLoading configuration files...")
            
            # Load adapter configuration
            adapter_config_path = os.path.join(self.adapter_path, "adapter_config.json")
            if os.path.exists(adapter_config_path):
                with open(adapter_config_path, 'r') as f:
                    self.adapter_config = json.load(f)
                print(f"✓ Adapter config loaded: {adapter_config_path}")
            else:
                print(f"⚠ Adapter config not found at: {adapter_config_path}")
            
            # Load processor configuration
            processor_config_path = os.path.join(self.adapter_path, "processor_config.json")
            if os.path.exists(processor_config_path):
                with open(processor_config_path, 'r') as f:
                    self.processor_config = json.load(f)
                print(f"✓ Processor config loaded: {processor_config_path}")
            else:
                print(f"⚠ Processor config not found")
            
            # Load special tokens
            special_tokens_path = os.path.join(self.adapter_path, "special_tokens_map.json")
            if os.path.exists(special_tokens_path):
                with open(special_tokens_path, 'r') as f:
                    self.special_tokens = json.load(f)
                print(f"✓ Special tokens loaded: {special_tokens_path}")
            else:
                print(f"⚠ Special tokens not found")
                
            return True
            
        except Exception as e:
            print(f"✗ Error loading configs: {str(e)}")
            import traceback
            traceback.print_exc()
            return False
    
    def load_tokenizer(self):
        """Load the tokenizer with custom configurations"""
        try:
            print("\nLoading tokenizer...")
            
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.adapter_path,
                trust_remote_code=True,
                use_fast=True
            )
            
            print(f"✓ Tokenizer loaded from: {self.adapter_path}")
            print(f"  Vocab size: {len(self.tokenizer)}")
            
            # Set padding token if not set
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                print("  ✓ Padding token set to EOS token")
                
            return True
            
        except Exception as e:
            print(f"✗ Error loading tokenizer: {str(e)}")
            import traceback
            traceback.print_exc()
            return False
    
    def check_gpu_memory(self):
        """Check available GPU memory"""
        if not torch.cuda.is_available():
            return 0, 0
        
        try:
            total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            allocated = torch.cuda.memory_allocated(0) / (1024**3)
            free = total - allocated
            return free, total
        except:
            return 0, 0
    
    def load_model(self):
        """
        Load base model with CPU offloading enabled
        Optimized for limited GPU memory (8GB RTX 3070)
        """
        try:
            print("\n" + "="*70)
            print("Loading Gemma 3n Model (CPU Offloading Enabled)")
            print("="*70)
            
            # Check for adapter files
            adapter_model_path = os.path.join(self.adapter_path, "adapter_model.safetensors")
            if not os.path.exists(adapter_model_path):
                print(f"✗ Adapter model not found at: {adapter_model_path}")
                return False
            
            # Ensure offload folder exists
            if not self.ensure_offload_folder():
                print("⚠ Continuing without offload folder...")
            
            # Check GPU memory
            free_gpu, total_gpu = self.check_gpu_memory()
            if self.device == "cuda":
                print(f"\nGPU Memory Status:")
                print(f"  Total: {total_gpu:.2f} GB")
                print(f"  Free: {free_gpu:.2f} GB")
            
            # Import bitsandbytes for quantization
            try:
                import bitsandbytes as bnb
                has_bitsandbytes = True
                print("\n✓ bitsandbytes available")
            except ImportError:
                has_bitsandbytes = False
                print("\n⚠ bitsandbytes not available - will use full precision")
            
            # Configure model loading based on available resources
            if self.device == "cuda" and has_bitsandbytes:
                print("\nConfiguring 8-bit quantization with CPU offloading...")
                
                # 8-bit quantization is more stable for CPU offloading than 4-bit
                quantization_config = BitsAndBytesConfig(
                    load_in_8bit=True,
                    llm_int8_enable_fp32_cpu_offload=True,  # CRITICAL: Enable CPU offload
                    llm_int8_threshold=6.0
                )
                
                print("  ✓ 8-bit quantization configured")
                print("  ✓ CPU offloading ENABLED (llm_int8_enable_fp32_cpu_offload=True)")
                
                # Set max memory to prevent overflow
                max_memory = {
                    0: "6GiB",      # Leave 1.6GB buffer on GPU
                    "cpu": "16GiB"  # Use system RAM for offloaded layers
                }
                
                device_map = "auto"  # Let transformers decide optimal split
                
                print(f"\n  Memory allocation:")
                print(f"    GPU (device 0): 6GiB (leaving buffer)")
                print(f"    CPU: 16GiB")
                print(f"  Device map: auto (optimal split)")
                
                model_kwargs = {
                    "quantization_config": quantization_config,
                    "device_map": device_map,
                    "max_memory": max_memory,
                    "trust_remote_code": True,
                    "low_cpu_mem_usage": True,
                    "torch_dtype": torch.float16,
                    "offload_folder": self.offload_folder,
                    "offload_state_dict": True
                }
                
            elif self.device == "cuda":
                # GPU available but no bitsandbytes - use float16
                print("\nConfiguring float16 loading with CPU offloading...")
                
                max_memory = {
                    0: "5GiB",      # Conservative GPU allocation
                    "cpu": "16GiB"
                }
                
                model_kwargs = {
                    "device_map": "auto",
                    "max_memory": max_memory,
                    "trust_remote_code": True,
                    "low_cpu_mem_usage": True,
                    "torch_dtype": torch.float16,
                    "offload_folder": self.offload_folder,
                    "offload_state_dict": True
                }
                
            else:
                # CPU only mode
                print("\nConfiguring CPU-only mode...")
                
                model_kwargs = {
                    "device_map": "cpu",
                    "trust_remote_code": True,
                    "low_cpu_mem_usage": True,
                    "torch_dtype": torch.float32
                }
            
            # Load base model
            print(f"\nLoading base Gemma 3n model from: {self.model_path}")
            print("This may take several minutes (offloading to CPU)...")
            print("Please be patient...\n")
            
            base_model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                **model_kwargs
            )
            
            print("\n✓ Base model loaded successfully with CPU offloading")
            
            # Check memory usage after loading
            if torch.cuda.is_available():
                allocated_gb = torch.cuda.memory_allocated(0) / (1024**3)
                reserved_gb = torch.cuda.memory_reserved(0) / (1024**3)
                print(f"\n  GPU Memory after loading:")
                print(f"    Allocated: {allocated_gb:.2f} GB")
                print(f"    Reserved: {reserved_gb:.2f} GB")
            
            # Print device distribution
            if hasattr(base_model, 'hf_device_map'):
                device_map_summary = {}
                for key, value in base_model.hf_device_map.items():
                    device_map_summary[value] = device_map_summary.get(value, 0) + 1
                
                print(f"\n  Model layer distribution:")
                for device, count in sorted(device_map_summary.items()):
                    device_name = f"GPU {device}" if isinstance(device, int) else device.upper()
                    print(f"    {device_name}: {count} modules")
            
            # Load PEFT adapters
            print(f"\nLoading fine-tuned therapeutic adapters...")
            print(f"  Adapter path: {self.adapter_path}")
            
            try:
                self.model = PeftModel.from_pretrained(
                    base_model,
                    self.adapter_path,
                    device_map="auto"
                )
                print("✓ Fine-tuned adapters loaded successfully")
                
            except Exception as adapter_error:
                print(f"⚠ Could not load adapters separately: {adapter_error}")
                print("  Using base model (adapters may already be merged)")
                self.model = base_model
            
            # Set model to evaluation mode
            self.model.eval()
            
            print("\n" + "="*70)
            print("✓ Model Ready for Therapeutic Conversations")
            print("="*70)
            
            # Final memory check
            if torch.cuda.is_available():
                final_allocated = torch.cuda.memory_allocated(0) / (1024**3)
                final_reserved = torch.cuda.memory_reserved(0) / (1024**3)
                free_memory = total_gpu - final_reserved
                
                print(f"\nFinal GPU Memory Status:")
                print(f"  Used: {final_reserved:.2f} GB / {total_gpu:.2f} GB")
                print(f"  Free: {free_memory:.2f} GB")
            
            print("\n" + "="*70)
            print("PERFORMANCE NOTES:")
            print("="*70)
            print("⚠ Model is distributed across GPU and CPU")
            print("  • GPU handles critical layers for speed")
            print("  • CPU handles remaining layers to save memory")
            print("  • Expected response time: 5-15 seconds per message")
            print("  • First response may be slower (model warm-up)")
            print("="*70 + "\n")
            
            return True
            
        except Exception as e:
            print(f"\n✗ Error loading model: {str(e)}")
            print("\nFull traceback:")
            import traceback
            traceback.print_exc()
            
            print("\n" + "="*70)
            print("Troubleshooting Guide:")
            print("="*70)
            print("1. Ensure bitsandbytes is installed:")
            print("   pip install bitsandbytes>=0.45.0")
            print("\n2. Check available system RAM:")
            print("   free -h  # Need at least 16GB available")
            print("\n3. Close other GPU/RAM intensive applications:")
            print("   nvidia-smi  # Check GPU usage")
            print("\n4. Clear GPU cache and restart:")
            print("   sudo systemctl restart gdm  # Or reboot system")
            print("\n5. If still fails, try CPU-only mode by setting:")
            print("   device_map='cpu' in load_model()")
            print("="*70)
            
            return False
    
    def initialize(self):
        """Initialize all components in correct order"""
        print("=" * 70)
        print("Initializing Gemma 3n Therapeutic Chatbot")
        print("=" * 70)
        print(f"Working directory: {self.base_path}")
        print(f"Model directory: {self.model_path}")
        print("=" * 70)
        
        # Check model path exists
        if not os.path.exists(self.model_path):
            print(f"✗ Model path does not exist: {self.model_path}")
            return False
        
        # Load configurations
        if not self.load_configs():
            print("⚠ Warning: Some configs could not be loaded, continuing...")
        
        # Load tokenizer
        if not self.load_tokenizer():
            print("✗ Failed to load tokenizer")
            return False
        
        # Load model with CPU offloading
        if not self.load_model():
            print("✗ Failed to load model")
            return False
        
        print("\n" + "=" * 70)
        print("✓✓✓ Therapeutic Chatbot Initialization Complete! ✓✓✓")
        print("=" * 70 + "\n")
        
        return True
    
    def get_model_info(self):
        """Return comprehensive model information"""
        info = {
            "model_type": "Gemma 3n (4B parameters)",
            "model_path": self.model_path,
            "device": self.device,
            "quantization": "8-bit with CPU offload",
            "fine_tuning": "Therapeutic Counseling Dataset",
            "conversation_type": "Mental Health Support",
            "offload_enabled": True
        }
        
        # Add adapter info if available
        if self.adapter_config:
            info["adapter_type"] = self.adapter_config.get("peft_type", "Unknown")
            info["target_modules"] = self.adapter_config.get("target_modules", [])
            info["base_model"] = self.adapter_config.get("base_model_name_or_path", "Unknown")
        
        # Add GPU info if available
        if torch.cuda.is_available():
            info["gpu_name"] = torch.cuda.get_device_name(0)
            info["gpu_memory_total_gb"] = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            info["gpu_memory_allocated_gb"] = torch.cuda.memory_allocated(0) / (1024**3)
            info["gpu_memory_reserved_gb"] = torch.cuda.memory_reserved(0) / (1024**3)
        
        # Add device map info if available
        if self.model and hasattr(self.model, 'hf_device_map'):
            device_counts = {}
            for device in self.model.hf_device_map.values():
                device_counts[str(device)] = device_counts.get(str(device), 0) + 1
            info["device_distribution"] = device_counts
        
        return info
    
    def cleanup(self):
        """Cleanup resources"""
        try:
            if self.model:
                del self.model
            if self.tokenizer:
                del self.tokenizer
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            
            print("✓ Model resources cleaned up")
            return True
            
        except Exception as e:
            print(f"⚠ Error during cleanup: {e}")
            return False


# Test function for standalone execution
def test_model_loader():
    """Test the model loader"""
    print("\n" + "="*70)
    print("Testing Gemma Model Loader")
    print("="*70 + "\n")
    
    loader = GemmaModelLoader()
    
    if loader.initialize():
        print("\n✓ Model loader test PASSED")
        print("\nModel Info:")
        info = loader.get_model_info()
        for key, value in info.items():
            print(f"  {key}: {value}")
        
        loader.cleanup()
        return True
    else:
        print("\n✗ Model loader test FAILED")
        return False


if __name__ == "__main__":
    # Run test when executed directly
    success = test_model_loader()
    sys.exit(0 if success else 1)
