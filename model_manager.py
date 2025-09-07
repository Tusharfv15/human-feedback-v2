"""
Shared model manager to prevent multiple LLM instantiations and OOM errors
"""

try:
    from unsloth import FastLanguageModel
    UNSLOTH_AVAILABLE = True
except ImportError:
    UNSLOTH_AVAILABLE = False


class ModelManager:
    """Singleton manager for shared LLM model instance"""
    
    _instance = None
    _model = None
    _tokenizer = None
    _initialized = False
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ModelManager, cls).__new__(cls)
        return cls._instance
    
    def initialize_model(
        self,
        model_name: str = "tusharParsai/semikong-finetuned",
        max_seq_length: int = 2048,
        dtype: str = "float16",
        load_in_4bit: bool = True,
        device_map: dict = None
    ):
        """Initialize model once for all agents"""
        if self._initialized:
            print("[CACHED] ModelManager: Using existing model instance")
            return self._model, self._tokenizer
        
        if device_map is None:
            device_map = {"": "cuda:0"}
        
        try:
            if UNSLOTH_AVAILABLE:
                print(f"[LOADING] ModelManager: Loading {model_name} (shared instance)")
                self._model, self._tokenizer = FastLanguageModel.from_pretrained(
                    model_name=model_name,
                    max_seq_length=max_seq_length,
                    dtype=dtype,
                    load_in_4bit=load_in_4bit,
                    device_map=device_map
                )
                FastLanguageModel.for_inference(self._model)
                print("[SUCCESS] ModelManager: Model loaded successfully with Unsloth 2x faster inference")
                self._initialized = True
            else:
                print("[WARNING] ModelManager: Unsloth not available, using fallback")
                self._model = None
                self._tokenizer = None
                self._initialized = True
                
        except Exception as e:
            print(f"[ERROR] ModelManager: Error loading model: {e}")
            self._model = None
            self._tokenizer = None
            self._initialized = True
        
        return self._model, self._tokenizer
    
    def get_model(self):
        """Get the shared model instance"""
        if not self._initialized:
            return self.initialize_model()
        return self._model, self._tokenizer
    
    def cleanup(self):
        """Clean up model resources"""
        if self._model is not None:
            del self._model
            del self._tokenizer
            self._model = None
            self._tokenizer = None
            self._initialized = False
            print("[CLEANUP] ModelManager: Model resources cleaned up")


# Global instance
model_manager = ModelManager()