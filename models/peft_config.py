"""
PEFT (LoRA and QLoRA) configuration utilities.
"""
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import BitsAndBytesConfig
import torch

from utils.config import LORA_R, LORA_ALPHA, LORA_DROPOUT, LORA_TARGET_MODULES


def get_lora_config(
    r: int = LORA_R,
    lora_alpha: int = LORA_ALPHA,
    lora_dropout: float = LORA_DROPOUT,
    target_modules: list = None
) -> LoraConfig:
    """
    Get LoRA configuration.
    
    Args:
        r: LoRA rank
        lora_alpha: LoRA alpha parameter
        lora_dropout: Dropout rate
        target_modules: Modules to apply LoRA to
    
    Returns:
        LoraConfig object
    """
    if target_modules is None:
        target_modules = LORA_TARGET_MODULES
    
    return LoraConfig(
        r=r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=target_modules,
        bias="none",
        task_type=None
    )


def get_qlora_config(
    r: int = LORA_R,
    lora_alpha: int = LORA_ALPHA,
    lora_dropout: float = LORA_DROPOUT,
    target_modules: list = None
) -> tuple:
    """
    Get QLoRA configuration (LoRA + 4-bit quantization).
    
    Args:
        r: LoRA rank
        lora_alpha: LoRA alpha parameter
        lora_dropout: Dropout rate
        target_modules: Modules to apply LoRA to
    
    Returns:
        Tuple of (LoraConfig, BitsAndBytesConfig)
    """
    lora_config = get_lora_config(r, lora_alpha, lora_dropout, target_modules)
    
    compute_dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
    
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=compute_dtype
    )
    
    return lora_config, bnb_config


def apply_lora(model: torch.nn.Module, lora_config: LoraConfig):
    """
    Apply LoRA to a model.
    
    Args:
        model: Model to apply LoRA to
        lora_config: LoRA configuration
    
    Returns:
        PEFT model
    """
    peft_model = get_peft_model(model, lora_config)
    return peft_model


def apply_qlora(model: torch.nn.Module, lora_config: LoraConfig):
    """
    Apply QLoRA to a model (LoRA + quantization).
    
    Args:
        model: Model to apply QLoRA to (should be loaded with quantization)
        lora_config: LoRA configuration
    
    Returns:
        PEFT model
    """
    model = prepare_model_for_kbit_training(model)
    
    peft_model = get_peft_model(model, lora_config)
    
    return peft_model


def count_trainable_parameters(model: torch.nn.Module) -> dict:
    """
    Count trainable parameters in a model.
    
    Args:
        model: Model to count parameters for
    
    Returns:
        Dictionary with parameter counts
    """
    trainable_params = 0
    all_params = 0
    
    for param in model.parameters():
        all_params += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    
    return {
        'trainable_params': trainable_params,
        'all_params': all_params,
        'trainable_percentage': 100 * trainable_params / all_params if all_params > 0 else 0
    }

