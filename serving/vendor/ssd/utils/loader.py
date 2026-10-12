import os
import json
import re
from glob import glob
import torch
from torch import nn
from safetensors import safe_open
from tqdm import tqdm

def default_weight_loader(param: nn.Parameter, loaded_weight: torch.Tensor):
    param.data.copy_(loaded_weight)


def load_embedding_from_target(model: nn.Module, target_path: str, target_hidden_size: int = None, draft_hidden_size: int = None) -> bool:
    """Try to load embedding weights from target model path (safetensors or bin)"""
    # Only load target embeddings if hidden sizes match (or if sizes not provided, assume compatible)
    if target_hidden_size is not None and draft_hidden_size is not None:
        if target_hidden_size != draft_hidden_size:
            print(f"[load_model] Skipping target embeddings: target hidden_size={target_hidden_size} != draft hidden_size={draft_hidden_size}")
            return False
    
    target_keys = ["model.embed_tokens.weight", "embed_tokens.weight"]
    
    # Try safetensors first
    safetensor_files = glob(os.path.join(target_path, "*.safetensors"))
    for file in safetensor_files:
        try:
            with safe_open(file, "pt", "cpu") as f:
                keys = f.keys()
                for key in target_keys:
                    if key in keys:
                        print(f"[load_model] Found embedding {key} in {file}")
                        tensor = f.get_tensor(key)
                        # Always load into model.embed_tokens.weight for EAGLE
                        param = model.get_parameter("model.embed_tokens.weight")
                        weight_loader = getattr(param, "weight_loader", default_weight_loader)
                        weight_loader(param, tensor)
                        return True
        except Exception as e:
            print(f"[load_model] Error reading safetensor {file}: {e}")
            continue

    # Try bin files
    bin_files = glob(os.path.join(target_path, "pytorch_model*.bin"))
    for file in bin_files:
        try:
            # Load only if necessary? No way to peek keys without loading header or whole file for bin.
            # But usually we can just load map_location=cpu
            print(f"[load_model] Checking {file} for embeddings...")
            state_dict = torch.load(file, map_location="cpu")
            for key in target_keys:
                if key in state_dict:
                    print(f"[load_model] Found embedding {key} in {file}")
                    tensor = state_dict[key]
                    param = model.get_parameter("model.embed_tokens.weight")
                    weight_loader = getattr(param, "weight_loader", default_weight_loader)
                    weight_loader(param, tensor)
                    return True
        except Exception as e:
            print(f"[load_model] Error reading bin {file}: {e}")
            continue
            
    return False


def load_eagle_model(model: nn.Module, path: str, packed_modules_mapping: dict, target_path: str = None, target_hidden_size: int = None):
    """Load EAGLE3 draft model weights from safetensors or pytorch_model.bin"""
    # Try safetensors first
    safetensor_files = glob(os.path.join(path, "*.safetensors"))
    state_dict = None
    
    if safetensor_files:
        print(f"[load_model] Detected EAGLE3 draft model, trying safetensors first")
        # Load all safetensors into a single state dict
        state_dict = {}
        for file in safetensor_files:
            try:
                with safe_open(file, "pt", "cpu") as f:
                    for key in f.keys():
                        state_dict[key] = f.get_tensor(key)
                print(f"[load_model] Loaded {len(state_dict)} weights from {file}")
                break  # For EAGLE, typically just one file
            except Exception as e:
                print(f"[load_model] Error reading safetensor {file}: {e}")
                continue
    
    # Fall back to pytorch_model.bin if safetensors didn't work
    if state_dict is None or len(state_dict) == 0:
        print(f"[load_model] Loading EAGLE3 draft model from pytorch_model.bin")
        bin_file = os.path.join(path, "pytorch_model.bin")
        if not os.path.exists(bin_file):
            raise FileNotFoundError(f"No safetensors or pytorch_model.bin found at {path} for EAGLE3 draft model")
        state_dict = torch.load(bin_file, map_location="cpu")
    
    # Load d2t and t2d dictionaries
    if hasattr(model, 'd2t') and 'd2t' in state_dict:
        d2t_tensor = state_dict['d2t']
        model.d2t = {i: int(d2t_tensor[i].item()) for i in range(len(d2t_tensor))}
        model.d2t_tensor = d2t_tensor.to('cuda').long()  # keep as tensor for fast indexing
        print(f"[load_model] Loaded d2t dictionary with {len(model.d2t)} entries")
    
    if hasattr(model, 't2d') and 't2d' in state_dict:
        t2d_tensor = state_dict['t2d']
        model.t2d = {i: int(t2d_tensor[i].item()) for i in range(len(t2d_tensor))}
        model.t2d_tensor = t2d_tensor.to('cuda').long()  # keep as tensor for fast indexing
        print(f"[load_model] Loaded t2d dictionary with {len(model.t2d)} entries")
    
    # Check for embedding layer
    found_embed_tokens = False
    found_any_embed = False
    for weight_name in state_dict.keys():
        if 'embed' in weight_name.lower():
            found_any_embed = True
            if 'embed_tokens' in weight_name:
                found_embed_tokens = True
                break
    
    if not found_embed_tokens:
        if target_path and target_hidden_size is not None:
            draft_hidden_size = model.config.hidden_size
            print(f"[load_model] 'embed_tokens' not found in draft weights. Attempting to load from target path: {target_path}")
            if load_embedding_from_target(model, target_path, target_hidden_size, draft_hidden_size):
                found_embed_tokens = True
        elif target_path:
            # For backward compatibility - if target_hidden_size not provided, try anyway (8B case)
            print(f"[load_model] 'embed_tokens' not found in draft weights. Attempting to load from target path: {target_path}")
            if load_embedding_from_target(model, target_path):
                found_embed_tokens = True
        
        if not found_embed_tokens:
            if found_any_embed:
                raise ValueError(
                    f"[load_model] ERROR: Found embedding layer(s) in EAGLE3 weights but not 'embed_tokens'. "
                    f"Available weights: {list(state_dict.keys())}"
                )
            else:
                raise ValueError(
                    f"[load_model] ERROR: No embedding layer found in EAGLE3 weights. "
                    f"Expected 'embed_tokens' or similar. Available weights: {list(state_dict.keys())}"
                )
    
    # Load model weights
    for weight_name, loaded_weight in tqdm(state_dict.items(), desc="Loading EAGLE3 weights"):
        # Skip the dictionary tensors as we've already processed them
        if weight_name in ['d2t', 't2d']:
            continue
        
        print(f'[load_model] loading weight {weight_name} with shape {loaded_weight.shape}')
        
        # Check if this weight should use packed module loading
        is_packed = False
        for k, (v, shard_id) in packed_modules_mapping.items():
            if k in weight_name:
                # Replace the module name but keep the .weight suffix
                param_name = weight_name.replace(k, v)
                param = model.get_parameter(param_name)
                weight_loader = getattr(param, "weight_loader")
                weight_loader(param, loaded_weight, shard_id)
                is_packed = True
                break
        
        if is_packed:
            continue
            
        # Map EAGLE3 weight names to our architecture for unpacked weights
        if weight_name == 'midlayer.hidden_norm.weight':
            # Special handling for conditioning feature layernorm
            param_name = 'model.layer.conditioning_feature_ln.weight'
        elif weight_name.startswith('midlayer.'):
            # The single layer is stored in model.layer (not layers list) in our implementation
            param_name = weight_name.replace('midlayer.', 'model.layer.')
        elif weight_name == 'norm.weight':
            # norm.weight -> final_norm.weight
            param_name = 'final_norm.weight'
        elif weight_name == 'embed_tokens.weight':
            # embed_tokens.weight -> model.embed_tokens.weight
            param_name = 'model.embed_tokens.weight'
        else:
            # fc.weight and lm_head.weight stay the same
            param_name = weight_name
        
        # Load the parameter
        param = model.get_parameter(param_name)
        weight_loader = getattr(param, "weight_loader", default_weight_loader)
        weight_loader(param, loaded_weight)


def load_safetensors_model(model: nn.Module, path: str, packed_modules_mapping: dict):
    """Load model weights from safetensors files"""
    safetensor_files = sorted(glob(os.path.join(path, "*.safetensors")))
    if not safetensor_files:
        raise FileNotFoundError(f"No safetensors weights found in {path}")
    index_path = os.path.join(path, 'model.safetensors.index.json')
    indexed_weights = None
    if os.path.isfile(index_path):
        with open(index_path, encoding='utf-8') as index_file:
            indexed_weights = json.load(index_file)['weight_map']
        expected_files = set(indexed_weights.values())
        if any(os.path.basename(name) != name or not name.endswith('.safetensors') for name in expected_files):
            raise ValueError(f'Unsafe safetensors shard name in {index_path}')
        missing_files = expected_files - {os.path.basename(file) for file in safetensor_files}
        if missing_files:
            raise FileNotFoundError(f"Missing safetensors shards: {', '.join(sorted(missing_files))}")
        safetensor_files = [os.path.join(path, name) for name in sorted(expected_files)]
    else:
        shard_names = [re.fullmatch(r'.+-(\d+)-of-(\d+)\.safetensors', os.path.basename(file))
                       for file in safetensor_files]
        if any(shard_names):
            if not all(shard_names) or len({int(match.group(2)) for match in shard_names}) != 1:
                raise ValueError(f'Inconsistent safetensors shards in {path}')
            total = int(shard_names[0].group(2))
            if {int(match.group(1)) for match in shard_names} != set(range(1, total + 1)):
                raise FileNotFoundError(f'Incomplete safetensors shards in {path}')
    loaded = set()
    loaded_shards = {}
    for file in tqdm(safetensor_files, desc="Loading model files"):
        with safe_open(file, "pt", "cpu") as f:
            for weight_name in f.keys():
                if weight_name in loaded:
                    raise ValueError(f'Duplicate safetensors weight: {weight_name}')
                if indexed_weights is not None and indexed_weights.get(weight_name) != os.path.basename(file):
                    raise ValueError(f'Safetensors index mismatch for {weight_name}')
                loaded.add(weight_name)
                for k in packed_modules_mapping:
                    if k in weight_name:
                        v, shard_id = packed_modules_mapping[k]
                        param_name = weight_name.replace(k, v)
                        param = model.get_parameter(param_name)
                        weight_loader = getattr(param, "weight_loader")
                        weight_loader(param, f.get_tensor(weight_name), shard_id)
                        loaded_shards.setdefault(param_name, set()).add(shard_id)
                        break
                else:
                    param = model.get_parameter(weight_name)
                    weight_loader = getattr(param, "weight_loader", default_weight_loader)
                    weight_loader(param, f.get_tensor(weight_name))
                    loaded_shards.setdefault(weight_name, set()).add(None)
    if indexed_weights is not None and loaded != set(indexed_weights):
        raise ValueError(f'Safetensors index lists weights missing from checkpoint: {sorted(set(indexed_weights) - loaded)[:5]}')
    missing = []
    for name, _ in model.named_parameters():
        required = {shard_id for _, (packed_name, shard_id) in packed_modules_mapping.items()
                    if packed_name in name} or {None}
        if not required.issubset(loaded_shards.get(name, set())):
            missing.append(name)
    if missing:
        raise ValueError(f'Missing model parameters in safetensors checkpoint: {missing[:10]}')


def load_model(model: nn.Module, path: str, target_path: str = None, target_hidden_size: int = None):
    print(f"[load_model] loading model from {path}")
    packed_modules_mapping = getattr(model, "packed_modules_mapping", {})
    
    # Check if this is an EAGLE3 draft model
    is_eagle = 'eagle' in path.lower()
    
    if is_eagle:
        load_eagle_model(model, path, packed_modules_mapping, target_path=target_path, target_hidden_size=target_hidden_size)
    else:
        load_safetensors_model(model, path, packed_modules_mapping)

    print(f"[load_model] finished loading model from {path}")
