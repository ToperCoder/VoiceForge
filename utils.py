import os

def setup_cuda_path():
    candidates = [
        "/usr/local/cuda/lib64",
        "/usr/lib/x86_64-linux-gnu/libcublas/12",
    ]
    dirs = [p for p in candidates if os.path.isdir(p)]
    if not dirs:
        return
    existing = [p for p in os.environ.get("LD_LIBRARY_PATH", "").split(":") if p]
    for p in dirs:
        if p not in existing:
            existing.insert(0, p)
    os.environ["LD_LIBRARY_PATH"] = ":".join(existing)

def clean_llm_output(text: str) -> str:
    """
    Cleans the model's output from technical artifacts like <think> blocks,
    template leaks, or role prefixes.
    """
    if not text:
        return ""
        
    result = text.strip()
    
    # 1. Remove <think> blocks (reasoning)
    if "<think>" in result and "</think>" in result:
        result = result.split("</think>")[-1].strip()
    elif "<think>" in result:
        # If the model forgot to close the tag
        result = result.split("<think>")[-1].strip()
        
    # 2. Remove common template leaks
    artifacts = ["<|im_end|>", "<|im_start|>", "assistant\n"]
    for art in artifacts:
        if art in result:
            result = result.replace(art, "").strip()
            
    # 3. Final cleanup of any trailing garbage
    return result.strip()
