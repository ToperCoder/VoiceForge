import os

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
