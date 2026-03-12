import os
import sys
from pathlib import Path

def setup_cuda_path():
    """
    Adds paths to NVIDIA DLL libraries (cublas, cudnn) to the Windows search system.
    Uses importlib.metadata to find package locations professionally.
    """
    if os.name != 'nt':
        return

    import importlib.metadata
    
    packages = ["nvidia-cublas-cu12", "nvidia-cudnn-cu12", "nvidia-cuda-runtime-cu12", "nvidia-cuda-nvrtc-cu12"]
    search_dirs = set()
    
    # 1. Professional discovery via importlib
    for pkg in packages:
        try:
            dist = importlib.metadata.distribution(pkg)
            # Find directories containing DLLs in the distribution
            for file in (dist.files or []):
                if file.name.lower().endswith(".dll"):
                    dll_path = Path(dist.locate_file(file)).parent
                    if (dll_path / "bin").exists():
                        search_dirs.add(dll_path / "bin")
                    search_dirs.add(dll_path)
        except importlib.metadata.PackageNotFoundError:
            continue

    # 2. Aggressive search for ANY folder containing a .dll in the venv (the final solution)
    venv_base = Path(sys.executable).parent.parent
    site_packages = venv_base / "Lib" / "site-packages"
    
    for root, _, files in os.walk(site_packages):
        if any(f.lower().endswith(".dll") for f in files):
            search_dirs.add(Path(root))
            
            # 2.1 Compatibility fix: if we have cublas64_12.dll but something wants v13
            if "nvidia" in root.lower() and "cublas" in root.lower():
                if "cublas64_12.dll" in files and "cublas64_13.dll" not in files:
                    try:
                        import shutil
                        shutil.copy2(Path(root) / "cublas64_12.dll", Path(root) / "cublas64_13.dll")
                        print(f"DEBUG: Created compatibility alias: cublas64_13.dll")
                    except Exception: pass

    added_count = 0
    # Filter and add
    final_dirs = sorted(list(search_dirs))
    for bin_dir in final_dirs:
        path_str = str(bin_dir.resolve())
        if bin_dir.is_dir():
            # Only add if it's an 'important' folder to avoid spamming system paths
            is_important = any(x in path_str.lower() for x in ["nvidia", "llama_cpp", "ctranslate2", "bin"])
            if is_important:
                if hasattr(os, "add_dll_directory"):
                    try:
                        # Register in Windows DLL Loader
                        os.add_dll_directory(path_str)
                        added_count += 1
                        # print(f"DEBUG: Registered: {path_str}")
                    except Exception:
                        pass
                
                if path_str not in os.environ["PATH"]:
                    os.environ["PATH"] = path_str + os.pathsep + os.environ["PATH"]
    
    if added_count > 0:
        print(f"DEBUG: DLL loading system initialized ({added_count} dirs).")

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
