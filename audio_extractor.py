import subprocess
import imageio_ffmpeg

def extract_audio(input_file: str, output_file: str):
    """
    Extracts audio from video or converts audio file to:
    - mono
    - 16kHz
    - wav
    """
    print("Extracting audio...")
    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
    command = [
        ffmpeg_exe, 
        "-y",               # Overwrite output file if it exists
        "-i", input_file,   # Input file
        "-ac", "1",         # Mono audio
        "-ar", "16000",     # 16kHz sample rate
        output_file         # Output file
    ]
    
    # Run ffmpeg silently
    subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
    return output_file
