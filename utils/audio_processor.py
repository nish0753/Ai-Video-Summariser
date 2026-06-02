import os
import subprocess
import glob
import uuid

# Prepend /opt/homebrew/bin to PATH on macOS to find Homebrew-installed binaries like FFmpeg
if os.path.exists("/opt/homebrew/bin") and "/opt/homebrew/bin" not in os.environ["PATH"]:
    os.environ["PATH"] = "/opt/homebrew/bin:" + os.environ["PATH"]

import yt_dlp

DOWNLOAD_DIR = "downloads"

def download_youtube_audio(url: str) -> str:
    """Download audio from YouTube.
    We use %(id)s as the filename to natively cache downloads.
    If the file already exists, yt-dlp will automatically skip downloading it.
    """
    os.makedirs(DOWNLOAD_DIR, exist_ok=True)
    
    # We no longer use a random UUID so that yt-dlp can recognize if the file already exists
    output_path = os.path.join(DOWNLOAD_DIR, "%(id)s.%(ext)s")
    
    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": output_path,
        "quiet": True,
        "noplaylist": True,
        "remote_components": ["ejs:github"],
        "postprocessors": [{
            "key": "FFmpegExtractAudio",
            "preferredcodec": "m4a",
        }],
    }
    
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        # Get the base filename, but because we force extraction to m4a, the final file ends in .m4a
        base_filename = ydl.prepare_filename(info)
        final_file = os.path.splitext(base_filename)[0] + ".m4a"
        return final_file


def convert_to_wav(input_path: str) -> str:
    """Legacy function, no longer strictly needed but kept for compatibility.
    Uses ffmpeg instead of pydub to avoid loading massive files into RAM.
    """
    output_path = os.path.splitext(input_path)[0] + "_converted.wav"
    if os.path.exists(output_path):
        return output_path

    subprocess.run([
        "ffmpeg", "-y", "-i", input_path, "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1", output_path
    ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return output_path


def chunk_audio(input_path: str, chunk_minutes: int = 10, chunk_seconds: int = None) -> list:
    """Split an audio file into chunks instantly using ffmpeg without decoding it."""
    duration_sec = chunk_seconds if chunk_seconds is not None else (chunk_minutes * 60)

    base_name = os.path.splitext(input_path)[0]
    ext = os.path.splitext(input_path)[1]

    # Include duration in filename so different chunk sizes don't collide in the cache
    out_pattern  = f"{base_name}_{duration_sec}s_chunk_%03d{ext}"
    glob_pattern = f"{base_name}_{duration_sec}s_chunk_*{ext}"

    # Check if chunks already exist for this file AND this chunk size
    existing_chunks = sorted(glob.glob(glob_pattern))
    if existing_chunks:
        print(f"Chunks ({duration_sec}s) already exist for {input_path}. Skipping chunking step.")
        return existing_chunks

    print(f"Running instant FFmpeg chunking on {input_path} ({duration_sec}s chunks)...")

    cmd = [
        "ffmpeg", "-y", "-i", input_path,
        "-c", "copy",
        "-f", "segment",
        "-segment_time", str(duration_sec),
        "-reset_timestamps", "1",
        out_pattern
    ]

    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    chunks = sorted(glob.glob(glob_pattern))
    return chunks


def enforce_cache_limit(max_size_mb: int = 2000):
    """Automatically manage disk space. 
    If the downloads folder exceeds max_size_mb, delete the oldest files until it's under the limit.
    """
    if not os.path.exists(DOWNLOAD_DIR):
        return
        
    max_bytes = max_size_mb * 1024 * 1024
    
    # Get all files in downloads with their size and modification time
    files = []
    total_size = 0
    for filename in os.listdir(DOWNLOAD_DIR):
        filepath = os.path.join(DOWNLOAD_DIR, filename)
        if os.path.isfile(filepath):
            size = os.path.getsize(filepath)
            mtime = os.path.getmtime(filepath)
            files.append({"path": filepath, "size": size, "mtime": mtime})
            total_size += size
            
    if total_size <= max_bytes:
        return
        
    print(f"Cache limit exceeded ({total_size / (1024*1024):.1f} MB). Cleaning up oldest files...")
    
    # Sort files by oldest first
    files.sort(key=lambda x: x["mtime"])
    
    for f in files:
        if total_size <= max_bytes:
            break
        try:
            os.remove(f["path"])
            total_size -= f["size"]
            print(f" - Deleted {os.path.basename(f['path'])}")
        except Exception as e:
            print(f"Failed to delete {f['path']}: {e}")

def process_input(source: str, chunk_minutes: int = 10) -> list:
    # Ensure our cache doesn't fill up the hard drive (Limit: 2 GB)
    enforce_cache_limit(max_size_mb=2000)

    if source.startswith("http://") or source.startswith("https://"):
        print("Detected Youtube URL. Downloading audio...")
        path = download_youtube_audio(source)

        print("Chunking audio instantly via FFmpeg...")
        chunks = chunk_audio(path, chunk_minutes=chunk_minutes)
        print(f"Audio ready - {len(chunks)} chunk(s) created.")
        return chunks

    # If it's a local file upload (e.g. from streamlit)
    print("Chunking local file instantly via FFmpeg...")
    chunks = chunk_audio(source, chunk_minutes=chunk_minutes)
    return chunks


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Download and chunk a YouTube URL via FFmpeg")
    parser.add_argument("url", nargs="?", help="YouTube URL to download")
    args = parser.parse_args()

    if args.url:
        print("Downloading...")
        path = download_youtube_audio(args.url)
        print("Chunking...")
        chunks = chunk_audio(path)
        print("Done. Chunks created:")
        for c in chunks:
            print(" -", c)
