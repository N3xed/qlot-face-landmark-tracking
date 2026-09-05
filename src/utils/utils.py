import os
import subprocess
import urllib.request
import urllib.parse
from pathlib import Path
from typing import Optional, Iterator, Any, Callable
try:
    from tqdm import tqdm
except ImportError:
    tqdm = None
import hashlib

cache_dir: str = ""

def set_cache_dir(dir: str):
    """
    Set the cache directory to a specific path.

    Args:
        dir (str): The path to set as the cache directory.
    """
    global cache_dir
    cache_dir = dir

def determine_cache_dir() -> str:
    """
    Find the an appropriate cache dir.

    1. If we're in a git repo, use: `<repo_root>/.cache`.
    2. Otherwise, use: `<os default cache dir>/pa-face-landmark-tracking`.

    Returns:
        str: The path to the cache directory.
    """
    global cache_dir
    if cache_dir != "":
        return cache_dir

    # Try to find git repo root
    try:
        # Set LC_ALL=C to ensure language-independent output
        env = os.environ.copy()
        env["LC_ALL"] = "C"

        result = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            capture_output=True,
            text=True,
            check=True,
            cwd=os.path.dirname(__file__),
            env=env
        )
        repo_root = result.stdout.strip()
        cache_dir = os.path.join(repo_root, ".cache")
    except (subprocess.CalledProcessError, FileNotFoundError):
        # Not in a git repo or git not available, use OS default cache dir
        if os.name == "nt":  # Windows
            base_cache = os.environ.get(
                "LOCALAPPDATA", os.path.expanduser("~/AppData/Local"))
        elif os.name == "posix":  # Unix-like (Linux, macOS)
            base_cache = os.environ.get(
                "XDG_CACHE_HOME", os.path.expanduser("~/.cache"))
        else:
            # Fallback
            base_cache = os.path.expanduser("~/.cache")
        cache_dir = os.path.join(base_cache, "pa-face-landmark-tracking")

    # Ensure the cache directory exists
    os.makedirs(cache_dir, exist_ok=True)
    return cache_dir


def get_cached_path(name: str) -> str:
    """
    Get the path to a cached file by name.

    Args:
        name (str): The name of the cached file.

    Returns:
        str: The full path to the cached file.
    """
    cache_dir = determine_cache_dir()
    return os.path.join(cache_dir, name)


def download_and_cache(url: str, name: Optional[str] = None, cache_dir: Optional[str] = None) -> str:
    """
    Download a file from a URL and cache it locally.

    Args:
        url (str): The URL to download the file from.
        name (str, optional): The name to save the file as. If None, use the last part of the URL.
        cache_dir (str, optional): The directory to cache the file in. Defaults to the result of `determine_cache_dir()`.

    Returns:
        str: The path to the cached file.
    """
    if cache_dir is None:
        cache_dir = determine_cache_dir()

    # Determine the filename
    if name is None:
        parsed_url = urllib.parse.urlparse(url)
        name = os.path.basename(parsed_url.path)
        if not name:
            # If no filename in URL, create name from hostname and hash
            import hashlib
            import re
            hostname = parsed_url.hostname or "unknown"
            # Sanitize hostname to contain only valid filesystem characters
            hostname = re.sub(r'[^\w\-_.]', '_', hostname)
            url_hash = hashlib.md5(url.encode()).hexdigest()[:10]
            name = f"{hostname}_{url_hash}"

    file_path = os.path.join(cache_dir, name)

    # Check if file already exists
    if os.path.exists(file_path):
        return file_path

    print(f"Downloading {url} to {file_path}...")

    # Download with progress bar if tqdm is available
    if tqdm is not None:
        def _download_with_progress():
            with urllib.request.urlopen(url) as response:
                total_size = int(response.headers.get('Content-Length', 0))

                with open(file_path, 'wb') as f:
                    if total_size > 0:
                        with tqdm(total=total_size, unit='B', unit_scale=True, desc=name) as pbar: # type: ignore
                            while True:
                                chunk = response.read(8192)
                                if not chunk:
                                    break
                                f.write(chunk)
                                pbar.update(len(chunk))
                    else:
                        # No content-length header, download without progress bar
                        with tqdm(unit='B', unit_scale=True, desc=name) as pbar: # type: ignore
                            while True:
                                chunk = response.read(8192)
                                if not chunk:
                                    break
                                f.write(chunk)
                                pbar.update(len(chunk))

        _download_with_progress()
    else:
        # Simple download without progress bar
        urllib.request.urlretrieve(url, file_path)

    return file_path


def list_files_in_dir(dir: str | Path, pattern: str | list[str] = "*") -> list[str]:
    """
    List all files in a directory matching a given pattern or patterns.

    Args:
        dir (str): The directory to search in.
        pattern (str | list[str]): The glob pattern(s) to match files. Can be a single pattern string or a list of patterns. Defaults to "*".

    Returns:
        list[str]: A list of file paths matching the pattern(s).
    """
    p = Path(dir)

    # Handle single pattern
    if isinstance(pattern, str):
        return [str(f) for f in p.glob(pattern) if f.is_file()]

    # Handle multiple patterns
    files = []
    for pat in pattern:
        files.extend([str(f) for f in p.glob(pat) if f.is_file()])

    # Remove duplicates while preserving order
    seen = set()
    unique_files = []
    for f in files:
        if f not in seen:
            seen.add(f)
            unique_files.append(f)

    return unique_files


def get_file_digest(file_path: str, use_contents=False) -> str:
    """
    Compute a SHA1 digest for a file based on its path, file time, and optionally its contents.

    Args:
        file_path (str): The path to the file.
        use_contents (bool): If True, include the file contents in the digest. Defaults to False.

    Returns:
        str: The SHA1 digest as a hexadecimal string.
    """
    hash_obj = hashlib.sha1()
    hash_obj.update(file_path.encode('utf-8'))
    try:
        mtime = os.path.getmtime(file_path)
        hash_obj.update(str(mtime).encode('utf-8'))
    except:
        pass
    if use_contents:
        with open(file_path, 'rb') as f:
            while True:
                chunk = f.read(8192)
                if not chunk:
                    break
                hash_obj.update(chunk)
    return hash_obj.hexdigest()

def png_encode(image, cvt_color: None | int = None) -> bytes:
    """
    Encode an BGR image (OpenCV compatible) to PNG format.
    
    Args:
        image: The input image in BGR format (numpy array).
        cvt_color (int, optional): OpenCV color conversion code to apply before encoding.

    Returns:
        bytes: The PNG-encoded image data.
    """
    import cv2
    if cvt_color is not None:
        image = cv2.cvtColor(image, cvt_color)
    is_success, encoded_image = cv2.imencode(".png", image)
    if not is_success:
        raise ValueError("Could not encode image to PNG format")
    return encoded_image.tobytes()


def format_dict(name: str, d: dict, inst: Optional[Any] = None) -> str:
    """
    Format a dictionary into a human-readable multi-line string.
    Args:
        name (str): The name/title to display at the top.
        d (dict): The dictionary to format.
        inst (Any, optional): An optional instance whose properties should be skipped.
    Returns:
        str: The formatted multi-line string.
    """

    import textwrap
    level = 0
    lines = []
    def lpush():
        nonlocal level
        level += 1
    def lpop():
        nonlocal level
        level -= 1
    def fstr(s: str):
        nonlocal level, lines
        lines.append(textwrap.indent(s, "    " * level))
    fstr(f"{name}:")
    lpush()
    for k, v in d.items():
        if k.startswith("_") or (inst is not None and isinstance(getattr(type(inst), k, None), property)):
            continue
        ks = f"{k}: "
        vs = str(v)
        if len(ks) + len(vs) > 96:
            fstr(ks)
            wrapped = textwrap.wrap(vs, width=100 - level * 4)
            if len(wrapped) > 1:
                lpush()
                fstr("\n".join(wrapped))
                lpop()
            else:
                fstr(vs)
        else:
            fstr(ks + vs)
    return "\n".join(lines)