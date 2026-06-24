import argparse
import sys
from pathlib import Path
import mimetypes
from google.genai import types as genai_types

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cli.hybrid_search_lib.utils import *


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Rewrite a text query using visual context from an image"
    )
    subparser = parser.add_subparsers(dest="command", help="Available commands")
    parser.add_argument(
        "--image",
        type=str,
        help="Path to the image file",
    )
    parser.add_argument(
        "--query",
        type=str,
        help="Text query to rewrite using context from the image",
    )

    args = parser.parse_args()
    mime, _ = mimetypes.guess_type(args.image)
    mime = mime or "image/jpeg"
    with open(args.image, "rb") as f:
        img = f.read()
    load_dotenv()
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY environment variable not set")

    client = genai.Client(api_key=api_key)
    contents = [
        """
        Given the included image and text query, rewrite the text query to improve search results from a movie database. Make sure to:
        - Synthesize visual and textual information
        - Focus on movie-specific details (actors, scenes, style, etc.)
        - Return only the rewritten query, without any additional commentary
     """,
        genai_types.Part.from_bytes(data=img, mime_type=mime),
        args.query.strip(),
    ]
    res = client.models.generate_content(model="gemma-4-31b-it", contents=contents)
    print(f"Rewritten query: {res.text.strip()}")
    if res.usage_metadata is not None:
        print(f"Total tokens:    {res.usage_metadata.total_token_count}")


if __name__ == "__main__":
    main()
