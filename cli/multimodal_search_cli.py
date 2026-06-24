import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cli.multimodal_search_lib.multimodal_search import MultiModelSearch


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Search movies by image using CLIP embeddings"
    )
    subparser = parser.add_subparsers(dest="command", help="Available commands")

    verify_image_parser = subparser.add_parser(
        "verify_image_embedding",
        help="Encode an image and print the embedding dimension",
    )
    verify_image_parser.add_argument(
        "imagePath", type=str, help="Path to the image file"
    )

    image_search_parser = subparser.add_parser(
        "image_search", help="Find movies visually similar to an image"
    )
    image_search_parser.add_argument(
        "imagePath", type=str, help="Path to the image file"
    )
    image_search_parser.add_argument(
        "--limit",
        type=int,
        nargs="?",
        default=5,
        help="Number of results to return (default: 5)",
    )

    args = parser.parse_args()
    match args.command:
        case "verify_image_embedding":
            multimodel_search = MultiModelSearch()
            embedding = multimodel_search.embed_image(args.imagePath)
            print(f"Embedding shape: {embedding.shape[1]} dimensions")

        case "image_search":
            multimodel_search = MultiModelSearch()
            multimodel_search.load_or_build()
            results = multimodel_search.search_with_image(args.imagePath, args.limit)
            for i, res in enumerate(results):
                print(
                    f"{i + 1}. {res['title']} (similarity: {res['score']:.3f})\n{res['description']}"
                )

        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
