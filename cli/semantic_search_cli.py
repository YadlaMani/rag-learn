#!/usr/bin/env python3

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cli.semantic_search_lib.utils import *
from cli.semantic_search_lib.chunking import *


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Search movies using semantic embeddings"
    )
    subparser = parser.add_subparsers(dest="command", help="Available commands")
    subparser.add_parser(
        "verify", help="Show the loaded model name and its max sequence length"
    )
    subparser.add_parser(
        "verify_embeddings",
        help="Build movie embeddings and verify the index",
    )
    embed_text_parser = subparser.add_parser(
        "embed_text", help="Encode a text string and print its embedding"
    )
    embed_text_parser.add_argument("text", type=str, help="Text to encode")

    embed_query_parser = subparser.add_parser(
        "embed_query", help="Encode a search query and print its embedding"
    )
    embed_query_parser.add_argument("query", type=str, help="Query to encode")

    search_parser = subparser.add_parser(
        "search", help="Search movies using semantic similarity"
    )
    search_parser.add_argument("query", type=str, help="Search query")
    search_parser.add_argument(
        "--limit", type=int, default=5, help="Number of results to return (default: 5)"
    )

    chunk_parser = subparser.add_parser(
        "chunk", help="Split text into fixed-size word chunks"
    )
    chunk_parser.add_argument("text", type=str, help="Text to split into chunks")
    chunk_parser.add_argument(
        "--chunk-size",
        type=int,
        default=200,
        help="Number of words per chunk (default: 200)",
    )
    chunk_parser.add_argument(
        "--overlap",
        type=int,
        default=2,
        help="Number of words to overlap between chunks (default: 2)",
    )

    semantic_chunk_parser = subparser.add_parser(
        "semantic_chunk", help="Split text into sentence-based semantic chunks"
    )
    semantic_chunk_parser.add_argument(
        "text", type=str, help="Text to split into chunks"
    )
    semantic_chunk_parser.add_argument(
        "--max-chunk-size",
        type=int,
        default=4,
        help="Max sentences per chunk (default: 4)",
    )
    semantic_chunk_parser.add_argument(
        "--overlap",
        type=int,
        default=0,
        help="Number of sentences to overlap between chunks (default: 0)",
    )

    subparser.add_parser(
        "embed_chunks", help="Build and cache chunk embeddings for all movies"
    )

    search_chunked__parser = subparser.add_parser(
        "search_chunked", help="Search movies using chunked semantic search"
    )
    search_chunked__parser.add_argument("query", type=str, help="Search query")
    search_chunked__parser.add_argument(
        "--limit", type=int, default=5, help="Number of results to return (default: 5)"
    )

    args = parser.parse_args()

    match args.command:
        case "verify":
            verify_model()
        case "embed_text":
            embed_text(args.text)
        case "verify_embeddings":
            verify_embeddings()
        case "embed_query":
            embed_query_text(args.query)
        case "search":
            search_movies(args.query, args.limit)
        case "chunk":
            print(f"Chunking {len(args.text)} characters")
            chunks = chunk_text(args.text, args.chunk_size, args.overlap)
            for i, chunk in enumerate(chunks):
                print(f"{i + 1}. {chunk}")

        case "semantic_chunk":
            print(f"Semantically chunking {len(args.text)} characters")
            semantic_chunks = semantic_chunk(
                args.text, args.max_chunk_size, args.overlap
            )
            for i, chunk in enumerate(semantic_chunks):
                print(f"{i + 1}. {chunk}")

        case "embed_chunks":
            embed_chunks()

        case "search_chunked":
            search_result = search_chunked(args.query, args.limit)
            for i, res in enumerate(search_result):
                print(f"\n{i + 1}. {res['title']} (score: {res['score']:.3f})")
                print(f"\t{res['document']}...")

        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
