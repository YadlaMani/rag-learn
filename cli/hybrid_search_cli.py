import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cli.hybrid_search_lib.utils import *


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Search movies by combining keyword and semantic search"
    )
    subparser = parser.add_subparsers(dest="command", help="Available commands")
    normalize_parser = subparser.add_parser(
        "normalize", help="Normalize a list of scores to the 0–1 range"
    )
    normalize_parser.add_argument(
        "scores", type=float, nargs="*", help="Space-separated scores to normalize"
    )

    weighted_search_parser = subparser.add_parser(
        "weighted-search",
        help="Search movies by blending BM25 and semantic scores with a weighted alpha",
    )
    weighted_search_parser.add_argument("query", type=str, help="Search query")
    weighted_search_parser.add_argument(
        "--alpha",
        type=float,
        nargs="?",
        default=0.5,
        help="Balance between keyword (0.0) and semantic (1.0) scoring (default: 0.5)",
    )
    weighted_search_parser.add_argument(
        "--limit",
        type=int,
        nargs="?",
        default=5,
        help="Number of results to return (default: 5)",
    )

    rrf_search_parser = subparser.add_parser(
        "rrf-search",
        help="Search movies using Reciprocal Rank Fusion of keyword and semantic results",
    )
    rrf_search_parser.add_argument("query", type=str, help="Search query")
    rrf_search_parser.add_argument(
        "-k",
        type=int,
        nargs="?",
        default=60,
        help="RRF smoothing constant; higher values reduce rank differences (default: 60)",
    )
    rrf_search_parser.add_argument(
        "--limit",
        type=int,
        nargs="?",
        default=5,
        help="Number of results to return (default: 5)",
    )
    rrf_search_parser.add_argument(
        "--enhance",
        type=str,
        choices=["spell", "rewrite", "expand"],
        help="Enhance query before search: 'spell' fixes typos, 'rewrite' rephrases, 'expand' adds synonyms",
    )

    rrf_search_parser.add_argument(
        "--rerank-method",
        type=str,
        choices=["individual", "batch", "cross_encoder"],
        help="Re-rank results after retrieval: 'individual' (LLM per doc), 'batch' (LLM batch), 'cross_encoder'",
    )

    rrf_search_parser.add_argument(
        "--evaluate",
        action="store_true",
        help="Score each result for relevance using an LLM (0–3 scale)",
    )

    args = parser.parse_args()
    match args.command:
        case "normalize":
            normalize(args.scores)
        case "weighted-search":
            weighted_search(args.query, args.alpha, args.limit)
        case "rrf-search":
            results = rrf_search(
                enhance_text(args.query, args.enhance),
                args.k,
                args.limit,
                args.rerank_method,
            )
            if args.evaluate:
                evaluate_results(args.query, results)

        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
