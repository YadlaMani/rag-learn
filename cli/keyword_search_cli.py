#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cli.keyword_search_lib.preprocessor import Preprocessor
from cli.keyword_search_lib.inverted_index import InvertedIndex

import math
import cli.keyword_search_lib.constants as constants


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Search movies using keyword-based retrieval (TF-IDF and BM25)"
    )
    subparser = parser.add_subparsers(dest="command", help="Available commands")
    search_parser = subparser.add_parser(
        "search", help="Search movies by keyword using the inverted index"
    )
    search_parser.add_argument("query", type=str, help="Search query")
    subparser.add_parser(
        "build", help="Build and cache the inverted index from movies data"
    )
    tf_parser = subparser.add_parser(
        "tf", help="Get the raw term frequency of a token in a document"
    )
    tf_parser.add_argument("doc_id", type=int, help="Document ID (1-indexed)")
    tf_parser.add_argument("token", type=str, help="Token to look up")
    idf_parser = subparser.add_parser(
        "idf", help="Get the inverse document frequency (IDF) of a term"
    )
    idf_parser.add_argument("term", type=str, help="Term to compute IDF for")
    tfidf_parser = subparser.add_parser(
        "tfidf", help="Compute the TF-IDF score of a term in a document"
    )
    tfidf_parser.add_argument("doc_id", type=int, help="Document ID (1-indexed)")
    tfidf_parser.add_argument("term", type=str, help="Term to compute TF-IDF for")

    bm25_idf_parser = subparser.add_parser(
        "bm25idf", help="Get the BM25-adjusted IDF score for a term"
    )
    bm25_idf_parser.add_argument("term", type=str, help="Term to compute BM25 IDF for")

    bm25_tf_parser = subparser.add_parser(
        "bm25tf", help="Get the BM25-adjusted TF score for a term in a document"
    )
    bm25_tf_parser.add_argument("doc_id", type=int, help="Document ID (1-indexed)")
    bm25_tf_parser.add_argument("term", type=str, help="Term to compute BM25 TF for")
    bm25_tf_parser.add_argument(
        "k1",
        type=float,
        nargs="?",
        default=constants.BM25_K1,
        help="BM25 k1 parameter, controls term frequency saturation (default: 1.5)",
    )
    bm25_tf_parser.add_argument(
        "b",
        type=float,
        nargs="?",
        default=constants.BM25_B,
        help="BM25 b parameter, controls document length normalization (default: 0.75)",
    )

    bm25search_parser = subparser.add_parser(
        "bm25search", help="Search movies using full BM25 scoring"
    )
    bm25search_parser.add_argument("query", type=str, help="Search query")
    bm25search_parser.add_argument(
        "limit",
        type=int,
        nargs="?",
        default=5,
        help="Number of results to return (default: 5)",
    )

    args = parser.parse_args()

    inverted_index = InvertedIndex()
    processor = Preprocessor()

    match args.command:
        case "search":
            try:
                inverted_index.load()
                query_tokens = processor.process(args.query)
                search_results = {}
                for token in query_tokens:
                    docs = inverted_index.get_documents(token)
                    for doc_id in docs:
                        search_results[doc_id] = inverted_index.docmap[doc_id]["title"]
                        if len(search_results) == 5:
                            break
                    if len(search_results) == 5:
                        break
                for result_id, result_title in search_results.items():
                    print(f"{result_id + 1} {result_title}")

            except FileNotFoundError:
                print("The Inverted Index is not build yet")
                exit(0)

        case "build":
            inverted_index.build()
            inverted_index.save()
        case "tf":
            inverted_index.load()
            doc_id = args.doc_id - 1
            search_token = args.token
            processed_token = processor.process(search_token)
            if len(processed_token) > 1:
                raise Exception("There should be only one search token")
            print(inverted_index.get_tf(doc_id, processed_token[0]))
        case "idf":
            inverted_index.load()
            search_term = args.term
            processed_term = processor.process(search_term)
            idf = math.log(
                len(inverted_index.docmap)
                / len(inverted_index.get_documents(processed_term[0]))
            )

            print(f"Inverse document frequency of '{args.term}': {idf:.3f}")

        case "tfidf":
            inverted_index.load()
            doc_id = args.doc_id - 1
            search_term = args.term
            processed_term = processor.process(search_term)
            if len(processed_term) > 1:
                raise Exception("There should be only one search term")
            tf = inverted_index.get_tf(doc_id, processed_term[0])
            idf = math.log(
                len(inverted_index.docmap)
                / len(inverted_index.get_documents(processed_term[0]))
            )
            tf_idf = tf * idf
            print(
                f"TF-IDF score of '{args.term}' in document '{args.doc_id}': {tf_idf:.3f}"
            )
        case "bm25idf":
            inverted_index.load()
            search_term = args.term
            processed_term = processor.process(search_term)
            bm25idf = inverted_index.get_bm25_idf(processed_term[0])
            print(f"BM25 IDF score of '{args.term}': {bm25idf:.3f}")
        case "bm25tf":
            inverted_index.load()
            doc_id = args.doc_id - 1
            search_term = args.term
            processed_term = processor.process(search_term)
            if len(processed_term) > 1:
                raise Exception("There should be only one search term")
            k1 = args.k1
            b = args.b
            bm25tf = inverted_index.get_bm25_tf(doc_id, processed_term[0], k1, b)
            print(
                f"BM25 TF score of '{args.term}' in document '{args.doc_id}': {bm25tf:.3f}"
            )

        case "bm25search":
            inverted_index.load()
            query = args.query
            limit = args.limit
            docs = inverted_index.bm25_search(query, limit)
            for i, doc_id in enumerate(docs):
                title = inverted_index.docmap[doc_id]["title"]
                print(f"{i + 1}. ({doc_id + 1}) {title} - Score: {docs[doc_id]:.3f}")

        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
