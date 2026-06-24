import argparse
import sys
from pathlib import Path
import os
from dotenv import load_dotenv
from google import genai

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from cli.hybrid_search_lib.utils import *


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Answer movie questions by retrieving relevant context (RAG)"
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    rag_parser = subparsers.add_parser(
        "rag", help="Find relevant movies and generate a natural-language answer"
    )
    rag_parser.add_argument("query", type=str, help="Search query")
    rag_parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="Number of retrieved movies to include in the prompt (default: 5)",
    )

    summarize_parser = subparsers.add_parser(
        "summarize",
        help="Find relevant movies and generate a combined summary",
    )
    summarize_parser.add_argument("query", type=str, help="Search query")
    summarize_parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="Number of retrieved movies to include in the prompt (default: 5)",
    )

    citations_parser = subparsers.add_parser(
        "citations",
        help="Find relevant movies and generate a summary with source citations",
    )
    citations_parser.add_argument("query", type=str, help="Search query")
    citations_parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="Number of retrieved movies to include in the prompt (default: 5)",
    )

    question_parser = subparsers.add_parser(
        "question", help="Answer a question conversationally using retrieved movie data"
    )
    question_parser.add_argument("query", type=str, help="Search query")
    question_parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="Number of retrieved movies to include in the prompt (default: 5)",
    )

    args = parser.parse_args()

    load_dotenv()
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY environment variable not set")

    client = genai.Client(api_key=api_key)
    match args.command:
        case "rag":
            query = args.query
            results = rrf_search(
                enhance_text(args.query, enhance="expand"), limit=5, verbose=False
            )

            contents = f"""You are a RAG agent for Hoopla, a movie streaming service.
                Your task is to provide a natural-language answer to the user's query based on documents retrieved during search.
                Provide a comprehensive answer that addresses the user's query.

                Query: {query}

                Documents:
                {chr(10).join(f"{i + 1}. {res['title']} - {res['description'][:200]}" for i, (_, res) in enumerate(results))}

                Answer:"""
            resp = client.models.generate_content(
                model="gemma-4-31b-it", contents=contents
            )
            text = resp.text
            if not text:
                print("No response from model")
                return
            print("Search Results:")
            for _, res in results:
                print(f"- {res['title']}")
            print(f"RAG Response:\n{text}")
        case "summarize":
            query = args.query
            results = rrf_search(
                enhance_text(args.query, enhance="expand"), limit=5, verbose=False
            )

            contents = f"""Provide information useful to the query below by synthesizing data from multiple search results in detail.

                    The goal is to provide comprehensive information so that users know what their options are.
                    Your response should be information-dense and concise, with several key pieces of information about the genre, plot, etc. of each movie.

                    This should be tailored to Hoopla users. Hoopla is a movie streaming service.

                    Query: {query}

                Search results:
                {chr(10).join(f"{i + 1}. {res['title']} - {res['description'][:200]}" for i, (_, res) in enumerate(results))}

                Provide a comprehensive 3–4 sentence answer that combines information from multiple sources:"""
            resp = client.models.generate_content(
                model="gemma-4-31b-it", contents=contents
            )
            text = resp.text
            if not text:
                print("No response from model")
                return
            print("Search Results:")
            for _, res in results:
                print(f"- {res['title']}")
            print(f"LLM Summary:\n{text}")
        case "citations":
            query = args.query
            results = rrf_search(
                enhance_text(args.query, enhance="rewrite"),
                limit=args.limit,
                verbose=False,
            )

            contents = f"""Provide information useful to the query below by synthesizing data from multiple search results in detail.

                    The goal is to provide comprehensive information so that users know what their options are.
                    Your response should be information-dense and concise, with several key pieces of information about the genre, plot, etc. of each movie.

                    This should be tailored to Hoopla users. Hoopla is a movie streaming service.

                    Query: {query}

                    Search results:
                   {chr(10).join(f"{i + 1}. {res['title']} - {res['description'][:800]}" for i, (_, res) in enumerate(results))}

                    Provide a comprehensive 3–4 sentence answer that combines information from multiple sources:"""
            resp = client.models.generate_content(
                model="gemma-4-31b-it", contents=contents
            )
            text = resp.text
            if not text:
                print("No response from model")
                return
            print("Search Results:")
            for _, res in results:
                print(f"- {res['title']}")
            print(f"LLM Summary:\n{text}")
        case "question":
            query = args.query
            results = rrf_search(
                enhance_text(args.query, enhance="spell"),
                limit=args.limit,
                verbose=False,
            )

            contents = f"""Answer the user's question based on the provided movies that are available on Hoopla, a streaming service.

            Question: {query}

            Documents:
            {chr(10).join(f"{i + 1}. {res['title']} - {res['description'][:800]}" for i, (_, res) in enumerate(results))}

            Instructions:
            - Answer questions directly and concisely
            - Be casual and conversational
            - Don't be cringe or hype-y
            - Talk like a normal person would in a chat conversation

            Answer:"""
            resp = client.models.generate_content(
                model="gemma-4-31b-it", contents=contents
            )
            text = resp.text
            if not text:
                print("No response from model")
                return
            print("Search Results:")
            for _, res in results:
                print(f"- {res['title']}")
            print(f"Answer:\n{text}")
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
