#!/usr/bin/env python3
"""Create a demo JSONL dataset with synthetic movie Q&A pairs.

Creates a JSONL file with ~20 synthetic movie Q&A pairs for testing the KG-RAG pipeline.
Each line is a JSON object with 'question' and 'answer' fields.

Example::

    python create_demo_dataset.py --output /tmp/demo.jsonl
"""

import argparse
from pathlib import Path

from mellea_contribs.kg.utils import save_jsonl


# Sample movie Q&A pairs for demo dataset
DEMO_QA_PAIRS = [
    {
        "question": "Who directed Avatar?",
        "answer": "James Cameron",
        "domain": "movie"
    },
    {
        "question": "What year was Titanic released?",
        "answer": "1997",
        "domain": "movie"
    },
    {
        "question": "Who are the main actors in The Dark Knight?",
        "answer": "Christian Bale, Heath Ledger, Michael Caine",
        "domain": "movie"
    },
    {
        "question": "What studio produced Inception?",
        "answer": "Warner Bros",
        "domain": "movie"
    },
    {
        "question": "In which year was Parasite released?",
        "answer": "2019",
        "domain": "movie"
    },
    {
        "question": "Who directed The Matrix?",
        "answer": "The Wachowskis",
        "domain": "movie"
    },
    {
        "question": "What is the director of Pulp Fiction?",
        "answer": "Quentin Tarantino",
        "domain": "movie"
    },
    {
        "question": "Which actor starred in both Forrest Gump and Cast Away?",
        "answer": "Tom Hanks",
        "domain": "movie"
    },
    {
        "question": "What year did Avatar: The Way of Water release?",
        "answer": "2022",
        "domain": "movie"
    },
    {
        "question": "Who directed Oppenheimer?",
        "answer": "Christopher Nolan",
        "domain": "movie"
    },
    {
        "question": "What is the budget of Avatar?",
        "answer": "$237 million",
        "domain": "movie"
    },
    {
        "question": "Who plays Spider-Man in the recent MCU films?",
        "answer": "Tom Holland",
        "domain": "movie"
    },
    {
        "question": "Which director made Interstellar?",
        "answer": "Christopher Nolan",
        "domain": "movie"
    },
    {
        "question": "What is the box office revenue of Avengers: Endgame?",
        "answer": "$2.798 billion",
        "domain": "movie"
    },
    {
        "question": "Who directed Dune?",
        "answer": "Denis Villeneuve",
        "domain": "movie"
    },
    {
        "question": "What award did Parasite win?",
        "answer": "Academy Award for Best Picture",
        "domain": "movie"
    },
    {
        "question": "Who composed the music for The Lord of the Rings?",
        "answer": "Howard Shore",
        "domain": "movie"
    },
    {
        "question": "What studio released The Lion King?",
        "answer": "Disney",
        "domain": "movie"
    },
    {
        "question": "Who is the director of Barbie?",
        "answer": "Greta Gerwig",
        "domain": "movie"
    },
    {
        "question": "What year was Jurassic Park released?",
        "answer": "1993",
        "domain": "movie"
    },
]


def main():
    """Create demo dataset and write to JSONL file."""
    parser = argparse.ArgumentParser(
        description="Create a demo JSONL dataset with synthetic movie Q&A pairs"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/demo_dataset.jsonl",
        help="Output file path (default: data/demo_dataset.jsonl)",
    )
    args = parser.parse_args()

    # Write demo Q&A pairs to JSONL using utility
    output_path = Path(args.output)
    save_jsonl(DEMO_QA_PAIRS, output_path)

    print(f"Created demo dataset with {len(DEMO_QA_PAIRS)} Q&A pairs")
    print(f"Saved to: {output_path}")


if __name__ == "__main__":
    main()
