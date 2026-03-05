"""CLI entry point for crane-gym."""

import argparse


def main():
    parser = argparse.ArgumentParser(description="crane-gym: hardware scalping toolkit")
    parser.add_argument("command", choices=["scrape", "analyze", "alert"], help="Command to run")
    parser.add_argument("--query", type=str, help="Search query (e.g. 'RTX 4090')")
    parser.add_argument("--category", type=str, default="gpu", help="Item category")
    args = parser.parse_args()
    print(f"crane-gym: {args.command} — not yet implemented")


if __name__ == "__main__":
    main()
