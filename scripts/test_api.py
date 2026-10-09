"""
Test script for Google Gemini API integration using the google-genai SDK.
This script verifies API key configuration and lists available Gemini models.
"""

import os
import sys
from dotenv import load_dotenv
from google import genai


def main():
    # 1. Load environment variables from .env file
    load_dotenv()

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key or api_key == "your_key_here":
        print(
            "Error: GEMINI_API_KEY is not set or still has the placeholder value in .env.",
            file=sys.stderr,
        )
        print("Please add your actual Gemini API key to .env before running this script.")
        sys.exit(1)

    # 2. Initialize the Gemini API client using the google-genai SDK
    # Notice: The key itself is passed to the client and never printed to stdout/stderr.
    client = genai.Client(api_key=api_key)

    # 3. Retrieve and print the list of available models
    print("Connecting to Gemini API and fetching available models...\n")
    try:
        model_count = 0
        for model in client.models.list():
            # Print each model's resource name
            print(f"- {model.name}")
            model_count += 1

        print(f"\nSuccessfully listed {model_count} models.")
    except Exception as e:
        print(f"Error fetching models from Gemini API: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

