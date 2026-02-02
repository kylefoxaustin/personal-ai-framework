#!/usr/bin/env python3
"""Test the LLM server"""

import requests
import json

def test_generation():
    url = "http://localhost:8080/generate"
    
    # Test 1: Simple generation
    payload = {
        "prompt": "Write a brief email response to: 'Can we meet tomorrow at 3pm?'",
        "max_tokens": 100,
        "temperature": 0.7
    }
    
    print("Testing simple generation...")
    response = requests.post(url, json=payload)
    print(f"Response: {response.json()['text']}\n")
    
    # Test 2: With context
    payload = {
        "prompt": "What is the status of the project?",
        "context": [
            "Project Alpha is currently in phase 2",
            "Expected completion date is March 2024",
            "Team size is 5 developers"
        ],
        "max_tokens": 150
    }
    
    print("Testing with RAG context...")
    response = requests.post(url, json=payload)
    print(f"Response: {response.json()['text']}\n")

if __name__ == "__main__":
    test_generation()
