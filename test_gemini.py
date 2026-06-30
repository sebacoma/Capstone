#!/usr/bin/env python3
"""
Test rápido de conexión a Gemini 3.5 Flash.
Hace UNA llamada simple y muestra la respuesta.

Uso:
    export GEMINI_API_KEY='...'
    python test_gemini.py
"""
import os
import sys
from google import genai
from google.genai import types


def main():
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("ERROR: Exporta GEMINI_API_KEY primero")
        print("  export GEMINI_API_KEY='...'")
        sys.exit(1)

    print("Conectando a Gemini 3.5 Flash...")
    client = genai.Client(api_key=api_key)

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents="Say hi in one short sentence",
        config=types.GenerateContentConfig(
            temperature=0.0,
            max_output_tokens=64,
        ),
    )

    print(f"Respuesta: {response.text}")
    print("\nTest exitoso. Proceder con rejudge_with_gemini.py")


if __name__ == "__main__":
    main()
