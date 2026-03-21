#!/usr/bin/env python
"""Cliente para el servidor de análisis de sentimiento IMDB.

Uso:
    python client.py "This movie was great!"
    python client.py "Terrible film." --url https://<id>-8081.cloudspaces.litng.ai
    python client.py --batch reviews.txt
"""
import argparse
import requests


def predict_one(url: str, text: str) -> None:
    resp = requests.post(f"{url}/predict", json={"text": text})
    resp.raise_for_status()
    result = resp.json()
    print(f"\nReseña: {result['text']}\n")
    for p in result["predictions"]:
        bar  = "█" * int(p["score"] * 40)
        icon = "✅" if p["label"] == "positive" else "❌"
        print(f"  {icon} {p['label']:<12} {p['score']:.2%}  {bar}")


def predict_batch(url: str, filepath: str) -> None:
    with open(filepath) as f:
        texts = [line.strip() for line in f if line.strip()]
    resp = requests.post(f"{url}/predict/batch", json={"texts": texts})
    resp.raise_for_status()
    print(f"{'Reseña':<60}  {'Label':<12}  {'Score'}")
    print("-" * 82)
    for item in resp.json()["results"]:
        top     = item["predictions"][0]
        preview = (item["text"][:57] + "...") if len(item["text"]) > 60 else item["text"].ljust(60)
        icon    = "✅" if top["label"] == "positive" else "❌"
        print(f"{preview}  {icon} {top['label']:<10}  {top['score']:.4f}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("text",    nargs="?",     help="Reseña a clasificar")
    parser.add_argument("--batch", metavar="FILE", help="Archivo con una reseña por línea")
    parser.add_argument("--url",   default="https://8081-01kjwjwt4perer9wherq2ga2z6.cloudspaces.litng.ai/")
    args = parser.parse_args()

    if args.batch:
        predict_batch(args.url, args.batch)
    elif args.text:
        predict_one(args.url, args.text)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
