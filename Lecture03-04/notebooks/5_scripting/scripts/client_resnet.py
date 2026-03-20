import argparse
import requests

URL = "https://8081-01kjwjwt4perer9wherq2ga2z6.cloudspaces.litng.ai"

def predict(image_path: str) -> None:
    with open(image_path, "rb") as f:
        response = requests.post(
            f"{URL}/predict",
            files={"file": f},
        )

    response.raise_for_status()
    predictions = response.json()["predictions"]

    print(f"\nImage: {image_path}")
    print("-" * 35)
    for p in predictions:
        bar = "█" * int(p["confidence"] * 30)
        print(f"  {p['class']:<12} {p['confidence']:.2%}  {bar}")


def main() -> None:
    parser = argparse.ArgumentParser(description="CIFAR-10 classifier client")
    parser.add_argument("image", help="Path to image file")
    args = parser.parse_args()
    predict(args.image)


if __name__ == "__main__":
    main()