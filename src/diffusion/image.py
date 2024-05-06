from datasets import load_dataset
from config import get_weights_file_path, get_config

def main():
    config = get_config()
    dataset = load_dataset(config["datasource"], config["subset"], split="train")

    print(dataset[0]["image"].format)

if __name__ == "__main__":
    main()