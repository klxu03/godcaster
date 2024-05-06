from pathlib import Path

def get_config():
    return {
        "batch_size": 8,
        "num_epochs": 1,
        "lr": 1e-4,
        "seq_len": 420,
        "img_size": 400,
        "img_width": 400,
        "d_model": 512,
        "heads": 8,
        "datasource": "animelover/scenery-images",
        "subset": "1-full",
        "lang": "en",
        "model_folder": "weights",
        "model_basename": "model_",
        "preload": None,
        "tokenizer_file": "tokenizer_{0}.json",
    }

def get_weights_file_path(config, epoch: str):
    model_folder = config['model_folder']
    model_basename = config["model_basename"]
    model_filename = f"{model_basename}{epoch}.pt"
    return str(Path(".") / model_folder / model_filename)