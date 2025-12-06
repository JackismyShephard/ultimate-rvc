import os
import pathlib

import torch


def change_info(path, info, name):
    try:
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        ckpt["info"] = info

        if not name:
            name = os.path.splitext(os.path.basename(path))[0]

        target_dir = os.path.join("logs", name)
        pathlib.Path(target_dir).mkdir(exist_ok=True, parents=True)

        torch.save(ckpt, os.path.join(target_dir, f"{name}.pth"))

        return "Success."

    except Exception as error:
        print(f"An error occurred while changing the info: {error}")
        return f"Error: {error}"
