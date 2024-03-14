import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import yaml
from tqdm import tqdm
import shutil


def main():
    file = "data/eval/plug_and_play/imagenetr-ti2i/imnetr-ti2i.yaml"
    output_dir = Path("data/eval/plug_and_play/imagenetr-ti2i")
    img_output_dir = output_dir / "imgs"

    img_output_dir.mkdir(parents=True, exist_ok=True)

    with open(file, "r") as f:
        data = yaml.safe_load(f)

    idx = 0
    out_data = []

    # most source prompts missing in dataset
    source_prompts_my = [
        'a sketch of a penguin',
        'an art of a penguin',
        'a painting of a penguin',
        'a sketch of a husky',
        'an art of a husky',
        'a toy of a husky',
        'a cartoon of a goldfish',
        'an origami of a goldfish',
        'a painting of a goldfish',
        'a sketch of a cat',
        'a sculpture of a cat',
        'a cartoon of a cat',
        'a sculpture of a jeep',
        'a painting of a jeep',
        'a toy of a jeep',
        'a cartoon of a castle',
        'a sculpture of a castle',
        'an embroidery of a castle',
        'a sculpture of a pizza',
        'a toy of a pizza',
        'a sketch of a pizza',
        'a painting of a violin',
        'a painting of a violin',
        'an origami of a violin',
        'a cartoon of a panda',
        'a sculpture of a panda',
        'a sketch of a panda',
        'an embroidery of a hummingbird',
        'a cartoon of a hummingbird',
        'an origami of a hummingbird'
    ]

    init_imgs_my = {"a cartoon of a panda": "/ImageNetR-TI2I/panda/cartoon_30.jpg"}

    for source_prompt_idx, sample in enumerate(tqdm(data)):
        source_prompt = sample.get("source_prompt", None)

        if source_prompt is None:
            assert len(source_prompts_my) == len(data), "Dataset length not matching"
            source_prompt = source_prompts_my[source_prompt_idx]

        init_img = sample.get("init_img", None)

        if init_img is None:
            init_img = init_imgs_my[source_prompt]

        img_file_src = "data/eval/plug_and_play" + init_img.lower()
        img_file_dst = img_output_dir / (source_prompt + ".png")
        shutil.copy(img_file_src, img_file_dst)

        for target_prompt in sample["target_prompts"][:3]:
            source_prompt_words = source_prompt.split(" ")
            target_prompt_words = target_prompt.split(" ")

            assert len(source_prompt_words) == len(target_prompt_words)

            words_diff = [[s, t] for s, t in zip(source_prompt_words, target_prompt_words) if s != t]
            words_diff = [[s, t] for s, t in words_diff if sorted([s, t]) not in (["a", "an"], )]

            assert len(words_diff) == 1

            out_data.append({
                "idx": idx,
                "source_prompt_idx": source_prompt_idx,
                "source_prompt": source_prompt,
                "target_prompt": target_prompt,
                "edit": [
                    ["replace", words_diff[0]]
                ],
            })


            idx += 1

    with open(output_dir / "prompts.yaml", "w") as f:
        yaml.dump(out_data, f)


if __name__ == "__main__":
    main()
