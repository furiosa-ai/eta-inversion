import argparse
from typing import Dict


def add_argument_choice_list(parser: argparse.ArgumentParser, name: str, title: str, choice_dict: Dict[str, str]) -> None:
    indent1 = 2
    indent2 = 25

    cont = f"\n".join(["".join([" " * indent1, k, " " * (indent2 - len(k)), v]) for k, v in choice_dict.items()])
    help = f"{title}\n{cont}"

    parser.add_argument("--" + name, metavar=name.upper(), choices=choice_dict.keys(), help=help)


def add_argparse_arg(parser: argparse.ArgumentParser, name: str) -> None:
    """Add predefined options to argument parser

    Args:
        name (str): Name of the argument to add
    """

    if name.startswith("--"):
        name = name[2:]

    helps = {
        "method": {
            "title": "Available inversion methods:",
            "choices": {
                "diffinv": "Naiv DDIM inversion",
                "nti": "Null text inversion",
                "npi": "Negative prompt inversion",
                "proxnpi": "Proximal negative prompt inversion",
                "edict": "EDICT inversion",
                "ddpminv": "DDPM inversion",
        }},
        "edit_method": {
            "title": "Available editing methods:",
            "choices": {
                "simple": "Simple denoising of inverted latent with target prompt",
                "ptp": "Prompt-to-prompt",
                "masactrl": "MasaControl",
                "pnp": "Plug-and-play",
        }},
    }

    helps["inv_method"] = helps["method"]

    add_argument_choice_list(parser, name, helps[name]["title"], helps[name]["choices"])
