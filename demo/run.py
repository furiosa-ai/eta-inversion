import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import gradio as gr

from demo.demo_utils import Demo


def main():
    demo = Demo()

    demo.launch()


if __name__ == "__main__":
    main()
