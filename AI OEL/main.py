"""
main.py – Entry point for the Agriculture Decision Support System
          Runs preprocessing → model training → (optional) GUI

Usage
-----
    python main.py            # full pipeline (no GUI)
    python main.py --gui      # full pipeline then launch GUI
    python main.py --gui-only # launch GUI (models must already exist)
"""

import sys
import os
import argparse
import subprocess

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR  = os.path.join(BASE_DIR, "src")


def run_script(name: str):
    """Run a src/ script and raise on failure."""
    path = os.path.join(SRC_DIR, name)
    print(f"\n{'='*55}")
    print(f"  Running  :  {name}")
    print(f"{'='*55}\n")
    result = subprocess.run([sys.executable, path], check=True)
    return result.returncode


def main():
    parser = argparse.ArgumentParser(
        description="Farm Yield AI – pipeline runner"
    )
    parser.add_argument(
        "--gui",
        action="store_true",
        help="Launch the Tkinter GUI after training",
    )
    parser.add_argument(
        "--gui-only",
        action="store_true",
        help="Skip training and open the GUI directly",
    )
    args = parser.parse_args()

    if not args.gui_only:
        run_script("preprocessing.py")
        run_script("model_training.py")
        print("\n🎉  Pipeline complete!  Results saved in results/")

    if args.gui or args.gui_only:
        print("\n🖥️   Launching GUI …")
        run_script("gui.py")


if __name__ == "__main__":
    main()
