import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from stam_flashsac.training import run_experiment


if __name__ == "__main__":
    run_experiment()
