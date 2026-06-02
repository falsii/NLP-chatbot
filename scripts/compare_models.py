import os
import subprocess
import sys


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def run_command(command):
    print("\n" + "=" * 80)
    print(f"Running: {' '.join(command)}")
    print("=" * 80)

    result = subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        capture_output=False,
        text=True
    )

    if result.returncode != 0:
        print(f"Command failed: {' '.join(command)}")
        sys.exit(result.returncode)


def main():
    python_executable = sys.executable

    print("\nComparing Bag-of-Words model and LSTM model")

    run_command([python_executable, "src/evaluate.py"])
    run_command([python_executable, "src/evaluate_lstm.py"])

    print("\nModel comparison completed.")


if __name__ == "__main__":
    main()