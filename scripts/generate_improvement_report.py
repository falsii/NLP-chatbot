import os
import subprocess
import sys


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def run_command(command):
    print(f"\nRunning: {' '.join(command)}")

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

    run_command([python_executable, "src/train.py"])
    run_command([python_executable, "src/evaluate.py"])

    print("\nRetrain and evaluation completed.")


if __name__ == "__main__":
    main()