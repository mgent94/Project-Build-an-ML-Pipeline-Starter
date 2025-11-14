import shlex
import subprocess
import sys

def main():
    # Expect two positional args from MLproject substitution: steps and hydra_options
    # MLflow may pass an empty string for hydra_options; handle that gracefully.
    if len(sys.argv) < 2:
        print("Usage: run_mlflow.py <steps> [hydra_options]")
        sys.exit(1)

    steps = sys.argv[1]
    hydra_options = sys.argv[2] if len(sys.argv) > 2 else ""

    # Normalize quotes that may be passed literally (e.g. '""', "''", or multiple quotes)
    # If, after removing surrounding whitespace and quote characters, nothing remains,
    # treat hydra_options as empty so we don't pass an empty token to Hydra.
    if isinstance(hydra_options, str):
        stripped = hydra_options.strip()
        # remove surrounding single/double quotes repeatedly
        stripped = stripped.strip('"\'"')
        if stripped == "":
            hydra_options = ""
        else:
            hydra_options = stripped

    args = [f"main.steps={steps}"]

    # Only extend with hydra options when non-empty after stripping
    if isinstance(hydra_options, str) and hydra_options.strip():
        try:
            extra = shlex.split(hydra_options)
        except ValueError:
            # fallback: split on whitespace
            extra = hydra_options.split()
        args.extend(extra)

    cmd = [sys.executable, "main.py"] + args
    print("Running:", " ".join(shlex.quote(p) for p in cmd))
    rc = subprocess.call(cmd)
    sys.exit(rc)

if __name__ == '__main__':
    main()
