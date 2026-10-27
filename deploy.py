#!/usr/bin/env python3
"""Build and deploy PulseHalo to Firebase Hosting."""

import json
import shutil
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent


def main():
    config_path = PROJECT_ROOT / ".firebaserc"
    with config_path.open(encoding="utf-8") as config_file:
        firebase_config = json.load(config_file)

    project_id = firebase_config.get("projects", {}).get("default")
    if not project_id:
        raise SystemExit(f"No default Firebase project is configured in {config_path}.")

    firebase_cli = shutil.which("firebase")
    if not firebase_cli:
        raise SystemExit(
            "Firebase CLI was not found. Install it with `npm install -g firebase-tools`, "
            "then sign in with `firebase login`."
        )

    print("Building the Firebase Hosting site...", flush=True)
    build_result = subprocess.run(
        [sys.executable, str(PROJECT_ROOT / "build_standalone.py")],
        cwd=PROJECT_ROOT,
        check=False,
    )
    if build_result.returncode:
        raise SystemExit(f"Site build failed with exit code {build_result.returncode}.")

    print(f"Deploying Hosting to Firebase project {project_id}...", flush=True)
    deploy_result = subprocess.run(
        [
            firebase_cli,
            "deploy",
            "--only",
            "hosting",
            "--project",
            project_id,
        ],
        cwd=PROJECT_ROOT,
        check=False,
    )
    if deploy_result.returncode:
        raise SystemExit(
            f"Firebase Hosting deployment failed with exit code {deploy_result.returncode}; "
            "no deployed URL is being reported."
        )

    hosting_url = f"https://{project_id}.web.app"
    print(f"Verifying deployed site at {hosting_url}...", flush=True)
    try:
        with urllib.request.urlopen(hosting_url, timeout=30) as response:
            if not 200 <= response.status < 300:
                raise SystemExit(
                    f"Firebase deployment completed, but the live URL returned "
                    f"HTTP {response.status}: {hosting_url}"
                )
    except urllib.error.URLError as error:
        raise SystemExit(
            f"Firebase deployment completed, but the live URL could not be verified: "
            f"{error}. Check {hosting_url}."
        ) from error

    print(f"\nFirebase deployed URL: {hosting_url}", flush=True)


if __name__ == "__main__":
    main()
