from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from glass_skull.llama_paths import DEFAULT_LLAMA_CPP_COMMIT, GLASS_SKULL_ROOT, MANAGED_LLAMA_BUILD_DIR, MANAGED_LLAMA_CPP_DIR


PATCH_DIR = GLASS_SKULL_ROOT / "patches" / "llama.cpp-glass"
BRANCH = "glass-skull/per-request-steering"


def run(cmd: list[str | Path], cwd: Path | None = None) -> None:
    printable = " ".join(str(part) for part in cmd)
    print(f"+ {printable}")
    subprocess.run([str(part) for part in cmd], cwd=cwd, check=True)


def output(cmd: list[str | Path], cwd: Path | None = None) -> str:
    return subprocess.check_output([str(part) for part in cmd], cwd=cwd, text=True).strip()


def ensure_clean(repo: Path) -> None:
    status = output(["git", "status", "--porcelain"], cwd=repo)
    if status:
        raise SystemExit(
            f"Managed llama.cpp clone is dirty: {repo}\n"
            "Commit, stash, or remove those changes before rerunning setup."
        )


def ensure_repo(repo: Path, remote: str) -> None:
    if repo.exists():
        if not (repo / ".git").exists():
            raise SystemExit(f"Managed path exists but is not a git repo: {repo}")
        ensure_clean(repo)
        return
    repo.parent.mkdir(parents=True, exist_ok=True)
    run(["git", "clone", remote, repo])


def ensure_commit(repo: Path, commit: str) -> None:
    try:
        run(["git", "cat-file", "-e", f"{commit}^{{commit}}"], cwd=repo)
    except subprocess.CalledProcessError:
        run(["git", "fetch", "--tags", "origin"], cwd=repo)
        run(["git", "cat-file", "-e", f"{commit}^{{commit}}"], cwd=repo)


def apply_patch(repo: Path, patch: Path) -> bool:
    try:
        run(["git", "apply", "--check", patch], cwd=repo)
    except subprocess.CalledProcessError:
        try:
            run(["git", "apply", "--reverse", "--check", patch], cwd=repo)
            print(f"patch already applied: {patch.name}")
            return False
        except subprocess.CalledProcessError as exc:
            raise SystemExit(f"Patch does not apply cleanly: {patch}") from exc
    run(["git", "apply", patch], cwd=repo)
    return True


def patch_repo(repo: Path, commit: str) -> None:
    ensure_clean(repo)
    ensure_commit(repo, commit)
    run(["git", "checkout", "-B", BRANCH, commit], cwd=repo)

    patches = sorted(PATCH_DIR.glob("*.patch"))
    if not patches:
        raise SystemExit(f"No patch files found under {PATCH_DIR}")

    changed = False
    for patch in patches:
        changed = apply_patch(repo, patch) or changed

    if output(["git", "status", "--porcelain"], cwd=repo):
        run(["git", "add", "."], cwd=repo)
        run(
            [
                "git",
                "-c",
                "user.name=Glass Skull",
                "-c",
                "user.email=glass-skull@example.invalid",
                "commit",
                "-m",
                "Apply Glass Skull llama.cpp patch series",
            ],
            cwd=repo,
        )
    elif not changed:
        print("managed clone already has the Glass Skull patch applied")


def build(repo: Path, build_dir: Path) -> None:
    run(
        [
            "cmake",
            "-S",
            repo,
            "-B",
            build_dir,
            "-DCMAKE_BUILD_TYPE=Release",
            "-DGGML_VULKAN=ON",
            "-DLLAMA_BUILD_SERVER=ON",
            "-DLLAMA_BUILD_TOOLS=ON",
        ]
    )
    run(["cmake", "--build", build_dir, "--target", "llama-server", "llama-cvector-generator", "-j"], cwd=repo)


def main() -> None:
    parser = argparse.ArgumentParser(description="Create and build the managed llama.cpp-glass checkout.")
    parser.add_argument("--repo", type=Path, default=MANAGED_LLAMA_CPP_DIR)
    parser.add_argument("--build-dir", type=Path, default=MANAGED_LLAMA_BUILD_DIR)
    parser.add_argument("--commit", default=DEFAULT_LLAMA_CPP_COMMIT)
    parser.add_argument("--remote", default="https://github.com/ggml-org/llama.cpp.git")
    parser.add_argument("--no-build", action="store_true")
    args = parser.parse_args()

    ensure_repo(args.repo.expanduser(), args.remote)
    patch_repo(args.repo.expanduser(), args.commit)
    if not args.no_build:
        build(args.repo.expanduser(), args.build_dir.expanduser())


if __name__ == "__main__":
    main()
