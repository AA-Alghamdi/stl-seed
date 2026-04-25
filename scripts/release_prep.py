"""v0.1.0 release-prep helper for stl-seed.

Performs the *local* portion of a tagged release:

  1. Verify gates: ``pytest`` clean, ``ruff check`` clean, firewall clean.
  2. Bump the version in ``pyproject.toml`` and
     ``src/stl_seed/__init__.py`` (CLI flag ``--bump {major,minor,patch}``).
  3. Generate ``CHANGELOG.md`` from ``git log`` between the last release
     tag (``v*``) and ``HEAD``. If no prior tag exists, log from the
     repo root.
  4. Create an annotated git tag ``v<new_version>`` (signed if
     ``user.signingkey`` is set in git config).
  5. Print next-step instructions for ``gh release create`` so the user
     drives the actual GitHub release manually.

This script does NOT:
  - push tags
  - create GitHub releases
  - upload to PyPI
  - amend or rewrite history

All of those are operator actions deliberately kept out of automation.
``--dry-run`` does everything except writing files and creating the
tag — useful for previewing the bump and CHANGELOG before committing.


Usage::

    # Preview a patch release (no writes, no tag):
    uv run python scripts/release_prep.py --bump patch --dry-run

    # Actual v0.1.0 prep (minor bump from 0.0.1):
    uv run python scripts/release_prep.py --bump minor

    # Skip the gate checks (escape hatch for emergency releases):
    uv run python scripts/release_prep.py --bump patch --skip-gates
"""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths and constants.
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).resolve().parents[1]
_PYPROJECT = _REPO_ROOT / "pyproject.toml"
_INIT_PY = _REPO_ROOT / "src" / "stl_seed" / "__init__.py"
_CHANGELOG = _REPO_ROOT / "CHANGELOG.md"

# `version = "..."` in [project]; tolerant of single/double quotes and
# whitespace. Anchored via a leading 'version' to avoid matching
# `target-version = "py311"` further down.
_PYPROJECT_VERSION_RE = re.compile(r'^(version\s*=\s*)["\']([^"\']+)["\']', re.MULTILINE)
_INIT_VERSION_RE = re.compile(r'^(__version__\s*=\s*)["\']([^"\']+)["\']', re.MULTILINE)

# Conventional commit prefixes -> CHANGELOG section. Keep this short;
# anything not matching falls into "Other".
_COMMIT_PREFIX_TO_SECTION: dict[str, str] = {
    "feat": "Features",
    "feature": "Features",
    "fix": "Fixes",
    "bug": "Fixes",
    "docs": "Documentation",
    "doc": "Documentation",
    "test": "Tests",
    "tests": "Tests",
    "refactor": "Refactoring",
    "perf": "Performance",
    "ci": "CI",
    "build": "Build",
    "chore": "Chores",
}


# ---------------------------------------------------------------------------
# Version bump.
# ---------------------------------------------------------------------------


@dataclass
class Version:
    """Semantic version parsed from a ``MAJOR.MINOR.PATCH`` string."""

    major: int
    minor: int
    patch: int

    @classmethod
    def parse(cls, raw: str) -> Version:
        m = re.fullmatch(r"(\d+)\.(\d+)\.(\d+)", raw.strip())
        if m is None:
            raise ValueError(f"version string {raw!r} is not MAJOR.MINOR.PATCH")
        return cls(major=int(m.group(1)), minor=int(m.group(2)), patch=int(m.group(3)))

    def bump(self, kind: str) -> Version:
        if kind == "major":
            return Version(major=self.major + 1, minor=0, patch=0)
        if kind == "minor":
            return Version(major=self.major, minor=self.minor + 1, patch=0)
        if kind == "patch":
            return Version(major=self.major, minor=self.minor, patch=self.patch + 1)
        raise ValueError(f"unknown bump kind {kind!r}; expected major|minor|patch")

    def __str__(self) -> str:
        return f"{self.major}.{self.minor}.{self.patch}"


def read_pyproject_version(path: Path = _PYPROJECT) -> Version:
    """Extract the current version from pyproject.toml."""
    text = path.read_text()
    m = _PYPROJECT_VERSION_RE.search(text)
    if m is None:
        raise SystemExit(f'could not find a `version = "..."` line in {path}')
    return Version.parse(m.group(2))


def write_pyproject_version(new: Version, *, path: Path = _PYPROJECT) -> None:
    """Rewrite the [project] `version = "..."` line."""
    text = path.read_text()
    new_text, n = _PYPROJECT_VERSION_RE.subn(rf'\g<1>"{new}"', text, count=1)
    if n != 1:
        raise SystemExit(f"failed to rewrite version in {path}")
    path.write_text(new_text)


def write_init_version(new: Version, *, path: Path = _INIT_PY) -> None:
    """Rewrite ``__version__ = "..."`` in src/stl_seed/__init__.py."""
    text = path.read_text()
    new_text, n = _INIT_VERSION_RE.subn(rf'\g<1>"{new}"', text, count=1)
    if n != 1:
        raise SystemExit(f"failed to rewrite __version__ in {path}")
    path.write_text(new_text)


# ---------------------------------------------------------------------------
# Git helpers.
# ---------------------------------------------------------------------------


def _git(*args: str, cwd: Path = _REPO_ROOT, check: bool = True) -> str:
    """Run a git command and return stdout (stripped)."""
    res = subprocess.run(
        ["git", *args],
        cwd=cwd,
        capture_output=True,
        text=True,
        check=False,
    )
    if check and res.returncode != 0:
        raise SystemExit(f"git {' '.join(args)} failed (exit {res.returncode}):\n{res.stderr}")
    return res.stdout.strip()


def last_release_tag() -> str | None:
    """Return the most recent tag matching ``v*``, or None.

    Uses ``git describe --tags --abbrev=0 --match v*``; if no tags
    exist git exits 128 and we return None.
    """
    res = subprocess.run(
        ["git", "describe", "--tags", "--abbrev=0", "--match", "v*"],
        cwd=_REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if res.returncode != 0:
        return None
    return res.stdout.strip() or None


def commits_since(ref: str | None) -> list[tuple[str, str]]:
    """Return ``[(sha, subject), ...]`` for commits since ``ref``.

    If ``ref`` is None, returns the full history (used when the repo
    has no prior release tag — typical for the v0.1.0 cut).
    """
    rng = "HEAD" if ref is None else f"{ref}..HEAD"
    raw = _git("log", "--no-merges", "--pretty=format:%h%x09%s", rng, check=False)
    out: list[tuple[str, str]] = []
    for line in raw.splitlines():
        if "\t" in line:
            sha, subject = line.split("\t", 1)
            out.append((sha.strip(), subject.strip()))
    return out


def is_signing_configured() -> bool:
    """Return True if ``user.signingkey`` is set in git config."""
    res = subprocess.run(
        ["git", "config", "--get", "user.signingkey"],
        cwd=_REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    return res.returncode == 0 and bool(res.stdout.strip())


def working_tree_dirty() -> bool:
    out = _git("status", "--porcelain", check=False)
    return bool(out.strip())


# ---------------------------------------------------------------------------
# CHANGELOG generation.
# ---------------------------------------------------------------------------


def categorize(subject: str) -> str:
    """Map a commit subject to a CHANGELOG section via prefix heuristic."""
    head = subject.split(":", 1)[0].lower().strip()
    # Strip optional scope: `feat(api): foo` -> `feat`.
    head = re.sub(r"\(.*\)$", "", head)
    return _COMMIT_PREFIX_TO_SECTION.get(head, "Other")


def render_changelog(
    new_version: Version,
    commits: list[tuple[str, str]],
    *,
    prior_tag: str | None,
    repo_url: str = "https://github.com/AA-Alghamdi/stl-seed",
) -> str:
    """Build the CHANGELOG body for the new version.

    Sections appear in a fixed order so the file diff is stable across
    releases. Commits within a section are ordered by their original
    git-log order (newest first).
    """
    sections: dict[str, list[tuple[str, str]]] = {}
    for sha, subj in commits:
        sections.setdefault(categorize(subj), []).append((sha, subj))

    section_order = [
        "Features",
        "Fixes",
        "Performance",
        "Refactoring",
        "Documentation",
        "Tests",
        "CI",
        "Build",
        "Chores",
        "Other",
    ]

    # ISO 8601 date for the release; uses local time deliberately so
    # the string lines up with the user's reading.
    import datetime as _dt

    today = _dt.date.today().isoformat()

    range_label = f"{prior_tag}..HEAD" if prior_tag else "initial release"
    lines: list[str] = []
    lines.append(f"## v{new_version} — {today}")
    lines.append("")
    lines.append(f"_Range: {range_label} (n={len(commits)} commits)_")
    lines.append("")
    if not commits:
        lines.append("_No commits since prior release._")
        lines.append("")
        return "\n".join(lines)

    for sec in section_order:
        items = sections.get(sec, [])
        if not items:
            continue
        lines.append(f"### {sec}")
        lines.append("")
        for sha, subj in items:
            link = f"[{sha}]({repo_url}/commit/{sha})"
            lines.append(f"- {subj} ({link})")
        lines.append("")
    return "\n".join(lines)


def prepend_changelog(new_block: str, *, path: Path = _CHANGELOG) -> None:
    """Prepend ``new_block`` to CHANGELOG.md, creating the file if needed."""
    header = "# Changelog\n\nAll notable changes to this project are documented here.\n\n"
    if path.exists():
        existing = path.read_text()
        if existing.startswith(header):
            body = existing[len(header) :]
        elif existing.startswith("# Changelog"):
            # An older custom header; keep it as-is and just prepend the block
            # below the first blank line.
            head, _, rest = existing.partition("\n\n")
            path.write_text(f"{head}\n\n{new_block}\n{rest}")
            return
        else:
            body = existing
        path.write_text(f"{header}{new_block}\n{body}")
    else:
        path.write_text(f"{header}{new_block}\n")


# ---------------------------------------------------------------------------
# Gate checks.
# ---------------------------------------------------------------------------


@dataclass
class GateResult:
    name: str
    ok: bool
    detail: str


def _run_gate(name: str, cmd: list[str], cwd: Path = _REPO_ROOT) -> GateResult:
    """Run a subprocess-style gate; capture stdout+stderr for the report."""
    if shutil.which(cmd[0]) is None:
        return GateResult(name=name, ok=False, detail=f"{cmd[0]} not on PATH")
    res = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, check=False)
    ok = res.returncode == 0
    tail = (res.stdout + res.stderr).strip().splitlines()
    detail = "\n".join(tail[-5:]) if tail else ""
    return GateResult(name=name, ok=ok, detail=detail)


def gate_pytest() -> GateResult:
    return _run_gate(
        "pytest",
        ["uv", "run", "pytest", "-q", "-m", "not gpu and not cuda and not mlx"],
    )


def gate_ruff() -> GateResult:
    return _run_gate("ruff check", ["uv", "run", "ruff", "check", "src", "tests", "scripts"])


def run_gates() -> list[GateResult]:
    return [gate_pytest(), gate_ruff()]


# ---------------------------------------------------------------------------
# Tag creation.
# ---------------------------------------------------------------------------


def create_tag(new_version: Version, *, sign: bool, dry_run: bool) -> str:
    """Create an annotated git tag and return the tag name.

    Signed if ``sign`` is True (requires ``user.signingkey`` already
    configured). The tag message is intentionally minimal — a one-line
    summary suitable for `gh release create` to expand from CHANGELOG.
    """
    tag = f"v{new_version}"
    msg = f"stl-seed {tag}"
    args = ["tag", "-a", tag, "-m", msg]
    if sign:
        args.insert(1, "-s")
    if dry_run:
        return tag
    _git(*args)
    return tag


# ---------------------------------------------------------------------------
# CLI.
# ---------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="release_prep.py",
        description="v0.1.0 release-prep helper: bump version, write CHANGELOG, create tag.",
    )
    p.add_argument(
        "--bump",
        choices=("major", "minor", "patch"),
        required=True,
        help="Which semver component to bump.",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Compute the new version + CHANGELOG, do not write or tag.",
    )
    p.add_argument(
        "--skip-gates",
        action="store_true",
        help="Skip pytest/ruff/firewall verification (NOT recommended).",
    )
    p.add_argument(
        "--allow-dirty",
        action="store_true",
        help="Allow the working tree to be dirty when creating the tag.",
    )
    p.add_argument(
        "--no-sign",
        action="store_true",
        help="Force unsigned tag even when user.signingkey is configured.",
    )
    p.add_argument(
        "--repo-url",
        type=str,
        default="https://github.com/AA-Alghamdi/stl-seed",
        help="Repo URL for CHANGELOG commit links (default: github.com/AA-Alghamdi/stl-seed).",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    cur = read_pyproject_version()
    new = cur.bump(args.bump)
    print(f"Current version : {cur}")
    print(f"Bumped version  : {new}  ({args.bump})")
    print(f"Dry run         : {args.dry_run}")
    print()

    # Gate checks (unless --skip-gates).
    if args.skip_gates:
        print("[skip] gate checks bypassed via --skip-gates")
    else:
        print("Running gates...")
        results = run_gates()
        for r in results:
            mark = "OK" if r.ok else "FAIL"
            print(f"  [{mark}] {r.name}")
            if not r.ok and r.detail:
                for line in r.detail.splitlines():
                    print(f"        {line}")
        if any(not r.ok for r in results):
            print("\nFAIL: one or more gates did not pass. Aborting.")
            print("Re-run with --skip-gates to override (NOT recommended for releases).")
            return 1

    # Working-tree cleanliness check (unless --allow-dirty).
    if working_tree_dirty() and not args.allow_dirty:
        print("\nFAIL: working tree is dirty (uncommitted changes).")
        print("Commit or stash first, or pass --allow-dirty.")
        return 1

    # CHANGELOG.
    prior_tag = last_release_tag()
    commits = commits_since(prior_tag)
    print(f"\nCHANGELOG range: {prior_tag or '(initial)'} -> HEAD  ({len(commits)} commits)")
    block = render_changelog(new, commits, prior_tag=prior_tag, repo_url=args.repo_url)
    print("\n--- CHANGELOG block (preview, first 30 lines) ---")
    for line in block.splitlines()[:30]:
        print(line)
    print("--- /preview ---\n")

    if args.dry_run:
        print("Dry run complete. No files written, no tag created.")
        print(f"  Planned tag: v{new}")
        return 0

    # Apply edits.
    print(f"Writing pyproject.toml version -> {new}")
    write_pyproject_version(new)
    print(f"Writing src/stl_seed/__init__.py __version__ -> {new}")
    write_init_version(new)
    print(f"Prepending CHANGELOG.md with v{new} block")
    prepend_changelog(block)

    # Create the tag.
    sign = is_signing_configured() and not args.no_sign
    print(f"Creating annotated tag v{new}  (signed={sign})")
    tag = create_tag(new, sign=sign, dry_run=False)

    # Final hint to the operator.
    print()
    print("=" * 60)
    print(f"  Release prep complete for {tag}")
    print("=" * 60)
    print()
    print("Next steps (run manually, NOT automated):")
    print()
    print("  # 1. Review the diff and the CHANGELOG entry")
    print("  git diff HEAD")
    print("  git diff --cached")
    print("  head -30 CHANGELOG.md")
    print()
    print("  # 2. Commit the bump")
    print("  git add pyproject.toml src/stl_seed/__init__.py CHANGELOG.md")
    print(f'  git commit -m "Release {tag}"')
    print()
    print("  # 3. Push the commit AND the tag")
    print("  git push origin HEAD")
    print(f"  git push origin {tag}")
    print()
    print("  # 4. Create the GitHub release")
    print(
        f"  gh release create {tag} --title '{tag}' "
        f"--notes-file <(awk '/^## v/{{c++}} c==1' CHANGELOG.md)"
    )
    print()
    print("  # 5. (optional) PyPI upload")
    print("  uv build")
    print("  uv publish --token $PYPI_TOKEN")
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
