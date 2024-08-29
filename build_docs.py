# build_docs.py
import subprocess

if __name__ == "__main__":
    subprocess.run(
        ["sphinx-apidoc", "-f", "-e", "-M", "--no-toc", "-o", "docs", "qute"],
        check=True,
    )
    subprocess.run(["make", "html"], cwd="docs", check=True)
