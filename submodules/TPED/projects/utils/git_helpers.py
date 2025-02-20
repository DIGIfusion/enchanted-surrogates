import subprocess

def get_git_root():
    try:
        # Run the git command and capture the output
        result = subprocess.check_output(["git", "rev-parse", "--show-toplevel"], universal_newlines=True)
        # Remove any trailing newline characters
        git_root = result.strip()
        return git_root
    except subprocess.CalledProcessError:
        # Handle the case where the current directory is not in a Git repository
        return None
