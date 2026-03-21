import yaml
import os

CONFIG_PATH = "plugins.yml"

def main():
    token = os.environ["GH_PAT"]

    with open(CONFIG_PATH, "r") as f:
        config = yaml.safe_load(f)

    os.makedirs("plugins", exist_ok=True)

    for plugin in config["plugins"]:
        name = plugin["name"]
        repo = plugin["repo"]

        cmd = (f'git clone https://x-access-token:{token}@github.com/{repo}.git plugins/{name}')
        

        print(f"[CLONE] {repo}")
        print(cmd)
        os.system(cmd)


if __name__ == "__main__":
    main()