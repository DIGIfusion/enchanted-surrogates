import sys, os

def append_es_to_path():
    # Dynamically calculate the path to the 'src' directory
    current_file = os.path.abspath(__file__)
    tests_dir = os.path.dirname(
        os.sep.join(os.path.normpath(current_file).split(os.sep)[:current_file.split(os.sep).index("tests") + 2])
    )
    print('debug tests dir:', tests_dir)
    src_path = os.path.join(os.path.dirname(tests_dir), "src")
    print('debug src path:', src_path)
    sys.path.append(src_path)

if __name__ == "__main__":
    append_es_to_path()
