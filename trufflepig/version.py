__version__ = "1.4.0"

version_string = f"v{__version__}"


def print_version():
    print(version_string)


def print_name_and_version():
    print(f"trufflepig {version_string}")


if __name__ == "__main__":
    print_version()
