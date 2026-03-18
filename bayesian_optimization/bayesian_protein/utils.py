import tempfile
from pathlib import Path


def is_temporary_directory(path: Path) -> bool:
    """
    Check if the given path is a temporary directory.

    :param path: A pathlib.Path object representing the path to check
    :return: True if the path is a temporary directory, False otherwise

    Example:
    >>> import tempfile
    >>> with tempfile.TemporaryDirectory() as tempdir:
    ...     is_temporary_directory(Path(tempdir))
    True
    >>> is_temporary_directory(Path("/non/temporary/path"))
    False
    """
    temp_dir = Path(tempfile.gettempdir())
    return temp_dir in path.parents or path == temp_dir


if __name__ == "__main__":
    import doctest

    doctest.testmod()
