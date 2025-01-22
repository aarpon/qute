#  ********************************************************************************
#  Copyright © 2022 - 2025, ETH Zurich, D-BSSE, Aaron Ponti
#  All rights reserved. This program and the accompanying materials
#  are made available under the terms of the Apache License Version 2.0
#  which accompanies this distribution, and is available at
#  https://www.apache.org/licenses/LICENSE-2.0.txt
#
#  Contributors:
#    Aaron Ponti - initial API and implementation
#  ******************************************************************************

import hashlib
from pathlib import Path
from typing import Union


def calculate_file_hash(
    file_path: Union[Path, str], hash_function: str = "sha256", chunk_size: int = 8192
) -> str:
    """Calculate the hash of a file.

    Parameters
    ----------

    file_path: Union[Path, str]
        The path to the file to be hashed.

    hash_function: str
        The name of the hash function to use (e.g., "sha256").

    chunk_size: int
        The size of each chunk to read from the file.

    Returns
    -------

    hash: str
        The hexadecimal hash of the file.
    """
    # Create a new hash object
    hash_obj = hashlib.new(hash_function)

    # Open the file in binary mode
    with open(file_path, "rb") as f:
        # Read the file in chunks
        while chunk := f.read(chunk_size):
            # Update the hash object with the file chunk
            hash_obj.update(chunk)

    # Return the hexadecimal digest of the hash
    return hash_obj.hexdigest()


def compare_file_hash(
    file_path: Union[Path, str],
    precalculated_hash: str,
    hash_function: str = "sha256",
    chunk_size: int = 8192,
) -> bool:
    """Compare the hash of a file with a precalculated hash.

    Parameters
    ----------

    file_path: str
        The path to the file to be hashed.

    precalculated_hash: str
        The precalculated hash to compare against.

    hash_function: str
        The name of the hash function to use (e.g., "sha256").

    chunk_size: int
        The size of each chunk to read from the file.

    Returns
    -------

    result: bool
        True if the file hash matches the precaculated hash.
    """

    # Calculate the hash of the file
    calculated_hash = calculate_file_hash(
        file_path=file_path, hash_function=hash_function, chunk_size=chunk_size
    )

    # Compare the calculated hash with the precalculated hash
    return calculated_hash == precalculated_hash


def write_hash_to_file(hash_value: str, output_file_path: Union[Path, str]):
    """Write the hash value to a file.

    Parameters
    ----------

    hash_value: str
        he hash value to write to the file.

    output_file_path: Union[Path, str]:
        The full path to the file where the hash will be written.
    """
    with open(output_file_path, "w") as f:
        f.write(hash_value)


def read_hash_from_file(hash_file_path):
    """
    Read the hash value from a file.

    Parameters
    ----------

    hash_file_path: Union[Path, str]
        The path to the file containing the hash.

    Returns
    -------

    hash_value: str
        The hash value read from the file.
    """
    with open(hash_file_path, "r") as f:
        return f.read().strip()
