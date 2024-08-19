# ******************************************************************************
# Copyright © 2022 - 2024, ETH Zurich, D-BSSE, Aaron Ponti
# All rights reserved. This program and the accompanying materials
# are made available under the terms of the Apache License Version 2.0
# which accompanies this distribution, and is available at
# https://www.apache.org/licenses/LICENSE-2.0.txt
#
# Contributors:
#   Aaron Ponti - initial API and implementation
# ******************************************************************************

import json
import os
import re
import shutil
import sys
import time
from pathlib import Path
from typing import Optional, Union

import numpy as np
import userpaths
from imio import load, save
from natsort import natsorted
from numpy.random import default_rng


def sample(
    image: np.ndarray,
    patch_size: tuple,
    y0: Optional[int] = None,
    x0: Optional[int] = None,
    seed: Optional[int] = None,
) -> tuple[np.ndarray, int, int]:
    """Returns a (random) subset of given shape from the passed 2D image.

    Parameters
    ----------

    image: numpy array
        Original intensity image.

    patch_size: tuple
        Size (y, x) of the subset of the image to be randomly extracted.

    y0: Optional[int]
        y component of the top left corner of the extracted region.
        If omitted (default), it will be randomly generated.

    x0: Optional[int]
        x component of the top left corner of the extracted region.
        If omitted (default), it will be randomly generated.

    seed: Optional[int]
        Random generator seed to reproduce the sampling. Omit to create a
        new random sample every time.

    Returns
    -------

    result: tuple[np.ndarray, int, int]
        Subset of the image of given size; y coordinate of the top-left corner of
        the extracted subset; x coordinate of the top-left corner of the extracted subset.
    """

    if image.ndim != 2:
        raise ValueError("The image must be 2D.")

    # Initialize random-number generator
    if seed is None:
        seed = time.time_ns()
    rng = np.random.default_rng(seed)

    # Get starting point
    max_y = image.shape[0] - patch_size[0]
    max_x = image.shape[1] - patch_size[1]
    if y0 is None:
        y0 = int(rng.uniform(0, max_y))
    if x0 is None:
        x0 = int(rng.uniform(0, max_x))

    # Return the subset and the starting coordinates
    return image[y0 : y0 + patch_size[0], x0 : x0 + patch_size[1]], y0, x0


def qute_to_msd_format(
    input_folder: Union[Path, str],
    output_folder: Union[Path, str],
    channel_names: tuple = ("fluorescence_microscopy",),
    label_names: tuple = ("background", "cell", "cell_border"),
    dataset_id: int = 1,
    dataset_name: Optional[str] = None,
    stem_name: str = "demo_",
    images_suffix: str = "_0000",
    to_nii_gz: bool = False,
    test_perc: float = 0.2,
    num_folds: int = 5,
    force: bool = False,
    seed=None,
) -> tuple[bool, str]:
    """Convert a qute dataset (with sub-folders "images" and "labels") to an MSD dataset.

    Make sure the images and labels have ONE numerical index only and that each is unique.

    Parameters
    ----------

    input_folder: Union[Path, str]
        Full path of the dataset in qute format.

    output_folder: Union[Path, str]
        Full path of the nnUnetv2-compatible dataset. It is recommended to set this to `$nnNet_raw`.

    channel_names: tuple
        Name of the channels, e.g. ("fluorescence_microscopy", )

    label_names: tuple
        Name of the segmentation classes, e.g. ("background", "cell", "cell_border")

    dataset_id: int
        Integer ID of the dataset.

    dataset_name: str
        Name of the dataset. If not specified, it will be derived from `input_folder`.

    stem_name: str
        Stem name of all images. In contrast to the qute format, all training images, labels and test
        images must have the same stem name.

    images_suffix: str (default is "_0000")
        Numeric suffix (as 0-padded string and leading _) for the images (it won't be applied to the labels).
        For instance, imagesTr/demo_001_0000.tif is matched by labelsTr/demo_001.tif.

    to_nii_gz: bool (default is False)
        Set to True to convert the TIFF images to nii.gz; otherwise, the TIFF files are just copied over.

    test_perc: float (default is 0.2)
        Fraction of the images to be used for testing.

    num_folds: int (default is 5)
        Number of folds for k-fold cross-validation during training.

    force: bool (default is False)
        Set to True to delete and recreate if a converted Dataset already exists in the `output_folder`, otherwise
        abort.

    seed: int (default is 2022)
        Seed for the random number generator (for training/test splitting).

    Returns
    -------

    res, msg: tuple(bool, str)
        res: True if the dataset was created, False otherwise
        msg: "" if successful; otherwise, contains error message.
    """

    # Check the structure of the input folder
    input_folder = Path(input_folder)
    if not input_folder.is_dir():
        return False, f"Input folder {input_folder} does not exit."

    # Check tha value of dataset_id
    if dataset_id != int(dataset_id) or dataset_id < 1:
        raise ValueError("`dataset_id` must be a positive integer.")

    # Check the value of test_perc
    if test_perc < 0.0 or test_perc > 1.0:
        raise ValueError("`test_perc` must be between 0 and 1.")

    # Make sure the images and labels sub-folders exist
    images_folder = input_folder / "images"
    if not images_folder.is_dir():
        return False, f"The `images` sub-folder does not exit."

    labels_folder = input_folder / "labels"
    if not labels_folder.is_dir():
        return False, f"The `labels` sub-folder does not exit."

    # Get the contents of images_folder and labels_folder
    images_files = list(images_folder.glob("*.tif*"))
    labels_files = list(labels_folder.glob("*.tif*"))
    if len(images_files) == 0 or len(labels_files) == 0:
        return False, f"No `images` or `labels` found."
    if len(images_files) != len(labels_files):
        return False, f"Unmatched number of `images` and `labels`."

    # Check whether the output folder already exists, otherwise create
    output_folder = Path(output_folder)
    output_folder.mkdir(exist_ok=True)

    # Check whether the dataset folder already exists, otherwise create
    if dataset_name is None:
        dataset_name = input_folder.name
    dataset_folder = output_folder / f"Dataset{dataset_id:03}_{dataset_name}"
    dataset_folder.mkdir(exist_ok=True)

    # Check whether the training and test sub-folders already exist, otherwise create
    images_tr_folder = dataset_folder / "imagesTr"
    images_tr_folder.mkdir(exist_ok=True)
    labels_tr_folder = dataset_folder / "labelsTr"
    labels_tr_folder.mkdir(exist_ok=True)
    images_ts_folder = dataset_folder / "imagesTs"
    images_ts_folder.mkdir(exist_ok=True)

    # Check if the sub-folders already contain images
    empty = True
    if len(list(images_ts_folder.glob("*.tif*"))) > 0:
        empty = False
    if len(list(labels_tr_folder.glob("*.tif*"))) > 0:
        empty = False
    if len(list(images_ts_folder.glob("*.tif*"))) > 0:
        empty = False
    if not empty:
        if not force:
            return False, "Dataset already existing. As requested, we abort here."

        # Delete sub-folders
        shutil.rmtree(images_tr_folder)
        shutil.rmtree(labels_tr_folder)
        shutil.rmtree(images_ts_folder)

        # Re-create them
        images_tr_folder.mkdir(exist_ok=True)
        labels_tr_folder.mkdir(exist_ok=True)
        images_ts_folder.mkdir(exist_ok=True)

    # Make sure that the filenames are sorted (naturally)
    images_files = natsorted(images_files)
    labels_files = natsorted(labels_files)

    # Shuffle a copy of the file names
    rng = default_rng(seed=seed)
    shuffled_indices = rng.permutation(len(images_files))
    shuffled_images_files = np.array(images_files.copy())[shuffled_indices].tolist()
    shuffled_labels_files = np.array(labels_files.copy())[shuffled_indices].tolist()

    # Split between training and testing sub-sets
    threshold = int(round((1.0 - test_perc) * len(images_files)))
    if threshold == 0 or threshold == len(images_files):
        return False, "The requested `test_perc` fails splitting the images properly."
    selected_training_images_files = shuffled_images_files[:threshold]
    selected_training_labels_files = shuffled_labels_files[:threshold]
    selected_test_images_files = shuffled_images_files[threshold:]

    def target_name(convert: bool, name: str, stem: str, suffix: str = ""):
        """Rename the target image to fit the nnUNetv2 expected pattern."""
        numbers = re.findall(r"(\d+).tif$", Path(name).name)
        if len(numbers) != 1:
            return name
        if convert:
            new_name = f"{stem}{numbers[0]}{suffix}.nii.gz"
        else:
            new_name = f"{stem}{numbers[0]}{suffix}.tif"
        return new_name

    # Prepare output file names
    renamed_selected_training_images_files = []
    renamed_selected_training_labels_files = []
    renamed_selected_test_images_files = []
    for f in selected_training_images_files:
        renamed_selected_training_images_files.append(
            f"{images_tr_folder}/{target_name(to_nii_gz, f, stem_name, images_suffix)}"
        )
    for f in selected_training_labels_files:
        renamed_selected_training_labels_files.append(
            f"{labels_tr_folder}/{target_name(to_nii_gz, f, stem_name)}"
        )
    for f in selected_test_images_files:
        renamed_selected_test_images_files.append(
            f"{images_ts_folder}/{target_name(to_nii_gz, f, stem_name, images_suffix)}"
        )

    # Now copy or convert the files
    for f, o in zip(
        selected_training_images_files, renamed_selected_training_images_files
    ):
        if to_nii_gz:
            save.to_nii(load.load_any(f), o)
        else:
            shutil.copy(f, o)
    for f, o in zip(
        selected_training_labels_files, renamed_selected_training_labels_files
    ):
        if to_nii_gz:
            save.to_nii(load.load_any(f), o)
        else:
            shutil.copy(f, o)
    for f, o in zip(selected_test_images_files, renamed_selected_test_images_files):
        if to_nii_gz:
            save.to_nii(load.load_any(f), o)
        else:
            shutil.copy(f, o)

    # Create the datalist
    datalist = {
        # Add all test images to datalist["testing"]
        "testing": [
            {"image": "./imagesTs/" + Path(file).name}
            for file in renamed_selected_test_images_files
        ],
        # Add all training images and labels as one fold
        "training": [
            {
                "image": f"./imagesTr/{Path(renamed_selected_training_images_files[i]).name}",
                "label": f"./labelsTr/{Path(renamed_selected_training_labels_files[i]).name}",
                "fold": 0,
            }
            for i in range(len(renamed_selected_training_images_files))
        ],
    }

    # Split training data into num_folds random folds
    fold_size = len(datalist["training"]) // num_folds
    for i in range(num_folds):
        for j in range(fold_size):
            datalist["training"][i * fold_size + j]["fold"] = i

    # Create dictionaries to store in the dataset.json file
    channel_names_dict = {}
    for i, channel in enumerate(channel_names):
        channel_names_dict[str(i)] = channel
    label_names_dict = {}
    for i, label in enumerate(label_names):
        label_names_dict[label] = i
    dataset = {
        "channel_names": channel_names_dict,
        "labels": label_names_dict,
        "numTraining": len(datalist["training"]),
        "file_ending": ".tif",
        "dataset_name": f"Dataset{dataset_id:03}_{dataset_name}",
    }
    with open(dataset_folder / "dataset.json", "w") as f:
        # write the dictionary to the file in JSON format
        json.dump(dataset, f)

    # Save the datalist
    datalist_file = (
        dataset_folder / f"msd_{Path(dataset_folder).name.lower()}_folds.json"
    )
    with open(datalist_file, "w", encoding="utf-8") as f:
        json.dump(datalist, f, ensure_ascii=False, indent=4)

    # Finally, create input.yaml
    with open(dataset_folder / "input.yaml", "w") as f:
        f.write(f"dataset_name_or_id: {dataset_id}\n")
        f.write(f"modality: CT\n")
        f.write(f"datalist: {datalist_file}\n")
        f.write(f"dataroot: {dataset_folder}\n")

    # Inform
    print(f"\n\n* * * All done! * * *")
    print(f"\nData and configuration files written to `{dataset_folder}`.")
    print(
        f"\nIf it is not already there, please make sure to copy/move `{dataset_folder.name}` to the folder "
    )
    print(
        f"pointed at by the environment variable `nnUNet_raw`. Then, assuming the dataset id is `1`\n"
        f"and the dataset is 2D, run the following (adapt accordingly): "
    )
    print(f"\n```sh")
    print(
        f"$ nnUNetv2_plan_and_preprocess -d 1 --verify_dataset_integrity -pl nnUNetPlannerResEncM -gpu_memory_target 12 -overwrite_plans_name nnUNetResEncUNetPlans_12G"
    )
    print(
        "$ for i in {0..4}; do nnUNetv2_train 1 2d $i --npz; done   # Fold 0 through 4, adapt as necessary"
    )
    print(f"$ nnUNetv2_find_best_configuration -d 1 -f 0 1 2 3 4 -c 2d")
    print(
        f"$ nnUNetv2_predict -d 1 -i $nnUNet_raw/$Dataset/imagesTs -o $nnUNet_results/$Dataset/inference -f 0 1 2 3 4 \\"
    )
    print(f"       -tr nnUNetTrainer -c 2d -p nnUNetPlans")
    print(
        f"$ nnUNetv2_apply_postprocessing -i $nnUNet_results/$Dataset/inference -o $nnUNet_results/$Dataset/postprocessing \\"
    )
    print(
        f"       -pp_pkl_file $nnUNet_results/$Dataset/nnUNetTrainer__nnUNetPlans__2d/crossval_results_folds_0_1_2_3_4/postprocessing.pkl \\"
    )
    print(
        f"     -np 8 -plans_json $nnUNet_results/$Dataset/nnUNetTrainer__nnUNetPlans__2d/crossval_results_folds_0_1_2_3_4/plans.json"
    )
    print(f"\n```")
    print(f"\nIn the commands above, replace `$Dataset` with the dataset name.")
    print(f"\nPlease see: ")
    print(
        f"  * https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/set_environment_variables.md"
    )
    print(
        f"  * https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/how_to_use_nnunet.md"
    )
    print(
        f"  * https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/resenc_presets.md"
    )
    print(
        f"  * https://transformhealthcare.medium.com/glioblastoma-brain-tumor-segmentation-part-6-neural-network-model-training-5de238e9b195"
    )
    print(
        f"  * https://transformhealthcare.medium.com/glioblastoma-brain-tumor-segmentation-part-7-inference-58d4287a040d"
    )

    # Return success
    return True, ""


if __name__ == "__main__":

    if len(sys.argv) == 1:

        # Path to qute demo segmentation dataset
        qute_dataset_folder = (
            Path(userpaths.get_my_documents())
            / "qute"
            / "data"
            / "demo_segmentation_3_classes/"
        )

    elif len(sys.argv) == 2:

        # Check if we have a valid folder as a second argument
        qute_dataset_folder = Path(sys.argv[1])

    else:

        sys.exit(
            f"Please use: python {sys.argv[0]} [folder_to_process] (omit to use demo dataset)."
        )

    if "nnUNet_raw" in os.environ:
        nnUnet_raw_folder = Path(os.environ["nnUNet_raw"])
    else:
        print("The environment variable `nnUNet_raw` is not defined.")
        sys.exit(1)

    qute_to_msd_format(
        input_folder=qute_dataset_folder,
        output_folder=nnUnet_raw_folder,
        force=True,
    )
