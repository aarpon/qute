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

from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np
import torch
from monai.data import DataLoader
from monai.inferers import SlidingWindowInferer
from monai.transforms import Transform
from monai.utils import BlendMode
from tifffile import TiffWriter

from qute.campaigns import CampaignTransforms
from qute.device import get_device
from qute.models.base_model import BaseModel


def full_inference(
    model: BaseModel,
    campaign_transforms: CampaignTransforms,
    data_loader: DataLoader,
    target_folder: Union[Path, str],
    roi_size: Tuple[int, ...],
    batch_size: int,
    overlap: float = 0.25,
    transpose: bool = True,
    output_dtype: Optional[Union[str, np.dtype]] = None,
    prefix: str = "pred_",
):
    """Run inference on full images using given model.

    Parameters
    ----------

    model: BaselModel
        MoOdel to be used for prediction.

    campaign_transforms: CampaignTransforms
        Campaign transforms to be applied (specifically, get_post_full_inference_transforms())

    data_loader: DataLoader
        DataLoader for the image files names to be predicted on.

    target_folder: Union[Path|str]
        Path to the folder where to store the predicted images.

    roi_size: Tuple[int, int]
        Size of the patch for the sliding window prediction. It must match the patch size during training.

    batch_size: int
        Number of parallel batches to run.

    overlap: float
        Fraction of overlap between rois.

    transpose: bool
        Whether to transpose the image before saving, to compensate for the default behavior of monai.transforms.LoadImage().

    output_dtype: Optional[np.dtype]
        Optional NumPy dtype for the output image. Omit to save the output of inference without casting.

    prefix: str = "pred_"
        Prefix to append to the file name. Set to "" to keep the original file name.

    Returns
    -------

    result: bool
        True if the inference was successful, False otherwise.
    """

    # Make sure the target folder exists
    Path(target_folder).mkdir(parents=True, exist_ok=True)

    # Retrieve file names from the dataloader
    if not hasattr(data_loader.dataset, "dataset") or not hasattr(
        data_loader.dataset.dataset, "data"
    ):
        raise ValueError("Incompatible dataset!")
    input_file_names = data_loader.dataset.dataset.data
    if len(input_file_names) == 0:
        print("No input files provided to process. Quitting.")
        return

    # Device
    device = get_device()

    # Make sure the model is on the device
    model.to(device)

    # Switch to evaluation mode
    model.eval()

    # Instantiate the inferer
    sliding_window_inferer = SlidingWindowInferer(
        roi_size=roi_size,
        sw_batch_size=batch_size,
        overlap=overlap,
        mode=BlendMode.GAUSSIAN,
        sigma_scale=0.125,
        device=device,
    )

    # Process all images
    c = 0
    with torch.no_grad():
        for images in data_loader:
            # Apply sliding inference over ROI size
            outputs = sliding_window_inferer(
                inputs=images.to(device),
                network=model,
            )

            # Apply post-transforms?
            outputs = campaign_transforms.get_post_full_inference_transforms()(outputs)

            # Retrieve the image from the GPU (if needed)
            preds = outputs.cpu().numpy()

            # Process one batch at a time
            for pred in preds:
                # Drop the channel singleton dimension
                if pred.shape[0] == 1:
                    pred = pred.squeeze(0)

                if transpose:
                    # Transpose to undo the effect of monai.transform.LoadImage(d)
                    pred = pred.T

                # Type-cast if needed
                if output_dtype is not None:
                    # Make sure not to wrap around
                    if np.issubdtype(output_dtype, np.integer):
                        info = np.iinfo(output_dtype)
                        pred[pred < info.min] = info.min
                        pred[pred > info.max] = info.max
                    pred = pred.astype(output_dtype)

                # Save prediction image as tiff file
                output_name = (
                    Path(target_folder) / f"{prefix}{input_file_names[c].stem}.tif"
                )
                c += 1
                with TiffWriter(output_name) as tif:
                    tif.write(pred, compression="zlib", compressionargs={"level": 9})

                # Inform
                print(f"Saved {output_name}.")

    print("Prediction completed.")

    # Return success
    return True


@staticmethod
def full_inference_ensemble(
    models: list,
    data_loader: DataLoader,
    target_folder: Union[Path, str],
    post_full_inference_transforms: Transform,
    roi_size: Tuple[int, ...],
    batch_size: int,
    voting_mechanism: str = "mode",
    weights: Optional[list] = None,
    overlap: float = 0.25,
    transpose: bool = True,
    save_individual_preds: bool = False,
    output_dtype: Optional[Union[str, np.dtype]] = None,
    prefix: str = "pred_",
    ensemble_prefix: str = "ensemble_",
):
    """Run inference on full images using an ensemble of models.

    Parameters
    ----------

    models: list
        List of trained models inheriting from BaseModel to use for ensemble prediction.

    data_loader: DataLoader
        DataLoader for the image files names to be predicted on.

    target_folder: Union[Path|str]
        Path to the folder where to store the predicted images.

    post_full_inference_transforms: Transform
        Composition of transforms to be applied to the result of the sliding window inference (whole image).

    roi_size: Tuple[int, int]
        Size of the patch for the sliding window prediction. It must match the patch size during training.

    batch_size: int
        Number of parallel batches to run.

    voting_mechanism: str = "mode"
        Voting mechanism to assign the final class among the predictions from the ensemble of models.
        One of "mode" (default) and "mean".
        "mode": pick the most common class among the predictions for each pixel.
        "mean": (rounded) weighted mean of the predicted classes per pixel. The `weights` argument defines
        the relative contribution of the models.

    weights: Optional[list]
        List of weights for each of the contributions. Only used if `voting_mechanism` is "mean".

    overlap: float
        Fraction of overlap between rois.

    transpose: bool
        Whether to transpose the image before saving, to compensate for the default behavior of
        monai.transforms.LoadImage().

    save_individual_preds: bool
        Whether to save the individual predictions of each model.

    output_dtype: Optional[np.dtype]
        Optional NumPy dtype for the output image. Omit to save the output of inference without casting.

    prefix: str = "pred_"
        Prefix to append to the file name. Set to "" to keep the original file name.

    ensemble_prefix: str = "ensemble_pred_"
        Prefix to append to the ensemble prediction file name. Set to "" to keep the original file name.

    Returns
    -------

    result: bool
        True if the inference was successful, False otherwise.
    """

    if voting_mechanism not in ["mode", "mean"]:
        raise ValueError("`voting mechanism` must be one of 'mode' or 'mean'.")

    if voting_mechanism == "mean":
        if len(models) != len(weights):
            raise ValueError("The number of weights must match the number of models.")

        # Turn the weights into a NumPy array of float 32 bit
        weights = np.array(weights, dtype=np.float32)
        weights = weights / weights.sum()

    # Check if models are instances of BaseModel
    if not isinstance(models[0], BaseModel) or not hasattr(models[0], "net"):
        raise ValueError(
            "The models must inherit from `BaseModel` and have a `net` attribute."
        )

    # Make sure the target folder exists
    Path(target_folder).mkdir(parents=True, exist_ok=True)

    # Retrieve file names from the dataloader
    if not hasattr(data_loader.dataset, "dataset") or not hasattr(
        data_loader.dataset.dataset, "data"
    ):
        raise ValueError("Incompatible dataset!")
    input_file_names = data_loader.dataset.dataset.data
    if len(input_file_names) == 0:
        print("No input files provided to process. Quitting.")
        return False

    # If needed, create the sub-folders for the individual predictions
    if save_individual_preds:
        for f in range(len(models)):
            fold_subfolder = Path(target_folder) / f"fold_{f}"
            Path(fold_subfolder).mkdir(parents=True, exist_ok=True)

    # Device
    device = get_device()

    # Switch to evaluation mode on all models
    for model in models:
        model.net.eval()

    # Instantiate the inferer
    sliding_window_inferer = SlidingWindowInferer(
        roi_size=roi_size,
        sw_batch_size=batch_size,
        overlap=overlap,
        mode=BlendMode.GAUSSIAN,
        sigma_scale=0.125,
        device=device,
    )

    c = 0
    with torch.no_grad():
        for images in data_loader:
            predictions = [[] for _ in range(len(models))]

            # Not process all models
            for n, model in enumerate(models):
                # Make sure the model is on the device
                model.to(device)

                # Apply sliding inference over ROI size
                outputs = sliding_window_inferer(
                    inputs=images.to(device),
                    network=model.net,
                )

                # Apply post-transforms?
                outputs = post_full_inference_transforms(outputs)

                # Retrieve the image from the GPU (if needed)
                preds = outputs.cpu().numpy()

                stored_preds = []
                for pred in preds:
                    # Drop the channel singleton dimension
                    if pred.shape[0] == 1:
                        pred = pred.squeeze(0)

                    if transpose:
                        # Transpose to undo the effect of monai.transform.LoadImage(d)
                        pred = pred.T
                    stored_preds.append(pred)

                # Store predictions for each model
                predictions[n] = stored_preds

            # Iterate over all images in the batch
            pred_dim = len(models)
            batch_dim = len(predictions[0])

            for b in range(batch_dim):
                # Apply selected voting mechanism
                if voting_mechanism == "mean":
                    # Apply weighted mean (and rounding) of the predictions per pixel

                    # Iterate over all predictions from the models
                    for p in range(pred_dim):
                        if p == 0:
                            ensemble_pred = weights[p] * predictions[p][b]
                        else:
                            ensemble_pred += weights[p] * predictions[p][b]
                    ensemble_pred = np.round(ensemble_pred).astype(np.int32)

                elif voting_mechanism == "mode":
                    # Select the mode of the predictions per pixel

                    # Store predictions in a stack
                    target = np.zeros(
                        (
                            pred_dim,
                            predictions[0][0].shape[0],
                            predictions[0][0].shape[1],
                        )
                    )

                    # Iterate over all predictions from the models
                    for p in range(pred_dim):
                        target[p, :, :] = predictions[p][b]
                    values, _ = torch.mode(torch.tensor(target), dim=0)
                    ensemble_pred = values.numpy().astype(np.int32)
                else:
                    raise ValueError(
                        "`voting mechanism` must be one of 'mode' or 'mean'."
                    )

                # Type-cast if needed
                if output_dtype is not None:
                    # Make sure not to wrap around
                    if np.issubdtype(output_dtype, np.integer):
                        info = np.iinfo(output_dtype)
                        ensemble_pred[ensemble_pred < info.min] = info.min
                        ensemble_pred[ensemble_pred > info.max] = info.max
                    ensemble_pred = ensemble_pred.astype(output_dtype)

                # Save ensemble prediction image as tiff file
                output_name = (
                    Path(target_folder)
                    / f"{ensemble_prefix}{input_file_names[c].stem}.tif"
                )
                with TiffWriter(output_name) as tif:
                    tif.write(
                        ensemble_pred,
                        compression="zlib",
                        compressionargs={"level": 9},
                    )

                # Inform
                print(f"Saved {output_name}.")

                # Save individual predictions?
                if save_individual_preds:
                    # Iterate over all predictions from the models
                    for p in range(len(predictions)):
                        # Save prediction image as tiff file
                        output_name = (
                            Path(target_folder)
                            / f"fold_{p}"
                            / f"{prefix}{input_file_names[c].stem}.tif"
                        )

                        # Get current prediction
                        current_pred = predictions[p][b]

                        # Type-cast if needed
                        if output_dtype is not None:
                            # Make sure not to wrap around
                            if np.issubdtype(output_dtype, np.integer):
                                info = np.iinfo(output_dtype)
                                current_pred[current_pred < info.min] = info.min
                                current_pred[current_pred > info.max] = info.max
                            current_pred = current_pred.astype(output_dtype)

                        # Save
                        with TiffWriter(output_name) as tif:
                            tif.write(
                                current_pred,
                                compression="zlib",
                                compressionargs={"level": 9},
                            )

                # Update global file counter c
                c += 1

    print("Ensemble prediction completed.")

    # Return success
    return True
