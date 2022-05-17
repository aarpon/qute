#  ********************************************************************************
#   Copyright © 2022-, ETH Zurich, D-BSSE, Aaron Ponti
#   All rights reserved. This program and the accompanying materials
#   are made available under the terms of the Apache License Version 2.0
#   which accompanies this distribution, and is available at
#   https://www.apache.org/licenses/LICENSE-2.0.txt
#
#   Contributors:
#       Aaron Ponti - initial API and implementation
#  ******************************************************************************/

from datetime import datetime
from tifffile import imwrite
import pytorch_lightning as pl
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from monai.losses import GeneralizedDiceLoss
import torchmetrics
from qute.data.dataloaders import CellSegmentationDemo
from qute.models.unet import UNet


SEED = 2022

if __name__ == "__main__":

    # Seeding
    seed_everything(SEED, workers=True)

    # Data module
    data_module = CellSegmentationDemo(seed=SEED)

    # Model
    model = UNet(num_res_units=4)

    # Callbacks
    early_stopping = EarlyStopping(monitor="val_loss")
    model_checkpoint = ModelCheckpoint(monitor="val_loss")

    # Loss
    criterion=GeneralizedDiceLoss(include_background=True, to_onehot_y=True, softmax=True, batch=True)

    # Metrics
    metrics=torchmetrics.JaccardIndex(num_classes=3, ignore_index=0)

    # Instantiate the Trainer
    trainer = pl.Trainer(
        gpus=1,
        precision=16,
        callbacks=[early_stopping, model_checkpoint],
        max_epochs=500,
        log_every_n_steps=1
    )
    trainer.logger._default_hp_metric = False

    # Train
    trainer.fit(model, data_module)

    # Load weights from best model
    model = UNet.load_from_checkpoint(model_checkpoint.best_model_path)

    # Test
    trainer.test(model, dataloaders=data_module.test_dataloader())

    # Predict on the test dataset
    predictions = trainer.predict(model, dataloaders=data_module.test_dataloader())

    # Save the predictions
    now = datetime.now().strftime("%Y%m%d_%H%M%S")
    predict_out_folder = data_module.data_dir / f"predictions_{now}"
    predict_out_folder.mkdir(parents=True, exist_ok=True)

    i = 0
    for prediction_batch in predictions:
        prediction_batch_cpu = prediction_batch.cpu().detach().numpy()
        for j in range(prediction_batch_cpu.shape[0]):
            out_filename = predict_out_folder / f"prediction_{data_module.test_labels[i].name}"
            imwrite(out_filename, prediction_batch_cpu[j])
            i += 1

    print(f"Saved {i} images to {predict_out_folder}.")
