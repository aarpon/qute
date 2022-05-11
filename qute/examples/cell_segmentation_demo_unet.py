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

import pytorch_lightning as pl
from qute.data.dataloaders import CellSegmentationDemo
from qute.models.unet import UNet


if __name__ == "__main__":

    # Data module
    data_module = CellSegmentationDemo()

    # Model
    model = UNet()

    # Early stopping
    early_stopping = pl.callbacks.early_stopping.EarlyStopping(
        monitor='val_loss',
    )

    # Instantiate the Trainer
    trainer = pl.Trainer(
        gpus=1,
        precision=16,
        callbacks=[early_stopping],
        max_epochs=500
    )
    trainer.logger._default_hp_metric = False

    # Train
    trainer.fit(model, data_module)
