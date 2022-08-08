## TF2 Keras Asymmetric Loss Function

reference: https://github.com/Alibaba-MIIL/ASL/blob/main/src/loss_functions/losses.py

This is the first one among the three versions of the loss: AsymmetricLoss, AsymmetricLossOptimized, ASLSingleLabel

Since this code is written in nested functions, not class, just copy and paste it before you apply loss function

ex) model.compile(loss=AsymmetricLoss(...), ...)
