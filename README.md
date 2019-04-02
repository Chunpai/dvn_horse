# dvn_horse
This is implementation of Weizmann horse segmentation with deep value network.

1. Pretrain the model with FCN, and save the weight into tmp/
2. Fine-tuning the weight with DVN
3. Adversarial + Inference (init=0) + Adam(lr=0.0001, weight_decay=0.001) +  Seed=0 Have best performance so far.

