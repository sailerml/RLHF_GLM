# RLHF_GLM

训练一个奖励模型，模型详细原理、训练步骤参考 [这里]()


训练脚本：

```sh
sh train_reward_model.sh
```

训练后log：

```python
...
global step 10, epoch: 1, loss: -0.51766, speed: 0.21 step/s
global step 20, epoch: 1, loss: -0.55865, speed: 0.22 step/s
global step 30, epoch: 1, loss: -0.60930, speed: 0.21 step/s
global step 40, epoch: 1, loss: -0.65024, speed: 0.21 step/s
global step 50, epoch: 1, loss: -0.67781, speed: 0.22 step/s
Evaluation acc: 0.50000
best F1 performence has been updated: 0.00000 --> 0.50000
global step 60, epoch: 1, loss: -0.69296, speed: 0.20 step/s
global step 70, epoch: 1, loss: -0.70710, speed: 0.20 step/s
...
```


推理脚本：
```sh
python inference_reward_model.py
```

训练数据：

路径  `data/reward_datasets/sentiment_analysis`
每一行是一个排序序列（用\t符号隔开），排在越前面的越偏「正向情绪」，排在越后面越「负向情绪」。


预训练模型：[nghuyong/ernie-3.0-base-zh · Hugging Face](https://huggingface.co/nghuyong/ernie-3.0-base-zh)

