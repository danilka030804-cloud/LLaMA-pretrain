## LLaMA Pretrain

My implementation of pretraining the [LLaMA](https://arxiv.org/abs/2302.13971) model from scratch based on the original Meta AI paper on the Transformer architecture.

### Architecture

* Model: LLaMA (Decoder-only Transformer)

* Parameters: ~197 M

* Tokenizer: Mistral-7B-v0.1

* Context Window: 256 tokens


### Training Optimizations

* Attention Acceleration: sdp_kernel for efficient self-attention computation

* Mixed Precision: FP16 (float16) to accelerate computation and save memory

* Gradient Accumulation: For efficient memory usage with large batch sizes

* Profiling the training step for bottle neck detection

* Dynamic Padding: Splitting long sequences into 256-token blocks to minimize padding


### Data

1/8 of the [`OpenWebText`](https://skylion007.github.io/OpenWebTextCorpus/) dataset was used for training. The size was specifically chosen so that the model would only need to be trained for a single epoch.

### Training Results

* Data Volume: 250 million tokens

* Final Loss: < 5.0

* Training time: ~7 hours

<table align="center">
  <!-- Первая строка картинок -->
  <tr>
    <td align="center">
      <img src="images/train_loss.png" alt="Train loss" width="400"/>
      <br>
      <sub><b>Train loss</b></sub>
    </td>
    <td align="center">
      <img src="images/train_learning_rate.png" alt="Learning rate" width="400"/>
      <br>
      <sub><b>Learning rate</b></sub>
    </td>
  </tr>
  
  <!-- Отступ между строками -->
  <tr><td colspan="2" style="height: 30px;"></td></tr>
  
  <!-- Вторая строка картинок -->
  <tr>
    <td align="center">
      <img src="images/train_grad_norm.png" alt="Grad norm" width="400"/>
      <br>
      <sub><b>Grad norm</b></sub>
    </td>
    <td align="center">
      <img src="images/train_tokens_per_step.png" alt="Tokens per step" width="400"/>
      <br>
      <sub><b>Tokens per step</b></sub>
    </td>
  </tr>
</table>
