<div align="center">

mechanical offering: an endlessly rising repository

![File:Adolph Menzel - Flötenkonzert Friedrichs des Großen in Sanssouci - Google Art Project.jpg](https://upload.wikimedia.org/wikipedia/commons/thumb/6/64/Adolph_Menzel_-_Fl%C3%B6tenkonzert_Friedrichs_des_Gro%C3%9Fen_in_Sanssouci_-_Google_Art_Project.jpg/800px-Adolph_Menzel_-_Fl%C3%B6tenkonzert_Friedrichs_des_Gro%C3%9Fen_in_Sanssouci_-_Google_Art_Project.jpg)

</div>

(inspired by Bach's [Endlessly Rising Canon](https://en.wikipedia.org/wiki/The_Musical_Offering#Canon_per_tonos_(endlessly_rising_canon)), as well as Douglas Hofstadter's [Gödel, Escher, Bach: an Eternal Golden Braid](https://en.wikipedia.org/wiki/G%C3%B6del,_Escher,_Bach). image from [Wikipedia](https://en.wikipedia.org/wiki/File:Adolph_Menzel_-_Fl%C3%B6tenkonzert_Friedrichs_des_Gro%C3%9Fen_in_Sanssouci_-_Google_Art_Project.jpg))

---

Notes:
- "Run and Debug" mode in VScode causes an error at the end of programs that use tinygrad, but normal VScode running or command line python running work normally
- tinygrad optimizers use Tensor.assign(self, x) which can only assign floats to x for some reason, so all params need to be float (???)
- `pip install datasets` for huggingface datasets
- `pip install tiktoken` for tiktoken tokenizer library
- `pip install flash-attn --no-build-isolation` for [flash-attention](https://github.com/Dao-AILab/flash-attention)
- `datasets/shakespeare` train shard is a renamed copy of val shard
- restarting computer helps free up gpu vram
- `nvidia-smi` to monitor gpu usage
- gitignore just stops working for some files/directories sometimes, and recloning the repo from github and moving all the old files over EXCEPT .GIT FOLDER fixes the problem (???)

Recipe to convert Transformer to normalized Transformer [[source](https://arxiv.org/abs/2410.01131)] [[source](https://github.com/NVIDIA/ngpt)]:
1. Remove all normalization layers (RMSNorm, LayerNorm, etc.), weight decay, and learning rate warmup.
2. After each training step, normalize all matrices ($E_{input} , E_{output} , W_q , W_k , W_v , W_o , W_{uv}, W_{oMLP}$) along their embedding dimension. ($W_{uv}$ and $W_{oMLP}$ are ff1 and ff2 for me)
3. Replace residual (aka skip) connection equations:\
$h ← h + ATTN(RMSNorm(h)),$\
$h ← h + MLP(RMSNorm(h))$ with\
$h ← Norm( h + α_A (Norm(h_A) − Norm(h)) ),$\
$h ← Norm( h + α_M (Norm(h_M) − Norm(h)) )$\
where $α_A$ (and also $α_M$ ) is treated with $α_{A,init} = 0.05$ (in order of $1/nlayers$ ) and $α_{A,scale} = 1/ \sqrt{d_{model}}$.
4. Change the softmax scaling factor in attention from $1/ \sqrt{d_k}$ to $\sqrt{d_k}$ . Implement scaling and normalization of attention:\
$q ← Norm(q)s_{qk}$\
$k ← Norm(k)s_{qk}$,\
where $s_{qk}$ is treated with $s_{qk,init} = 1$ and $s_{qk,scale} = 1/ dmodel$ .
5. Implement scaling of MLP:\
$uv ← uvs_{uv}\sqrt{d_{model}}$\
where $s_{uv}$ is treated with $s_{uv,init} = 1$ and $s_{uv,scale} = 1$ .
1. Implement the rescaling of output logits\
$z ← zs_z$\
where $s_z$ is treated with $s_{z,init} = 1$ and
$s_{z,scale} = 1/ dmodel$ .