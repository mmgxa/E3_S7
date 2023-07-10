from typing import Any, Optional

from lightning import LightningModule

import torch
from torch import nn
from einops.layers.torch import Rearrange
from torch.nn import functional as F

from torchmetrics import MinMetric, MeanMetric
from torchmetrics.classification.accuracy import Accuracy

import tiktoken


cl100k_base = tiktoken.get_encoding("cl100k_base")

# In production, load the arguments directly instead of accessing private attributes
# See openai_public.py for examples of arguments for specific encodings
enc = tiktoken.Encoding(
    # If you're changing the set of special tokens, make sure to use a different name
    # It should be clear from the name what behaviour to expect.
    name="cl100k_im",
    pat_str=cl100k_base._pat_str,
    mergeable_ranks=cl100k_base._mergeable_ranks,
    special_tokens={
        **cl100k_base._special_tokens,
        "<|im_start|>": 100264,
        "<|im_end|>": 100265,
    }
)


class MultiHeadAttention(nn.Module):
    def __init__(self, n_heads, n_dim, dropout):
        super(MultiHeadAttention, self).__init__()

        self.n_heads = n_heads
        self.n_dim = n_dim
        self.h_dim = n_dim // n_heads

        self.keys = nn.Sequential(
            nn.Linear(n_dim, self.h_dim * self.n_heads),
            Rearrange('b time (nh dim) -> nh b time dim', nh=self.n_heads)
        )
        self.queries = nn.Sequential(
            nn.Linear(n_dim, self.h_dim * self.n_heads),
            Rearrange('b time (nh dim) -> nh b time dim', nh=self.n_heads)
        )
        self.values = nn.Sequential(
            nn.Linear(n_dim, self.h_dim * self.n_heads),
            Rearrange('b time (nh dim) -> nh b time dim', nh=self.n_heads)
        )

        self.proj = nn.Linear(n_dim, n_dim)
        self.layer_norm = nn.LayerNorm(n_dim)
        self.attn_dropout = nn.Dropout(p=dropout)

        self.rearrange_out = Rearrange(
            'nh b vt dim -> b vt (nh dim)'
        )

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None):
        key = self.keys(x)
        query = self.queries(x)
        value = self.values(x)

        energies = torch.einsum(
            'nbqd,nbkd->nbqk',
            query,
            key,
        )

        if mask is not None:
            fill_value = 1e-20
            energies = energies.masked_fill(mask, fill_value)

        attn = F.softmax(energies, dim=-1)
        attn = self.attn_dropout(attn)

        out = torch.einsum(
            'nbqk,nbkd->nbqd',
            attn,
            value,
        )

        out = self.rearrange_out(out)
        out = self.proj(out)

        return out

class ResidualAdd(nn.Module):
    def __init__(self, fn):
        super(ResidualAdd, self).__init__()
        self.fn = fn

    def forward(self, x):
        res = x
        out = self.fn(x)
        out += res
        return out
    
class FeedForwardBlock(nn.Sequential):
    def __init__(self, emb_size, drop_p, expansion = 4):
        super(FeedForwardBlock, self).__init__(
            nn.Linear(emb_size, expansion * emb_size),
            nn.GELU(),
            nn.Dropout(drop_p),
            nn.Linear(expansion * emb_size, emb_size)
        )

class GPTDecoderBlock(nn.Module):
    def __init__(
        self,
        emb_size = 768,
        drop_p = 0.0,
        forward_expansion = 4,
        n_heads=4
    ):
        super(GPTDecoderBlock, self).__init__()

        self.ln = nn.LayerNorm(emb_size)
        self.mha = MultiHeadAttention(n_heads=n_heads, n_dim=emb_size, dropout=drop_p)
        self.drop = nn.Dropout(drop_p)

        self.out_block = ResidualAdd(
            nn.Sequential(
                nn.LayerNorm(emb_size),
                FeedForwardBlock(emb_size=emb_size, drop_p=drop_p, expansion=forward_expansion),
                nn.Dropout(drop_p)
            )
        )

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None):
        residual = x

        out = self.ln(x)
        out = self.mha(out, mask)
        out = self.drop(out)
        out = x + out
        out = self.out_block(out)

        return out

class GPT(nn.Module):
    def __init__(self, vocab_size, block_size, n_embed, n_heads, drop_p, n_decoder_blocks):
        super(GPT, self).__init__()

        self.block_size = block_size

        self.token_embedding_table = nn.Embedding(vocab_size, n_embed)
        self.position_embedding_table = nn.Embedding(block_size, n_embed)
        self.blocks = nn.ModuleList()
        for _ in range(n_decoder_blocks):
            self.blocks.append(GPTDecoderBlock(emb_size=n_embed, n_heads=n_heads, drop_p=drop_p))

        self.ln = nn.LayerNorm(n_embed)
        self.ffwd = FeedForwardBlock(emb_size=n_embed, drop_p=drop_p)
        self.lm_head = nn.Linear(n_embed, vocab_size)

        # query: what am i looking for?
        # key: what do i contain?

    def forward(self, idx: torch.Tensor, targets: Optional[torch.Tensor]=None, mask: Optional[torch.Tensor]=None):
        B, T = idx.shape

        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(torch.arange(T, device=idx.device))
        x = tok_emb + pos_emb
        for block in self.blocks:
            x = block(x, mask)
        x = self.ln(x)
        x = self.ffwd(x)
        logits = self.lm_head(x)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)
        return logits, loss

    @torch.jit.export
    def generate(self, idx: torch.Tensor, max_new_tokens: int, temperature: float = 1.0, top_k: Optional[int] = None):
        for _ in range(max_new_tokens):
            # if the sequence context is growing too long we must crop it at block_size
            idx_cond = idx if idx.size(1) <= self.block_size else idx[:, -self.block_size:]
            # forward the model to get the logits for the index in the sequence
            logits, _ = self(idx_cond, targets=None, mask=None)
            # pluck the logits at the final step and scale by desired temperature
            logits = logits[:, -1, :] / temperature
            # optionally crop the logits to only the top k options
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            # apply softmax to convert logits to (normalized) probabilities
            probs = F.softmax(logits, dim=-1)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)
            # append sampled index to the running sequence and continue
            idx = torch.cat((idx, idx_next), dim=1)

        return idx

class GPTLitModule(LightningModule):
    def __init__(
        self,
        learning_rate,
        n_embed,
        block_size,
        n_heads,
        drop_p,
        n_decoder_blocks,
        
    ):
        super().__init__()

        self.n_embed = n_embed
        self.block_size = block_size
        self.n_heads = n_heads
        self.drop_p = drop_p
        self.n_decoder_blocks = n_decoder_blocks
        self.learning_rate = learning_rate

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        # ignoring net as the model weights themselves are not a hyperparam
        self.save_hyperparameters(logger=False, ignore=['model'])

        self.model = GPT(
            vocab_size=enc.n_vocab,
            block_size=self.hparams.block_size,
            n_embed=self.hparams.n_embed,
            n_heads=self.hparams.n_heads,
            drop_p=self.hparams.drop_p,
            n_decoder_blocks=self.hparams.n_decoder_blocks
        )
       
        # for averaging loss across batches
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()

        # for tracking best so far validation loss
        self.val_loss_best = MinMetric()

        self.register_buffer("mask", torch.tril(torch.ones(block_size, block_size)) == 0)

    def forward(self, x: torch.Tensor, targets: Optional[torch.Tensor] = None):
        mask = self.mask if targets is not None else None
        return self.model(x, targets=targets, mask=mask)

    def on_train_start(self):
        # by default lightning executes validation step sanity checks before training starts,
        # so it's worth to make sure validation metrics don't store results from these checks
        self.val_loss.reset()
        self.val_loss_best.reset()

    def model_step(self, batch: Any):
        x, y = batch
        logits, loss = self.forward(x, targets=y)
        return loss

    def training_step(self, batch: Any, batch_idx: int):
        loss = self.model_step(batch)

        # update and log metrics
        self.train_loss(loss)
        self.log("train/loss", self.train_loss,
                 on_step=False, on_epoch=True, prog_bar=True)

        # return loss or backpropagation will fail
        return loss

    def on_train_epoch_end(self):
        pass

    def validation_step(self, batch: Any, batch_idx: int):
        loss = self.model_step(batch)

        # update and log metrics
        self.val_loss(loss)
        self.log("val/loss", self.val_loss, on_step=False,
                 on_epoch=True, prog_bar=True)

    def on_validation_epoch_end(self):
        loss = self.val_loss.compute()  # get current val loss
        self.val_loss_best(loss)  # update best so far val loss
        # log `val_acc_best` as a value through `.compute()` method, instead of as a metric object
        # otherwise metric would be reset by lightning after each epoch
        self.log("val/loss_best", self.val_loss_best.compute(), sync_dist=True, prog_bar=True)

    def configure_optimizers(self):
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        Examples:
            https://lightning.ai/docs/pytorch/latest/common/lightning_module.html#configure-optimizers
        """
        optimizer = torch.optim.Adam(params=self.parameters(), lr=self.learning_rate)

        return {"optimizer": optimizer}

