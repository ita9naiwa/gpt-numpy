import sys
import json
import struct
import numpy as np
import tensorflow as tf
from scipy.special import softmax


def gelu(x):
    return 0.5*x*(1+np.tanh(np.sqrt(2/np.pi)*(x+0.044715*np.power(x, 3))))

def linear(emb, w, b):
    return np.matmul(emb, w) + b[np.newaxis, :]

def normalize(x, g, b, eps=1e-10):
    mean = x.mean(axis=-1, keepdims=True)
    var = x.var(axis=-1, keepdims=True)
    sqrtd = np.sqrt(eps + var)
    h = (x - mean) / sqrtd
    return h * g[np.newaxis, :] + b[np.newaxis, :]

def attention_mask(nd, ns):
    i = np.arange(nd)[:,None]
    j = np.arange(ns)
    m = i >= j - ns + nd
    return m


class gpt2Model():
    def __init__(self):
        self.n_vocab = None
        self.n_ctx = None
        self.n_embd = None
        self.n_head = None
        self.n_layer = None
        self.k_memory = None
        self.v_memory = None
        self.tensors = None
        self.shapes = None
        self.eps = 1e-6

    def load(self, dir_model):
        print("loading hyper-parameters...")
        with open(dir_model + "/hparams.json", "r", encoding="utf-8") as f:
            hparams = json.load(f)
        self.n_vocab = hparams["n_vocab"]
        self.n_ctx = hparams["n_ctx"]
        self.n_embd = hparams["n_embd"]
        self.n_head = hparams["n_head"]
        self.n_layer = hparams["n_layer"]
        self.k_memory = np.zeros(shape=(self.n_layer, self.n_ctx, self.n_embd), dtype=np.float32)
        self.v_memory = np.zeros(shape=(self.n_layer, self.n_ctx, self.n_embd), dtype=np.float32)

        self.tensors = {}
        self.shapes = {}
        list_vars = tf.train.list_variables(dir_model)
        for tensor_name, shape in list_vars:
            print("loading varaible %s" % tensor_name)
            data = tf.train.load_variable(dir_model, tensor_name)
            transpose_list = ["/attn/c_attn/w", "/attn/c_proj/w", "/mlp/c_fc/w", "/mlp/c_proj/w"]
            if np.any([tensor_name.endswith(x) for x in transpose_list]):
                data = data.squeeze(0)
            shape = data.shape
            # print("processing %s\tshape: " % tensor_name, shape)
            tensor_name = tensor_name.split("model/")[1]
            self.tensors[tensor_name] = data
        self.shapes = {key: tensor.shape for (key, tensor) in self.tensors.items()}
        print("processing done!")

    def forward_pass(self, tokens, n_past):
        n_head = self.n_head
        n_embd = self.n_embd

        n = len(tokens)
        pos_indices = np.zeros(n, dtype=np.int32)
        for i in range(n):
            pos_indices[i] = n_past + i
        pos_emb = self.tensors["wpe"][pos_indices]
        token_emb = self.tensors["wte"][tokens]
        input_layer = token_emb + pos_emb

        for l in range(self.n_layer):
            """
            attn input layer normalization
            h%d/ln_1/g shape [n_embd]
            h%d/ln_1/b shape [n_embd]
            """
            h = normalize(input_layer, self.tensors["h%d/ln_1/g" % l], self.tensors["h%d/ln_1/b" % l])

            """
            self attention layer
            h%d/attn/c_attn/w [3 * n_embd, n_embd]
            h%d/attn/c_attn/b [3 * n_embd]
            """
            attn = linear(h, self.tensors["h%d/attn/c_attn/w" % l], self.tensors["h%d/attn/c_attn/b" % l])
            Q =      attn[:, 0 * n_embd:1 * n_embd]
            K_curr = attn[:, 1 * n_embd:2 * n_embd]
            V_curr = attn[:, 2 * n_embd:3 * n_embd]
            self.k_memory[l, n_past:n_past + n, :] = K_curr
            self.v_memory[l, n_past:n_past + n, :] = V_curr

            # Q shape: [n_head, n, dim]
            Q = Q.reshape(n, n_head, n_embd // n_head).transpose((1, 0, 2))
            # K shape: [n_head, n_past + n, dim]
            K = self.k_memory[l, :n_past + n, :].reshape(n_past + n, n_head, n_embd // n_head).transpose(1, 0, 2)
            # V shape: [n_head, n_past + n, dim]
            V = self.v_memory[l, :n_past + n, :].reshape(n_past + n, n_head, n_embd // n_head).transpose(1, 0, 2)
            W = np.einsum("hnd,hpd->hnp", Q, K) / np.sqrt(V.shape[-1])
            mask = attention_mask(W.shape[1], W.shape[2])
            W = W - (1.0 - mask) * 1e10
            W = softmax(W, axis=-1)
            h = np.einsum("hab,hby->hay", W, V).transpose((1, 0, 2)).reshape(n, -1)

            # projection layer
            """
                h%d/attn/c_proj/w [n_embd, n_embd]
                h%d/attn/c_proj/b [n_embd]
            """
            h = linear(h, self.tensors["h%d/attn/c_proj/w" % l], self.tensors["h%d/attn/c_proj/b" % l])

            # residual connection
            input_ff = input_layer + h
            """
            feed-foward input normalization
            h%d/ln_1/g shape [n_embd]
            h%d/ln_1/b shape [n_embd]
            """
            h = normalize(input_ff, self.tensors["h%d/ln_2/g" % l],  self.tensors["h%d/ln_2/b" % l])

            """
            feed-forward layer
            h%d/mlp/c_fc/w [n_embd, n_embd * 3]
            h%d/mlp/c_fc/b [n_embd * 3]
            h%d/mlp/c_proj/w [n_embd * 3, n_embd]
            h%d/mlp/c_proj/b [n_embd]
            """
            h = linear(h, self.tensors["h%d/mlp/c_fc/w" % l], self.tensors["h%d/mlp/c_fc/b" % l])
            h = gelu(h)
            h = linear(h, self.tensors["h%d/mlp/c_proj/w" % l], self.tensors["h%d/mlp/c_proj/b" % l])
            input_layer = h + input_ff

        emb = input_layer.copy()
        # final normalization
        emb = normalize(emb, self.tensors["ln_f/g"], self.tensors["ln_f/b"] )
        # head is tied with wte in gpt-2 model
        lm_head = self.tensors["wte"].T
        logits = np.matmul(emb[-1], lm_head)
        return logits

if __name__ == "__main__": #test
    dir_model = "./models/gpt-2-117M"
    model = gpt2Model()
    model.load(dir_model)