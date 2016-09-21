import numpy as np
import theano
from theano import tensor

floatX = theano.config.floatX
np.random.seed(4444)


def tanh(x):
    return tensor.tanh(x)


def relu(x):
    return tensor.nnet.relu(x)


def linear(x):
    return x


def init_param(inp_size, out_size=None, name='W', scale=0.01, ortho=False):
    if out_size is None:
        out_size = inp_size
    if ortho and inp_size == out_size:
        u, s, v = np.linalg.svd(np.random.randn(inp_size, inp_size))
        W = u.astype('float32')
    else:
        W = scale * np.random.randn(inp_size, out_size).astype(floatX)
    return theano.shared(W, name=name)


def init_bias(layer_size, name):
    return theano.shared(np.zeros(layer_size, dtype=floatX), name=name)


def _p(p, q, r):
    return '{}_{}_{}'.format(p, q, r)


class DropoutLayer(object):
    def __init__(self, p, trng):
        self.p = p
        self.trng = trng

    def fprop(self, x):
        retain_prob = 1. - self.p
        x *= self.trng.binomial(x.shape, p=retain_prob, dtype=x.dtype)
        x /= retain_prob
        return x


class Sequence(object):
    def __init__(self, layers=None):
        self.layers = layers
        if layers is None:
            self.layers = list()

    def fprop(self, inp, **kwargs):
        z = inp
        for i, layer in enumerate(self.layers):
            z = layer(z, **kwargs)
        return z

    def get_params(self):
        params = []
        for layer in self.layers:
            params += layer.get_params()
        return params

    def add(self, layer):
        if isinstance(layer, list):
            self.layers += layer
        else:
            self.layers.append(layer)


class DenseLayer(object):
    def __init__(self, nin, dim, activ=None, prefix='ff',
                 postfix='0', scale=0.01, ortho=False, add_bias=True,
                 dropout=0., trng=None):
        self.activ = 'lambda x: tensor.tanh(x)' if activ is None else activ
        self.add_bias = add_bias
        self.W = init_param(nin, dim, _p(prefix, 'W', postfix),
                            scale=scale, ortho=ortho)
        if add_bias:
            self.b = init_bias(dim, _p(prefix, 'b', postfix))

        self.dropout_layer = None
        if dropout > 0. and dropout is not None:
            self.dropout_layer = DropoutLayer(dropout, trng)

    def fprop(self, state_below, use_noise=False):
        pre_act = tensor.dot(state_below, self.W) + \
            (self.b if self.add_bias else 0.)
        z = eval(self.activ)(pre_act)
        if use_noise and self.dropout_layer is not None:
            z = self.dropout_layer.fprop(z)
        return z

    def get_params(self):
        params = {self.W.name: self.W}
        if self.add_bias:
            params[self.b.name] = self.b
        return params


class MultiLayer(object):
    def __init__(self, nin, dims, dropout=None, activ=None, **kwargs):
        if dropout is None:
            dropout = [None for _ in range(len(dims))]
        if activ is None:
            activ = [None for _ in range(len(dims))]
        self.layers = []
        for i, dim in enumerate(dims):
            self.layers.append(
                DenseLayer(nin, dim, postfix=i, dropout=dropout[i], activ=activ[i],
                           **kwargs))
            nin = dim

    def fprop(self, inp, **kwargs):
        for i, layer in enumerate(self.layers):
            inp = layer.fprop(inp, **kwargs)
        return inp

    def get_params(self):
        params = {}
        for layer in self.layers:
            params.update(**layer.get_params())
        return params


class AttentionLayer(object):
    def __init__(self, dim, att_dim=100, prefix='att', **kwargs):
        self.dim = dim
        self.U = init_param(dim, 1, name=_p(prefix, 'U', 0))
        self.W = init_param(dim, name=_p(prefix, 'W', 0))
        self.b = init_bias(dim, name=_p(prefix, 'b', 0))
        self.params = [self.U, self.W, self.b]

    def fprop(self, inps, **kwargs):
        if len(inps) == 1:
            return inps[0]
        tbd = tensor.stack(inps)  # time x batch x dim
        ptbd = tensor.dot(tbd, self.W) + self.b
        alpha = tensor.dot(ptbd, self.U)
        alpha = alpha.reshape([alpha.shape[0], alpha.shape[1]])
        alpha = tensor.exp(alpha)
        alpha = alpha / alpha.sum(0, keepdims=True)
        wa = (tbd * alpha[:, :, None]).sum(0)  # weighted average
        return wa, alpha

    def get_params(self):
        return {p.name: p for p in self.params}


class TreeLayer(object):
    def __init__(self, dims, prefix='tree', **kwargs):
        self.dims = dims
        self.W = init_param(dims, name=_p(prefix, 'W', 0))
        self.b = init_bias(dims, name=_p(prefix, 'b', 0))
        self.params = [self.U, self.W, self.b]

    def fprop(self, inps, **kwargs):
        if len(inps) == 1:
            return inps[0]
        tbd = tensor.stack(inps)  # time x batch x dim
        ptbd = tensor.dot(tbd, self.W) + self.b
        alpha = tensor.dot(ptbd, self.U)
        alpha = alpha.reshape([alpha.shape[0], alpha.shape[1]])
        alpha = tensor.exp(alpha)
        alpha = alpha / alpha.sum(0, keepdims=True)
        wa = (tbd * alpha[:, :, None]).sum(0)  # weighted average
        return wa, alpha

    def get_params(self):
        return {p.name: p for p in self.params}


class Merger(object):
    def __init__(self, dims, op='sum', **kwargs):
        self.dims = dims
        self.op = op
        self.params = {}
        self.mlp = None
        if op == 'attention-sum':
            self.mlp = AttentionLayer(dims, prefix='att', **kwargs)
            self.params.update(self.mlp.get_params())
        elif op == 'weighted-sum':
            self.mlp = TreeLayer(dims, prefix='tree', **kwargs)
            self.params.update(self.mlp.get_params())

    def fprop(self, inps, *args, **kwargs):
        return self._merge(inps, *args, **kwargs)

    def _merge(self, inps, axis=0, **kwargs):
        if self.op == 'sum':
            merged = inps[0]
            for i in range(1, len(inps)):
                merged += inps[i]
            return merged
        elif self.op == 'attention-sum':
            if len(inps) == 1:
                return inps[0]
            merged, alpha = self.mlp.fprop(inps, **kwargs)
            return merged, alpha
        elif self.op == 'weighted-sum':
            pass
        else:
            raise ValueError("Unrecognized merge operation!")

    def get_params(self):
        return self.params


class Selector(object):
    def __init__(self, dims, op='mean', **kwargs):
        self.dims = dims
        self.op = op

    def fprop(self, inps, *args, **kwargs):
        pass
