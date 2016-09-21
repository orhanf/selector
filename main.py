import theano
from theano import tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

from layers import MultiLayer, Merger
from optimizer import Model, get_optimizer
from training import train

floatX = theano.config.floatX


def merge_dicts(*dict_args):
    result = {}
    for dictionary in dict_args:
        result.update(dictionary)
    return result


def build_model(xl, xr, y, learning_rate, linput_dim, rinput_dim, llayer_dims,
                rlayer_dims, tlayer_dims, merged_layer_dim=None, lbranch=None,
                rbranch=None, tbranch=None, trng=None, **kwargs):

    print('Building training model')
    use_noise = True
    alpha = None
    op = 'weighted-sum'
    f_l = MultiLayer(linput_dim, llayer_dims, trng=trng, **lbranch)  # left
    f_r = MultiLayer(rinput_dim, rlayer_dims, trng=trng, **rbranch)  # right
    f_t = MultiLayer(merged_layer_dim, tlayer_dims, trng=trng, **tbranch)
    merger = Merger(merged_layer_dim, op=op)  # merger layer

    fl = f_l.fprop(xl, use_noise=use_noise)
    fr = f_r.fprop(xr, use_noise=use_noise)

    fl_only = merger.fprop([fl], use_noise=use_noise)
    fr_only = merger.fprop([fr], use_noise=use_noise)
    f_both = merger.fprop([fl, fr], axis=1, use_noise=use_noise)
    if op == 'weighted-sum':
        f_both, alpha = f_both

    yl_probs = T.nnet.softmax(f_t.fprop(fl_only, use_noise=use_noise))
    yr_probs = T.nnet.softmax(f_t.fprop(fr_only, use_noise=use_noise))
    yb_probs = T.nnet.softmax(f_t.fprop(f_both, use_noise=use_noise))

    cost_l = T.nnet.categorical_crossentropy(yl_probs, y).mean()
    cost_r = T.nnet.categorical_crossentropy(yr_probs, y).mean()
    cost_b = T.nnet.categorical_crossentropy(yb_probs, y).mean()

    cost_l.name = 'cost_l'
    cost_r.name = 'cost_r'
    cost_b.name = 'cost_b'

    params_l = merge_dicts(f_l.get_params(),
                           f_t.get_params())
    params_r = merge_dicts(f_r.get_params(),
                           f_t.get_params())
    params_b = merge_dicts(f_l.get_params(),
                           f_r.get_params(),
                           f_t.get_params(),
                           merger.get_params())

    print('Computing gradients')
    grads_l = [theano.grad(cost_l, p) for k, p in params_l.items()]
    grads_r = [theano.grad(cost_r, p) for k, p in params_r.items()]
    grads_b = [theano.grad(cost_b, p) for k, p in params_b.items()]

    print('Building validation model')
    fl = f_l.fprop(xl, use_noise=False)
    fr = f_r.fprop(xr, use_noise=False)

    fl_only = merger.fprop([fl], use_noise=False)
    fr_only = merger.fprop([fr], use_noise=False)
    f_both = merger.fprop([fl, fr], op='sum', axis=1, use_noise=False)
    if op == 'weighted-sum':
        f_both, alpha_val = f_both

    yl_probs_val = T.nnet.softmax(f_t.fprop(fl_only, use_noise=False))
    yr_probs_val = T.nnet.softmax(f_t.fprop(fr_only, use_noise=False))
    yb_probs_val = T.nnet.softmax(f_t.fprop(f_both, use_noise=False))

    acc_l = T.mean(T.eq(T.argmax(yl_probs_val, axis=1), y), dtype=floatX)
    acc_r = T.mean(T.eq(T.argmax(yr_probs_val, axis=1), y), dtype=floatX)
    acc_b = T.mean(T.eq(T.argmax(yb_probs_val, axis=1), y), dtype=floatX)

    model_l = Model(cost=cost_l, params=params_l, grads=grads_l, acc=acc_l)
    model_r = Model(cost=cost_r, params=params_r, grads=grads_r, acc=acc_r)
    model_b = Model(cost=cost_b, params=params_b, grads=grads_b, acc=acc_b, alpha=alpha)

    return model_l, model_r, model_b


options = {
    'lbatch_sz': 128,
    'rbatch_sz': 128,
    'bbatch_sz': 128,
    'nb_classes': 10,
    'linput_dim': 392,
    'rinput_dim': 392,
    'llayer_dims': [512, 512],
    'rlayer_dims': [512, 512],
    'tlayer_dims': [512, 10],
    'merged_layer_dim': 512,
    'lbranch': {'dropout': [0.5, 0.2], 'activ': ['relu', 'relu'], 'prefix': 'left'},
    'rbranch': {'dropout': [0.5, 0.2], 'activ': ['relu', 'relu'], 'prefix': 'right'},
    'tbranch': {'dropout': [0.2, None], 'activ': ['relu', 'linear'], 'prefix': 'top'},
    'lr': .0001,
    'optimizer': 'uAdam',
    'num_epochs': 200
}


def main():
    # spawn theano vars
    xs = [T.imatrix('x%d' % i) for i in range(options['max_src'])]
    y = T.ivector('y')
    learning_rate = T.scalar('learning_rate')
    trng = RandomStreams(4321)

    # use test values
    """
    import numpy as np
    batch_size = 10
    theano.config.compute_test_value = 'raise'
    xl.tag.test_value = np.random.randn(batch_size, 392).astype(floatX)
    xr.tag.test_value = np.random.randn(batch_size, 392).astype(floatX)
    y.tag.test_value = np.random.randint(8, size=batch_size).astype(np.int32)
    learning_rate.tag.test_value = 0.5
    """

    # build cgs
    model = build_model(
        xs, y, learning_rate, trng=trng,
        **options)

    # compile
    opt = get_optimizer(options['optimizer'])
    f_train = opt(learning_rate, model, xs + [y], return_alpha=True)

    # compile validation/test functions
    f_valid = theano.function(
        xs + [y], [model.cost, model.acc], on_unused_input='warn')

    # training loop
    train(f_train,  f_valid,
          xs, y, **options)

if __name__ == "__main__":
    main()
