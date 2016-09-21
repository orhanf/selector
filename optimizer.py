import numpy as np
import theano

from theano import tensor


class Model(object):
    def __init__(self, **kwargs):
        self.__dict__.update(**kwargs)


def get_optimizer(opt_name):
    return eval(opt_name)


def rmsprop(lr, model, inp):
    tparams = model.params
    cost = model.cost
    grads = model.grads

    zipped_grads = [theano.shared(p.get_value() * np.float32(0.),
                                  name='%s_grad' % k)
                    for k, p in tparams.iteritems()]
    running_grads = [theano.shared(p.get_value() * np.float32(0.),
                                   name='%s_rgrad' % k)
                     for k, p in tparams.iteritems()]
    running_grads2 = [theano.shared(p.get_value() * np.float32(0.),
                                    name='%s_rgrad2' % k)
                      for k, p in tparams.iteritems()]

    zgup = [(zg, g) for zg, g in zip(zipped_grads, grads)]
    rgup = [(rg, 0.95 * rg + 0.05 * g) for rg, g in zip(running_grads, grads)]
    rg2up = [(rg2, 0.95 * rg2 + 0.05 * (g ** 2))
             for rg2, g in zip(running_grads2, grads)]

    updir = [theano.shared(p.get_value() * np.float32(0.),
                           name='%s_updir' % k)
             for k, p in tparams.iteritems()]
    updir_new = [(ud, 0.9 * ud - 1e-4 * zg / tensor.sqrt(rg2 - rg ** 2 + 1e-4))
                 for ud, zg, rg, rg2 in zip(updir, zipped_grads, running_grads,
                                            running_grads2)]
    param_up = [(p, p + udn[1]) for p, udn in zip(tparams.values(), updir_new)]
    f_update = theano.function(
        inputs=[lr] + inp, outputs=cost,
        updates=zgup+rgup+rg2up+updir_new+param_up,
        on_unused_input='ignore')

    return f_update


def uAdam(lr, model, inp, b1=0.9, b2=0.999, e=1e-8, return_alpha=False):
    tparams = model.params
    cost = model.cost
    grads = model.grads

    gshared = [theano.shared(p.get_value() * 0.,
                             name='%s_grad' % k)
               for k, p in tparams.iteritems()]
    gsup = [(gs, g) for gs, g in zip(gshared, grads)]

    updates = []
    step_rule_updates = []

    i = theano.shared(np.float32(0.), 'time')
    i_t = i + 1.
    fix1 = b1**(i_t)
    fix2 = b2**(i_t)
    lr_t = lr * (tensor.sqrt(1 - fix2) / (1 - fix1))

    for p, g in zip(tparams.values(), gshared):
        m = theano.shared(p.get_value() * 0., p.name + '_mean')
        v = theano.shared(p.get_value() * 0., p.name + '_variance')

        m_t = b1 * m + (1. - b1) * g
        v_t = b2 * v + (1. - b2) * g**2
        m_t_hat = m_t / (1 - fix1)
        v_t_hat = v_t / (1 - fix2)
        g_t = lr_t * m_t_hat / (tensor.sqrt(v_t_hat) + e)
        p_t = p - g_t

        updates.append((m, m_t))
        updates.append((v, v_t))
        updates.append((p, p_t))
        step_rule_updates.append(m)
        step_rule_updates.append(v)

    updates.append((i, i_t))
    step_rule_updates.append(i)

    outputs = cost
    if return_alpha:
        outputs = [cost, model.alpha]

    f_update = theano.function(
        inputs=[lr] + inp, outputs=outputs,
        updates=gsup+updates,
        on_unused_input='ignore')
    return f_update
