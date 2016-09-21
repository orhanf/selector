import numpy
import time

from data_iterator import load_dataset, iterate_minibatches


def train(f_train_l, f_train_r, f_train_b,
          f_valid_l, f_valid_r, f_valid_b,
          xl, xr, y, lr=1., num_epochs=20,
          lbatch_sz=128, rbatch_sz=128, bbatch_sz=128,
          op='weighted-sum'
          **kwargs):

    # load data
    X_train, y_train, X_val, y_val, X_test, y_test = load_dataset()

    print("Starting training...")
    for epoch in range(num_epochs):

        train_err_l = 0
        train_err_r = 0
        train_err_b = 0
        tr_batches = 0
        alphas = []
        start_time = time.time()
        for lbatch, rbatch, bbatch in zip(
                iterate_minibatches(X_train, y_train, lbatch_sz, shuffle=True),
                iterate_minibatches(X_train, y_train, rbatch_sz, shuffle=True),
                iterate_minibatches(X_train, y_train, bbatch_sz, shuffle=True)
                ):
            _train_err_b, alpha = f_train_b(lr, bbatch[0], bbatch[1], bbatch[2])
            train_err_b += _train_err_b
            alphas.append(alpha)
            train_err_r += f_train_r(lr, rbatch[1], rbatch[2])
            train_err_l += f_train_l(lr, lbatch[0], lbatch[2])
            tr_batches += 1

        # And a full pass over the validation data:
        lval_err = 0
        rval_err = 0
        bval_err = 0
        lval_acc = 0
        rval_acc = 0
        bval_acc = 0
        val_batches = 0
        for batch in iterate_minibatches(
                X_val, y_val, 500, shuffle=False):
            lbatch, rbatch, targets = batch
            lerr, lacc = f_valid_l(lbatch, targets)
            rerr, racc = f_valid_r(rbatch, targets)
            berr, bacc = f_valid_b(lbatch, rbatch, targets)
            lval_err += lerr
            rval_err += rerr
            bval_err += berr
            lval_acc += lacc
            rval_acc += racc
            bval_acc += bacc
            val_batches += 1

        # Then we print the results for this epoch:
        print(("Epoch {:>4} of {} took {:.3f}s" +
               "  train_loss - l:[{:.6f}] r:[{:.6f}] b:[{:.6f}] " +
               "  valid_loss - l:[{:.6f}] r:[{:.6f}] b:[{:.6f}] " +
               "  valid_acc - l:[{:.2f} %] r:[{:.2f} %] b:[{:.2f} %] alphas:[{}]").format(
            epoch + 1, num_epochs, time.time() - start_time,
            train_err_l / tr_batches, train_err_r / tr_batches,
            train_err_b / tr_batches, lval_err / val_batches,
            rval_err / val_batches, bval_err / val_batches,
            lval_acc / val_batches * 100,
            rval_acc / val_batches * 100,
            bval_acc / val_batches * 100,
            (numpy.vstack([aa.mean(1) for aa in alphas]).mean(0))))

    # After training, we compute and print the test error:
    ltest_err = 0
    rtest_err = 0
    btest_err = 0
    ltest_acc = 0
    rtest_acc = 0
    btest_acc = 0
    test_batches = 0
    for batch in iterate_minibatches(X_test, y_test, 500, shuffle=False):
        lbatch, rbatch, targets = batch
        lerr, lacc = f_valid_l(lbatch, targets)
        rerr, racc = f_valid_r(rbatch, targets)
        berr, bacc = f_valid_b(lbatch, rbatch, targets)
        ltest_err += lerr
        rtest_err += rerr
        btest_err += berr
        ltest_acc += lacc
        rtest_acc += racc
        btest_acc += bacc
        test_batches += 1
    print("Final results:")
    print("  l test loss:\t\t\t{:.6f}"
          .format(ltest_err / test_batches))
    print("  r test loss:\t\t\t{:.6f}"
          .format(rtest_err / test_batches))
    print("  b test loss:\t\t\t{:.6f}"
          .format(btest_err / test_batches))
    print("  l test accuracy:\t\t{:.2f} %"
          .format(ltest_acc / test_batches * 100))
    print("  r test accuracy:\t\t{:.2f} %"
          .format(rtest_acc / test_batches * 100))
    print("  b test accuracy:\t\t{:.2f} %"
          .format(btest_acc / test_batches * 100))
