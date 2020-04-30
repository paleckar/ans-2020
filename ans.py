import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch
from torch.nn import functional as F
from torchvision import transforms
from tqdm.auto import tqdm


class Layer(object):

    def __init__(self):
        super().__init__()
        self.params = {}
        self.training = True
        self._cache = None
    
    def forward(self, *args, **kwargs):
        raise NotImplementedError
    
    def backward(self, *args, **kwargs):
        raise NotImplementedError

    def parameters(self):
        return (par for name, par in self.params.items())
    
    def train(self):
        self.training = True
    
    def eval(self):
        self.training = False

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)
    
    def __getattr__(self, key):
        return self.__dict__.get(key, self.params[key])


class SGD(object):

    def __init__(self, params, learning_rate=1e-3, weight_decay_l2=1e-3):
        """
        params ... slovnik mapujici jmena parametru na jejich hodnoty
        """
        super().__init__()
        self.params = params
        self.learning_rate = learning_rate
        self.weight_decay_l2 = weight_decay_l2
    
    def step(self, grads):
        """
        Updatuje parametry gradienty specifikovanimi argumentem `grads`.

        grads ... slovnik mapujici jmena parametru na jejich gradienty
        """

        for name, grad in grads.items():
            # pokud regularizace, pak upravime gradient
            if self.weight_decay_l2 > 0.:
                grad = grad + 2. * self.weight_decay_l2 * self.params[name]
            
            # stochastic gradient descent update
            self.params[name] -= self.learning_rate * grad


class BatchLoader(object):
    
    def __init__(self, X_data, y_data, batch_size, subset_name, shuffle=True):
        super().__init__()
        
        self.X_data = X_data
        self.y_data = y_data
        self.batch_size = batch_size
        self.subset_name = subset_name
        self.shuffle = shuffle
    
    def __iter__(self):
        # V prvnim cviceni jsme vybirali vzorky do batche nahodne s opakovanim,
        # neboli behem jedne epochy se mohly nektere obrazky opakovat, zatimco
        # na jine se vubec nedostalo. Zde budeme prochazet obrazky opet v nahodnem
        # poradi, ovsem tak, ze za jednu epochu uvidime kazdy obrazek prave jednou.
        # Toho docilime tak, ze data pred pruchodem nahodne zprehazime.
        if self.shuffle:
            perm = torch.randperm(self.X_data.shape[0])
        else:
            perm = torch.arange(self.X_data.shape[0])
        
        for n in range(len(self)):
            batch_ids = perm[n*self.batch_size : (n+1)*self.batch_size]
            yield self.X_data[batch_ids], self.y_data[batch_ids]
        
    def __len__(self):
        return self.X_data.shape[0] // self.batch_size


class Stats(object):
    
    def __init__(self, smooth_coef=0.99):
        super().__init__()
        
        self._epochs = []
        self._running_avg = {}
        self.smooth_coef = smooth_coef
    
    def __len__(self):
        return len(self._epochs)
    
    def append_batch_stats(self, subset, **stats):
        if subset not in self._epochs[-1]:
            self._epochs[-1][subset] = {}
        for k, v in stats.items():
            if k not in self._epochs[-1][subset]:
                self._epochs[-1][subset][k] = []
            self._epochs[-1][subset][k].append(v)

        if subset not in self._running_avg:
            self._running_avg[subset] = {}
        for k, v in stats.items():
            self._running_avg[subset][k] = self.smooth_coef * self._running_avg[subset].get(k, v) + (1. - self.smooth_coef) * v
    
    def new_epoch(self):
        self._epochs.append({})
        
    def epoch_average(self, epoch, subset, metric):
        try:
            return np.mean([s for s in self._epochs[epoch][subset][metric]])
        except (KeyError, TypeError):
            return np.nan
    
    def ravg(self, subset, metric):
        return self._running_avg[subset][metric]
      
    def summary(self, epoch=-1):
        epoch = self._epochs.index(self._epochs[epoch])
        
        res = {}
        for subset, stats in self._epochs[epoch].items():
            for metric in stats:
                if metric not in res:
                    res[metric] = {}
                res[metric][subset] = self.epoch_average(epoch, subset, metric)
        
        return pd.DataFrame(res).rename_axis('Epoch {:02d}'.format(epoch + 1), axis='columns')
    
    def best_epoch(self, key=None):
        if key is None:
            key = lambda i: self.epoch_average(i, 'valid', 'acc')
        return max(range(len(self)), key=key)
    
    def best_results(self, key=None):
        return self.summary(self.best_epoch(key=key))

    def _plot(self, xdata, yleft, yright, ax=None, xlabel='batch', ylabels=None, **kwargs):
        """
        vykresli prubeh lossu a prip. do stejneho grafu zakresli i prubeh acc
        """
        
        if ax is None:
            fig, ax = plt.subplots()

        colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
        styles = ['-', '--', ':']
        labels = ylabels if ylabels is not None else [None] * len(styles)

        def filter_plot_data(x, y):
            ok = ~np.isnan(y)
            return x[ok], y[ok]
        
        if yleft is not None:
            for i, ydata in enumerate(yleft):
                x, y = filter_plot_data(xdata, np.array(ydata))
                ax.plot(x, y, color=colors[0], linestyle=styles[i], label=labels[i], **kwargs)
            ax.set_ylabel('loss', color=colors[0])
            ax.tick_params(axis='y', colors=colors[0])

        if yright is not None:
            ax2 = ax.twinx()
            for i, ydata in enumerate(yright):
                x, y = filter_plot_data(xdata, np.array(ydata))
                ax2.plot(x, y, color=colors[1], linestyle=styles[i], label=labels[i], **kwargs)
            ax2.set_ylabel('acc', color=colors[1])
            ax2.tick_params(axis='y', colors=colors[1])

        ax.set_xlabel(xlabel)
        ax.figure.tight_layout()
        if ylabels is not None:
            ax.legend()
        
    def plot_by_batch(self, ax=None, subset='train', left_metric='loss', right_metric='acc', block_len=1):
        if ax is None:
            fig, ax = plt.subplots()
        
        yleft = None
        if left_metric is not None:
            yleft = [v for i, ep in enumerate(self._epochs) for v in ep[subset][left_metric]]
        yright = None
        if right_metric is not None:
            yright = [v for i, ep in enumerate(self._epochs) for v in ep[subset][right_metric]]
        len_fn = lambda x: len(x) if x is not None else 0
        xdata = 1 + np.arange(max(len_fn(yleft), len_fn(yright)))

        if block_len is not None:
            if yleft is not None:
                yleft = [np.mean(np.reshape(yleft[:block_len * (len(yleft) // block_len)], (-1, block_len)), axis=1)]
            if yright is not None:
                yright = [np.mean(np.reshape(yright[:block_len * (len(yright) // block_len)], (-1, block_len)), axis=1)]
            xdata = xdata[len(xdata) % block_len : : block_len]

        self._plot(xdata, yleft, yright, ax=ax, xlabel='batch')
    
    def plot_by_epoch(self, ax=None, subsets=('train', 'valid'), left_metric='loss', right_metric='acc'):
        if ax is None:
            fig, ax = plt.subplots()

        xdata = 1 + np.arange(len(self))
        yleft, yright= [], []
        
        for i, ss in enumerate(subsets):
            yleft.append([])
            yright.append([])
            for j, ep in enumerate(self._epochs):
                yleft[-1].append(self.epoch_average(j, ss, left_metric))
                yright[-1].append(self.epoch_average(j, ss, right_metric))
        
        self._plot(xdata, yleft, yright, ax=ax, xlabel='epoch', ylabels=subsets)


def train(model, crit, loader, optimizer, stats, subset_name='train'):
    """
    Trenovani dopredneho modelu metodou minibatch SGD
    
    vstup:
        model     ... objekt implementujici metody `forward` a `backward`
        crit      ... kriterium, vraci loss (skalar); musi byt objekt implementujici metody `forward` a `backward`
        loader    ... objekt tridy `ans.BatchLoader`, ktery prochazi data po davkach
        optimizer ... objekt updatujici parametry modelu a implementujici metodu `step`
        stats     ... objekt typu `ans.Stats`
    """
    
    pb = tqdm(loader, desc='epoch {:02d} {}'.format(len(stats), subset_name))
    for X_batch, y_batch in pb:  
        
        # dopredny pruchod
        score = model.forward(X_batch)
        
        # loss
        loss = crit.forward(score, y_batch)

        # zpetny pruchod
        dscore, _ = crit.backward()
        _, dparams = model.backward(dscore)

        # update parametru
        optimizer.step(dparams)

        # vyhodnotime presnost
        _, pred = score.max(dim=1)
        acc = torch.sum(pred == y_batch).float() / X_batch.shape[0]
        
        stats.append_batch_stats(subset_name, loss=float(loss), acc=float(acc))
        pb.set_postfix(
            loss='{:.3f}'.format(stats.ravg(subset_name, 'loss')),
            acc='{:.3f}'.format(stats.ravg(subset_name, 'acc'))
        ) 


def validate(model, crit, loader, stats, subset_name='valid'):
    """
    Validace modelu
    
    vstup:
        model  ... objekt tridy Layer; musi implementovat metody forward a backward
        crit      ... kriterium, vraci loss (skalar); musi byt objekt implementujici metody `forward` a `backward`
        loader ... objekt tridy `ans.BatchSampler`, ktery prochazi data po davkach
    """

    model.eval()
    device = next(model.parameters()).device
    
    pb = tqdm(loader, desc='epoch {:02d} {}'.format(len(stats), subset_name))
    for X_batch, y_batch in pb:
        # zajistit, aby model i data byla na stejnem zarizeni (cpu vs gpu)
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)

        # dopredny pruchod
        score = model.forward(X_batch)
        
        # loss
        loss = crit.forward(score, y_batch)
        
        # vyhodnotime presnost
        _, pred = score.max(dim=1)
        acc = torch.sum(pred == y_batch).float() / X_batch.shape[0]
        
        stats.append_batch_stats(subset_name, loss=float(loss), acc=float(acc))
        pb.set_postfix(
            loss='{:.3f}'.format(stats.ravg(subset_name, 'loss')),
            acc='{:.3f}'.format(stats.ravg(subset_name, 'acc'))
        )


def eval_numerical_gradient_tensor(f, x, df, h=None):
    """
    Vypocte numericky gradient funkce `f`.
    Puvodni funkce `eval_numerical_gradient_array` ze Stanford cs231n byla upravena a prevedena do pytorch.
    """
    
    # automaticky vypocte krok v zavislosti na std. odchylce parametru
    if h is None:
        h = max(1e-6, 0.1 * x.std())
    
    x_ = x.flatten()  # sdili s `x` pamet
    dx = torch.zeros_like(x_)
    
    for i in range(x_.shape[0]):
        v = float(x_[i])  # musi byt obaleno `float`, jinak vraci `Tensor`, ktery sdili pamet
        x_[i] = v + h
        p = f(x)
        x_[i] = v - h
        n = f(x)
        x_[i] = v
        
        dx[i] = torch.sum((p - n) * df) / (2. * h)
    
    return dx.reshape(x.shape)


def rel_error(x, y):
    """
    Prevzato z Standford cs231n a prevedeno do pytorch.
    """
    return float(torch.max(torch.abs(x - y) / (torch.clamp(torch.abs(x) + torch.abs(y), min=1e-8))))


def check_gradients(model: Layer, inputs, doutputs, input_names=None, h=None):
    """
    Funkce vyzkousi dopredny a zpetny pruchod pro kazdy parametr zadaneho modelu
    a zkontroluje gradienty vuci numericke diferenci
    
    vstupy:
        model ... objekt, ktery implementuje metody `forward` a `backward`
        inputs ... `tuple` vstupu vrstvy; gradient bude pocitan pouze pro vstupy s realnymi cisly
        doutputs ... matice N x H gradientu na vystup site
        input_names ... seznam stejne jako `inputs` udavajici jmena vstupu
    vystupy:
        grads ... gradienty vypoctene `model.backward(doutputs)`
        grads_num ... gradienty vypoctene numericky
    """
    
    # vypocti gradienty analyticky (potencialne s bugy)
    out = model.forward(inputs)
    dinputs, dparams = model.backward(doutputs)
    grads = {'inputs': dinputs, **dparams}
    
    # numericky gradient (diference) na vstup (je vzdy az na toleranci spravne)
    grads_num = {}
    
    # vrstva muze mit libovolny pocet vstupu 
    if not isinstance(inputs, tuple):
        inputs = (inputs,)
    if input_names is None:
        input_names = tuple(f'input{i + 1}' for i in range(len(inputs)))
    
    # checkni gradient pro kazdy pro kazdy vstup
    for i, x in enumerate(inputs):
        # gradient pocitame, pouze pokud jsou vstupem realna cisla (ne integery apod.)
        if x.dtype in (torch.float16, torch.float32, torch.float64):
            grads_num['inputs'] = eval_numerical_gradient_tensor(lambda _: model.forward(*inputs), x, doutputs, h=h)
            print(f'd{input_names[i]} error: ', rel_error(grads['inputs'], grads_num['inputs']))
    
    # numericky gradient na parametry modelu
    for name in model.params:
        grads_num[name] = eval_numerical_gradient_tensor(lambda _: model.forward(*inputs), model.params[name], doutputs, h=h)
        print(f'd{name} error: ', rel_error(grads[name], grads_num[name]))
    
    return grads, grads_num


def predict_and_show(rgb, model, transform, classes=None):
    """
    vstupy:
        rgb ... numpy.ndarray formatu vyska x sirka x kanaly a typu np.uint8
        model ... objekt typu nn.Module
        transform ... predzpracovani obrazku, ktere provede pred pruchodem siti
        classes ... seznam trid, bude `None` nebo `list` stejne dlouhy jako pocet vystupnich skore `model`u
    """
    # prepnout model do testovaciho rezimu
    model.eval()
    
    # prevod do torch
    x = transform(rgb)
    x = x.to(next(model.parameters()).device)
    x = x[None]
    
    # dopredny pruchod
    score = model(x)
    prob = F.softmax(score, dim=1)
    
    # prevod do numpy
    score = score.detach().cpu().numpy().squeeze()
    prob = prob.detach().cpu().numpy().squeeze()

    # tridy
    if classes is None:
        classes = [str(i) for i in range(prob.shape[0])]
    
    # vykresleni matplotlib
    plt.figure(figsize=(5, 5))
    plt.imshow(np.array(rgb))
    ids = np.argsort(-score)
    for i, ci in enumerate(ids[:10]):
        text = '{:>5.2f} %  {}'.format(100. * prob[ci], classes[ci])
        if len(text) > 40:
            text = text[:40] + '...'
        plt.gcf().text(1., 0.8 - 0.075 * i, text, fontsize=24)
    plt.subplots_adjust()
