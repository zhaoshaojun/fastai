from fastai.script import *
from fastai.vision import *
from fastai.callbacks import *
from fastai.distributed import *
from fastai.callbacks.tracker import *
torch.backends.cudnn.benchmark = True
import time
from fastai.general_optimizer import *

def get_data(path, size, bs, workers):
    tfms = ([
        flip_lr(p=0.5),
        #brightness(change=(0.4,0.6)),
        #contrast(scale=(0.7,1.3))
    ], [])
    return (ImageList.from_folder(path).split_by_folder(valid='val')
            .label_from_folder().transform(tfms, size=size)
            .databunch(bs=bs, num_workers=workers)
            .presize(size, scale=(0.35,1))
            .normalize(imagenet_stats))

def bn_and_final(m):
    ll = flatten_model(m)
    last_lin = next(o for o in reversed(ll) if isinstance(o, bias_types))
    idx = [i for i,o in enumerate(ll) if
           (i>50 and isinstance(o, bn_types)) or o==last_lin]
    l1 = [o for i,o in enumerate(ll) if i not in idx]
    l2 = [ll[i] for i in idx]
    return split_model(splits=[l1,l2])

class SGDX(GeneralOptimizer):
    def make_step(self, p, group, group_idx):
        p.data.add_(-group['lr'], self.state[p]['mom'])

class RMSpropX(GeneralOptimizer):
    def make_step(self, p, group, group_idx):
        p.data.addcdiv_(-group['lr'], self.state[p]['mom_buffer'], self.state[p]['alpha_buffer'].sqrt() + 1e-7)

@call_parse
def main( gpu:Param("GPU to run on", str)=None ):
    """Distributed training of Imagenette.
    Fastest multi-gpu speed is if you run with: python -m fastai.launch"""
    bs,tot_epochs,lr = 256,5,0.0003

    # Pick one of these
    path,size = untar_data(URLs.IMAGENETTE_160),128
    #path,size = untar_data(URLs.IMAGENETTE_320),224
    #path,size = untar_data(URLs.IMAGENETTE),224

    gpu = setup_distrib(gpu)
    n_gpus = num_distrib() or 1

    workers = min(8, num_cpus()//n_gpus)
    data = get_data(path, size, bs, workers)
    #opt_func = partial(optim.Adam, betas=(0.9,0.99), eps=1e-7)
    #opt_func = partial(optim.RMSprop, alpha=0.9)
    #opt_func = optim.SGD
    #opt_func = partial(SGDX, stats=[AvgStatistic('avgstat', init=0, beta=0.9),])
    #"""
    opt_func = partial(RMSpropX, stats=[
        AvgStatistic('mom', 0.9),
        AvgSquare('alpha', 0.9),
    ])
    #"""
    #learn = (cnn_learner(data, models.xresnet50, pretrained=False, concat_pool=False, lin_ftrs=[], split_on=bn_and_final,
    learn = (Learner(data, models.xresnet50(),
             metrics=[accuracy,top_k_accuracy], wd=1e-3, opt_func=opt_func,
             bn_wd=False, true_wd=True, loss_func = LabelSmoothingCrossEntropy())
        .mixup(alpha=0.2)
        .to_fp16(dynamic=True)
        #.split(bn_and_final)
    )
    if gpu is None: learn.to_parallel()
    else:           learn.to_distributed(gpu)

    # Using bs 256 on single GPU as baseline, scale the LR linearly
    bs_rat = bs/256
    lr *= bs_rat
    learn.fit_one_cycle(tot_epochs, lr, div_factor=20, pct_start=0.5, moms=(0.9,0.9))
    learn.save('nette')

