from fastai.script import *
from fastai.vision import *
from fastai.callbacks import *
from fastai.distributed import *
from fastai.callbacks.tracker import *
torch.backends.cudnn.benchmark = True

def get_data(path, size, bs, workers):
    tfms = ([
        flip_lr(p=0.5),
        brightness(change=(0.4,0.6)),
        contrast(scale=(0.7,1.3))
    ], [])
    train = ImageList.from_csv(path, 'train.csv')#.use_partial_data(0.001)
    valid = ImageList.from_csv(path, 'valid.csv')#.use_partial_data()
    lls = ItemLists(path, train, valid).label_from_df().transform(
            tfms, size=size).presize(size, scale=(0.25, 1.0))
    return lls.databunch(bs=bs, num_workers=workers).normalize(imagenet_stats)

@call_parse
def main( gpu:Param("GPU to run on", str)=None ):
    """Distributed training of Imagenet. Fastest speed is if you run with: python -m fastai.launch"""
    path = Path('/mnt/fe2_disk/')
    tot_epochs = 90
    epoch_fn = path/'imagenet'/'models'/'epoch'
    epoch = 0
    if epoch_fn.exists(): epoch = int(epoch_fn.open().read())
    epoch += 1
    if epoch>tot_epochs: return

    bs,lr = 256,0.1
    gpu = setup_distrib(gpu)
    n_gpus = int(os.environ.get("WORLD_SIZE", 1))
    workers = min(32, num_cpus()//n_gpus)
    data = get_data(path/'imagenet', 224, bs, workers)
    opt_func = partial(optim.SGD, momentum=0.9)
    learn = Learner(data, models.xresnet50(), metrics=[accuracy,top_k_accuracy], wd=1e-5,
        opt_func=opt_func, bn_wd=True, true_wd=False
        , loss_func = LabelSmoothingCrossEntropy()).mixup(alpha=0.2)
    learn.callback_fns += [
        partial(TrackEpochCallback, epoch_offset=epoch-1),
        partial(SaveModelCallback, every='epoch', name='model')
    ]
    learn.split(lambda m: (children(m)[-2],))
    if epoch>1: learn.load(f'model_{epoch-1}', purge=False, device=gpu)
    if gpu is None: learn.model = nn.DataParallel(learn.model)
    else:           learn.distributed(gpu)
    learn.to_fp16(dynamic=True)

    # Using bs 256 on single GPU as baseline, scale the LR linearly
    tot_bs = bs*n_gpus
    bs_rat = tot_bs/256
    lr *= bs_rat
    learn.fit_one_cycle(tot_epochs-epoch+1, lr, div_factor=5, pct_start=0.05, moms=(0.9,0.9), start_epoch=epoch)
    learn.save('done')
    return

    learn.freeze()
    learn.fit_one_cycle(1, lr/100, div_factor=1, final_div=1, moms=(0.9,0.9))
    learn.unfreeze()
    gc.collect()
    time.sleep(1)
    learn.save('done3')

    learn.fit_one_cycle(20, lr/200, div_factor=5, pct_start=0.05, moms=(0.9,0.9))
    learn.save('done2')

