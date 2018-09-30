"`vision.data` manages data input pipeline - folders -> transform -> batch input. Includes support for classification, segmentation and bounding boxes"
from ..torch_core import *
from .image import *
from .transform import *
from ..data import *

__all__ = ['DatasetTfm', 'ImageDataset', 'ImageMultiDataset', 'ObjectDetectDataset', 'SegmentationDataset', 'denormalize',
           'get_image_files', 'image_data_from_csv', 'image_data_from_folder', 'normalize', 'normalize_funcs',
           'show_image_batch', 'show_images', 'show_xy_images', 'transform_datasets', 'cifar_norm', 'cifar_denorm', 'imagenet_norm',
           'imagenet_denorm']

TfmList = Collection[Transform]

image_extensions = set(k for k,v in mimetypes.types_map.items() if v.startswith('image/'))

def get_image_files(c:Path, check_ext:bool=True)->FilePathList:
    "Return list of files in `c` that are images. `check_ext` will filter to `image_extensions`."
    return [o for o in list(c.iterdir())
            if not o.name.startswith('.') and not o.is_dir()
            and (not check_ext or (o.suffix in image_extensions))]

def show_image_batch(dl:DataLoader, classes:Collection[str], rows:int=None, figsize:Tuple[int,int]=(12,15),
                     denorm:Callable=None) -> None:
    "Show a few images from a batch."
    x,y = next(iter(dl))
    if rows is None: rows = int(math.sqrt(len(x)))
    x = x[:rows*rows].cpu()
    if denorm: x = denorm(x)
    show_images(x,y[:rows*rows].cpu(),rows, classes, figsize)

def show_images(x:Collection[Image],y:int,rows:int, classes:Collection[str], figsize:Tuple[int,int]=(9,9))->None:
    "Plot images (`x[i]`) from `x` titled according to `classes[y[i]]`."
    fig, axs = plt.subplots(rows,rows,figsize=figsize)
    for i, ax in enumerate(axs.flatten()):
        show_image(x[i], ax=ax)
        ax.set_title(classes[y[i]])
    plt.tight_layout()

def show_xy_images(x:Tensor,y:Tensor,rows:int,figsize:tuple=(9,9)):
    "Show a selection of images and targets from a given batch."
    fig, axs = plt.subplots(rows,rows,figsize=figsize)
    for i, ax in enumerate(axs.flatten()): show_image(x[i], y=y[i], ax=ax)
    plt.tight_layout()

class ImageDataset(LabelDataset):
    "Dataset for folders of images in style {folder}/{class}/{images}."
    def __init__(self, fns:FilePathList, labels:ImgLabels, classes:Optional[Classes]=None):
        self.classes = ifnone(classes, list(set(labels)))
        self.class2idx = {v:k for k,v in enumerate(self.classes)}
        self.x = np.array(fns)
        self.y = np.array([self.class2idx[o] for o in labels], dtype=np.int64)

    def __getitem__(self,i): return open_image(self.x[i]),self.y[i]

    @staticmethod
    def _folder_files(folder:Path, label:ImgLabel, check_ext=True)->Tuple[FilePathList,ImgLabels]:
        "From `folder` return image files and labels. The labels are all `label`. `check_ext` means only image files."
        fnames = get_image_files(folder, check_ext=check_ext)
        return fnames,[label]*len(fnames)

    @classmethod
    def from_single_folder(cls, folder:PathOrStr, classes:Classes, check_ext=True):
        "Typically used for test set. label all images in `folder` with `classes[0]`."
        fns,labels = cls._folder_files(folder, classes[0], check_ext=check_ext)
        return cls(fns, labels, classes=classes)

    @classmethod
    def from_folder(cls, folder:Path, classes:Optional[Classes]=None,
                    valid_pct:float=0., check_ext:bool=True) -> Union['ImageDataset', List['ImageDataset']]:
        "Dataset of `classes` labeled images in `folder`. Optional `valid_pct` split validation set."
        if classes is None: classes = [cls.name for cls in find_classes(folder)]

        fns,labels = [],[]
        for cl in classes:
            f,l = cls._folder_files(folder/cl, cl, check_ext=check_ext)
            fns+=f; labels+=l

        if valid_pct==0.: return cls(fns, labels, classes=classes)
        return [cls(*a, classes=classes) for a in random_split(valid_pct, fns, labels)]

#Draft, to check
class ImageMultiDataset(LabelDataset):
    def __init__(self, fns:FilePathList, labels:ImgLabels, classes:Optional[Classes]=None):
        self.classes = ifnone(classes, uniqueify(np.concatenate(labels)))
        self.class2idx = {v:k for k,v in enumerate(self.classes)}
        self.x = np.array(fns)
        self.y = [np.array([self.class2idx[o] for o in l], dtype=np.int64)
                  for l in labels]

    def encode(self, x:Collection[int]):
        "One-hot encode the target."
        res = np.zeros((self.c,), np.float32)
        res[x] = 1.
        return res

    def get_labels(self, idx:int) -> ImgLabels: return [self.classes[i] for i in self.y[idx]]
    def __getitem__(self,i:int) -> Tuple[Image, ImgLabels]: return open_image(self.x[i]), self.encode(self.y[i])

    @classmethod
    def from_single_folder(cls, folder:PathOrStr, classes:Classes, check_ext=True):
        "Typically used for test set; label all images in `folder` with `classes[0]`."
        fnames = get_image_files(folder, check_ext=check_ext)
        labels = [[classes[0]]] * len(fnames)
        return cls(fnames, labels, classes=classes)

    @classmethod
    def from_folder(cls, path:PathOrStr, folder:PathOrStr, fns:pd.Series, labels:ImgLabels, valid_pct:float=0.2,
        classes:Optional[Classes]=None):
        train,valid = random_split(valid_pct, f'{path}/{folder}/' + fns, labels)
        train_ds = cls(*train, classes=classes)
        return [train_ds,cls(*valid, classes=train_ds.classes)]

class SegmentationDataset(DatasetBase):
    "A dataset for segmentation task."
    def __init__(self, x:Collection[PathOrStr], y:Collection[PathOrStr]):
        assert len(x)==len(y)
        self.x,self.y = np.array(x),np.array(y)

    def __getitem__(self, i:int) -> Tuple[Image,ImageMask]:
        return open_image(self.x[i]), open_mask(self.y[i])

@dataclass
class ObjectDetectDataset(Dataset):
    "A dataset with annotated images."
    x_fns:Collection[Path]
    bbs:Collection[Collection[int]]
    def __post_init__(self): assert len(self.x_fns)==len(self.bbs)
    def __repr__(self) -> str: return f'{type(self).__name__} of len {len(self.x_fns)}'
    def __len__(self) -> int: return len(self.x_fns)
    def __getitem__(self, i:int) -> Tuple[Image,ImageBBox]:
        x = open_image(self.x_fns[i])
        return x, ImageBBox.create(self.bbs[i], *x.size)

class DatasetTfm(Dataset):
    "`Dataset` that applies a list of transforms to every item drawn."
    def __init__(self, ds:Dataset, tfms:TfmList=None, tfm_y:bool=False, **kwargs:Any):
        "this dataset will apply `tfms` to `ds`"
        self.ds,self.tfms,self.kwargs,self.tfm_y = ds,tfms,kwargs,tfm_y
        self.y_kwargs = {**self.kwargs, 'do_resolve':False}

    def __len__(self)->int: return len(self.ds)

    def __getitem__(self,idx:int)->Tuple[ItemBase,Any]:
        "returns tfms(x),y"
        x,y = self.ds[idx]
        x = apply_tfms(self.tfms, x, **self.kwargs)
        if self.tfm_y: y = apply_tfms(self.tfms, y, **self.y_kwargs)
        return x, y

    def __getattr__(self,k):
        "passthrough access to wrapped dataset attributes"
        return getattr(self.ds, k)

def transform_datasets(train_ds:Dataset, valid_ds:Dataset, test_ds:Optional[Dataset]=None,
                       tfms:Optional[Tuple[TfmList,TfmList]]=None, **kwargs:Any):
    "Create train, valid and maybe test DatasetTfm` using `tfms` = (train_tfms,valid_tfms)."
    res = [DatasetTfm(train_ds, tfms[0],  **kwargs),
           DatasetTfm(valid_ds, tfms[1],  **kwargs)]
    if test_ds is not None: res.append(DatasetTfm(test_ds, tfms[1],  **kwargs))
    return res

def normalize(x:TensorImage, mean:FloatTensor,std:FloatTensor)->TensorImage:
    "Normalize `x` with `mean` and `std`."
    return (x-mean[...,None,None]) / std[...,None,None]
def denormalize(x:TensorImage, mean:FloatTensor,std:FloatTensor)->TensorImage:
    "Denormalize `x` with `mean` and `std`."
    return x*std[...,None,None] + mean[...,None,None]

def _normalize_batch(b:Tuple[Tensor,Tensor], mean:FloatTensor, std:FloatTensor, do_y:bool=False)->Tuple[Tensor,Tensor]:
    "`b` = `x`,`y` - normalize `x` array of imgs and `do_y` optionally `y`."
    x,y = b
    x = normalize(x,mean,std)
    if do_y: y = normalize(y,mean,std)
    return x,y

def normalize_funcs(mean:FloatTensor, std:FloatTensor, do_y=False, device=None)->[Callable,Callable]:
    "Create normalize/denormalize func using `mean` and `std`, can specify `do_y` and `device`."
    if device is None: device=default_device
    return (partial(_normalize_batch, mean=mean.to(device),std=std.to(device)),
            partial(denormalize,     mean=mean,           std=std))

cifar_stats = (tensor([0.491, 0.482, 0.447]), tensor([0.247, 0.243, 0.261]))
cifar_norm,cifar_denorm = normalize_funcs(*cifar_stats)
imagenet_stats = tensor([0.485, 0.456, 0.406]), tensor([0.229, 0.224, 0.225])
imagenet_norm,imagenet_denorm = normalize_funcs(*imagenet_stats)

def _create_with_tfm(train_ds, valid_ds, test_ds=None,
               path='.', bs=64, ds_tfms=None, num_workers=default_cpus,
               tfms=None, device=None, size=None, **kwargs)->'DataBunch':
        "`DataBunch` factory. `bs` batch size, `ds_tfms` for `Dataset`, `tfms` for `DataLoader`."
        datasets = [train_ds,valid_ds]
        if test_ds is not None: datasets.append(test_ds)
        if ds_tfms: datasets = transform_datasets(*datasets, tfms=ds_tfms, size=size, **kwargs)
        dls = [DataLoader(*o, num_workers=num_workers) for o in
               zip(datasets, (bs,bs*2,bs*2), (True,False,False))]
        return DataBunch(*dls, path=path, device=device, tfms=tfms)

DataBunch.create = _create_with_tfm

def image_data_from_folder(path:PathOrStr, train:PathOrStr='train', valid:PathOrStr='valid',
                          test:Optional[PathOrStr]=None, **kwargs:Any) -> DataBunch:
    "Create `DataBunch` from imagenet style dataset in `path` with `train`,`valid`,`test` subfolders."
    path=Path(path)
    train_ds = ImageDataset.from_folder(path/train)
    datasets = [train_ds, ImageDataset.from_folder(path/valid, classes=train_ds.classes)]
    if test: datasets.append(ImageDataset.from_single_folder(
        path/test,classes=train_ds.classes))
    return DataBunch.create(*datasets, path=path, **kwargs)

def _get_fns(ds, path):
    "List of all file names relative to `path`."
    return [str(fn.relative_to(path)) for fn in ds.x]

def _labels_to_csv(self, dest:str):
    "Save file names and labels in `data` as CSV to file name `dest`."
    fns = _get_fns(self.train_ds)
    y = list(self.train_ds.y)
    fns += _get_fns(self.valid_ds)
    y += list(self.valid_ds.y)
    if hasattr(self,'test_dl') and data.test_dl:
        fns += _get_fns(self.test_ds)
        y += list(self.test_ds.y)
    df = pd.DataFrame({'name': fns, 'label': y})
    df.to_csv(dest, index=False)

DataBunch.labels_to_csv = _labels_to_csv

def uniqueify(x:Series) -> List[Any]: return list(OrderedDict.fromkeys(x).keys())

def csv_to_fns_labels(csv_path:PathOrStr, fn_col:int=0, label_col:int=1,
                      label_delim:str=' ', header:Optional[Union[int,str]]='infer', suffix:Optional[str]=None):
    df = pd.read_csv(csv_path, header=header)
    df.iloc[:,label_col] = list(csv.reader(df.iloc[:,label_col], delimiter=label_delim))
    labels = df.iloc[:,label_col]
    fnames = df.iloc[:,fn_col]
    if suffix: fnames = fnames + suffix
    return fnames, labels

def image_data_from_csv(path:PathOrStr, folder:PathOrStr='.', csv_labels:PathOrStr='labels.csv', valid_pct:float=0.2,
                        test:Optional[PathOrStr]=None, suffix:str=None, **kwargs:Any) -> DataBunch:
    fnames, labels = csv_to_fns_labels(csv_labels, suffix=suffix)
    path=Path(path)
    datasets = ImageMultiDataset.from_folder(path, folder, fnames, labels, valid_pct=valid_pct)
    if test: datasets.append(ImageMultiDataset.from_single_folder(path/test, classes=datasets[0].classes))
    return DataBunch.create(*datasets, path=path, **kwargs)

