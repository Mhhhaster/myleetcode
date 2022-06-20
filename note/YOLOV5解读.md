# YOLOV5解读

## 1.logging学习

1.logging对象的创建

- Loggers 有以下的属性和方法。注意 *永远* 不要直接实例化 Loggers，应当通过模块级别的函数 `**logging.getLogger(name)**` 。多次使用相同的名字调用 [`getLogger()`](https://docs.python.org/zh-cn/3.7/library/logging.html#logging.getLogger) 会一直**返回相同**的 Logger 对象的引用。

- 推荐的结构**logging.getLogger(name)**

- `info`(*msg*, **args*, ***kwargs*)

  在此记录器上记录 `INFO` 级别的消息

  `warning`(*msg*, **args*, ***kwargs*)

  在此记录器上记录 `WARNING` 级别的消息。

  `error`(*msg*, **args*, ***kwargs*)

  在此记录器上记录 `ERROR` 级别的消息

  ```
  import logging
  logging.warning('Watch out!')  # will print a message to the console
  logging.info('I told you so')  # will not print anything
  ```

  

  日志的级别

  | `CRITICAL` | 50   |
  | ---------- | ---- |
  | `ERROR`    | 40   |
  | `WARNING`  | 30   |
  | `INFO`     | 20   |
  | `DEBUG`    | 10   |
  | `NOTSET`   | 0    |

设置消息的格式

```python
import logging
logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)
logging.debug('This message should appear on the console')
logging.info('So should this')
logging.warning('And this, too')
```

显示时间

```python
import logging
logging.basicConfig(format='%(asctime)%(message)s')
logging.warning('is when this event was logged')

#2022-05-16 12:12:29,915 is when this event was logged
```

在每个使用日志记录的模块中(每个python文件)使用模块级记录器，命名如下:

```
logger = logging.getLogger(__name__)
```

进阶的使用

```python
import logging

# create logger
logger = logging.getLogger('simple_example')
logger.setLevel(logging.DEBUG)

# create console handler and set level to debug
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)

# create formatter
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# add formatter to ch
ch.setFormatter(formatter)

# add ch to logger
logger.addHandler(ch)

# 'application' code
logger.debug('debug message')
logger.info('info message')
logger.warning('warn message')
logger.error('error message')
logger.critical('critical message')

# 2005-03-19 15:10:26,618 - simple_example - DEBUG - debug message
# 2005-03-19 15:10:26,620 - simple_example - INFO - info message
# 2005-03-19 15:10:26,695 - simple_example - WARNING - warn message
# 2005-03-19 15:10:26,773 - simple_example - CRITICAL - critical message
```

使用配置文件

```python
import logging
import logging.config

logging.config.fileConfig('logging.conf')

# create logger
logger = logging.getLogger('simpleExample')

# 'application' code
logger.debug('debug message')
logger.info('info message')
logger.warning('warn message')
logger.error('error message')
logger.critical('critical message')	
```

yaml

```yaml
version: 1
formatters:
  simple:
    format: '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
handlers:
  console:
    class: logging.StreamHandler
    level: DEBUG
    formatter: simple
    stream: ext://sys.stdout
loggers:
  simpleExample:
    level: DEBUG
    handlers: [console]
    propagate: no
root:
  level: DEBUG
  handlers: [console]
```

conf配置

```conf
[loggers]
keys=root,simpleExample

[handlers]
keys=consoleHandler

[formatters]
keys=simpleFormatter

[logger_root]
level=DEBUG
handlers=consoleHandler

[logger_simpleExample]
level=DEBUG
handlers=consoleHandler
qualname=simpleExample
propagate=0

[handler_consoleHandler]
class=StreamHandler
level=DEBUG
formatter=simpleFormatter
args=(sys.stdout,)

[formatter_simpleFormatter]
format=%(asctime)s - %(name)s - %(levelname)s - %(message)s
datefmt=
```

## 2.os.path的使用

```python
import os
 
print( os.path.basename('/root/runoob.txt') )   # 返回文件名
print( os.path.dirname('/root/runoob.txt') )    # 返回目录路径
print( os.path.split('/root/runoob.txt') )      # 分割文件名与路径
print( os.path.join('root','test','runoob.txt') )  # 将目录和文件名合成一个路径
```

## 3.hashlib的使用

```
import hashlib
md = hashlib.md5()
```

## 4.[YOLOV5](https://so.csdn.net/so/search?q=YOLOV5&spm=1001.2101.3001.7020)代码理解类权重系数和图像权重系数(解决类别不均衡的trick)

### 类权重系数

- 当训练图像的所有类个数不相同时,我们可以更改类权重, 即而达到更改图像权重的目的.然后根据图像权重新采集数据，这在图像类别不均衡的数据下尤其重要。

- 使用yolov5训练自己的数据集时，各类别的标签数量难免存在不平衡的问题，在训练过程中为了就减小类别不平衡问题的影响，yolov5中引入了类别权重和图像权重的设置。

  ```python
  model.class_weights = labels_to_class_weights(dataset.labels, nc).to(device)
  ```

  ```python
  def labels_to_class_weights(labels, nc=80):
      # Get class weights (inverse frequency) from training labels
      if labels[0] is None:  # no labels loaded
          return torch.Tensor()
  
      labels = np.concatenate(labels, 0)  # labels.shape = (866643, 5) for COCO
      classes = labels[:, 0].astype(np.int)  # labels = [class xywh]
      weights = np.bincount(classes, minlength=nc)  # occurrences per class返回0-25每类出现的次数
  
      # Prepend gridpoint count (for uCE training)
      # gpi = ((320 / 32 * np.array([1, 2, 4])) ** 2 * 3).sum()  # gridpoints per image
      # weights = np.hstack([gpi * len(labels)  - weights.sum() * 9, weights * 9]) ** 0.5  # prepend gridpoints to start
  
      weights[weights == 0] = 1  # replace empty bins with 1，出现出现次数为0的类，将其设为1
      weights = 1 / weights  # number of targets per class
      weights /= weights.sum()  # normalize
      return torch.from_numpy(weights)
  ```

### 图像权重

在训练过程中，当设置参数–image_weights为True时，会计算图像采集的权重，若图像权重越大,那么该图像被采样的概率也越大。后面遍历图像时,则按照重新采集的索引dataset.indices进行计算

```
parser.add_argument('--image-weights', action='store_true', default=True, help='use weighted image selection for training')  # 加载图像权重
```

```python
if opt.image_weights:
            # Generate indices
            if rank in [-1, 0]:
                cw0 = model.class_weights.cpu().numpy()  # ([0.64486, 0.12426, 0.23088])
                cw = cw0*(1 - maps) ** 2  # class weights
                iw = labels_to_image_weights(dataset.labels, nc=nc, class_weights=cw)  # image weights
                """indices[7, 49, 44, 14, 29, 26, 38, 46, 5, 1, 48, 25, 44, 0, 26, 42, 13, 54, 
                52, 1, 1, 31, 54, 22, 12, 24, 1, 12, 25, 29, 13, 13, 12, 26, 17, 1, 48, 32, 37, 
                10, 57, 50, 6, 19, 42, 41, 54, 24, 48, 39, 17, 34, 51, 49, 29, 34, 1, 14]"""
                dataset.indices = random.choices(range(dataset.n), weights=iw, k=dataset.n)  # rand weighted idx
```

## 5.Dataloader

### 1.矫正图片的orientation

```python
# 获取orientation
for orientation in ExifTags.TAGS.keys():
    if ExifTags.TAGS[orientation] == 'Orientation':
        break
        
def exif_size(img):
    """有exif标签的情况下获取图片的信息获取图片的宽、高信息"""
    # Returns exif-corrected PIL size
    s = img.size  # (width, height)
    try:
        rotation = dict(img._getexif().items())[orientation]
        if rotation == 6:  # rotation 270
            s = (s[1], s[0])
        elif rotation == 8:  # rotation 90
            s = (s[1], s[0])
    except:
        pass

    return s

def exif_transpose(image):
    """
    Transpose a PIL image accordingly if it has an EXIF Orientation tag.
    From https://github.com/python-pillow/Pillow/blob/master/src/PIL/ImageOps.py

    :param image: The image to transpose.
    :return: An image.
    """
    exif = image.getexif()
    orientation = exif.get(0x0112, 1)  # default 1
    if orientation > 1:
        method = {2: Image.FLIP_LEFT_RIGHT,
                  3: Image.ROTATE_180,
                  4: Image.FLIP_TOP_BOTTOM,
                  5: Image.TRANSPOSE,
                  6: Image.ROTATE_270,
                  7: Image.TRANSVERSE,
                  8: Image.ROTATE_90,
                  }.get(orientation)
        if method is not None:
            image = image.transpose(method)
            del exif[0x0112]
            image.info["exif"] = exif.tobytes()
    return image
```

### 2.dataloader

```python
def create_dataloader(path, imgsz, batch_size, stride, single_cls=False, hyp=None, augment=False, cache=False, pad=0.0,
                      rect=False, rank=-1, workers=8, image_weights=False, quad=False, prefix=''):
    """根据LoadImagesAndLabels创建dataloader
    参数解析：
    path：包含图片路径的txt文件或者包含图片的文件夹路径
    imgsz：网络输入图片大小
    batch_size: 批次大小
    stride：网络下采样最大总步长
    single_cls：是否为单类
    hyp：网络训练时的一些超参数，包括学习率等，这里主要用到里面一些关于数据增强(旋转、平移等)的系数
    augment：是否进行数据增强
    cache：是否提前缓存图片到内存，以便加快训练速度
    pad：设置矩形训练的shape时进行的填充
    rect：是否进行矩形训练
    rank: 多卡训练时的进程编号
    workers: 加载数据时的cpu进程数
    image_weights:训练时是否对图片进行采样的权重
    quad: 是否使用collate_fn4作为dataloader的选择函数
    prefix: 一个标志，多为train/val，处理标签时保存cache文件会用到
    """
    # Make sure only the first process in DDP process the dataset first, and the following others can use the cache
    with torch_distributed_zero_first(rank):
        dataset = LoadImagesAndLabels(path, imgsz, batch_size,
                                      augment=augment,  # augment images
                                      hyp=hyp,  # augmentation hyperparameters
                                      rect=rect,  # rectangular training
                                      cache_images=cache,
                                      single_cls=single_cls,
                                      stride=int(stride),
                                      pad=pad,
                                      image_weights=image_weights,
                                      prefix=prefix)
    # 要么batchsize 要么是数据集的长度
    batch_size = min(batch_size, len(dataset))
    # 三个取其一 求最小
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, workers])  # number of workers
    # 分布式采样
    sampler = torch.utils.data.distributed.DistributedSampler(dataset) if rank != -1 else None
    loader = torch.utils.data.DataLoader if image_weights else InfiniteDataLoader
    # Use torch.utils.data.DataLoader() if dataset.properties will update during training else InfiniteDataLoader()
    dataloader = loader(dataset,
                        batch_size=batch_size,
                        num_workers=nw,
                        sampler=sampler,
                        pin_memory=True,
                        collate_fn=LoadImagesAndLabels.collate_fn4 if quad else LoadImagesAndLabels.collate_fn)
    return dataloader, dataset
```

