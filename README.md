# DLMB
A collection of gan implementations. DC-WGAN-GP and c-DC-WGAN-GP 
(
conditional deep convolutional Wasserstein GAN with gradient penalty
[<sup>1</sup>](#wgan) [<sup>2</sup>](#wgan_gp) [<sup>4</sup>](#dcgan)
). 
Dataset we use HAM10000, also tested at MNIST and CIFAR 10. 

##### Toy task

Also we implemented vanilla toy task in folder toy task vanilla gan(MLP layer), 
vanilla DCGAN, vanilla WGAN-GP(MLP layer) and vanilla cWGAN-GP(MLP layer). Besides we add 
VAE as awell as semi supervised VAE(ssvae) implementation of 
pyro-example[<sup>3</sup>](#pyro-tutorial)as reference.


## Implementation
TODO

## Training
Assume that you have the folder $DATA_ROOT/HAM10000/img that saves 
all the HAM10000 images.  Besides you need corresponding file meta.csv.
The data could be download here[<sup>5</sup>](#HAM1000).

A example of training:
```
python cwagan_gp.py --data HAM10000 --root $data --csv-file $csv_file
                    --depth 5 --img-size 64 --batch-size 64 --checkpoint-factor 10
```
To check the learning process use tensorboard, (default in `./runs`), if the image doesn't change with 
step continuously, try to add `--samples_per_plugin images=0`
 
```
tensorboard --logdir=$DLMB_FOLDER
```

To continue the training with checkpoint, try this command, note that if you recover
from ckpt file, the other parameters will be disabled, and use the old parameter to train.
```
python cwagan_gp.py --data HAM10000 --recover --checkpoint-file $ckpt_file.pth
```

The entry to run the file is cwgan_gp.py. The parameters are defined as 
```
usage: cwgan_gp.py [-h] --data DATA [--root ROOT] [--csv-file CSV_FILE] [--n-epoc N_EPOC] 
                   [--d-step D_STEP] [--batch-size BATCH_SIZE] [--z-dim Z_DIM] 
                   [--ndf NDF] [--ngf NGF]
                   [--depth DEPTH] [--img-size IMG_SIZE] [--lr-g LR_G] [--lr-d LR_D] 
                   [--lr-beta1 LR_BETA1] [--lr-beta2 LR_BETA2] 
                   [--data-percentage DATA_PERCENTAGE] [--data-aug]
                   [--condition] [--embedding-dim EMBEDDING_DIM] [--recover] 
                   [--checkpoint-file CHECKPOINT_FILE] 
                   [--checkpoint-factor CHECKPOINT_FACTOR]

parse args

optional arguments:
  -h, --help            show this help message and exit
  --data DATA           MNIST|CIFAR10|HAM10000, name of dataset used for training, 
                        only support MNIST CIFAR10 and HAM10000
  --root ROOT           root folder for dataset, for HAM10000, 
                        you are supposed to have a folder with $DATA_ROOT/HAM10000/img/ 
                        that contains all the HAM10000 images. The --root requires
                        the upper path $DATA_ROOT,for MNIST and CIFAR10 will automatically
                        download in give root path if there is no corresponding dataset
  --csv-file CSV_FILE   only used for HAM10000, the corresponding metadata file
  --n-epoc N_EPOC       epochs for training
  --d-step D_STEP       steps for training discriminator per gan training step
  --batch-size BATCH_SIZE
                        batch size for training
  --z-dim Z_DIM         noise shape for generator
  --ndf NDF             number of discriminator feature, check the introduction of network
  --ngf NGF             number of generator feature, check the introduction of network
  --depth DEPTH         how deep the network is. note that it must be more than three, 
                        also it must correspond the --img-size, 
                        i.e.depth:img_size, 5:64, 6:128, 7:256...
  --img-size IMG_SIZE   resize the original image, it must correspond to 
                        --depth, i.e. depth:img_size, 5:64, 6:128, 7:256...
  --lr-g LR_G           learning rate of generator
  --lr-d LR_D           learning rate of discriminator
  --lr-beta1 LR_BETA1   beta1 for ADAM optimizer
  --lr-beta2 LR_BETA2   beta2 for ADAM optimizer
  --data-percentage DATA_PERCENTAGE
                        (0.0, 1.0], how many percentage of data will be used for training
  --data-aug            if use data augmentation or not, augmentation includes 
                        random horizontal/vertical flip, random resized crop
  --condition           if use conditional training
  --embedding-dim EMBEDDING_DIM
                        embedding dim, used for embedding layer of label: disease type
  --recover             if continue training from prior checkpoint,if use this, 
                        all the other parameters are disabled, the parameter of 
                        checkpoints parameter will be used
  --checkpoint-file CHECKPOINT_FILE
                        checkpoint file path
  --checkpoint-factor CHECKPOINT_FACTOR
                        the model will be checked every $ckpt-factor epoch

```

## TEST-GUI
TODO
## Reference
<div id="wgan"></div>
- [1] [Arjovsky M, Chintala S, Bottou L. 
Wasserstein generative adversarial networks[C]//International 
conference on machine learning. PMLR, 2017: 214-223.]
<div id="wgan_gp"></div>
- [2] [Gulrajani I, Ahmed F, Arjovsky M, et al. Improved 
training of wasserstein gans[J]. arXiv preprint arXiv:1704.00028, 2017.]
<div id ='pyro-tutorial'></div>
- [3] [Pyro-Tutorial-SSVAE](https://pyro.ai/examples/ss-vae.html)
<div id ='dcgan'></div>
- [4] [Radford A, Metz L, Chintala S. Unsupervised representation 
learning with deep convolutional generative adversarial networks[J]. 
arXiv preprint arXiv:1511.06434, 2015.]
<div id ='HAM10000'></div>
- [5] [Tschandl, P., Rosendahl, C. & Kittler, H. The HAM10000 dataset, 
a large collection of multi-source dermatoscopic images of common pigmented 
skin lesions. Sci. Data 5, 180161 (2018). doi: 10.1038/sdata.2018.161]

