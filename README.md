Information Maximizing GAN in Tensorflow
========================================

Tensorflow implementation of [Information Maximizing GAN](https://arxiv.org/abs/1606.03657) MNIST handwritten digit dataset.

Prerequisites
-------------

This code requires [Tensorflow](https://www.tensorflow.org/). The MNIST dataset is stored in the 'MNIST_data' directory. The files will be automatically downloaded if the dataset does not exist.
    
If you want to use `--show_progress True` option, you need to install python package `progress`.

    $ pip install progress
    
Usage
-----

To train a vanilla InfoGAN with z dimension 14, categorical code dimension 10, and continuous code dimension 2, run the following command:

    $ python main.py --gan_type VanillaInfoGAN --z_dim 14 --c_cat 10 --c_cont 2

To see all training options, run:

    $ python main.py --help

which will print:

    usage: main.py [-h] [--input_dim INPUT_DIM] [--z_dim Z_DIM] [--c_cat C_CAT]
               [--c_cont C_CONT] [--d_update D_UPDATE]
               [--batch_size BATCH_SIZE] [--nepoch NEPOCH] [--lr LR]
               [--max_grad_norm MAX_GRAD_NORM] [--gan_type GAN_TYPE]
               [--checkpoint_dir CHECKPOINT_DIR] [--image_dir IMAGE_DIR]
               [--use_adam [USE_ADAM]] [--nouse_adam]
               [--show_progress [SHOW_PROGRESS]] [--noshow_progress]

    optional arguments:
      -h, --help            show this help message and exit
      --input_dim INPUT_DIM
                            dimension of the discriminator input placeholder [784]
      --z_dim Z_DIM         dimension of the generator input noise variable z [14]
      --c_cat C_CAT         dimension of the categorical latent code [10]
      --c_cont C_CONT       dimension of the continuous latent code [2]
      --d_update D_UPDATE   update the discriminator weights [d_update] times per
                            generator/Q network update [2]
      --batch_size BATCH_SIZE
                            batch size to use during training [128]
      --nepoch NEPOCH       number of epochs to use during training [100]
      --lr LR               learning rate of the optimizer to use during training
                            [0.001]
      --max_grad_norm MAX_GRAD_NORM
                            clip L2-norm of gradients to this threshold [40]
      --gan_type GAN_TYPE   input "VanillaInfoGAN" to use Vanilla InfoGAN;
                            otherwise, input "InfoDCGAN" [VanillaInfoGAN]
      --checkpoint_dir CHECKPOINT_DIR
                            checkpoint directory [./checkpoints]
      --image_dir IMAGE_DIR
                            directory to save generated images to [./images]
      --use_adam [USE_ADAM]
                            if True, use Adam optimizer; otherwise, use SGD [True]
      --nouse_adam
      --show_progress [SHOW_PROGRESS]
                            print progress [False]
      --noshow_progress

(Optional) If you want to see a progress bar, install `progress` with `pip`:

    $ pip install progress
    $ python main.py --nhop 3 --mem_size 50 --show_progress True

After training is finished, you can test and validate with:

    $ python main.py --is_test True --show_progress True


Acknowledgements
----------------

Majority of the source code in VanillaInfoGAN.py is from: Agustinus Kristiadi / [@wiseodd](http://wiseodd.github.io/).


Notes
-----

I have also written the tensorflow implementation of the InfoGAN as introduced in the original paper (See Appendix C). However, due to the lack of computational power, I was not able to test it out; therefore, the convergence of the generator and the discriminator for InfoDCGAN is not guranteed.