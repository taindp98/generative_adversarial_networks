# Methods

GANs are a kind of implicit generative model, which means we train a neural net to produce samples. 

The implicit generative models don’t let us query the probability of an observation.

To rephrase this, we simultaneously train two different networks:

1. The generator network G, which tries to generate realistic samples.
2. The discriminator network D, which is a binary classification network which tries to classify real vs. fake samples. It takes an input x and computes D(x), the probability it assigns to x being real.

The two networks are trained competitively: the generator is trying to fool the discriminator, and the discriminator is trying not to be fooled by the generator.

![Screen Shot 2021-09-05 at 12.42.34.png](https://s3.us-west-2.amazonaws.com/secure.notion-static.com/a15c17a8-6473-47ae-b1a3-9e4a8239237a/Screen_Shot_2021-09-05_at_12.42.34.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIAT73L2G45O3KS52Y5%2F20210911%2Fus-west-2%2Fs3%2Faws4_request&X-Amz-Date=20210911T031325Z&X-Amz-Expires=86400&X-Amz-Signature=43d771e060065fe60070d2a76573be74354f0a7cf323d67b389f86575ac1270d&X-Amz-SignedHeaders=host&response-content-disposition=filename%20%3D%22Screen%2520Shot%25202021-09-05%2520at%252012.42.34.png%22)
If the discriminator has low cross entropy, that means it can easily distinguish real from fake; if it has high cross-entropy, that means it can’t. Therefore, the most straightforward criterion for the generator is to maximize the discriminator’s cross-entropy.

![Screen Shot 2021-09-05 at 12.48.20.png](https://s3.us-west-2.amazonaws.com/secure.notion-static.com/70e1f032-7e86-43de-a199-b3fa0ef059b0/Screen_Shot_2021-09-05_at_12.48.20.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIAT73L2G45O3KS52Y5%2F20210911%2Fus-west-2%2Fs3%2Faws4_request&X-Amz-Date=20210911T031351Z&X-Amz-Expires=86400&X-Amz-Signature=369e78c27e91ad5b75efd9b6f96ad0174635384019d97c978ba9c2ceb9c2edc9&X-Amz-SignedHeaders=host&response-content-disposition=filename%20%3D%22Screen%2520Shot%25202021-09-05%2520at%252012.48.20.png%22)

![Screen Shot 2021-09-05 at 22.32.06.png](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/1e8b4cf2-7e94-4168-9c61-76b91785d919/Screen_Shot_2021-09-05_at_22.32.06.png)

![Screen Shot 2021-09-05 at 13.43.57.png](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/2719fd5c-3b84-4392-a6c2-dfffb2bc6f74/Screen_Shot_2021-09-05_at_13.43.57.png)

## Deep Convolutional GAN (DCGAN)

The objective function of the discriminator becomes:

![Screen Shot 2021-09-05 at 13.54.12.png](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/f8038a07-83d9-432e-818c-f474d3fa3c98/Screen_Shot_2021-09-05_at_13.54.12.png)

the objective function of the generator becomes:

![Screen Shot 2021-09-05 at 13.54.51.png](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/f96b8d37-489c-4b0e-b224-7a5b23760dc6/Screen_Shot_2021-09-05_at_13.54.51.png)

![Screen Shot 2021-09-05 at 14.05.58.png](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/1c6682b0-a715-4363-89a2-693ae3d18a8e/Screen_Shot_2021-09-05_at_14.05.58.png)

![Screen Shot 2021-09-05 at 14.05.00.png](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/e96e7a9d-7e8f-4b4d-a2c5-0be80443dcc1/Screen_Shot_2021-09-05_at_14.05.00.png)

![Screen Shot 2021-09-05 at 14.06.26.png](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/8e8e89fd-8dd2-4e09-b6fc-38c41fcc712a/Screen_Shot_2021-09-05_at_14.06.26.png)

## Conditional GAN (cGAN)

![Screen Shot 2021-09-06 at 09.53.00.png](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/d6193902-2958-427e-8f94-58113bd81832/Screen_Shot_2021-09-06_at_09.53.00.png)

In a cGAN (conditional GAN), the discriminator is given data/label pairs instead of just data, and the generator is given a label in addition to the noise vector, indicating which class the image should belong to. The addition of labels forces the generator to learn multiple representations of different training data classes, allowing for the ability to explicitly control the output of the generator. When training the model, the label is usually combined with the data sample for both the generator and discriminator.

Here, labels can be one-hot encoded to remove ordinality and then input to both the discriminator and generator as additional layers, where they are then concatenated with their respective image inputs (i.e., concatenated with noise for the generator, and with the training set for the generator). Thus, both of the neural networks are conditioned on image class labels during training.

## Cycle Consistent GAN (CycleGAN)

Architecture for doing style transfer of images. Recall that the style transfer task is to take an image in one style (such as a photograph) and transform it to be a different style.

The cycle is one to go from Style A to Style B, and one to go from Style B to Style A.

![Screen Shot 2021-09-05 at 20.21.29.png](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/bf32e9c4-337a-42ee-8643-9b1b275ab81b/Screen_Shot_2021-09-05_at_20.21.29.png)

A recently-introduced method for image-to-image translation called CycleGAN is particularly interesting because it allows us to use un-paired training data. This means that in order to train
it to translate images from domain X to domain Y , we do not have to have exact correspondences
between individual images in those domains.

![Screen Shot 2021-09-05 at 19.59.54.png](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/36d429fe-5129-4556-9e41-bbf0f1cc81bf/Screen_Shot_2021-09-05_at_19.59.54.png)

![Screen Shot 2021-09-05 at 20.00.32.png](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/432638e0-6d94-4a6a-9712-bfa37a174dcd/Screen_Shot_2021-09-05_at_20.00.32.png)

## Super Resolution GAN

![Screen Shot 2021-09-06 at 09.55.49.png](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/bee84325-0bc1-4c04-87d0-087bd0d735d1/Screen_Shot_2021-09-06_at_09.55.49.png)

An SRGAN uses the adversarial nature of GANs, in combination with deep neural networks, to learn how to generate upscaled images (up to four times the resolution of the original). These resulting super resolution images have better accuracy and generally garner high mean opinion scores (MOS).

To train an SRGAN, a high-resolution image is first downsampled into a lower resolution image and input into a generator. The generator then tries to upsample that image into super resolution. The discriminator is used to compare the generated super-resolution image to the original high-resolution image. The GAN loss from the discriminator is then back propagated into both the discriminator and generator as shown above.

The generator uses a number of convolution neural networks (CNNs) and ResNets, along with batch-normalization layers, and ParametricReLU for the activation function. These first downsample the image before upsampling it to generate a super-resolution image. Similarly, the discriminator uses a series of CNNs along with dense layers, a Leaky ReLU, and a sigmoid activation to determine if an image is the original high-resolution image, or the super-resolution image output by the generator.

![Screen Shot 2021-09-06 at 10.02.47.png](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/f7012986-89e5-4040-b16d-674a306708ad/Screen_Shot_2021-09-06_at_10.02.47.png)

The convolution layer with “k3n64s1” stands for 3x3 kernel filters outputting 64 channels with stride 1.

### Loss function

The loss function for the generator composes of the content loss (reconstruction loss) and the adversarial loss.

![Screen Shot 2021-09-06 at 10.05.33.png](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/25a01d3d-f01c-4036-9f94-d38234d524b3/Screen_Shot_2021-09-06_at_10.05.33.png)

here, content loss is MSELoss (feature space)

The adversarial loss is defined as:

![Screen Shot 2021-09-06 at 10.06.03.png](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/ec060ed7-1a7e-417a-82e3-04d3b4adceac/Screen_Shot_2021-09-06_at_10.06.03.png)

and adversarial loss is BCEWithLogitsLoss

> Propose the use of features extracted from a pre-trained VGG network instead of low-level pixel-wise error measures.

### Evaluations

Standard metrics such as the Structural Similarity (SSIM) or the Peak Signal to Noise Ratio (PSNR) used as quantitative measurements, which are common used in evaluating SR image. GAN-based super-resolved images could have higher errors in terms of the PSNR and SSIM metrics, but still generate more appealing images.

The PSNR computes the peak signal-to-noise ratio, in decibels, between two images. This ratio is used as a quality measurement between the original and a compressed image. The higher the PSNR, the better the quality of the compressed, or reconstructed image.

The mean-square error (MSE) and the peak signal-to-noise ratio (PSNR) are used to compare image compression quality. The MSE represents the cumulative squared error between the compressed and the original image, whereas PSNR represents a measure of the peak error. The lower the value of MSE, the lower the error.

![Screen Shot 2021-09-11 at 09.50.33.png](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/17bb8d36-1a3e-489a-b000-1297b540c803/Screen_Shot_2021-09-11_at_09.50.33.png)

![Screen Shot 2021-09-11 at 09.50.43.png](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/a52a353f-5b91-4bf8-a492-2a1b27df33ad/Screen_Shot_2021-09-11_at_09.50.43.png)

### Luminance Channel

The color mapping algorithm assumes that only the spatial distribution of the luminance channel is known, and the purpose of the algorithm is to derive three-dimensional spatial distributions of the red, green and blue channels from a one-dimensional channel.

![Screen Shot 2021-09-11 at 09.51.39.png](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/7c48ed9e-834a-42e8-83cc-a0589be0b07a/Screen_Shot_2021-09-11_at_09.51.39.png)

Different approaches exist for computing the PSNR of a color image. Because the human eye is most sensitive to Luminance information, you can compute the PSNR for color images by converting the image to a color space that separates the intensity (Luminance) channel, such as YCbCr. The Y (Luminance), in YCbCr represents a weighted average of R, G, and B. G is given the most weight, again because the human eye perceives it most easily. Compute the PSNR only on the Luminance channel.

### Multi-scale GANs for Memory-efficient Generation of High Resolution Medical Images

A progressive learning strategy for GANs that starts with low resolution and adds finer details
throughout the training.

This approach combines a progressive multi-scale learning strategy with a patch-wise approach,
where low-resolution image content is learned first, and image patches at higher resolutions are conditioned on the previous scales to preserve global intensity information.

> **What is the difference between patch-wise training and fully convolutional training in ?**
Basically, fully convolutional training takes the whole MxM image and produces outputs for all sub-images in a single Conv-Net forward pass. Patch-wise training explicitly crops out the sub-images and produces outputs for each sub-image in independent forward passes. Therefore, fully convolutional training is usually substantially faster than patch-wise training.
So, for fully convolutional training, you make updates like this:
Input whole MxM image (or multiple images). Push through ConvNet -> get an entire map of outputs (maximum size MxM per image, possibly smaller). Make updates using the loss of all outputs. Now while this is quite fast, it restricts your training sampling process compared to patch-wise training: You are forced to make a lot of updates on the same image (actually, all possible updates for all sub-images) during one step of your training. That's why they write that fully convolutional training is only identical to patch-wise training, if each receptive field (aka sub-image) of an image is contained in a training batch of the patch-wise training procedure (for patch-wise training, you also could have two of ten possible sub-images from image A, three of eight possible sub-images from image B, etc. in one batch). Then, they argue that by not using all outputs during fully convolutional training, you get closer to patch-wise training again (since you are not making all possible updates for all sub-images of an image in a single training step). However, you waste some of the computation. Also, in Section 4.4/Figure 5, they describe that making all possible updates works just fine and there is no need to ignore some outputs.

It applies a conditional GAN for unsupervised domain adaptation. GANs are generative models that learn to map a random noise vector z to an output image y using a generator function. The **conditional GANs**, that learn the mapping from an observed image x additionally, G : {x, z} → y.

### Multi-scale Conditional GANs

In the lowest resolution (LR) GAN, the whole low-resolution edge image is used as an input. In the higher-resolution (HR) GANs, two conditions are used: a patch from the image of the previous resolution, upscaled to the size of the current scale; and a patch from the edge image of the current scale.

### Architecture and Training

The LR GAN uses a U-Net architecture, which is able to filter out many unimportant details and generalize better due to its bottleneck.

Three common GAN architectures are chosen as baselines: DCGAN, Pix2Pix, and progressive growing GAN (PGGAN).

> **In CNN, are upsampling and transpose convolution the same?**
The opposite of the convolutional layers are the transposed convolution layers (also known as deconvolution, but correctly mathematically speaking this is something different). They work with filters, kernels, strides just as the convolution layers but instead of mapping from e.g. 3x3 input pixels to 1 output they map from 1 input pixel to 3x3 pixels. Of course, also back-propagation works a little bit different.
The opposite of the pooling layers are the upsampling layers which in their purest form only resize the image (or copy the pixel as many times as needed). A more advanced technique is un-pooling which reverts max-pooling by remembering the location of the maxima in the max-pooling layers and in the un-pooling layers copy the value to exactly this location.

### Implementations

1. Data Thorax X-ray: [https://www.kaggle.com/raddar/chest-xrays-indiana-university](https://www.kaggle.com/raddar/chest-xrays-indiana-university)

    In this dataset, 1500 frontal images of size 20482 are used.

2. Experiments: based on unpaired image domain translation, so the trained GANs are conditioned on the object’s edges.

    ![Screen Shot 2021-09-05 at 22.28.08.png](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/806427b6-1741-4a09-8713-532d22f25546/Screen_Shot_2021-09-05_at_22.28.08.png)

    LR generator architecture

![Screen Shot 2021-09-05 at 22.29.11.png](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/b0a90edc-bb0d-4ad6-9e68-bea5085eecff/Screen_Shot_2021-09-05_at_22.29.11.png)

HR generator architecture

![Screen Shot 2021-09-05 at 22.29.42.png](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/a6fe71e5-aa8f-4b6e-a55e-869385b0dfa1/Screen_Shot_2021-09-05_at_22.29.42.png)

LR and HR networks discriminator architectures

# References

[https://junyanz.github.io/CycleGAN/](https://junyanz.github.io/CycleGAN/)

[https://arxiv.org/pdf/1703.10593.pdf](https://arxiv.org/pdf/1703.10593.pdf)

[https://arxiv.org/pdf/1611.07004.pdf](https://arxiv.org/pdf/1611.07004.pdf)

[http://www.cs.toronto.edu/~rgrosse/courses/csc321_2018/assignments/a4-handout.pdf](http://www.cs.toronto.edu/~rgrosse/courses/csc321_2018/assignments/a4-handout.pdf)

[https://towardsdatascience.com/image-to-image-translation-using-cyclegan-model-d58cfff04755](https://towardsdatascience.com/image-to-image-translation-using-cyclegan-model-d58cfff04755)

[https://towardsdatascience.com/using-conditional-deep-convolutional-gans-to-generate-custom-faces-from-text-descriptions-e18cc7b8821](https://towardsdatascience.com/using-conditional-deep-convolutional-gans-to-generate-custom-faces-from-text-descriptions-e18cc7b8821)

[https://github.com/evanhu1/pytorch-CelebA-faCeGAN](https://github.com/evanhu1/pytorch-CelebA-faCeGAN)

[https://arxiv.org/pdf/1907.01376v2.pdf](https://arxiv.org/pdf/1907.01376v2.pdf)

[https://github.com/hristina-uzunova/MEGAN](https://github.com/hristina-uzunova/MEGAN)

[https://jonathan-hui.medium.com/gan-super-resolution-gan-srgan-b471da7270ec](https://jonathan-hui.medium.com/gan-super-resolution-gan-srgan-b471da7270ec)

[https://arxiv.org/pdf/1609.04802.pdf](https://arxiv.org/pdf/1609.04802.pdf)

[https://sci-hub.se/https://ieeexplore.ieee.org/document/9153436](https://sci-hub.se/https://ieeexplore.ieee.org/document/9153436)
