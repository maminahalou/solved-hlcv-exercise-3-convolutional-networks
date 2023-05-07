Download Link: https://assignmentchef.com/product/solved-hlcv-exercise-3-convolutional-networks
<br>
<h1>Question 1: Implement Convolutional Network</h1>

In this question we will implement a five-layered convolutional neural network architecture as well as the loss function to train it. Starting from the main file ex3 convnet.py. Refer to the comments in the code to the exact places where you need to fill in the code.

Figure 1: Figure depicting the convolutional neural networks.

<ol>

 <li>Our architecture is shown in Fig 1. It has five convolution blocks. Each block is consist of convolution, max pooling and ReLU operation in that order. We will use 3×3 kernels in all convolutional layers. Set the padding and stride of the convolutional layers so that they maintain the spatial dimensions. Max pooling operations are done with 2×2 kernels, with a stride of 2, thereby halving the spatial resolution each time. Finally, five stacking these five blocks leads to a 512 × 1 × 1 feature map. Classification is achieved by a fully connected layer. We will train convolutional neural networks on CIFAR-10 dataset. Implement a class ConvNet to define the model described. The ConvNet takes 32×32 color images as inputs and has 5 hidden layers with 128, 512, 512, 512, 512 filters, and produces 10-class classification. The code to train the model is already provided. Train the above model and report the training and validation accuracies.</li>

 <li>Implement a function PrintModelSize, which calculates and prints the number of parameters of a neural network. PrintModelSize takes a model as input, and returns the parameters of a model. This gives us a measure of model capacity. Report the number of parameters for the described model.</li>

 <li>Implement a function VisualizeFilter, which visualizes the filters of the first convolution layer implemented in Q1.a. In other words, you need to show 128 filters with size 3 × 3 as color images (since each filter has 3 input channels). Stack these into 3×3 color images into one large image. You can use the imshow function from the matplotlib library to visualize the weights. See an example in figure 2 Compare the filters before and after training. Do you see any patterns ?</li>

</ol>

Figure 2: Example visualization of trained filters in the first conv layers.

<h1>Question 2: Improve training of Convolutional Networks</h1>

<ol>

 <li>Batch normalization is a widely used operation in neural networks, which will increase the speed of convergence, as well as reach higher performance. Read the paper “Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift” for more theoretical details.</li>

</ol>

In practise, this operation are implemented in most popular toolbox, such as PyTorch, TensorFlow. Add batch normalization in the model of Q1.a. Please keep other hyperparameters same, but only adding batch normalization. The ConvNet with batch normalization still uses the same class with Q1.a, but different arguments. Check the code for details. In each block, the computations are in the order of convolution, batch normalization, pooling, ReLU. Compare the loss curves and accuracy using batch normalization to its counterpart in Q1.a.

<ol>

 <li>Early stopping is a method to alleviate overfitting. Instead of reporting the performance of final model, early stopping also saves the best model on the validation set during training. Increase the training epochs to 50 in Q1.a and Q2.a, and compare the best model and latest model on the training set. Due to the randomness, you can train multiple times to verify and observe overfitting and early stopping.</li>

</ol>

<h1>Question 3: Improve generalization of Convolutional Networks</h1>

e saw in q2 that the model can start over-fitting to the training set, if we continue training for long. To prevent over-fitting, there are two main paradigms we can focus one. First is to get more training data. This might be difficult and expensive process involving significant effort. However, it is usually the most effective way to learn more general models. A cheaper alternative is to perform data augmentation to increase the effective data size. Second approach is to regularize the model. In the following sub-questions we will experiment with each of these appraoches and measure the effect on the model generalization

<ol>

 <li>Data augmentation is the process of creating more training data by applying certain transformations to the training set images. Usually the underlying assumption is that the label of the image does not change under the applied transformations. This includes geometric transformations like translation, rotation, scaling, flipping, random cropping and color transforamtions like greyscale, colorjitter. For every image in the training batch, a random transformation is sampled from the possible ones (for eg. random number of pixels to translate the image by) and is applied onto the image. While designing the data input pipeline, we must choose the hyper-parameters for these transformations (for eg. limits of translation or rotation) based on things we expect to see in the test-set/real world. Your task in this question is to implement the data augmentation for the CIFAR-10 classification task.</li>

</ol>

Many of these transformations are implemented in the torchvision.transforms package. Familiarize yourself with the apis of these transforms, and functions to compose multiple transforms or randomly sample them. Next, implement geometric and color space data-augmentations for the CIFAR-10 dataset, by choosing the right functions and order of application. Tune the hyper-parameters of these data-augmentations to improve the validation performance. You will need to train the model a bit longer (20-30 epochs) with data augmentation, as the training data is effectively larger now. Discuss which augmentations work well for you in the report.

<ol>

 <li>Dropout is a popular scheme to regularize the model to improve generalization. Dropout layer works by setting the input activations randomly to zero at the output. You can implement Dropout by adding the nn.Dropout layer between the conv blocks in your model. The layer has a single hyper-parameteres <em>p</em>, which is the probability of dropping the input activations. High values of <em>p </em>regularize the model heavily and decrease model capacity, but with low values, the model might overfit. Find the right hyper-parameter for <em>p </em>by training the model for different values of <em>p </em>and comparaing training and validation accuracies. You can use the same parameter <em>p </em>for all layers. You can also disable the data-augmentation from previous step while running this experiment, to clearly see the benefit of dropout. Show the plot of training and validation accuracies for different values of dropout (0.1 – 0.9) in the report.</li>

</ol>

<h1>Question 4: Use pretrained networks</h1>

It has become standard practice in computer vision tasks related to images, to use a convolutional network pre-trained as the backbone feature extraction network and train new layers on top for the target task. In this question we will implement such a model. We will use the VGG 11 bn network from the torchvision.models library as our backbone network. This model has been trained on ImageNet achieving top-5 error rate of 10.19%. It consists of 8 convolutional layers followed by adaptive average pooling and fully-connected layers to perform the classification. We will get rid of the averate pooling and fully-connected layers from the VGG 11 bn model and attach our own fully connected layers to perform the Cifar-10 classifcation.

<ol>

 <li>Instantiate a pretrained version of the VGG 11 bn model with ImageNet pre-trained weights. Add two fully connected layers on top, with Batch Norm and ReLU layers in between them to build the CIFAR-10 10-class classifier. Note that you will need to set the correct mean and variance in the data-loader, to match the mean and varaince the data was normalized with when the VGG 11 bn was trained. Train only the newly added layers while disabling gradients for rest of the network. Each parameter in PyTorch has a requires grad flag, which can be turned off to disable gradient computation for it. Get familiar with this gradient control mechanism in PyTorch and train the above model. As a reference point you will see validation accuracies in the range (61-65%) if implemented correctly.</li>

 <li>We can see that while the ImageNet features are useful, just learning the new layers does not yield better performance than training our own network from scratch. This is due to the domain-shift between the ImageNet dataset (224×224 resolution images) and the CIFAR-10 dataset (32×32 images). To improve the performance we can fine-tune the whole network on the CIFAR-10 dataset, starting from the ImageNet initialization. To do this, enable gradient computation to the rest of the network, and update all the model parameters. Additionally train a baseline model where the same entire network is trained from scratch, without loading the ImageNet weights. Compare the two models training curves, validation and testing performance in the report.</li>

</ol>