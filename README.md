# Deep Learning Experiments

The course projects for Deep Learning, including:

## CNN experiments on CIFAR-10
In order to train fast (with FLOPs as few as possible) and train better (with accuracy as high as possible), tuned a variety of hyperparameters, searched for optimizations and tricks.

## Novel Image Captioning
Using encoder-decoder framework and automatic weight transfer tricks, trained on COCO 2014 an image captioning model which can describe objects which never appeared in the paired training data.

Reference:

Hendricks, L.A., Venugopalan, S., Rohrbach, M., Mooney, R.J., Saenko, K., & Darrell, T. (2015). Deep Compositional Captioning: Describing Novel Object Categories without Paired Training Data. *2016 IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*, 1-10.

## Functa
Using MAML framework to train an ideal SIREN initialization and modulation mapping on MNIST and CIFAR-10, which can quickly fit an INR on every data point. Leveraged Meta-SGD to further enhance the meta-learning effect.

Reference:

Dupont, E., Kim, H., Eslami, S.M., Jimenez Rezende, D., & Rosenbaum, D. (2022). From data to functa: Your data point is a function and you can treat it like one. *International Conference on Machine Learning*.
