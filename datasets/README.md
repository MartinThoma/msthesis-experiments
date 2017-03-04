Every dataset contains

* meta: A dict which contains...
    * n_classes: the number of classes,
    * image_width, image_height, image_depth: the dimensions of the images
* distorted_inputs(data_dir, batch_size): A function which reads the data
  and returns a generator. The generator returns data+labels for training.
* 