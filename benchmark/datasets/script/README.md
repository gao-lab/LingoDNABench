## Large dataset
### TFBS/DNA accessibility/Histone modification
Given the large size of the dataset, the embeddings are converted into the TFRecord format for training. This process involves splitting the data into appropriately sized chunks, storing them as TFRecords, and subsequently training them through a TensorFlow pipeline.
