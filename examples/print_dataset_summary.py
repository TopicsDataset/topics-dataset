import numpy as np
from collections import Counter


def main():
    sample = np.load('data/dataset_sample.npy')
    print('Data shape:', sample.shape)
    topic_ids = Counter(sample[:, 0].astype(np.int32))
    print('Record counts by topic id:', topic_ids)
    text_embeding_size = 300
    text_embeddings = sample[:, 1:1 + text_embeding_size]
    records_with_text = np.sum(text_embeddings.sum(axis=1) > 0)
    print('proportion of records with text: ', records_with_text / len(sample))

    image_embedings = sample[:, 1 + text_embeding_size:]
    records_with_image = np.sum(image_embedings.sum(axis=1) > 0)
    print('proportion of records with image: ', records_with_image / len(sample))


if __name__ == '__main__':
    main()
