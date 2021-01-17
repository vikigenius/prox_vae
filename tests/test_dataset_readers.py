import pytest
from src.data.dataset_readers import SentenceSimilarityDatasetReader


@pytest.mark.skip(reason='TODO')
def test_sentence_similarity_reader():
    num_negative_samples = 2
    reader = SentenceSimilarityDatasetReader(num_negative_samples=num_negative_samples)
    instances = reader.read('tests/fixtures/snli_samples.txt')
    for instance in instances:
        assert 'negatives' in instance
        assert len(instance['negatives']) == 2
