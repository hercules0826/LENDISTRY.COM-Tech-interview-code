import math

import pytest

from classifier import centroid, classify_vector, cosine_similarity

GROUP_A = [
    [0.9, 0.1, 0.2],
    [0.8, 0.2, 0.1],
    [0.95, 0.1, 0.15],
]
GROUP_B = [
    [0.1, 0.9, 0.2],
    [0.2, 0.8, 0.1],
    [0.1, 0.95, 0.15],
]


def test_sample_query_belongs_to_group_a():
    assert classify_vector([0.85, 0.15, 0.2], GROUP_A, GROUP_B) == "0"


def test_query_like_group_b_belongs_to_group_b():
    assert classify_vector([0.15, 0.85, 0.2], GROUP_A, GROUP_B) == "1"


def test_returns_string_labels():
    assert classify_vector(GROUP_A[0], GROUP_A, GROUP_B) in {"0", "1"}


def test_magnitude_does_not_change_the_answer():
    query = [0.85, 0.15, 0.2]
    scaled = [x * 100 for x in query]
    assert classify_vector(scaled, GROUP_A, GROUP_B) == classify_vector(query, GROUP_A, GROUP_B)


def test_higher_dimensional_vectors():
    group_a = [[1.0, 0.0, 0.0, 0.1, 0.0], [0.9, 0.1, 0.0, 0.0, 0.1]]
    group_b = [[0.0, 0.0, 1.0, 0.1, 0.0], [0.1, 0.0, 0.9, 0.0, 0.1]]
    assert classify_vector([0.8, 0.0, 0.1, 0.0, 0.0], group_a, group_b) == "0"
    assert classify_vector([0.0, 0.1, 0.8, 0.0, 0.0], group_a, group_b) == "1"


def test_tie_goes_to_group_a():
    assert classify_vector([1.0, 1.0], [[1.0, 0.0]], [[0.0, 1.0]]) == "0"


def test_centroid_is_component_wise_mean():
    assert centroid([[1.0, 2.0], [3.0, 4.0]]) == [2.0, 3.0]


def test_cosine_similarity_known_values():
    assert cosine_similarity([1.0, 0.0], [1.0, 0.0]) == pytest.approx(1.0)
    assert cosine_similarity([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)
    assert cosine_similarity([1.0, 0.0], [-1.0, 0.0]) == pytest.approx(-1.0)
    assert cosine_similarity([1.0, 1.0], [1.0, 0.0]) == pytest.approx(1 / math.sqrt(2))


@pytest.mark.parametrize(
    "query, group_a, group_b",
    [
        ([0.5, 0.5], [], [[0.0, 1.0]]),            # empty group A
        ([0.5, 0.5], [[1.0, 0.0]], []),            # empty group B
        ([], [[1.0, 0.0]], [[0.0, 1.0]]),          # empty query
        ([0.5, 0.5, 0.5], [[1.0, 0.0]], [[0.0, 1.0]]),  # query dimension mismatch
        ([0.5, 0.5], [[1.0, 0.0], [1.0]], [[0.0, 1.0]]),  # ragged group
        ([0.0, 0.0], [[1.0, 0.0]], [[0.0, 1.0]]),  # zero query vector
    ],
)
def test_invalid_input_raises_value_error(query, group_a, group_b):
    with pytest.raises(ValueError):
        classify_vector(query, group_a, group_b)
