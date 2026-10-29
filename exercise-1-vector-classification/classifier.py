"""Classify a query vector as belonging to Group A ("0") or Group B ("1").

Approach: nearest centroid by cosine similarity, in pure Python.

1. Average each group into a single centroid vector.
2. Compute the cosine similarity between the query and each centroid.
3. Return the label of the more similar centroid.

Cosine similarity compares direction, not magnitude. That suits data like the
sample, where each group is defined by which dimensions are large ("mostly the
first component" vs "mostly the second"), not by vector length.
"""

import math
from typing import Sequence

Vector = Sequence[float]

GROUP_A = "0"
GROUP_B = "1"


def dot(a: Vector, b: Vector) -> float:
    return sum(x * y for x, y in zip(a, b))


def norm(v: Vector) -> float:
    return math.sqrt(dot(v, v))


def cosine_similarity(a: Vector, b: Vector) -> float:
    """Cosine of the angle between a and b, in [-1, 1]."""
    norm_a, norm_b = norm(a), norm(b)
    if norm_a == 0 or norm_b == 0:
        raise ValueError("Cosine similarity is undefined for a zero vector.")
    return dot(a, b) / (norm_a * norm_b)


def centroid(vectors: Sequence[Vector]) -> list[float]:
    """Component-wise mean of a non-empty list of equal-length vectors."""
    count = len(vectors)
    return [sum(components) / count for components in zip(*vectors)]


def _validate(query: Vector, group_a: Sequence[Vector], group_b: Sequence[Vector]) -> None:
    if not group_a or not group_b:
        raise ValueError("Both groups must contain at least one vector.")
    dimension = len(query)
    if dimension == 0:
        raise ValueError("The query vector must not be empty.")
    for name, group in (("group_a", group_a), ("group_b", group_b)):
        for i, vector in enumerate(group):
            if len(vector) != dimension:
                raise ValueError(
                    f"{name}[{i}] has {len(vector)} dimensions; "
                    f"the query has {dimension}."
                )


def classify_vector(query: Vector, group_a: Sequence[Vector], group_b: Sequence[Vector]) -> str:
    """Return "0" if query is closer to group_a, or "1" if closer to group_b.

    An exact tie goes to group_a ("0").

    Raises ValueError if a group is empty, the dimensions do not match, or
    the query or a centroid is a zero vector (cosine similarity is undefined).
    """
    _validate(query, group_a, group_b)

    similarity_a = cosine_similarity(query, centroid(group_a))
    similarity_b = cosine_similarity(query, centroid(group_b))

    return GROUP_A if similarity_a >= similarity_b else GROUP_B


if __name__ == "__main__":
    group_a = [
        [0.9, 0.1, 0.2],
        [0.8, 0.2, 0.1],
        [0.95, 0.1, 0.15],
    ]
    group_b = [
        [0.1, 0.9, 0.2],
        [0.2, 0.8, 0.1],
        [0.1, 0.95, 0.15],
    ]
    query_vector = [0.85, 0.15, 0.2]

    print(classify_vector(query_vector, group_a, group_b))  # "0"
