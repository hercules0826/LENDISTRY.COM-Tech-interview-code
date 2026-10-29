# Exercise 1: Vector Classification

`classify_vector(query, group_a, group_b)` returns `"0"` if the query belongs
to Group A and `"1"` if it belongs to Group B.

## Approach: nearest centroid by cosine similarity

1. **Centroid.** Average each group component-wise into one representative vector.
2. **Similarity.** Compute the cosine similarity between the query and each centroid:
   `cos(q, c) = (q · c) / (‖q‖ ‖c‖)`.
3. **Decision.** Return the label of the more similar centroid.

Everything is written by hand with the standard library (`math.sqrt` only), as the
brief requires: no NumPy, scikit-learn or similarity libraries.

### Why this approach

- **It fits the data.** The groups differ by *direction*: Group A vectors are
  dominated by the first component, Group B by the second. Cosine similarity
  measures direction and ignores length, so `[0.85, 0.15, 0.2]` and
  `[85, 15, 20]` are classified the same way.
- **It is simple and fast.** Building the centroids is O(n·d), and each query is
  O(d) after that. It is easy to read and explain.
- **It resists noise.** Averaging smooths out individual outlier vectors.

### Alternatives considered

| Option | Why I did not choose it |
| --- | --- |
| k-nearest neighbours (compare with every vector) | Handles irregular cluster shapes better, but needs a choice of `k` and costs O(n·d) per query. A centroid is enough for clearly separated groups like these. |
| Euclidean distance to centroid | Sensitive to vector magnitude. Use it if magnitude carries meaning in your data. |
| Mean similarity to every member of the group | Close to the centroid approach, but more work per query. |

## Assumptions

- All vectors, including the query, have the same dimension. A mismatch raises `ValueError`.
- Both groups are non-empty.
- A zero query vector, or a zero centroid, has no direction, so cosine
  similarity is undefined and `ValueError` is raised rather than guessing.
- An exact tie goes to Group A (`"0"`). The brief does not specify a rule, so
  I picked a deterministic one.
- The return value is the string `"0"` or `"1"`, matching the expected usage.

## Run

```bash
cd exercise-1-vector-classification
python classifier.py        # runs the sample data, prints 0
python -m pytest -q         # runs the tests
```

```python
from classifier import classify_vector

classify_vector([0.85, 0.15, 0.2], group_a, group_b)  # "0"
```
