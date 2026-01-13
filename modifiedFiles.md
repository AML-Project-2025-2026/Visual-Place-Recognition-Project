# Modified/added files with respect to the original Repository

This document lists the files that were modified or added with respect to the original repository, along with a brief description of the changes.

## Modified files

* **`Visual-Place-Recognition-Project/VPR-methods-evaluation/main.py`**
  Modified to use **dot product** instead of **L2 distance** for similarity computation.

* **`Visual-Place-Recognition-Project/VPR-methods-evaluation/vpr_models/mixvpr.py`**
  Modified with the implementation of **ResNet-50**.

* **`Visual-Place-Recognition-Project/match_queries_preds.py`**
  Modified to compute also the **average processing time per query**.

* **`Visual-Place-Recognition-Project/reranking.py`**
  Modified, given the change just above, to extract results from a **dictionary containing a list**, rather than directly from a list.

## Added files

* **`Visual-Place-Recognition-Project/adaptive_reranking.py`**
  More details can be found in the file.

* **`Visual-Place-Recognition-Project/adaptive_reranking_eval.py`**
  More details can be found in the file.
