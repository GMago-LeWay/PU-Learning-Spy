# PU Learning

本项目源码大部分参考了仓库https://github.com/trokas/pu_learning , 在此基础上做出了些许调整。

The code of this repository was mostly from trokas/pu_learning, on the basis of which we simplify the experiments because we want to focus on Spy Method other than Bagging Method.

此项目仅供学习使用，如想要详细参考，请转向https://github.com/trokas/pu_learning

This repository is only for study. If you want to refer or use the repository, please go to trokas/pu_learning.

## Usage

Two-Step Spy Method, you need to use two model to initiate.

```python
from pu_learning import spies
model = spies(SVM(), SVM())
model.fit(X, y)
model.predict(X)
```

