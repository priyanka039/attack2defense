# Where the trained weights go

Checkpoints are not in git (they are tens of megabytes each). Download them from Google Drive folder `Attack2Defense_results` into:

```
checkpoints/resnet18_clean/best.pt
checkpoints/resnet18_fgsm/best.pt
checkpoints/resnet18_pgd/best.pt
```

Copy `config.json` from the matching results folder next to each `best.pt` if it is not already inside the checkpoint.

The API also reads reported metrics from `results/` (the unzipped Colab archive) even when weights are missing. Attacks and new evaluations need `best.pt`.
