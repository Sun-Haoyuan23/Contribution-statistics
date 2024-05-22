# Way to put datasets

**The default `--data_root` is `./data`. Put datasets in this directory.**

Each dataset should be stored in a separate directory: 

```
data/
  ├── cifar10/
  ├── cifar100/
  └── imagenet1k/
```

It is noteworthy that all directory names are in **lowercase**.

If you would like to change the data root, please set the argument `--data_root`.
