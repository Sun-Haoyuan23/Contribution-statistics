from .deit import *
from .tnt import *
from .vit import *

__vars__ = vars()


def build_model(args):
    from termcolor import cprint
    from .. import datasets
    from ..utils.misc import is_main_process

    model_lib = args.model_lib.lower()
    model_name = args.model.lower()

    if 'num_classes' in args.model_kwargs.keys():
        cprint(f"Warning: Do NOT set 'num_classes' in 'args.model_kwargs'. "
               f"Now fetching the 'num_classes' registered in 'reconstruction/datasets/__init__.py'.", 'light_yellow')

    try:
        num_classes = datasets.num_classes[args.dataset.lower()]
    except KeyError:
        print(f"KeyError: 'num_classes' for the dataset '{args.dataset.lower()}' is not found. "
              f"Please register your dataset's 'num_classes' in 'reconstruction/datasets/__init__.py'.")
        exit(1)

    args.model_kwargs['num_classes'] = num_classes

    pretrained = not args.no_pretrain and is_main_process()

    if model_lib == 'torchvision-ex':
        try:
            model = __vars__[model_name](**args.model_kwargs)
        except KeyError:
            print(f"KeyError: Model '{model_name}' is not found.")
            exit(1)

        return model

    if model_lib == 'timm':
        import timm
        return timm.create_model(model_name=model_name, pretrained=pretrained, **args.model_kwargs)

    raise ValueError(f"Model lib '{model_lib}' is not found.")
