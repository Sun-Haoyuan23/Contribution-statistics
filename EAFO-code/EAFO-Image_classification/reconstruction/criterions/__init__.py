from .cross_entropy import CrossEntropy

def build_criterion(args):
    criterion_name = args.criterion.lower()

    losses = ['labels']
    weight_dict = {'loss_ce': 1}
    
    return CrossEntropy(losses=losses, weight_dict=weight_dict)
