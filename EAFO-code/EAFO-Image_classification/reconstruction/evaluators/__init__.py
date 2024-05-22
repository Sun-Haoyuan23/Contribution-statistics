from .default import Evaluators


def build_evaluator(args):
    evaluator_name = args.evaluator.lower()

    if evaluator_name in ['default']:
        metrics = ['acc', 'recall', 'precision', 'f1']
        return Evaluators(metrics)

    raise ValueError(f"Evaluator '{evaluator_name}' is not found.")
