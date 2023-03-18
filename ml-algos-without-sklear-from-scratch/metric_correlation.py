

#!/usr/bin/env python3
import argparse
import dataclasses

import numpy as np

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--bootstrap_samples", default=100, type=int, help="Bootstrap samples")
parser.add_argument("--data_size", default=1000, type=int, help="Data set size")
parser.add_argument("--plot", default=False, const=True, nargs="?", type=str, help="Plot the predictions")
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--seed", default=42, type=int, help="Random seed")
# For these and any other arguments you add, ReCodEx will keep your default value.


class ArtificialData:
    @dataclasses.dataclass
    class Sentence:
        """ Information about a single dataset sentence."""
        gold_edits: int  # Number of required edits to be performed.
        predicted_edits: int  # Number of edits predicted by a model.
        predicted_correct: int  # Number of correct edits predicted by a model.
        human_rating: int  # Human rating of the model prediction.

    def __init__(self, args: argparse.Namespace):
        generator = np.random.RandomState(args.seed)

        self.sentences = []
        for _ in range(args.data_size):
            gold = generator.poisson(2)
            correct = generator.randint(gold + 1)
            predicted = correct + generator.poisson(0.5)
            human_rating = max(0, int(100 - generator.uniform(5, 8) * (gold - correct)
                                      - generator.uniform(8, 13) * (predicted - correct)))
            self.sentences.append(self.Sentence(gold, predicted, correct, human_rating))



def main(args: argparse.Namespace) -> tuple[float, float]:
    def pearsonr(x, y):
        sum_x = float(sum(x))
        sum_y = float(sum(y))
        sum_xx = 0
        sum_yy = 0
        for i in x:
            sum_xx+= i * i

        for i in y:
            sum_yy+= i * i
        p_sum = sum(xi * yi for xi, yi in zip(x, y))
        nominator = p_sum - (sum_x * sum_y / len(x))
        denominator = pow((sum_xx - (sum_x ** 2) / len(x)) * (sum_yy - (sum_y ** 2) / len(x)), 0.5)
        if denominator == 0: return 0
        return nominator / denominator

    # Create the artificial data.
    data = ArtificialData(args)

    # Create `args.bootstrap_samples` bootstrapped samples of the dataset by
    # sampling sentences of the original dataset, and for each compute
    # - average of human ratings,
    # - TP, FP, FN counts of the predicted edits.
    human_ratings, predictions = [], []
    generator = np.random.RandomState(args.seed)
    for _ in range(args.bootstrap_samples):
        # Bootstrap sample of the dataset.
        sentences = generator.choice(data.sentences, size=len(data.sentences), replace=True)


        # TODO: Append the average of human ratings of `sentences` to `human_ratings`.
        avg = []
        for sentence in sentences:
            avg.append(sentence.human_rating)

        human_ratings.append(sum(avg)/ len(avg))

        # TODO: Compute TP, FP, FN counts of predicted edits in `sentences`
        TP = 0
        FP = 0
        FN = 0

        for sentence in sentences:
            TP += sentence.predicted_correct
            FP += sentence.predicted_edits - sentence.predicted_correct
            FN += sentence.gold_edits - sentence.predicted_correct

        # and append them to `predictions`.
        predictions.append([TP,FP,FN])


    betas, correlations = [], []
    for beta in np.linspace(0, 2, 201):
        betas.append(beta)
        fscores =[]
        for i in range(len(predictions)):
            TP = predictions[i][0]
            FP = predictions[i][1]
            FN = predictions[i][2]

            recall = TP / (TP + FN)
            precision = TP / (TP + FP)
            fscore = (1 + (beta ** 2)) * ((precision * recall) / (((beta ** 2) * precision) + recall))
            fscores.append(fscore)

        # TODO: For each bootstrap dataset, compute the F_beta score using
        # the counts in `predictions` and then manually compute the Pearson
        # correlation between the computed scores and `human_ratings`. Append
        # the result to `correlations`.

        pearsons_coefficient = pearsonr(human_ratings,fscores)
        correlations.append(pearsons_coefficient)

    if args.plot:
        import matplotlib.pyplot as plt
        plt.plot(betas, correlations)
        plt.xlabel(r"$\beta$")
        plt.ylabel(r"Pearson correlation of $F_\beta$-score and human ratings")
        plt.show() if args.plot is True else plt.savefig(args.plot, transparent=True, bbox_inches="tight")

    # TODO: Assign the highest correlation to `best_correlation` and
    # store corresponding beta to `best_beta`.
    best_beta = betas[np.asarray(correlations).argmax()]
    print(np.max(np.asarray(correlations)))
    best_correlation = np.max(np.asarray(correlations))

    return best_beta, best_correlation


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    best_beta, best_correlation = main(args)

    print("Best correlation of {:.3f} was found for beta {:.2f}".format(
        best_correlation, best_beta))