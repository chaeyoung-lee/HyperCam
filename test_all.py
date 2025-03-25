"""
HyperCam Evaluation Script - all classifiers and datasets
"""

from HyperCam.Classifier import *
from utils import *
import argparse


def main():
    classifiers = ["vanillahdc", "countsketch", "bloomfilter"]
    datasets = ["mnist", "fashion-mnist"]
    
    results = {"mnist": [], "fashion-mnist": []}
    for dataset in datasets:
        for classifier in classifiers:
            # Parse configuration
            config = parse_configuration(f'config/{classifier}.yaml')

            # Get data
            x, y, test_x, test_y, config = read_dataset(
                config, dataset=dataset, flatten=True
            )

            # Get classifier
            classifier = Classifier(config)

            # Train and test results
            classifier.train(x, y)
            acc = classifier.test(test_x, test_y)
            results[dataset].append((classifier, acc))
            
            print(f"Config: {classifier} -> Accuracy: {acc:.4f}")
    
    # Display results
    print("\n### Final Results Summary ###")
    for dataset, acc_list in results.items():
        print(f"\nDataset: {dataset}")
        for config, acc in acc_list:
            print(f"  - {config}: {acc:.4f}")

    return 0


if __name__ == "__main__":
    main()