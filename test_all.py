"""
HyperCam Evaluation Script - all classifiers and datasets
"""

from HyperCam.Classifier import *
from utils import *
import argparse


def main():
    classifiers = ["vanillahdc", "countsketch", "bloomfilter"]
    datasets = ["mnist", "fashion-mnist"]
    num_runs = 5
    
    results = {dataset: {classifier: [] for classifier in classifiers} for dataset in datasets}
    
    with open("results_all.txt", "w") as f:
        for run in range(1, num_runs + 1):
            print(f"\n### Run {run}/{num_runs} ###")
            f.write(f"\n### Run {run}/{num_runs} ###\n")
            
            for dataset in datasets:
                print(f"\nEvaluating on {dataset.upper()}...")
                f.write(f"\nEvaluating on {dataset.upper()}...\n")
                
                for classifier_name in classifiers:
                    # Parse configuration
                    config = parse_configuration(f'config/{classifier_name}.yaml')
                    
                    # Get data
                    x, y, test_x, test_y, config = read_dataset(
                        config, dataset=dataset, flatten=True
                    )
                    
                    # Get classifier
                    classifier = Classifier(config)
                    
                    # Train and test results
                    classifier.train(x, y)
                    acc = classifier.test(test_x, test_y)
                    results[dataset][classifier_name].append(acc)
                    
                    result_str = f"Config: {classifier_name} -> Accuracy: {acc:.4f}"
                    print(result_str)
                    f.write(result_str + "\n")
        
        # Final results summary
        print("\n### Final Results Summary ###")
        f.write("\n### Final Results Summary ###\n")
        
        for dataset, classifier_results in results.items():
            print(f"\nDataset: {dataset.upper()}")
            f.write(f"\nDataset: {dataset.upper()}\n")
            
            for classifier_name, acc_list in classifier_results.items():
                avg_acc = sum(acc_list) / len(acc_list)
                acc_list_str = ", ".join([f"{acc:.4f}" for acc in acc_list])
                
                summary_str = f"  - {classifier_name}: {acc_list_str} | Avg: {avg_acc:.4f}"
                print(summary_str)
                f.write(summary_str + "\n")

    return 0


if __name__ == "__main__":
    main()
