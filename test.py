"""
HyperCam Evaluation Script
"""

from HyperCam.Classifier import *
from utils import *
import argparse


def main(args):
    # Parse configuration
    config = parse_configuration(args.config)

    # Get data
    x, y, test_x, test_y, config = read_dataset(
        config, dataset=config["data"], flatten=True
    )

    # Get classifier
    classifier = Classifier(config)

    # Train and test results
    classifier.train(x, y)
    classifier.test(test_x, test_y)

    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c",
        "--config",
        default="config/vanillahdc.yaml",
        type=str,
        help="Specify the filepath of the YAML configuration file",
    )
    args = parser.parse_args()
    main(args)
