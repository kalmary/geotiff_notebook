# simple pipeline to run all programs in correct order
from DataAugmenter import augment_data
from DetectDecrease import detect_decrease
from Evaluation import evaluate_results, summarize_results
from DataClustering import cluster_data


def main():
    # Step 1: Data Augmentation
    augment_data("data")

    # Step 2: Detect Decrease
    detect_decrease("data")

    # Step 3: Evaluation
    evaluate_results("data")
    summarize_results("data")
    
    # Step 4: Data Clustering
    cluster_data("data")

if __name__ == "__main__":
    main()
