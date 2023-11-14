from autogluon.tabular import TabularDataset, TabularPredictor
import time

import time
import logging

log_filename = 'E:/vscode/automl_research_file/automl file/result_file/Autogluon_log_file.log'
logging.basicConfig(filename=log_filename, level=logging.INFO, format='%(asctime)s - %(levelname)s: %(message)s')

# 훈련 데이터와 테스트 데이터 불러오기
train_data = TabularDataset('E:/vscode/automl_research_file/dataset/train_data_selected.csv')
test_data = TabularDataset('E:/vscode/automl_research_file/dataset/test_data_selected.csv')




# Lists to store metrics across models
f1_scores = []
accuracies = []
precisions = []
recalls = []
training_times = []

# Train multiple AutoGluon models
num_models_to_train = 10

for i in range(num_models_to_train):
    # Create a TabularPredictor and measure training time
    start_time = time.time()
    predictor = TabularPredictor(label='DIS').fit(train_data)
    end_time = time.time()
    
    # Evaluate the model on the test data
    performance = predictor.evaluate(test_data)
    
    # Store F1 score and other metrics
    f1_score = performance['f1']
    accuracy = performance['accuracy']
    precision = performance['precision']
    recall = performance['recall']
    
    f1_scores.append(f1_score)
    accuracies.append(accuracy)
    precisions.append(precision)
    recalls.append(recall)
    training_times.append(end_time - start_time)

    log_message = (
        f"Iteration: {i + 1}\n"
        f"F1 Score: {f1_score}\n"
        f"Accuracy: {accuracy}\n"
        f"Precision: {precision}\n"
        f"Recall: {recall}\n"
        f"Training Time: {training_times} seconds\n"
    )
    logging.info(log_message)
    


# Calculate statistics on model performances
max_f1 = max(f1_scores)
min_f1 = min(f1_scores)
avg_f1 = sum(f1_scores) / num_models_to_train

max_accuracy = max(accuracies)
min_accuracy = min(accuracies)
avg_accuracy = sum(accuracies) / num_models_to_train

max_precision = max(precisions)
min_precision = min(precisions)
avg_precision = sum(precisions) / num_models_to_train

max_recall = max(recalls)
min_recall = min(recalls)
avg_recall = sum(recalls) / num_models_to_train

max_training_time = max(training_times)
min_training_time = min(training_times)
avg_training_time = sum(training_times) / num_models_to_train



# Save the log message to a file in the current directory
log_message = (
    f"Best F1 Score: {max_f1}\n"
    f"Minimum F1 Score: {min_f1}\n"
    f"Average F1 Score: {avg_f1}\n\n"
    f"Best Accuracy: {max_accuracy}\n"
    f"Minimum Accuracy: {min_accuracy}\n"
    f"Average Accuracy: {avg_accuracy}\n\n"
    f"Best Precision: {max_precision}\n"
    f"Minimum Precision: {min_precision}\n"
    f"Average Precision: {avg_precision}\n\n"
    f"Best Recall: {max_recall}\n"
    f"Minimum Recall: {min_recall}\n"
    f"Average Recall: {avg_recall}\n\n"
    f"Maximum Training Time: {max_training_time} seconds\n"
    f"Minimum Training Time: {min_training_time} seconds\n"
    f"Average Training Time: {avg_training_time} seconds\n"
)

logging.info(log_message)
