import time
import pandas as pd
from tpot import TPOTClassifier
#from sklearn.model_selection import train_test_split
from sklearn.metrics import *
import logging

# Configure the logging module
log_filename = 'E:/vscode/automl_research_file/automl file/result_file/Tpot_log_file.log'
logging.basicConfig(filename=log_filename, level=logging.INFO, format='%(asctime)s - %(levelname)s: %(message)s')

# Lists to store metrics across models
accuracies = []
precisions = []
recalls = []
f1_scores = []
training_times = []

# 데이터 불러오기
data = pd.read_csv('E:/vscode/automl_research_file/dataset/train_data_selected.csv')
test_data = pd.read_csv('E:/vscode/automl_research_file/dataset/test_data_selected.csv')

# 특성과 타겟 분리
X_train = data.drop(columns=['DIS'])  
y_train = data['DIS'] 

# TPOT 설정
tpot = TPOTClassifier(generations=5, population_size=20, verbosity=2, cv=10, scoring='f1', max_time_mins=10)

#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
# 10번 반복
for i in range(10):
    
    # TPOT 모델 훈련 
    start_time = time.time()
    tpot.fit(X_train, y_train)
    end_time = time.time()
    training_time = end_time - start_time

    y_pred = tpot.predict(test_data.drop(columns=['DIS']))
    accuracy = accuracy_score(test_data['DIS'], y_pred)
    precision = precision_score(test_data['DIS'], y_pred)
    recall = recall_score(test_data['DIS'], y_pred)
    f1 = f1_score(test_data['DIS'], y_pred)
    
    # 지표 및 훈련 시간을 리스트에 추가
    accuracies.append(accuracy)
    precisions.append(precision)
    recalls.append(recall)
    f1_scores.append(f1)
    training_times.append(training_time)
    
    # 로그 파일에 결과를 남김
    log_message = (
        f"Iteration {i + 1}:\n"
        f"Test Data Accuracy: {accuracy}\n"
        f"Test Data Precision: {precision}\n"
        f"Test Data Recall: {recall}\n"
        f"Test Data F1 Score: {f1}\n"
        f"TPOT Model Training Time: {training_time} seconds\n"
    )
    logging.info(log_message)
    logging.info("-" * 30)

# 평균, 최소, 최대값 계산
avg_accuracy = sum(accuracies) / len(accuracies)
max_accuracy = max(accuracies)
min_accuracy = min(accuracies)

avg_precision = sum(precisions) / len(precisions)
max_precision = max(precisions)
min_precision = min(precisions)

avg_recall = sum(recalls) / len(recalls)
max_recall = max(recalls)
min_recall = min(recalls)

avg_f1 = sum(f1_scores) / len(f1_scores)
max_f1 = max(f1_scores)
min_f1 = min(f1_scores)

avg_training_time = sum(training_times) / len(training_times)
max_training_time = max(training_times)
min_training_time = min(training_times)

# 결과를 로그 파일에 남김
log_message = (
    f"Average Accuracy: {avg_accuracy}\n"
    f"Best Accuracy: {max_accuracy}\n"
    f"Minimum Accuracy: {min_accuracy}\n\n"
    
    f"Average Precision: {avg_precision}\n"
    f"Best Precision: {max_precision}\n"
    f"Minimum Precision: {min_precision}\n\n"
    
    f"Average Recall: {avg_recall}\n"
    f"Best Recall: {max_recall}\n"
    f"Minimum Recall: {min_recall}\n\n"
    
    f"Average F1 Score: {avg_f1}\n"
    f"Best F1 Score: {max_f1}\n"
    f"Minimum F1 Score: {min_f1}\n\n"
    
    f"Average Training Time: {avg_training_time} seconds\n"
    f"Maximum Training Time: {max_training_time} seconds\n"
    f"Minimum Training Time: {min_training_time} seconds\n"
)
logging.info(log_message)