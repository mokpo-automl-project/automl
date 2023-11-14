import numpy as np
import pandas as pd
#from sklearn.model_selection import train_test_split
import h2o
from h2o.automl import H2OAutoML
from sklearn.metrics import *
import time
import logging

h2o.init(max_mem_size='1G')  # -1은 모든 CPU 스레드 사용, max_mem_size는 메모리 크기 설정
# 로그 파일 설정
log_filename = 'E:/vscode/automl_research_file/automl file/result_file\H20_log.log'
logging.basicConfig(filename=log_filename, level=logging.INFO, format='%(asctime)s - %(levelname)s: %(message)s')

# 훈련 데이터와 테스트 데이터 로드 
train_data = pd.read_csv('E:/vscode/automl_research_file/dataset/train_data_selected.csv')
test_data = pd.read_csv('E:/vscode/automl_research_file/dataset/test_data_selected.csv')
# 타겟 열 지정 (성별 열)
target_column = 'DIS'  # 내가 원하는걸 여기에 넣음 
x = list(train_data.columns)
y = target_column

x.remove(y)

# 결과를 저장할 리스트 초기화
f1_scores = []
accuracy_scores = []
precision_scores = []
recall_scores = []
training_times = []  # 모델 훈련 시간 저장을 위한 리스트 추가

# 모델 실행 및 결과 저장을 위한 반복
num_iterations = 10
max_runtime_secs = 600  # 최대 훈련 시간을 초 단위로 설정 (예: 3600초 = 1시간)

results = []
#train, valid = train_test_split(train_data, test_size=0.3, shuffle=False)
for i in range(num_iterations):
    

    h2o_train = h2o.H2OFrame(train_data)
    h2o_valid = h2o.H2OFrame(test_data)

    h2o_train[y] = h2o_train[y].asfactor()
    h2o_valid[y] = h2o_valid[y].asfactor()

    # 모델 훈련 시작 시간 기록
    start_time = time.time()

    aml = H2OAutoML(sort_metric='F1', exclude_algos=['XGBoost', 'StackedEnsemble'],max_runtime_secs=max_runtime_secs)
    aml.train(x=x, y=y, training_frame=h2o_train, leaderboard_frame=h2o_valid)

    # 모델 훈련 종료 시간 기록
    end_time = time.time()

    # 모델 훈련 시간 계산 및 저장
    training_time = end_time - start_time
    training_times.append(training_time)

    leaderboard = aml.leaderboard
    
    model_id = aml.leader.model_id

    h2o_test = h2o.H2OFrame(test_data)

    predictions = aml.leader.predict(h2o_test)

    predictions_df = predictions.as_data_frame()
    column_names = predictions_df.columns
    predicted_classes = predictions_df['predict']

    actual_classes = test_data['DIS']

    f1_score_value = f1_score(actual_classes, predicted_classes)
    accuracy = accuracy_score(actual_classes, predicted_classes)
    precision = precision_score(actual_classes, predicted_classes)
    recall = recall_score(actual_classes, predicted_classes)

    result = {
        "Iteration": i + 1,
        "Model_ID": model_id,
        "F1_Score": f1_score_value,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "Training_Time": training_time  # 모델 훈련 시간을 결과에 추가
    }

    results.append(result)

    f1_scores.append(f1_score_value)
    accuracy_scores.append(accuracy)
    precision_scores.append(precision)
    recall_scores.append(recall)

    # 결과 출력 및 로그 파일에 저장
    log_message = (
        f"Iteration: {i + 1}\n"
        f"Model ID: {model_id}\n"
        f"F1 Score: {f1_score_value}\n"
        f"Accuracy: {accuracy}\n"
        f"Precision: {precision}\n"
        f"Recall: {recall}\n"
        f"Training Time: {training_time} seconds\n"  # 모델 훈련 시간을 출력
    )

    print(log_message)
    logging.info(log_message)

# 각 지표의 최고점, 최저점, 평균 계산
max_f1 = max(f1_scores)
min_f1 = min(f1_scores)
avg_f1 = sum(f1_scores) / len(f1_scores)

max_accuracy = max(accuracy_scores)
min_accuracy = min(accuracy_scores)
avg_accuracy = sum(accuracy_scores) / len(accuracy_scores)

max_precision = max(precision_scores)
min_precision = min(precision_scores)
avg_precision = sum(precision_scores) / len(precision_scores)

max_recall = max(recall_scores)
min_recall = min(recall_scores)
avg_recall = sum(recall_scores) / len(recall_scores)

# 모델 훈련 시간의 최고점, 최저점, 평균 계산
max_training_time = max(training_times)
min_training_time = min(training_times)
avg_training_time = sum(training_times) / len(training_times)

# 최고점, 최저점, 평균을 로그 파일에 저장
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