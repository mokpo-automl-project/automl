import os
import pandas as pd
from pycaret.classification import *
from sklearn.metrics import *
import time
import logging

# 로그 파일 설정
log_filename = 'E:/vscode/automl_research_file/automl file/result_file/pycaret_log.log'
logging.basicConfig(filename=log_filename, level=logging.INFO, format='%(asctime)s - %(levelname)s: %(message)s')

# CSV 파일을 읽어옵니다.
train_data = pd.read_csv('E:/vscode/automl_research_file/dataset/train_data_selected.csv')
test_data = pd.read_csv('E:/vscode/automl_research_file/dataset/test_data_selected.csv')

# 필요한 열을 정의합니다.
cause_columns = 'DIS'  # 여기에 분류할 열 이름을 지정
best_models = []  # 최고의 모델들을 저장할 리스트
output_dir = 'Py_model'
os.makedirs(output_dir, exist_ok=True)

# F1 점수와 정확도를 저장할 리스트 초기화
f1_scores = []
accuracy_scores = []
precision_scores = []
recall_scores = []

# 모델 학습 시간을 저장할 리스트 초기화
training_times = []

for i in range(10):
    setup(data=train_data, target=cause_columns,verbose=False)
    # 모델 학습 시작 시간 기록
    start_time = time.time()
    
    #비교를 해서 제일 좋은 모델을 가져옴 
    current_model = compare_models(fold=10, sort='F1', n_select=1)
    
    # 모델 학습 종료 시간 기록
    end_time = time.time()
    
    # 모델 학습 시간 계산
    training_time = end_time - start_time
    training_times.append(training_time)

    #테스트 데이터로 테스트
    predictions = predict_model(current_model, data=test_data)
    y_true = test_data[cause_columns]
    y_pred = predictions['prediction_label']
    f1 = f1_score(y_true, y_pred)
    f1_scores.append(f1)
    
    # 정확도 계산
    accuracy = accuracy_score(y_true, y_pred)
    accuracy_scores.append(accuracy)
    
    # 정밀도 계산
    precision = precision_score(y_true, y_pred)
    precision_scores.append(precision)
    
    # 재현율 계산
    recall = recall_score(y_true, y_pred)
    recall_scores.append(recall)
    log_message = (
        f"Iteration: {i + 1}\n"
        f"F1 Score: {f1}\n"
        f"Accuracy: {accuracy}\n"
        f"Precision: {precision}\n"
        f"Recall: {recall}\n"
        f"Training Time: {training_time} seconds\n"
    )
    logging.info(log_message)
    
# 각 평가 메트릭의 최고값, 최저값, 평균값 계산
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

# 학습 시간 관련 통계 계산
max_training_time = max(training_times)
min_training_time = min(training_times)
avg_training_time = sum(training_times) / len(training_times)

# 결과 출력 및 로그 파일에 저장
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

print(log_message)
logging.info(log_message)
