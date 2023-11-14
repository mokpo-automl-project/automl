from autogluon.tabular import TabularPredictor,TabularDataset
from sklearn.metrics import accuracy_score, f1_score
import os

# 모델 폴더 경로
model_folder = "D:\VSCODE\AutogluonModels"  # 모델 폴더 경로를 적절히 수정

# 훈련 데이터와 테스트 데이터 불러오기
train_data = TabularDataset('D:/VSCODE/automl_research_file/dataset/train_data_selected.csv')  # 훈련 데이터 파일 경로
test_data = TabularDataset('D:/VSCODE/automl_research_file/dataset/test_data_selected.csv')    # 테스트 데이터 파일 경로

# 모델 폴더에 있는 모든 모델 불러오기
model_list = [os.path.join(model_folder, model_name) for model_name in os.listdir(model_folder) if os.path.isdir(os.path.join(model_folder, model_name))]

# 모델별 성능 측정
for model_path in model_list:
    print(f"Evaluating model: {model_path}")
    predictor = TabularPredictor.load(model_path)

    # 테스트 데이터에 대한 예측 수행
    predictions = predictor.predict(test_data.drop(columns=['DIS']))  # 'DIS' 열 제외

    # 정확도 계산
    accuracy = accuracy_score(test_data['DIS'], predictions)

    # F1 점수 계산 (이진 분류에서는 F1 점수를 계산할 수 있음)
    f1 = f1_score(test_data['DIS'], predictions)

    # 결과 출력
    print("정확도:", accuracy)
    print("F1 점수:", f1)
