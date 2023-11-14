from autogluon.tabular import TabularDataset, TabularPredictor

# 훈련 데이터와 테스트 데이터 불러오기
train_data = TabularDataset('D:/VSCODE/automl_research_file/dataset/train_data_selected.csv')  # 훈련 데이터 파일 경로
test_data = TabularDataset('D:/VSCODE/automl_research_file/dataset/test_data_selected.csv')    # 테스트 데이터 파일 경로
for i in range (0,30):
# AutoGluon 이진 분류 모델 학습
    predictor = TabularPredictor(label='DIS').fit(train_data)

# 테스트 데이터에 대한 예측 수행
    predictions = predictor.predict(test_data)

# 모델 저장
predictor.save("model_folder")  # 모델 폴더에 저장

# 결과 출력
print(predictions)
