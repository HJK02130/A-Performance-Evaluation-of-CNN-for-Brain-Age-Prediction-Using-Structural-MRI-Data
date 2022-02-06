# <div align=center> 자기공명영상 데이터를 이용한 합성곱 신경망 기반 <br /> 뇌연령 예측 모델의 성능 평가 </div>
### <div align=center> A Performance Evaluation of CNN for Brain Age Prediction Using Structural MRI Data
 </div>

<div align=center>
	<br />
	<br />
	<img alt="GitHub repo size" src="https://img.shields.io/github/repo-size/HJK02130/A-Performance-Evaluation-of-CNN-for-Brain-Age-Prediction-Using-Structural-MRI-Data?style=flat-square">
	<img alt="GitHub code size in bytes" src="https://img.shields.io/github/languages/code-size/HJK02130/A-Performance-Evaluation-of-CNN-for-Brain-Age-Prediction-Using-Structural-MRI-Data?style=flat-square">
	<img alt="GitHub language count" src="https://img.shields.io/github/languages/count/HJK02130/A-Performance-Evaluation-of-CNN-for-Brain-Age-Prediction-Using-Structural-MRI-Data?style=flat-square">
	<br />
	<img alt="GitHub Repo stars" src="https://img.shields.io/github/stars/HJK02130/A-Performance-Evaluation-of-CNN-for-Brain-Age-Prediction-Using-Structural-MRI-Data?style=social">
	<img alt="GitHub watchers" src="https://img.shields.io/github/watchers/HJK02130/A-Performance-Evaluation-of-CNN-for-Brain-Age-Prediction-Using-Structural-MRI-Data?style=social">
	<img alt="GitHub issues" src="https://img.shields.io/github/issues/HJK02130/A-Performance-Evaluation-of-CNN-for-Brain-Age-Prediction-Using-Structural-MRI-Data">
	<br />
	<img alt="GitHub commit activity" src="https://img.shields.io/github/commit-activity/m/HJK02130/A-Performance-Evaluation-of-CNN-for-Brain-Age-Prediction-Using-Structural-MRI-Data?style=flat-square">
	<img alt="GitHub last commit" src="https://img.shields.io/github/last-commit/HJK02130/A-Performance-Evaluation-of-CNN-for-Brain-Age-Prediction-Using-Structural-MRI-Data?style=flat-square">
	</div>
<br />

### <div align=center> :computer: Language & Development Environment :computer: </div>
<div align=center>
	<img src="https://img.shields.io/badge/Python-3766AB?style=flat-square&logo=Python&logoColor=white"/> 
	<img src="https://img.shields.io/badge/GoogleColab-F9AB00?style=flat-square&logo=GoogleColab&logoColor=white"/> </div>

<br />

### <div align=center> :keyboard: Developer : Hyun Ji Kim :keyboard: </div>
<div align=center>
	<a href="mailto:hjk02130@gmail.com"> <img src ="https://img.shields.io/badge/Gmail-EA4335.svg?&style=flat-squar&logo=Gmail&logoColor=white"/> </a> 
	<a href = "https://github.com/HJK02130"> <img src ="https://img.shields.io/badge/Github-181717.svg?&style=flat-squar&logo=Github&logoColor=white"/> </a> </div>
	
<br />
<br />

> ### 데이터

**<div align=center> ![](https://lh3.googleusercontent.com/5XcKpqI6HX87bEzfQkRu3mhLDJ9SE1RYE-DAx2rE4xspQhZhss6jMVXiFzR9lApnThjuhVAElx3IxMvF_iQfRRGmDCd9WnXB-dIR_RQU5QZ1Bg2WVwsfgVhyFIJXgPz8y8l5nXlm) </div>** 
raw T1 강조 MRI 영상의 INU(Information Non-Uniformity)를 교정하고 두개골을 제거하는 전처리를 수행한 최종 579명의 실제연령 정보 및 뇌 MRI 데이터를 7:1:2의 비율로 각각 training set(n=405), validation set(n=58), test set(n=116)으로 나누었다. 이 때, 모든 연령대의 데이터가 각각의 data set에 비슷한 비율로 분포하도록 data set을 나누었다.

<br />
<br />

> ### 모델 : 3D-CNN, 2D-CNN, VGGNet, ResNet18, ResNet34, ResNet50, 2D-RNN

각 모델은 Adam optimizer을 사용하였고 100 epoch 학습하였다. 3D-CNN, VGGNet, ResNet모델의 batch size는 16, weight decay와 learning rate는 각각 0.0006으로 설정하였고, 2D-CNN 모델의 batch size는 8, weight decay와 learning rate는 각각 0.0001로 설정하였다. 각 epoch마다 학습이 완료된 모델을 validation set을 적용하여 Mean Absolute Error(MAE) loss를 계산하고 가장 MAE 값이 낮은 모델을 이용하여 test set에 대한 성능 평가를 수행하였다.

<br />
<br />

> ### 성능평가지표

각각의 모델에서 예측된 뇌연령과 실제연령의 MAE와 상관계수(Pearson correlation coefficient, R)를 계산하여 각 모델의 성능을 비교하였다.

<br />
<br />

> ### 결과

**<div align=center> ![](https://lh5.googleusercontent.com/sTtAcOGKzeEOnNiNxr5YvS955DiRc5Px_UCa4Psi79b2HkJBR49OWdfVFFlC4xxwelr2uO8KA1asbTWLynDlMKgaJ4wWtTE0WjM4nbWvHSONbPyzu6b44sqPUzbwt0cimTEm-lJV) </div>**
 테스트한 6가지 딥러닝 모델 중 2D-CNN 모델이 가장 높은 성능을 보였다(MAE = 5.77, R = 0.88). 반대로 ResNet 모델이 가장 낮은 성능을 보였다(MAE = 18.15, R = 0.09).

<br />
<br />

> ### 결론

본 연구에서는 다양한 CNN 기반 모델들의 성능 결과를 정량적으로 제시하였으며 추후 딥러닝 기반 뇌연령 예측 모델 개발을 위해 활용될 수 있을 것이다. T1 강조 MRI 데이터를 이용한 딥러닝 기반 뇌연령 예측 모델의 성능 평가를 통해 뇌연령 예측 모델의 정확도를 개선하고 신경퇴행성 질환 및 정신질환 환자의 가속 노화 예측력을 향상시킬 수 있을 것으로 기대한다.
