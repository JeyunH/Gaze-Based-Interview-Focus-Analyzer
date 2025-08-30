# 면접 집중도 분석을 위한 시선 및 머리 움직임 추정 및 집중도 정량화 알고리즘 <br>(Gaze and Head Pose Estimation and Concentration Quantification Algorithm for Interview Analysis)

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange?logo=pytorch)](https://pytorch.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green?logo=opencv)](https://opencv.org/)
[![MediaPipe](https://img.shields.io/badge/MediaPipe-0.10%2B-red?logo=google)](https://developers.google.com/mediapipe)

AI 면접 연습 웹 서비스에 사용될 **면접 집중도 분석** AI 모듈입니다. 면접 영상을 프레임 단위로 분석하여 사용자의 **시선 방향(Gaze)** 과 **머리 움직임(Head Pose)** 을 추정하고, 이를 통해 면접 집중도를 정량적으로 분석하는 것을 목표로 합니다.

<br>

## 주요 기능 (Key Features)

- **👀 시선 추정 (Gaze Estimation)**
  - L2CS-Net 모델을 활용하여 사용자의 시선이 화면 중앙을 향하는지, 혹은 다른 곳으로 분산되는지를 추정합니다.
  - Pitch (상하) 및 Yaw (좌우) 각도를 계산하여 시선의 방향을 정량화하고 시선 벡터를 통해 실시간으로 시각화합니다.

- **🌝 머리 자세 추정 (Head Pose Estimation)**
  - MediaPipe Face Mesh를 기반으로 3D 얼굴 랜드마크를 감지하고, 이를 `cv2.solvePnP` 함수와 결합하여 머리의 3차원 회전 각도(Yaw, Pitch, Roll)를 계산합니다.
  - **지수 이동 평균 필터, 칼만 필터(Kalman Filter)** 를 적용하여 머리 움직임 값의 노이즈와 떨림 현상을 최소화하고, 부드럽고 안정적인 추정 결과를 제공합니다.
  - 3차원 회전 각도를 기반으로 머리의 자세를 3축 벡터로 실시간 렌더링합니다.

- **📊 집중도 분석 (Concentration Analysis)**
  - 논문, 연구 등 문헌정보를 기반으로 면접에서의 집중도를 평가하는 기준을 정립합니다.
  - 시선과 머리 자세에 대한 실시간 데이터를 종합하여 집중도를 정량화하는 알고리즘을 개발/적용합니다.

    1. 집중 / 분산 상태 정의 (Defining States):
    
    시선 각도와 머리 자세의 안정성을 기준으로 '집중 상태(Engaged State)'와 '주의 분산 상태(Distracted State)'를 판단하기 위한 임계값(Threshold)을 설정합니다. 예를 들어, 시선이 화면 중앙에서 15도 이상 벗어나거나, 머리가 특정 각도 이상으로 자주 움직일 경우 '주의 분산'으로 감지합니다.
    
    2. 시간적 데이터 분석 (Temporal Analysis):
    
    단순히 한 프레임의 데이터가 아닌, 시간의 흐름에 따라 **주의 분산 상태의 지속 시간(Duration)과 빈도(Frequency)**를 함께 분석합니다. 짧고 순간적인 움직임과 오랫동안 시선이 다른 곳을 향하는 것을 구분하여 분석의 정확도를 높입니다.
    
    3. 집중도 점수화 (Scoring Algorithm):
    
    '집중 상태' 유지 시간, '주의 분산'의 빈도 및 지속 시간 등 여러 요소에 각각 가중치(Weight)를 부여합니다. 이 가중치를 종합하여 최종적으로 사용자가 이해하기 쉬운 '집중도 점수(Concentration Score)'를 산출하여 분석결과에 반영합니다.

<br>

## AI 모델 데모 (AI Model Demo)

이 프로젝트는 다음과 같이 사용자의 시선과 머리 움직임을 실시간으로 추정하고 시각화할 수 있습니다.

| 머리 움직임 추정 (Head Pose Estimation) | 시선 추정 (Gaze Estimation) |
| :-------------------------: | :------------------------------------: |
| ![Head Pose Estimation Demo](etc/head_pose_demo.gif) | ![Gaze Estimation Demo](etc/gaze_demo.gif) |

*`MediaPipe`와 지수이동평균(or 칼만) 필터를 통해 머리 방향(3D 축)을 추정하고, `L2CS-Net`을 통해 시선 방향(빨간색 벡터)을 추정하여 시각화한 모습입니다.*

<br>

## 기술 스택 (Tech Stack)

| 구분 (Category) | 기술 (Technology) |
| :--- | :--- |
| **Main Language** | `Python` |
| **Core Libraries** | `OpenCV`, `NumPy`, `MediaPipe` |
| **Deep Learning** | `PyTorch` |
| **Gaze Estimation** | `L2CS-Net` (ResNet50-based) |
| **Head Pose Estimation** | `MediaPipe Face Mesh` + `cv2.solvePnP` |
| **Data Smoothing** | `Kalman Filter`, `EMA(Exponential Moving Average)` |

<br>

## 시작하기 (Getting Started)

### 1. 환경 설정 (Prerequisites)

먼저 Anaconda를 통해 Python 가상환경을 설정하는 것을 권장합니다.

```bash
# 가상환경 생성
conda create -n gaze_env python=3.11
conda activate gaze_env
```

### 2. 필요 라이브러리 설치 (Installation)

이 프로젝트는 PyTorch를 기반으로 하며, GPU(CUDA) 환경에서 테스트되었습니다.

```bash
# 1. PyTorch 설치 (자신의 CUDA 버전에 맞게 설치)
# (예: CUDA 11.8)
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu118

# 2. L2CS-Net 및 기타 라이브러리 설치
pip install opencv-python mediapipe numpy
pip install git+https://github.com/edavalosanaya/L2CS-Net.git@main
```
라이브러리 설치를 위한 코드는 각 `ipynb` 파일에 포함되어 있으니 참고바랍니다.

### 3. 모델 가중치 파일 다운로드 (Download Model Weights)

시선 추정을 위해 사전 훈련된 L2CS-Net 모델 가중치 파일이 필요합니다.

- [L2CSNet_gaze360.pkl](https://drive.google.com/drive/folders/17p6ORr-JQJcw-eYtG2WGNiuS_qVKwdWd?usp=sharing) 파일을 다운로드하여 `models/` 폴더 내에 위치시킵니다.

### 4. 영상 데이터 준비 (Prepare Data)

AI 추론에 사용할, 얼굴과 눈이 보이는 영상을 준비하여 프로젝트 폴더에 `webcam.mp4`라는 이름으로 저장<br>
확장자 또는 파일 이름이 다를 경우, 코드 내 `webcam.mp4`를 대체하여 사용

### 5. 코드 실행 (Execution)

데모 코드를 실행하여 웹캠이나 비디오 파일로 기능을 테스트할 수 있습니다.

```python
# L2CS-Net (시선 추정) 테스트 코드 실행
Gaze-Estimation_L2CS-Net.ipynb

# MediaPipe (머리 움직임) 테스트 코드 실행
Head-Pose-Estimation_MediaPipe.ipynb
```

<br>

## 향후 계획 (Future Work)

- [ ] 문헌 조사를 통한 집중도 정량화 알고리즘 개발
- [ ] AI 면접 연습 웹 서비스 연동으로 면접 집중도 분석 결과 시각화

---
