# AI 면접 연습을 위한 시선 및 머리 움직임 분석기 <br>(Gaze & Head Pose Analyzer for AI Interview Practice)

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange?logo=pytorch)](https://pytorch.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green?logo=opencv)](https://opencv.org/)
[![MediaPipe](https://img.shields.io/badge/MediaPipe-0.10%2B-red?logo=google)](https://developers.google.com/mediapipe)

AI 면접 연습 웹 서비스에 사용될 **사용자 집중도 분석** AI 모듈입니다. 웹캠 영상을 실시간으로 분석하여 사용자의 **시선 방향(Gaze)**과 **머리 움직임(Head Pose)**을 추적하고, 이를 통해 면접 집중도를 정량적으로 분석하는 것을 목표로 합니다.

<br>

## 주요 기능 (Key Features)

- **👀 실시간 시선 추정 (Real-time Gaze Estimation)**
  - L2CS-Net 모델을 활용하여 사용자의 시선이 화면 중앙을 향하는지, 혹은 다른 곳으로 분산되는지를 실시간으로 추정합니다.
  - Pitch (상하) 및 Yaw (좌우) 각도를 계산하여 시선의 방향을 정량화합니다.

- **🌝 안정적인 머리 자세 추정 (Robust Head Pose Estimation)**
  - MediaPipe Face Mesh를 기반으로 3D 얼굴 랜드마크를 감지하고, 이를 `cv2.solvePnP` 함수와 결합하여 머리의 3차원 회전 각도(Yaw, Pitch, Roll)를 계산합니다.
  - **칼만 필터(Kalman Filter)**를 적용하여 머리 움직임 값의 노이즈와 떨림 현상을 최소화하고, 부드럽고 안정적인 추적 결과를 제공합니다.

- **📊 데이터 시각화 (Data Visualization)**
  - 사용자가 직관적으로 자신의 상태를 파악할 수 있도록 시선 벡터와 머리 방향 축(3D Axis)을 웹캠 영상 위에 실시간으로 렌더링합니다.
  - 분석된 각도 값을 화면에 텍스트로 표시하여 즉각적인 피드백을 제공합니다.

<br>

## 데모

이 프로젝트는 다음과 같이 사용자의 시선과 머리 움직임을 실시간으로 분석하고 시각화합니다.

| 시선 추정 (Gaze Estimation) | 머리 움직임 추정 (Head Pose Estimation) |
| :-------------------------: | :------------------------------------: |
| ![Head Pose Estimation Demo](etc/head_pose_demo.gif) | ![Gaze Estimation Demo](etc/gaze_demo.gif) |

*`L2CS-Net`을 통해 시선 방향(빨간색 벡터)을 추정하고, `MediaPipe`와 지수이동평균(or 칼만) 필터를 통해 머리 방향(3D 축)을 추정하는 모습입니다.*

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

먼저 Python 가상환경을 설정하는 것을 권장합니다.

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

### 3. 모델 가중치 파일 다운로드 (Download Model Weights)

시선 추정을 위해 사전 훈련된 L2CS-Net 모델 가중치 파일이 필요합니다.

- [L2CSNet_gaze360.pkl](https://drive.google.com/drive/folders/17p6ORr-JQJcw-eYtG2WGNiuS_qVKwdWd?usp=sharing) 파일을 다운로드하여 `models/` 폴더 내에 위치시킵니다.

### 4. 코드 실행 (Execution)

데모 코드를 실행하여 웹캠이나 비디오 파일로 기능을 테스트할 수 있습니다.

```python
# L2CS-Net (시선 추정) 테스트 코드 실행
python gaze_estimation_test.py

# MediaPipe (머리 움직임) 테스트 코드 실행
python head_pose_estimation_test.py
```

<br>

## 향후 계획 (Future Work)

- [ ] 분석 데이터의 시계열 저장을 통한 면접 후 종합 리포트 생성 기능
- [ ] 시선과 머리 움직임 데이터를 종합하여 '집중도 점수'를 산출하는 알고리즘 고도화
- [ ] 웹 서비스 연동을 위한 FastAPI 기반 API 서버 구축
- [ ] 모델 경량화를 통한 웹 브라우저 환경(WebAssembly)에서의 실시간 처리 연구

---
