# Korean_AU 프로젝트

이 프로젝트는 혐오 발언(hate speech) 분류 모델을 학습하고 추론하는 과정을 포함하고 있습니다. 각 모듈의 역할 및 GCP 환경 설정에 대한 설명을 다음과 같이 정리하였습니다.

## 프로젝트 코드 구조
```plaintext
├── LICENSE
├── NIKL_AU_2023_COMPETITION_v1.0
│   ├── dev.csv
│   ├── test.csv
│   └── train.csv
├── README.md
├── best_model
│   ├── config.json
│   └── model.safetensors
├── data.py
├── inference.py
├── main.py
├── model.py
├── model
│   └── results
│       └── checkpoint
│           ├── config.json
│           ├── model.safetensors
│           ├── optimizer.pt
│           ├── rng_state.pth
│           ├── scheduler.pt
│           ├── trainer_state.json
│           └── training_args.bin
├── prediction
│   └── result.csv
├── preprocessing.ipynb
├── requirements.txt
├── sh_for_gcp
│   ├── clone_git_repo.sh
│   ├── cuda_install.sh
│   ├── dependencies_install.sh
│   ├── full_install.sh
│   ├── pyenv_setup.sh
│   └── python_virtualenv.sh
├── utils.py
└── wandb
    └── [다양한 실험 로그 파일들]
```

## 목차
1. [코드에 대한 설명](#1-코드에-대한-설명)
    - 1.1 [데이터 처리 모듈 (data.py)](#11-데이터-처리-모듈-datapy)
    - 1.2 [모델 모듈 (model.py)](#12-모델-모듈-modelpy)
    - 1.3 [유틸리티 모듈 (utils.py)](#13-유틸리티-모듈-utilspy)
    - 1.4 [메인 모듈 (main.py)](#14-메인-모듈-mainpy)
    - 1.5 [추론 모듈 (inference.py)](#15-추론-모듈-inferencepy)
2. [GCP에서 환경 설정](#2-gcp에서-환경-설정)
    - 2.1 [CUDA 및 NVIDIA 드라이버 설치 (cuda_install.sh)](#21-cuda-및-nvidia-드라이버-설치-cuda_installsh)
    - 2.2 [Pyenv 종속성 설치 (dependencies_install.sh)](#22-pyenv-종속성-설치-dependencies_installsh)
    - 2.3 [Pyenv 설치 및 환경 변수 설정 (pyenv_setup.sh)](#23-pyenv-설치-및-환경-변수-설정-pyenv_setupsh)
    - 2.4 [Python 및 가상 환경 설정 (python_virtualenv.sh)](#24-python-및-가상-환경-설정-python_virtualenvsh)
    - 2.5 [Git 리포지토리 클론 (clone_git_repo.sh)](#25-git-리포지토리-클론-clone_git_reposh)
    - 2.6 [전체 스크립트 실행 (full_install.sh)](#26-전체-스크립트-실행-full_installsh)

---

## 1. 코드에 대한 설명

### 1.1 데이터 처리 모듈 (data.py)

데이터를 준비하고 처리하는 모듈입니다. 주요 클래스 및 함수는 다음과 같습니다:

- **hate_dataset class**  
  토크나이징된 입력을 받아 데이터셋 클래스로 반환하는 역할을 합니다.

- **load_data**  
  CSV 파일로부터 데이터를 읽어와서 데이터프레임으로 반환하는 함수입니다.

- **construct_tokenized_dataset**  
  데이터프레임을 입력으로 받아 토크나이징한 후 반환하는 함수입니다.

- **prepare_dataset**  
  CSV 파일로부터 데이터를 읽어와서 토크나이징된 데이터셋으로 반환하는 함수입니다.

### 1.2 모델 모듈 (model.py)

모델 및 토크나이저를 관리하고 학습을 진행하는 모듈입니다:

- **load_tokenizer_and_model_for_train**  
  Hugging Face로부터 사전학습된 토크나이저와 모델을 불러와 반환하는 함수입니다. 이때, `config.num_labels`를 2로 수정합니다.

- **load_model_for_inference**  
  모델과 토크나이저를 반환하는 함수로, 학습된 모델 체크포인트로부터 불러옵니다.

- **load_trainer_for_train**  
  모델과 데이터셋을 입력으로 받아 `Trainer`를 반환하는 함수입니다.

- **train**  
  모델, 토크나이저, 데이터셋을 받아와 `Trainer`를 통해 학습을 진행하고, 최종적으로 최상의 모델을 저장하는 함수입니다.

### 1.3 유틸리티 모듈 (utils.py)

여러 작업에 도움이 되는 유틸리티 함수들이 포함되어 있습니다:

- **compute_metrics**  
  `Trainer`에서 메트릭을 계산하기 위해 사용되는 함수입니다.

### 1.4 메인 모듈 (main.py)

모델 학습 및 추론에 필요한 설정(config)을 관리합니다:

- **parse_args**  
  모델 학습 및 추론에 쓰일 설정(config)을 관리하는 함수입니다.

### 1.5 추론 모듈 (inference.py)

학습된 모델을 통해 결과를 추론하는 기능을 담당합니다:

- **inference**  
  학습된(trained) 모델을 통해 결과를 추론하는 함수입니다.

- **infer_and_eval**  
  학습된 모델로 추론을 진행하고, 예측한 결과를 반환하는 함수입니다.

---

## 2. GCP에서 환경 설정

이 섹션에서는 GCP VM 환경에서 CUDA 설치 및 pyenv 설정 등 필요한 환경을 자동으로 구성하기 위한 쉘 스크립트를 설명합니다.

### 2.1 CUDA 및 NVIDIA 드라이버 설치 (`cuda_install.sh`)

- **기능**: CUDA와 NVIDIA 드라이버를 설치한 뒤, 시스템을 재부팅합니다.
- **주의사항**: 재부팅 후에 나머지 스크립트를 실행해야 합니다.

### 2.2 Pyenv 종속성 설치 (`dependencies_install.sh`)

- **기능**: `pyenv` 설치에 필요한 종속성들을 설치합니다.
- **설치 항목**: `make`, `build-essential`, `libssl-dev` 등 다양한 패키지들이 포함됩니다.

### 2.3 Pyenv 설치 및 환경 변수 설정 (`pyenv_setup.sh`)

- **기능**: `pyenv`를 설치하고, 환경 변수 설정을 추가합니다.
- **환경 파일 수정**: `~/.bashrc` 파일에 `pyenv` 관련 설정을 추가합니다.

### 2.4 Python 및 가상 환경 설정 (`python_virtualenv.sh`)

- **기능**: Python 버전 3.11.8을 설치하고, 가상환경을 생성하고 활성화합니다.
- **가상환경 이름**: 기본적으로 `"my_env"`라는 이름으로 생성됩니다.

### 2.5 Git 리포지토리 클론 (`clone_git_repo.sh`)

- **기능**: curl, git, vim 등을 설치한 후, 환경 변수로 제공된 Git 리포지토리 URL을 사용해 리포지토리를 클론합니다.
- **환경 변수 사용**: 스크립트를 실행할 때 `GIT_REPO_URL` 환경 변수를 설정해주어야 합니다.

### 2.6 전체 스크립트 실행 (`full_install.sh`)

- **기능**: 위의 모든 쉘 스크립트를 순차적으로 실행하여 전체 환경을 설정합니다.
- **사용법**: Git 리포지토리 URL을 환경 변수로 전달하여 실행합니다.
    ```bash
    GIT_REPO_URL="https://github.com/your/repository.git" bash full_install.sh
    ```

---

## GitHub 링크

이 프로젝트의 전체 코드는 GitHub에서 확인할 수 있습니다. 각 쉘 파일과 Python 스크립트의 최신 버전은 [GitHub Repository](https://github.com/your/repository)에 업로드되어 있습니다. 자세한 내용은 해당 링크를 참조하세요.
