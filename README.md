Whisper/ Wav2Vec2 모델을 주로 이용하는 Audio Classification Pytorchn 코드입니다.
사용 가능한 모델은 다음과 같습니다.

- Whisper base: Whisper encoder + projection layer(256) + classifier/regressor
- Wav2Vec 2.0 base: wav2vec2 + projection layer(256) + classifier/regressor
- Resnet 50: Resnet basic block * 4 + classifier

학습시: train 파일
- train.py 모델 학습하는 메인 train 파일입니다.
- train_diff.py Balancing Loss 와 Difficulty based Loss 를 이용할 때 맞추어진 train 파일입니다. 코드 정리 후에 train.py 와 통합할 예정입니다.

Custom Loss functions (Util Folder 안) 
- Balancing Loss (단어별 정상:비정상 비율 맞추는 값을 CE Loss 에 곱함)
- Contrastive Loss (현재 학습하는 똑같은 타겟단어인 음성 두 개(정상과 비정상)을 불러와 얻은 임베딩과 Cosine Similarity 을 계산하여 Loss 구함)

타겟 단어 Dictonary 파일 (메인 폴더 안)
- target_text_list_norm.py : 타겟단어를 스팩트로그램 마지막 줄에 숫자로 소수로 encoding되어 들어갈 매핑 dictionary
- word_to_difficulty.py : encoding 된 타겟단어 key와 그 단어의 해당하는 채점된 난이도 difficulty(util 안 word_difficulty.py 파일에서 채점 됨) 를 매핑하는 dictionary

디버깅 시:
- main.py 위에 debugging 코드를 un-comment 해주세요.

허깅페이스/토치비전 모델을 불러와 사용하는 파이토치 코드입니다.   
데이터 전처리, 모델 빌딩 관련은 허깅페이스 내부 코드와 거의 같습니다.
- 음성 데이터 전처리: 2채널 데이터를 평균값으로 변환
- 256 길이의 feature projection layer 추가 
   

# 사용 방법
## requirments
pytorch, torchvision, torchaudio, transformers, scikit-laern, pandas    
실제 사용한 버전은 다음과 같습니다.
- python==3.10.13
- pytorch==2.4
- torchvision==0.19
- torchaudio==2.4.0
- transformers==4.40.2

## command
실제 사용하는 스크립트는 main.py입니다.
``` shell
python main.py --args...
```
## main arguments
### command line arguments
데이터 관련
- --model_embedding: 나이 정보를 스팩트로그램 마지막 두번째 줄에 채웁니다. 
- --model_text_embedding: 타겟 단어 key 정보를 스팩트로그램 마지막 줄에 채웁니다. (나이 정보와 와 타겟단어 Key 둘다 사용시, n_mels=78 이여아 합니다.)
- --data_length: 음성 데이터의 최대 길이 (초 단위)
- --data_type: wave 혹은 spectrogram (모델의 요구사항에 따라 자동으로 맞춰집니다.)
- --target: 데이터 파일의 헤더에서 타겟으로 사용할 컬럼명

데이터셋 관련
- --seed: 데이터 분할과 셔플에 사용하는 시드 번호
- --filter_dataset: 일정 타겟단어만 걸러내어 학습할 시 이용해주세요.
- --word : filter_dataset 을 설정했을 시 어떤 단어를 걸러낼 지 명시해주세요.
- --test_best_model: 사용하지 않으면 항상 맨 마지막 에폭의 모델을 불러옵니다.
- --splitted_data_files: train, valid, test 데이터의 파일이 전부 분할되어 있어야 하고, 각각의 파일명이 주어져야 합니다.
  - valid_filname이 없을 경우 train data에서 분할합니다. (전체 데이터셋에서의 valid_size만큼)
  - train_filename과 test_filename은 반드시 존재해야 합니다.
- --test_size, --valid_size: 전체 데이터에서 분할할 비율
- --k_fold: 지정된 숫자만큼 k-fold cross validation을 수행합니다.
  - 이 때 --valid_size는 무시됩니다.
  - train data와 test 데이터를 test_size에 맞춰 분할한 후 train data를 k 개로 분할합니다.
- --no_shuffle: 사용할 경우 데이터셋 분할, 데이터 로더에서 셔플이 일어나지 않습니다.
  - 사용할 시 **학습이 잘 안되는 경우**가 많습니다.
    
학습 관련
- --checkpoint: 모델 파라미터 저장 시 파일명
- --test: 학습 없이 모델 evaluation 만 하기 원할 시, 저장된 모델을 checkpoint로 설정한 후 켜주세요.
- 사용 가능한 loss function, optimizer, scheduler, model은 help를 참고해 주세요.
  - 각각에 대해 pytorch에서 지원하고 자주 사용되는 파라미터들은 구현되어 있습니다. (weight decay 등)
  - focal, clacss balanced loss의 파라미터는 util 내부의 원 코드를 참고해 주세요.
  - SAM optimizer와 cosine annealing scheduler도 내부 코드를 확인해 주세요.
- --lr: 스케줄러의 max_lr
- --warmup_ratio: 전체 스텝 대비 웜업의 비율 (스텝 수는 내부에서 자동으로 계산됩니다.)
- --batch_size, --epochs: 각각의 기본값은 16과 10입니다.
     
스펙트로그램 관련     
데이터 프레임 수를 length라고 할 때, 변환된 스펙트로그램의 크기는 n_mels * (length // hop_length + 1) 입니다.
- --n_mels: resnet50의 경우 224, whisper의 경우 80이 기본값입니다.
  - **이보다 작을 경우 남은 크기만큼 나이 데이터로 채워집니다.**
  - 나이 데이터의 값은 (전체 14개의 나이를 0과 1사이로 정규화한 값) * age_momentum 입니다.
- --hop_length: 기본값은 허깅페이스 whisper config의 값과 같습니다.
  - resnet50 모델 사용 시 데이터 크기 요구사항에 맞춰 자동으로 계산됩니다.
  - whisper 사용 시 제한이 없습니다. (내부 임베딩 테이블 크기를 자동으로 계산합니다.)
- --transform: SpecAugment를 사용할 수 있습니다.
  - --time_mask: 타임 마스킹을 사용합니다. 파라미터는 코드 내부에서 수정해야 합니다!!
  - --freq_mask: 주파스 마스킹을 사용합니다. 파라미터는 코드 내부에서 수정해야 합니다!!

### 코드 내부에서 수정해야 할 것 (!)
- SpecAugment parameters
- callback parameters

written by Gahye K. & Selina S.