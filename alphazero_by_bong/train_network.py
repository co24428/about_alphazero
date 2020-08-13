# ====================
# 파라미터 갱신 파트
# ====================

# 패키지 임포트
from dual_network import DN_INPUT_SHAPE
from tensorflow.keras.callbacks import LearningRateScheduler, LambdaCallback
from tensorflow.keras.models import load_model
from tensorflow.keras import backend as K
from pathlib import Path
import numpy as np
import pickle

# 파라미터 준비
RN_EPOCHS = 100  # 학습 횟수

# 셀프 플레이 파트에서 저장한 데이터 로드
# 학습 데이터 로드
def load_data():
    history_path = sorted(Path('./data').glob('*.history'))[-1]
    with history_path.open(mode='rb') as f:
        return pickle.load(f)

# 듀얼 네트워크 학습
def train_network():
    # 학습 데이터 로드
    history = load_data()
    '''
    로드한 데이터의 형태
    [
      [ [자신의돌의위치[9], 상대방의돌의위치[9]], 정책, 가치 ],
      [ [자신의돌의위치[9], 상대방의돌의위치[9]], 정책, 가치 ],
      ...
      [ [자신의돌의위치[9], 상대방의돌의위치[9]], 정책, 가치 ]
    ]
    '''
    # 지도학습 데이터 형태로 구성됨
    # [[자신의돌의위치[9], 상대방의돌의위치[9]],...], [정책,...], [가치,...]
    xs, y_policies, y_values = zip(*history)

    # 학습을 위한 입력 데이터 셰이프로 변환
    a, b, c     = DN_INPUT_SHAPE # (3,3,2)
    xs          = np.array(xs)   # 배열생성
    # (입력데이터수, 2, 9) => (500, 3,3,2)
    # (500, 2, 3, 3) => (500, 3, 3, 2)
    xs          = xs.reshape(len(xs), c, a, b).transpose(0, 2, 3, 1)
    # (500,)
    y_policies  = np.array(y_policies)
    # (500,)
    y_values    = np.array(y_values)

    # 베스트 플레이어 모델 로드
    model       = load_model('./model/best.h5')

    # 모델 컴파일
    # 알파제로 오리지널은 SGD를 사용, 여기서는 속도 때문에 adam 선택
    model.compile(loss=['categorical_crossentropy', 'mse'], optimizer='adam')

    # 학습률
    def step_decay(epoch):
      '''
        오리지널 알파제로 바둑
        0   step => 0.02
        300 step => 0.002
        500 step => 0.0002
        체스/장기
        0   step => 0.2
        100 step => 0.02
        300 step => 0.002
        500 step => 0.0002
      '''
      x = 0.001
      if epoch >= 50: x = 0.0005
      if epoch >= 80: x = 0.00025
      return x

    # 콜백함수
    # 원하는 입맛대로 학습률을 조절하는 함수
    lr_decay = LearningRateScheduler(step_decay)

    # 콜백함수
    # 출력
    # 1 게임마다 경과 출력
    print_callback = LambdaCallback(
        on_epoch_begin=lambda epoch, logs:
        print('\rTrain {}/{}'.format(epoch + 1, RN_EPOCHS), end=''))

    # 학습 실행
    # 입력, 정답, 배치사이즈, 학습횟수, ..
    model.fit(xs, [y_policies, y_values], batch_size=128, epochs=RN_EPOCHS,
              verbose=0, callbacks=[lr_decay, print_callback])
    print('')

    # 최신 플레이어 모델 저장
    model.save('./model/latest.h5')

    # 모델 파기
    K.clear_session()
    del model

# 동작 확인
if __name__ == '__main__':
    train_network()























