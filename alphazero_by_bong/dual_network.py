# ====================
# 듀얼 네트워크 생성 => 모델 생성
# ====================
'''
ResNet 
- 망이 깊어지면 성능이 좋아진다는 개념이, 어느 정도 이상 가면 성능이 더 떨어지는 결과가 나오면서 나온 방식
  - 무작정 늘리면 오히려 성능이 약화된다
- 레지듀얼 블럭(regidual block)이라는 숏컷 구조를 이용하여 대응
'''

# 패키지 임포트
from tensorflow.keras.layers import Activation, Add, BatchNormalization, Conv2D, Dense, GlobalAveragePooling2D, Input
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2
from tensorflow.keras import backend as K
import os

# 파라미터 준비
DN_FILTERS      = 128         # 컨볼루션 레이어 커널 수(오리지널 알파제로는 256）
DN_RESIDUAL_NUM = 16          # 레지듀얼 블록 수(오리지널 알파제로는 19)
DN_INPUT_SHAPE  = (3, 3, 2)   # 입력 셰이프          => (30, 6, ?(틱텍토는 둔곳에는 못둔다. 플래닝은 둘수 있다))
DN_OUTPUT_SIZE  = 9           # 행동 수(배치 수(3*3)) => 30*6

# 컨볼루션 레이어 생성
def conv(filters):
    return Conv2D(filters,   # 커널수
                  3,         # 커널사이즈
                  padding='same', # 패딩 보정은 동일크기 -> 결과물의 크기가 작아지지 않음
                  use_bias=False, # 바이어스 미사용
                  kernel_initializer='he_normal', # 커널 가중치(weight) 행렬의 초기값, he_normal:정규분포에 따른 초기값(성능이 잘 나옴)
                  kernel_regularizer=l2(0.0005))  # kernel의 가중치에 적용할 정규화. 어느정도 튀는값에 잘 대응함 

# 레지듀얼 블록 생성
def residual_block():
    def f(x):
        sc = x
        
        x = conv(DN_FILTERS)(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)

        x = conv(DN_FILTERS)(x)
        x = BatchNormalization()(x)
        # 출력에 2개를 입력하고
        x = Add()([x, sc])
        # 활성화 함수 통과하여 리턴
        x = Activation('relu')(x)
        return x

    return f

# 듀얼 네트워크 생성
def dual_network():
  # 모델 생성이 완료된 경우 처리하지 않음, 바로 종료 --------------------------------------
  if os.path.exists('./model/best.h5'):
      return
  # [입력레이어 => 컨볼류전 레이어 => 레지듀얼 블록x16 => 풀링 레이어 => 정책 출력, 가치 출력]
  # 입력 레이어 -------------------------------------------------------------------
  # (3, 3, 2)
  input = Input(shape=DN_INPUT_SHAPE)

  # 컨볼루션 레이어 
  x = conv(DN_FILTERS)(input)
  x = BatchNormalization()(x)
  x = Activation('relu')(x)

  # 레지듀얼 블록 x 16
  for i in range(DN_RESIDUAL_NUM):
      x = residual_block()(x)

  # 풀링 레이어
  x = GlobalAveragePooling2D()(x)

  # policy 출력
  # DN_OUTPUT_SIZE : 9, L2정규화, softmax(9개값들이 차지하는 비율)
  p = Dense(DN_OUTPUT_SIZE, kernel_regularizer=l2(0.0005),
            activation='softmax', name='pi')(x)

  # value 출력, 현재 국면에서 승리 예측(0~1), L2정규화
  v = Dense(1, kernel_regularizer=l2(0.0005))(x)
  v = Activation('tanh', name='v')(v)

  # 모델 생성
  model = Model(inputs=input, outputs=[p, v])

  # 모델 저장
  os.makedirs('./model/', exist_ok=True)  # 폴더가 없는 경우 생성
  model.save('./model/best.h5')  # 베스트 플레이어 모델

  # 모델 파기
  K.clear_session()
  del model


# 동작 확인
if __name__ == '__main__':
    dual_network()
