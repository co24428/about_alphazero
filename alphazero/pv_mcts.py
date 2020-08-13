# ====================
# 몬테카를로 트리 탐색 생성
# 뉴럴 네트워크에서 국면의 가치를 취득함->이전 케이스에서는 플레이아웃에서 국면의가치를 취득함
# ====================

# 패키지 임포트
from game import State
from dual_network import DN_INPUT_SHAPE
from math import sqrt
from tensorflow.keras.models import load_model
from pathlib import Path
import numpy as np

# 파라미터 준비
PV_EVALUATE_COUNT = 50  # 추론 1회당 시뮬레이션 횟수(오리지널: 1600회)


# 추론(예측) -> 뉴럴 네트워크의 추론(예측)을 수행
# 1개의 입력 데이터로 추론을 수행하고자 한다 
def predict(model, state):
    # 추론을 위한 입력 데이터 셰이프 변환
    a, b, c = DN_INPUT_SHAPE # (3,3,2)
    # (내 돌의 위치, 상대방 돌의 위치)를 기초로 -> (1,3,3,2) 로 변환한다
    # (2, 9)
    x = np.array([state.pieces, state.enemy_pieces])
    # (2, 9)-> (2,3,3) -> (3,3,2) -> (1,3,3,2) 
    x = x.reshape(c, a, b).transpose(1, 2, 0).reshape(1, a, b, c)

    # 추론
    # 입력 데이터가 1개 이므로 batch_size = 1
    y = model.predict(x, batch_size=1)

    # 정책 얻기
    # 입력 데이터가 1개 이므로 => y[0][0]에 정책이 한개
    policies = y[0][0][list(state.legal_actions())]    # 합법적인 수만-> 돌을 넣을수 있는 공간
    # 둘수 있는 수만을 추출하여, 합게로 나누어 둘수 있는 수만의 확률 분포로 변환 처리
    # 각 자리에 대한 확률 분포가 생성됨
    policies /= sum(policies) if sum(policies) else 1  # 합계 1의 확률분포로 변환

    # 가치 얻기
    # 배치 사이즈가 1이므로, y[1][0]에서 가치가 하나씩 출력됨
    value = y[1][0][0]
    return policies, value


# 노드 리스트를 시행 횟수 리스트로 변환
def nodes_to_scores(nodes):
    scores = []
    for c in nodes:
        scores.append(c.n)
    return scores


# 몬테카를로 트리 탐색 스코어 얻기
def pv_mcts_scores(model, state, temperature):
  # 몬테카를로 트리 탐색 노드 정의
  # 노드 관리를 쉽게 하기 위해서 Node 클레스로 정리
  # 노드 => 30x6개의 블럭 한개, 한개
  class Node:
    # 노드 초기화
    def __init__(self, state, p):
      # 맴버 변수
      self.state = state        # 상태 (State)
      self.p = p                # 정책 (ndarray)
      self.w = 0                # 누계 가치 (int)
      self.n = 0                # 시행 횟수 (int)
      self.child_nodes = None   # 자녀 노드군 (list, 자식들은 Node)

    # 국면 가치 계산
    def evaluate(self):
        # 게임 종료 시 : 패배:-1, 무승부 :0
        if self.state.is_done():
            # 승패 결과로 가치 얻기
            value = -1 if self.state.is_lose() else 0

            # 누계 가치와 시행 횟수 갱신
            self.w += value # 누계 가치 추가
            self.n += 1     # 시행 회수 증가
            return value

        # 자녀 노드가 존재하지 않는 경우
        if not self.child_nodes:
            # 뉴럴 네트워크 추론을 활용한 정책과 가치 얻기
            policies, value = predict(model, self.state)

            # 누계 가치와 시행 횟수 갱신
            self.w += value # 누계 가치 추가 
            self.n += 1     # 시행 횟수 증가

            # 자녀 노드 전개
            self.child_nodes = []
            # 둘수 있는 노드들에 정책을 넣어서 세트로 뽑아서 
            for action, policy in zip(self.state.legal_actions(), policies):
              # 자식 노드 추가
              self.child_nodes.append(Node(self.state.next(action), policy))
            return value

        # 자녀 노드가 존재하는 경우
        else:
            # 아크 평가값이 가장 큰 자녀 노드의 평가로 가치 얻기
            value = -self.next_child_node().evaluate()

            # 누계 가치와 시행 횟수 갱신
            self.w += value
            self.n += 1
            return value

    # 아크 평가가 가장 큰 자녀 노드 얻기
    # 이전 게임 에서는 UCBI에 평가
    # C_PUCT(승률과 수의 예측 확률 X 바이어스의 벨런스를 조정하기 위한 정수)는 1.0으로 고정
    def next_child_node(self):
          # 아크 평가 계산
          C_PUCT = 1.0
          # 노드별 시행 횟수를 리스트로 구해서 전체 합을 t에 대입
          t = sum(nodes_to_scores(self.child_nodes))
          pucb_values = []
          # 노드를 돌아가면서
          for child_node in self.child_nodes:
            # 아크 평가 계산 => 공식대로
            pucb_values.append((-child_node.w / child_node.n if child_node.n else 0.0) +
                                  C_PUCT * child_node.p * sqrt(t) / (1 + child_node.n))

          # 아크 평가값이 가장 큰 자녀 노드 반환
          # 그중 가장 큰 인덱스 리턴
          return self.child_nodes[np.argmax(pucb_values)]

  # 1. 현재 국면의 노드 생성
  # state(게임환경) 기반으로 노드 생성
  root_node = Node(state, 0)

  # 여러 차례 가치 평가 실행
  # PV_EVALUATE_COUNT 횟수만큼 몬테카를로 트리 탐색 시뮬레이션을 실행
  # 시행 횟수가 높은 자식 노드의 가치가 가장 높다
  for _ in range(PV_EVALUATE_COUNT):
      root_node.evaluate()

  # 합법적인 수의 확률 분포
  # 지식 노드들의 시행 횟수를 리스트로 반환
  # 이 리스트는 둘수 있는 확률 분포를 나타낸다
  scores = nodes_to_scores(root_node.child_nodes)
  # 뉴럴 네트워크상에서 입력이 같으면 출력도 같다 
  # 따라서, 둘수 있는 수의 확률 분포를 사용하여 셀프 플레이를 수행하면 같은 수만 두게 된다
  # 이에 학습 데이터 변화에 변동을 주어야 한다 =>
  # 알파 제로는 이를 위해 볼츠만 분포를 사용한다
  # temperature는 온도 파라미터라는 볼츠만 분포의 분산된 정도를 지정
  # 온도 파라미터가 1인 경우 시행 횟수가 가장 많은 수를 100% 선택하도록 최대값만 1이 되도록 한다 
  if temperature == 0:  # 최대값인 경우에만 1
      action = np.argmax(scores)
      scores = np.zeros(len(scores))
      scores[action] = 1
  else:  # 볼츠만 분포를 기반으로 분산 추가
      scores = boltzman(scores, temperature)
  return scores


# 몬테카를로 트리 탐색을 활용한 행동 선택
def pv_mcts_action(model, temperature=0):
  # 계산된 값이 세팅굄 함수를 리턴
  def pv_mcts_action(state):
    scores = pv_mcts_scores(model, state, temperature)
    return np.random.choice(state.legal_actions(), p=scores)

  return pv_mcts_action


# 볼츠만 분포 계산
# (둘수 있는 수의 확률 분포, 온도 파라미터)
def boltzman(xs, temperature):
    xs = [x ** (1 / temperature) for x in xs]
    return [x / sum(xs) for x in xs]


# 동작 확인
if __name__ == '__main__':
    # 모델 로드
    path  = sorted(Path('./model').glob('*.h5'))[-1]
    model = load_model(str(path))

    # 상태 생성
    state = State()

    # 몬테카를로 트리 탐색을 활용해 행동을 얻는 함수 생성
    next_action = pv_mcts_action(model, 1.0)

    # 게임 종료 시까지 반복
    while True:
        # 게임 종료 시
        if state.is_done():
            break

        # 행동 얻기
        action = next_action(state)

        # 다음 상태 얻기
        state = state.next(action)

        # 문자열 출력
        print(state)







