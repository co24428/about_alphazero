# ====================
# 컨넥트4
# Role
# 2 플레이어가 교대로 7x6 보드면에서 아래부터 돌을 쌓는다
# 가로, 세로, 대각선중 4개의 돌을 나란히 놓으면 승리
# 전제, 클릭으로 돌을 넣고, 사람이 무조건 선수
# 게임판
# (7,6,2), 돌을 놓았다면 1, 않놓았으면 0
# 액션
# 돌을 떨어뜨릴 열을 찾고(0~6:7개), 선택열에서 아직 돌이 놓이지 않은 가장 아래 위치에 돌을 놓는다
# 행동수는 7
# ====================

# 패키지 임포트
import random
import math


# 게임 상태
class State:
    # 게임 상태 초기화
    def __init__(self, pieces=None, enemy_pieces=None):
        # 돌의 배치 : (7x6) => 42의 1차원 배열 생성
        # 플래닝에서는 (2,30,6) or (2,6,30) or (30,6):덱위아래를 같이 보는 방식
        self.pieces         = pieces        if pieces != None       else [0] * 42
        self.enemy_pieces   = enemy_pieces  if enemy_pieces != None else [0] * 42

    # 돌의 수 얻기
    def piece_count(self, pieces):
        # 총개수
        count = 0
        # 게임판을 돌면서
        for i in pieces:
            # 1과 일치하면
            if i == 1:
                # 카운트 증가
                count += 1
        # 카운트 리턴
        return count

    # 패배 여부 판정
    # 플래닝은 짐을 다 싣었다면, 누가 리워드가 더 높은지로 체크(기타 조건이 더 들어갈수 있음)
    def is_lose(self):
        # 돌 4개 연결 여부 판정 -> 체크
        def is_comp(x, y, dx, dy):
            for k in range(4):
                # 아래 경우는 무조건 패배
                if y < 0 or 5 < y or x < 0 or 6 < x or \
                        self.enemy_pieces[x + y * 7] == 0:
                    return False
                x, y = x + dx, y + dy
            # 통과했으면 승리 
            return True

        # 패배 여부 판정
        for j in range(6):
            for i in range(7):
                # 승리 조건
                if is_comp(i, j, 1, 0) or is_comp(i, j, 0, 1) or \
                        is_comp(i, j, 1, -1) or is_comp(i, j, 1, 1):
                    return True
        # 패배
        return False

    # 무승부 여부 판정
    def is_draw(self):
        # 승리,패배가 아직 완성이 않되었는데, 놓아진 돌수의 합이 42이면 완료
        # 플래닝에서는 무승부가 없음
        return self.piece_count(self.pieces) + self.piece_count(self.enemy_pieces) == 42

    # 게임 종료 여부 판정
    def is_done(self):
        # 이기거나, 졌거나, 비기면 종료 
        return self.is_lose() or self.is_draw()

    # 다음 상태 얻기
    def next(self, action):
        # 해당 상태에서 게임판 카피
        # 여기서 램덤으로 갈지, 아니면 각 노드의 포인ㅌ 가 높은 순으로 놓을지, 아니면 하위 덱 부 터 채울지 판단 필요
        pieces = self.pieces.copy()
        for j in range(5, -1, -1):
            # 둘다 0이면,
            if self.pieces[action + j * 7] == 0 and self.enemy_pieces[action + j * 7] == 0:
                # 해당위치에 돌을 넣는다
                pieces[action + j * 7] = 1
                break
        # 이렇게 만듫어진 state를 넣어서 다시 생성
        return State(self.enemy_pieces, pieces)

    # 둘수 있는 수 취득
    def legal_actions(self):
        actions = []
        # 열을 돌면서, 빈곳을 찾는다
        for i in range(7):
            if self.pieces[i] == 0 and self.enemy_pieces[i] == 0:
                actions.append(i)
        return actions

    # 선 수 여부 확인
    def is_first_player(self):
        # 둘이 놓은 돌수가 같으면 나의 턴, 
        # 플래닝에서는 턴이 없음 무조건 내턴
        return self.piece_count(self.pieces) == self.piece_count(self.enemy_pieces)

    # 문자열 표시
    def __str__(self):
        # 돌이 놓여진 화면 표시
        ox = ('o', 'x') if self.is_first_player() else ('x', 'o')
        str = ''
        for i in range(42):
            if self.pieces[i] == 1:
                str += ox[0]
            elif self.enemy_pieces[i] == 1:
                str += ox[1]
            else:
                str += '-'
            if i % 7 == 6:
                str += '\n'
        return str


# 랜덤으로 행동 선택
def random_action(state):
    # 돌을 놓을 수 있는 리스트를 구해서
    legal_actions = state.legal_actions()
    # 랜덤으로 놓을곳을 찾는다
    return legal_actions[random.randint(0, len(legal_actions) - 1)]


# 동작 확인
if __name__ == '__main__':
    # 상태 생성
    state = State()

    # 게임 종료 시까지 반복
    while True:
        # 게임 종료 시
        if state.is_done():
            break

        # 다음 상태 얻기
        # 다은 놓아야 하는 위치를 찾아서 세팅하고
        # 게임 환경을 받아서
        state = state.next(random_action(state))

        # 문자열 표시
        print(state)
        print()
