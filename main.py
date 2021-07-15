import numpy as np
from time import time
import pickle
from perceptron import Perceptron

# and 학습
def setp1_learning():
    # 학습 테스트를 위해 사용할 데이터
    X = np.array([[0,0], [0,1], [1,0], [1,1]])
    y = np.array([-1, -1, -1, 1])

    #퍼셉트론 객체를 생성
    ppn = Perceptron(eta = 0.1)
    # 학습
    stime = time()
    ppn.fit(X, y)
    etime = time()
    print('학습에 걸린 시간 : ', (etime - stime))
    print("학습 중 오차가 난 개수 : ", ppn.errors_)

    #학습이 완료된 객체를 파일로 저장한다.
    with open('perception.pickle', 'wb') as fp:
        pickle.dump(ppn, fp)
    print('학습 완료, 잘 저장 돼었습니다.')
   
def setp2_using():
    # 파일로부터 객체를 복원
    with open('perception.pickle', 'rb') as fp:
        ppn = pickle.load(fp)
    while True:
        a1 = input('첫 번째 2진수를 입력 해 주세요 :')
        a2 = input('두 번째 2진수를 입력 해 주세요 :')
        X = np.array([int(a1), int(a2)])
        # 계산된 결과를 가져온다.
        result = ppn.predict(X)
        if result ==1 :
            print('결과 : 1')
        else :
            print('결과 : 0')

#setp1_learning()
setp2_using()