import tensorflow as tf   #텐서플로우를 임포트한다.
import numpy as np
tf.enable_eager_execution()  #즉시실행한다

x_data = [1, 2, 3, 4, 5]   #x는 입력값이다 x=1,2,3,4,5
y_data = [1, 2, 3, 4, 5]   #y는 출력값이다 y=1,2,3,4,5

W = tf.Variable(2.9)    #w의 초기값을 2.9로 한다
b = tf.Variable(0.5)    #b의 초기값을 0.5 로 한다.

Learning_rate = 0.01

for i in range(100+1):                #경사 하강 알고리즘법을 100번 반복한다.
    with tf.GradientTape() as tape:   #경사하강 알고리즘 구현 tape에 변수들에 대한 정보 기록
        hypothesis = W * x_data + b   #hypothsis의 함수식
        cost = tf.reduce_mean(tf.square(hypothesis - y_data))         # 가설과 실제데이터의 차이, 즉 에러 제곱의 평균 
    W_grad, b_grad = tape.gradient(cost, [W, b])                      #cost함수에대해  w,b에 대한 미분값을 각각 구해 w,b를 업데이트한다.
    W.assign_sub(learning_rate * W_grad)                              #assign_sb 호출 ,gradient 얼마나 반영할지 learning_rate가 결정.
    b.assign_sub(learning_rate * b_grad)                              #여기까지가 w,b한번 업데이트 되는과정
    if i % 10 == 0:
      print("{:5}|{:10.4f}|{:10.4f}|{:10.6f}".format(i, W.numpy(), b.numpy(), cost))  #w,b,cost 중간중간 얼마나 변하는지 확인 if값이 10의 배수가 될때마다 내용을 출력한다.

print()

