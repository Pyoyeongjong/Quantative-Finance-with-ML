import matplotlib as plt

# 데이터 예제
data = [1, 2, 2, 3, 4, 5, 5, 5, 6, 7, 8, 9, 10]

# 히스토그램 생성
plt.hist(data, bins=10, alpha=0.75, color='blue')


# 그래프 제목 및 라벨 추가
plt.title('Example Histogram')
plt.xlabel('Values')
plt.ylabel('Frequency')

# 그래프 보여주기
plt.show()