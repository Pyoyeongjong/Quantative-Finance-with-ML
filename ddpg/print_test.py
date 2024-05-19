import matplotlib.pyplot as plt
import pandas as pd
import bitcoinA2Cenv

tickers = bitcoinA2Cenv.tickers

def change_curr_to_tick():
    for ticker in tickers:
        trade_data = pd.read_csv(f"{ticker}_test_result.csv").drop(columns='Unnamed: 0').dropna()
        data_1h = pd.read_csv(f"../../candle_datas/{ticker}_1h_sub.csv").drop(columns='Unnamed: 0').dropna()

    

def print_result():
    for ticker in tickers:
        # 각 test 폴더에서 실행한다 가정
        trade_data = pd.read_csv(f"{ticker}_test_result.csv").drop(columns='Unnamed: 0').dropna()
        data_1h = pd.read_csv(f"../../candle_datas/{ticker}_1h_sub.csv").drop(columns='Unnamed: 0').dropna()


        profit_loss_long = trade_data[trade_data['position'] == 0]['percent']
        profit_loss_short = trade_data[trade_data['position'] == 1]['percent']

        budget_list = trade_data['budget']
        hold_time = trade_data['end']-trade_data['start']

        # 롱/숏 수익률 계산
        long_budget = 10000
        for p in profit_loss_long:
            long_budget = long_budget * (1 + p/100)
        short_budget = 10000
        for p in profit_loss_short:
            short_budget = short_budget * (1 + p/100)
        
        print(long_budget)
        print(short_budget)


        

        # 히스토그램, 그래프 출력
        # 손익 히스토그램 생성
        pl_bins_list = list(range(-11,50))
        plt.hist(profit_loss_long[(profit_loss_long > 1) | (profit_loss_long < -1)], bins=pl_bins_list, alpha=0.75, color='blue')

        # 그래프 제목 및 라벨 추가
        plt.title(f'[{ticker}] Profit-Loss-Long')
        plt.xlabel('PF/LS')
        plt.ylabel('Frequency')
        plt.show()  # 그래프 표시

        # 히스토그램, 그래프 출력
        # 손익 히스토그램 생성
        pl_bins_list = list(range(-11,50))
        plt.hist(profit_loss_short[(profit_loss_short > 1) | (profit_loss_short < -1)], bins=pl_bins_list, alpha=0.75, color='blue')

        # 그래프 제목 및 라벨 추가
        plt.title(f'[{ticker}] Profit-Loss-Short')
        plt.xlabel('PF/LS')
        plt.ylabel('Frequency')
        plt.show()  # 그래프 표시

        # budget 그래프 그리기
        index = list(range(len(budget_list)))

        # 그래프 그리기
        plt.figure(figsize=(10, 4))  # 그래프 크기 설정
        plt.plot(index, budget_list, marker=None)  # 선 그래프 그리기
        plt.title(f'[{ticker}] Budget Over Time')  # 그래프 제목
        plt.xlabel('Time')  # x축 라벨
        plt.ylabel('Budget ($)')  # y축 라벨
        plt.grid(True)  # 그리드 표시
        plt.xticks(rotation=45)  # x축 라벨 회전
        plt.tight_layout()  # 레이아웃 조정
        plt.show()  # 그래프 표시

        # 홀딩시간 히스토그램 생성
        hold_bins_list = list(range(0, 100, 5))
        plt.hist(hold_time[hold_time > 3], bins=hold_bins_list, alpha=0.75, color='blue')

        # 그래프 제목 및 라벨 추가
        plt.title(f'[{ticker}] HoldTime')
        plt.xlabel('HoldTime')
        plt.ylabel('Frequency')
        plt.show()

def main():
    print_result()

if __name__=='__main__':
    main()