### Legacy (not used now, replaced by queue.PriorityQueue) ###
# Description: Priority Queue class

class MinHeap:
    """each item is a tuple (value, id)"""
    def __init__(self, size):
        self.inf = 10**9        # 十分に大きい数 
        self.size = size + 1    # self.array[0]は使わない
        # ヒープを構成する配列
        self.array = [(self.inf,-1)]*self.size
        self.last = 0 # 現在までに入っているデータ数
    def add(self, value: float, id: int):  # 木構造の一番右の末端ノードの隣の葉が空いていたら追加
        if self.last != self.size:
            # 一番右の葉ノードとして追加
            self.last += 1
            self.array[self.last] = (value, id)
            # 制約を満たしているかチェックをする 
            self.check_after_add(self.last)
    def remove(self):           # 最小値の取り出し
        if self.last != 0:      # root Node に value があればそれが最小値 
            removed = self.array[1]
            # 一番右の葉ノードを根ノードに移動 
            self.array[1] = self.array[self.last]
            self.array[self.last] = (self.inf,-1)
            self.last -= 1
            # 制約を満たしているかチェックをする 
            self.check_after_remove(1)
            return removed
    def check_after_add(self, i):
        if i < 2:               # 根ノードまで行ったら (root Node or No data) 終了(再帰の限界)
            return
        elif self.array[i//2][0] > self.array[i][0]:  # 親の方が大きい
            tmp = self.array[i]
            self.array[i] = self.array[i//2]
            self.array[i//2] = tmp
            self.check_after_add(i//2)
        else:                                   # すでに入れ替え完了
            return
    def check_after_remove(self, i):
        if i*2 > self.last:     # 葉ノードまで行ったら終了(再帰の限界)
            return
        elif self.array[i*2][0] < self.array[i][0] and self.array[i*2+1][0] > self.array[i*2][0]:   # 左の子が最小
            tmp = self.array[i]
            self.array[i] = self.array[i*2]
            self.array[i*2] = tmp
            self.check_after_remove(i*2)
        elif self.array[i*2+1][0] < self.array[i][0] and self.array[i*2+1][0] < self.array[i*2][0]: # 右の子が最小
            tmp = self.array[i]
            self.array[i] = self.array[i*2+1]
            self.array[i*2+1] = tmp
            self.check_after_remove(i*2+1)
        else:                   # すでに入れ替え完了
            return