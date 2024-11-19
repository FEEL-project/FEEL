import numpy as np
from queue import PriorityQueue	# MinHeap (heapq is not thread-safe)
import faiss
import torch
import torch.nn as nn
from torch.utils.data import WeightedRandomSampler, DataLoader
from .database import VectorDatabase
from .event_dataset import EventDataset
from .episode import Episode

class Hippocampus():
	"""海馬: カスタムデータローダー的な位置付け
		# DataLoaderでは、RandomSampler/SequentialSamplerが使われていたのを変更する
	以下の機能を担う
	1. eventを、(id, characteristics, evaluation)という形でmemoryに格納する
	2. 新規eventについて、関連性の高いeventを検索し、それらを加味したepisodeを生成する
	3. memory内のeventを、計算された重みで決まる確率に従ってreplayする
	"""
	def __init__(self, dimension, ):
		# hyper-parameters
		self.cnt_events = 0 # 現在までの総event数
		self.num_events = 0 # 現在memoryに存在するeventの数
		self.num_replay = 0 # これまでの総replay数
		self.loss_interval = 10 # データセットからサンプルの削除頻度
		self.loss_rate = 0.1 # データセットから削除する割合
		self.minimal_events = 100 # replayが可能な最低限のevent数
		self.max_events = 1000 # memoryに格納可能な最大event数
		self.replay_iteration = 5 # 一度のreplayで生成するepisodeの数
		
		# memory(ベクトルデータベース), 類似度検索用
		nlist = 100 # FAISSインデックスのクラスタ数
		self.STM = VectorDatabase(dimension, index_type="IVF", nlist=40) # memory本体(Short-Term-Memory)
		self.event_dataset = EventDataset([], [], []) # memoryのデータセット
		
		# relevant dictionary
		self.id_to_memory = {} # key:id, value:(characteristics, evaluation)
		self.id_to_priority = {} # key:id, value:priority
		self.list_priority = [] # index:id, data:priority
		
		# priorityの低いデータを削除するためのqueue
		self.priority = PriorityQueue() # priorityによる優先度つきのeventキュー(MinHeap)
		
		# generator
		self.replay_generator = None
		self.DataLoader = None
		self.batch_size = 1
		
	def calc_priority(self, event_id, evaluation):
		"""
		時間(event_id)と感情(evaluation)から、memory内での優先度を計算する
		"""
		
	def __iter__(self):
		"""
		一つサンプルを生成する
		"""
		if self.num_events <= self.minimal_events:
			return None
		if self.replay_generator == None or self.num_replay % self.loss_interval == 0:
			# 一定の頻度でsamplerに使うweightsを更新する
			weights = torch.tensor(self.list_priority)
			self.replay_generator = WeightedRandomSampler(weights, self.replay_iteration)
			self.DataLoader = DataLoader(self.event_dataset, sampler=self.replay_generator, batch_size=1)
		return next(iter(self.DataLoader))
	
	def get_characteristics(self, event_id):
		"""
		event_idからcharacteristicsを取得する
		"""
		characteristics = self.id_to_memory(event_id)
		return characteristics
		
	def search(self, k=5, event_id=-1, characteristics=None):
		"""
		event_idで与えたeventに類似するeventをmemoryから検索し、そのevent_idのリストを返す
		"""
		if characteristics == None:
			if event_id == -1:
				raise ValueError("Neither event_id nor characteristics is given.")
			characteristics = self.id_to_memory(event_id)
		return self.STM.search(characteristics, k)
		
	def memory_loss(self):
		"""
		memory中の比重が小さすぎるサンプルを削除する
		例: 一定数のreplayの度に、memory中の比重が小さすぎるサンプルを削除する
		"""
		if self.num_events <= self.minimal_events:
			return
		num_removed = int((self.num_events-self.minimal_events)*self.loss_rate) # 削除するeventの数
		list_removed = [] # 削除されたeventのidリスト
		for i in range(num_removed):
			id_removed = self.priority.get()
			list_removed.append(id_removed)
			self.id_to_priority[id_removed] = 0
			
	def replay(self):
		"""
		1. memoryから自発的にeventをサンプリングし、前頭前野へと出力する
			# eventからの経過時間
			# eventのインパクト
		2. 必要に応じて、weightが小さすぎるサンプルを削除する
		"""
		# eventのサンプリング
		event = self.__iter__()
		# 更新・不要サンプルの削除
		self.num_replay += 1
		if self.num_replay % self.loss_interval == 0:
			self.memory_loss()
		return event
		
	def generate_episode(self, event_id=-1, characteristics=None):
		"""
		eventから過去の類似eventを検索して、episodeを1つ生成する
		event==None(新たなeventが発生していないとき):
			event = replay()
		"""
		if event_id == -1 and characteristics == None:
			event_id, characteristics = self.replay()
		id_list = self.search(k=self.size_episode, 
											event_id=event_id, 
											characteristics=characteristics)
		episode = []
		for i in range(len(id_list)):
			episode.append(self.get_characteristics(id_list[i]))
		episode = torch.tensor(episode)
		return episode
		
	def receive(self, event_id, characteristics, evaluation):
		"""
		知覚したeventの特徴量を一次感覚野から受け取る
		1. 受け取ったeventの特徴量を、以下と対応させてmemoryに格納
			# 現時刻(event_id)
			# 扁桃体からの感情
		2. 関連するeventを検索し、対応づけたepisodeを生成する
		"""
		# memoryに格納
		self.STM.add(event_id, characteristics)
		priority = self.calc_priority(event_id, evaluation)
		self.priority.put((event_id, priority))
		
		# episode生成
		if self.num_events > self.minimal_events:
			# 一定数のeventがmemoryにある場合、関連eventを検索してepisode生成する
			episode = self.generate_episode(event_id, characteristics)
			return episode # torch.tensor
		else:
			return None
			
		