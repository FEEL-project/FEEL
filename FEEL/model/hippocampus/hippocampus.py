from typing import List, Tuple, Optional, Dict, Union
import os
import pickle
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
	def __init__(self, dimension=768, replay_rate=10, replay_iteration=5, 
              size_episode=3, minimal_to_generate=8, minimal_to_loss=100,
              minimal_to_replay=100, replay_batch_size=5):
		# hyper-parameters
		self.cnt_events = 0 # 現在までの総event数
		self.num_events = 0 # 現在memoryに存在するeventの数
		self.num_replay = 0 # これまでの総replay数
		self.size_episode = size_episode 		# 生成するepisodeの長さ
		self.minimal_to_generate = minimal_to_generate 	# episodeを生成するための最低限のevent数
		# memory-loss
		self.minimal_to_loss = minimal_to_loss 	# memory-lossが発生する最小のevent数
		self.loss = True			# .no_loss()でFalse, .loss()でTrueに変更
		self.loss_interval = 10 	# データセットからサンプルの削除頻度
		self.loss_rate = 0.1 		# データセットから削除する割合
		self.max_events = 1000 		# memoryに格納可能な最大event数
		# replay parameters
		self.minimal_to_replay = minimal_to_replay	# replayが可能な最低限のevent数
		self.replay_rate = replay_rate 		# replayの頻度
		self.replay_batch_size = replay_batch_size 	# replay時のbatch_size
		self.replay_iteration = replay_iteration 	# 一度のreplayで生成する episode minibatch の数
		
		# memory(ベクトルデータベース), 類似度検索用
		self.STM = VectorDatabase(dimension, index_type="Flat") # memory本体(Short-Term-Memory)
		self.event_dataset = EventDataset() # memoryのデータセット, get_by_id()でアクセス可能
		
		# priority hyper-parameters
		self.base_priority = 100	# 新規eventのpriorityの基準値
		self.priority_method = ['base', 'rate']	# [0]:priorityの初期化方法, [1]:priorityの更新方法
		
		# priorityの低いデータを削除するためのqueue
		self.priority = PriorityQueue() # priorityによる優先度つきのeventキュー(MinHeap)
		""" issue: priorityの初期値の与え方
		1. すべてのeventに同じpriorityを与える
		2. その時点でのmemory内のpriorityの最大値を基準にする
		"""
		
		# generator
		self.replay_generator = None
		self.DataLoader = None
		self.batch_size = 1
	
	def no_loss(self) -> bool:
		self.loss = False
  
	def loss(self) -> bool:
		self.loss = True
  
	def generate_id(self) -> int:
		"""
		新規eventに付与するidを生成する
		"""
		new_id = self.cnt_events
		self.cnt_events += 1 
		return new_id

	def init_priority(self, event_id, evaluation, method:str='base'):
		"""
		時間(event_id)と感情(evaluation)から、memory内での優先度を計算する
		"""
		if method == 'base':
			return torch.norm(evaluation)+self.base_priority	# issue: np.absの使用が適切か->torch.normに変更
		else:
			raise ValueError("Invalid method.")
		### issue: priorityの初期化方法を増やす

	def update_priority(self, event_ids, method:str='rate', evaluation2s=None, rate=1.0):
		"""
		methodを指定して、eventのpriorityを更新する
		1. rate: priorityにevaluation1のrate倍を加算する
		"""
		if evaluation2s != None and len(event_ids) != len(evaluation2s):
			raise ValueError("Size of event_ids is not consistent with that of evaluation2s.")
		if method == 'replace' and evaluation2s == None:
			raise ValueError("evaluation2s is required when method is 'replace'.")
		for i, event_id in enumerate(event_ids):
			if isinstance(event_id, torch.Tensor):
				event_id = event_id.item()
			self.event_dataset.update_priority(event_id, method=method, 
                                     evaluation2=evaluation2s[i] if method=='replace' else None,
                                     rate=rate)
		### issue: 追加された更新方法に対応する引数を渡す->'rate'ならevaluation2はNoneで良い
		
	def sample(self, batch_size=1)->Dict[str, torch.tensor]:
		"""
		一つサンプルを生成する
		"""
		if self.num_events <= self.minimal_to_replay:
			return None
		if self.replay_generator == None or self.num_replay % self.loss_interval == 0:
			# 一定の頻度でsamplerに使うweightsを更新する
			weights = self.event_dataset.get_priority()
			self.replay_generator = WeightedRandomSampler(weights, self.replay_iteration)
			self.DataLoader = DataLoader(self.event_dataset, sampler=self.replay_generator, batch_size=batch_size)
		return next(iter(self.DataLoader))
  
	def get_event(self, event_id):
		"""
		event_idからcharacteristicsを取得する
		"""
		event = self.event_dataset.get_by_id(event_id)
		if event == None:
			raise ValueError("Invalid event_id.")
		return event
		
	def search(self, k=5, event_id=-1, characteristics=None):
		"""
		event_idで与えたeventに類似するeventをmemoryから検索し、そのevent_idのリストを返す
		"""
		if characteristics == None:
			if event_id == -1:
				raise ValueError("Neither event_id nor characteristics is given.")
			characteristics = self.event_dataset.get_by_id(event_id)['characteristics']	# 要変更
		return self.STM.search(characteristics, k)	#  
		
	def memory_loss(self):
		"""
		memory中の比重が小さすぎるサンプルを削除する
		例: 一定数のreplayの度に、memory中の比重が小さすぎるサンプルを削除する
		"""
		if self.num_events <= self.minimal_to_loss or not self.loss:
			# memoryが一定数未満の場合、またはlossがFalseの場合、何もしない
			return
		num_removed = int((self.num_events-self.minimal_to_loss)*self.loss_rate) # 削除するeventの数
		list_removed = [] # 削除されたeventのidリスト
		for i in range(num_removed):
			id_removed, priority_removed = self.priority.get()
			list_removed.append(id_removed)
			self.event_dataset.remove_by_id(id_removed)	# memory (EventDataset) から削除
			self.STM.remove(id_removed)					# ShortTermMemory (VectorDatabase) から削除
			
	def replay(self, batch_size=1)->List[Dict[str, Union[int, torch.tensor]]]:
		"""
		1. memoryから自発的にeventをサンプリングし、前頭前野へと出力する
			# eventからの経過時間
			# eventのインパクト
		2. 必要に応じて、weightが小さすぎるサンプルを削除する
		"""
		if self.num_events <= self.minimal_to_replay:
			raise ValueError("Hippocampus.replay: Required number of events are not stored yet.")
		# eventのサンプリング
		events = self.sample(batch_size) # -> {id: torch.tensor([]), characteristics: torch.tensor([]), evaluation1: torch.tensor([]), evaluation2: torch.tensor([])}
		print("Hippocampus.replay: \n", events)
		if events == None:
			raise ValueError("Hippocampus.replay: No event is sampled.")
		self.update_priority(events['id'], method=self.priority_method[1], evaluation2s=events['evaluation2'], rate=0.5) ### issue: idの形式
		### issue: receiveと出力形式が異なる
		new_events = []
		for i in range(batch_size):
			event = {}
			event['id'] = events['id'][i].item()
			event['characteristics'] = events['characteristics'][i]
			event['evaluation1'] = events['evaluation1'][i]
			event['evaluation2'] = events['evaluation2'][i]
			new_events.append(event)
		###
		# 更新・不要サンプルの削除
		self.num_replay += batch_size
		if self.loss and self.num_replay % self.loss_interval == 0 or self.num_events > self.max_events:
			self.memory_loss()
		return new_events
		
	def generate_episode(self, events=None, batch_size=1) -> torch.tensor:
		"""
		search events relevant to initiating event, and generate episode
		episode: characteristics of initiating event, that of relevant events
		episode_batch : torch.tensor([episode1, episode2, ...]) (size=(B, size_episode, 768))
		"""
		### issue: mini-batchへの対応->解消
		if self.num_events < self.minimal_to_generate:
			return None
		if len(events) != batch_size:
			raise ValueError("Size of batch is not consistent.")
		episode_batch = []
		for i in range(batch_size):
			event = events[i]
			if event.get('id') == None:
				raise ValueError("event_id is inappropriate.")
			result_list = self.search(k=self.size_episode-1, 
											characteristics=event['characteristics']) # List[Tuple[int, float]]
			episode = [event.get('characteristics')] # initiating event of episode i
			associated_id = []
			associated_priority = []
			for j in range(self.size_episode-1):
				event = self.get_event(result_list[j][0])
				episode.append(event['characteristics'])	# result_list[j][0]: event_id, result_list[j][1]: distance
				associated_id.append(result_list[j][0])
				associated_priority.append(event['evaluation2'])
			### issue: searchにより連想されたeventのpriorityを更新する->以下で対応
			self.update_priority(associated_id, method=self.priority_method[1],evaluation2s=associated_priority, rate=0.5)
   
			episode_tensor = torch.stack(episode)
			print(episode_tensor.shape)
			episode_batch.append(episode_tensor)
		out = torch.stack(episode_batch)
		print(out.shape)
		return out
		
	def receive(self, characteristics=None, evaluation1=None)->List[Dict[str, Union[int, torch.tensor]]]:
		"""
		知覚したeventに関連する情報を受け取り、辞書フォーマット(のリスト)で返す
		1. characteristics: torch.tensor, size=(B,768)
		2. evaluation1: torch.tensor, size=(B,1)
		-> [{'id': event_id, 'characteristics': characteristics, 'evaluation1': evaluation1},{..},...] (len=B)
		"""
		batch_size = characteristics.shape[0]
		if evaluation1 is not None and evaluation1.shape[0] != batch_size:
			raise ValueError("Size of batch is not consistent.")
		events = []
		for i in range(batch_size):
			event = {}
			event['id'] = self.generate_id()
			if characteristics is not None:
				event['characteristics'] = characteristics[i]
			if evaluation1 is not None:
				event['evaluation1'] = evaluation1[i]
			events.append(event)
		return events # [{'id': event_id, 'characteristics': characteristics, 'evaluation1': evaluation1},{..},...]
		# evaluation2 is calculated in Prefrontal-Cortex, Evaluation-Controller
			
	def save_to_memory(self, event=None,
                    event_id: int =-1, characteristics = None,
                    evaluation1 = None, evaluation2 = None, 
                    priority: float = 0.0):
		"""
		save new event to memory
		0. event: {'id': event_id, 'characteristics': characteristics,
					'evaluation1': evaluation1, 'evaluation2': evaluation2}
		以下 event の不足分を補完する
		1. event_id: id of new event
		2. characteristics: characteristics found by Sensory-Cortex (torch.tensor, size=(B,768))
		3. evaluation1: evaluation of new event by Subcortical-Pathway (torch.tensor, size=(B,1))
		4. evaluation2: evaluation of new event by Evaluation-Controller (torch.tensor, size=(B,8))
		5. priority: priority of new event
		"""
		### issue: error handling
		if event_id == -1 and event.get('id') != None:
			event_id = event['id']
			if isinstance(event_id, torch.Tensor):
				event_id = event_id.item()
		if characteristics == None and 'characteristics' in event:
			characteristics = event['characteristics']
		if evaluation1 == None and 'evaluation1' in event:
			evaluation1 = event['evaluation1']
		if evaluation2 == None and 'evaluation2' in event:
			evaluation2 = event['evaluation2']
		if priority == 0.0:
			priority = self.init_priority(event_id, evaluation1, method=self.priority_method[0])
   
		# memoryに格納
		self.event_dataset.add_item(event_id, characteristics, evaluation1, evaluation2, priority)
		# print(f"event_id: {event_id}, characteristics: {characteristics.shape}")
		self.STM.add(event_id, characteristics)
		self.priority.put((event_id, priority))
		self.num_events += 1
  
	def save_to_file(self, file_path):
		"""
		save memory to file
		USAGE:
			hippocampus.save_to_file('memory.pkl')
		"""
		base, _ = os.path.splitext(file_path)
		dataset_file = f"{base}_dataset.pkl"
		self.event_dataset.save_to_file(dataset_file) # save memory to file
		db_file = f"{base}_db.pkl"
		self.STM.save_to_file(db_file)	# save memory to file
		with open(file_path, 'wb') as f:
			# Save the rest of the object
			pickle.dump({
				'cnt_events': self.cnt_events,
				'num_events': self.num_events,
				'num_replay': self.num_replay,
				'size_episode': self.size_episode,
				'minimal_to_generate': self.minimal_to_generate,
				'minimal_to_loss': self.minimal_to_loss,
				'loss': self.loss,
				'loss_interval': self.loss_interval,
				'loss_rate': self.loss_rate,
				'max_events': self.max_events,
				'minimal_to_replay': self.minimal_to_replay,
				'replay_rate': self.replay_rate,
				'replay_iteration': self.replay_iteration,
				'base_priority': self.base_priority,
				'priority_method': self.priority_method,
				'priority': self.priority
			}, f)

	@staticmethod
	def load_from_file(file_path):
		"""
		load memory from file
		USAGE:
			hippocampus = Hippocampus.load_from_file('memory.pkl')
		"""
		base, _ = os.path.splitext(file_path)
		dataset_file = f"{base}_dataset.pkl"
		db_file = f"{base}_db.pkl"
		hippocampus = Hippocampus()
		hippocampus.event_dataset = EventDataset.load_from_file(dataset_file)
		hippocampus.STM = VectorDatabase.load_from_file(db_file)
		with open(file_path, 'rb') as f:
			obj = pickle.load(f)
			hippocampus.cnt_events = obj['cnt_events']
			hippocampus.num_events = obj['num_events']
			hippocampus.num_replay = obj['num_replay']
			hippocampus.size_episode = obj['size_episode']
			hippocampus.minimal_to_generate = obj['minimal_to_generate']
			hippocampus.minimal_to_loss = obj['minimal_to_loss']
			hippocampus.loss = obj['loss']
			hippocampus.loss_interval = obj['loss_interval']
			hippocampus.loss_rate = obj['loss_rate']
			hippocampus.max_events = obj['max_events']
			hippocampus.minimal_to_replay = obj['minimal_to_replay']
			hippocampus.replay_rate = obj['replay_rate']
			hippocampus.replay_iteration = obj['replay_iteration']
			hippocampus.base_priority = obj['base_priority']
			hippocampus.priority_method = obj['priority_method']
			hippocampus.priority = obj['priority']
		return hippocampus