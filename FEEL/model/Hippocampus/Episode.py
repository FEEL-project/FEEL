

class Episode():
	"""
	episode: series of events
	1. list of event_id
	2. tensor of characteristics
	3. tensor of evaluation (of replayed event)
	"""
	def __init__(self, length=3, event_id_list=[], characteristics=None, evaluation=None):
		self.length = length
		self.event_id_list = event_id_list
		self.characteristics = characteristics
		self.evaluation = evaluation
	def __len__(self):
		return self.length
			