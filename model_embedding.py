import math

import tensorflow as tf
import numpy as np
import math

from collections import defaultdict


class Embedding:
	@property
	def num_train(self):
		return self.__num_train
	@property
	def num_test(self):
		return self.__num_test

	@property
	def h_rt(self):
		return self.__h_rt

	@property
	def id2entity(self):
		return self.__id2entity

	@property
	def id2relation(self):
		return self.__id2relation


	def __init__(self,
				dir_entity2id,
				dir_relation2id,
				dir_train,
				dir_test,
				dimension):
		self.__dimension = dimension

		self.__variables = []

		# read entity2id, relation2id files
		self.__entity2id, self.__id2entity, self.__num_entities = self._load_item2id(dir_entity2id)
		self.__relation2id, self.__id2relation, self.__num_relations = self._load_item2id(dir_relation2id)


		# read train/test triples: train triples are scene-item triples
		self.__h_rt, self.__triples_train, self.__items, self.__scenes = self._read_triples(dir_train)
		_, self.__triples_test, _, _ = self._read_triples(dir_test)
		print(self.__triples_test[:50])

		self.__num_train = len(self.__triples_train)
		self.__num_test = len(self.__triples_test)
		self.__num_scene = len(self.__scenes)
		self.__num_item = len(self.__items)

		# output basic information
		print('# entities:', len(self.__entity2id))
		print('# relations:', len(self.__relation2id))
		print('# train triples:', len(self.__triples_train))
		print('# test triples:', len(self.__triples_test))
		print('# items:', len(self.__items))
		print('# scenes:', len(self.__scenes))


		# variables in embedding models
		bound = 6 / math.sqrt(self.__dimension)

		with tf.device('/gpu'):
			self.__embedding_entities = tf.get_variable('embedding_entities',
													  [self.__num_entities, self.__dimension],
													  initializer=tf.random_uniform_initializer(minval=-bound,
																								maxval=bound))

			self.__embedding_relations = tf.get_variable('embedding_relations', [self.__num_relations, self.__dimension],
												initializer=tf.random_uniform_initializer(minval=-bound, maxval=bound,
																							))

			self.__variables.append(self.__embedding_entities)
			self.__variables.append(self.__embedding_relations)
			#self.__variables.append(self.__embedding_entities_scene)
			print('finishing initializing')


	def _load_item2id(self, dir):
		item2id = defaultdict()
		id2item = defaultdict()
		for line in open(dir, 'r', encoding='utf-8').readlines():
			item, id_str = line.strip().split('\t')
			item2id[item] = int(id_str)
			id2item[int(id_str)] = item
		return item2id, id2item, len(item2id)

	def _read_triples(self, dir):
		# h_rt stores the neighbors of h
		items_set = set()
		scenes_set = set()
		h_rt = defaultdict(list)
		triples = list()
		for line in open(dir, 'r', encoding='utf-8').readlines():
			h_str, r_str, t_str = line.strip().split('\t')
			h_id = self.__entity2id[h_str]
			t_id = self.__entity2id[t_str]
			r_id = self.__relation2id[r_str]
			if r_str != 'scene':
				h_rt[h_id].append([r_id, t_id])
			if r_str == 'scene':
				items_set.add(h_id)
				scenes_set.add(t_id)
				triples.append((h_id, r_id, t_id))
		scenes_list = list(scenes_set)
		items_list = list(items_set)

		return h_rt, triples,items_list, scenes_list

	# return the basic information of h_rt
	def check_h_rt(self):
		max_rt = 0
		min_rt = len(self.__triples_train)
		h_rt_distribution = defaultdict(int)
		for item in self.__items:
			h = item
			rt = self.__h_rt[h]
			num_rt = len(rt)
			h_rt_distribution[int(num_rt / 10)] += 1
			if num_rt<min_rt:
				min_rt = num_rt
				min_item = self.__id2entity[h]
			if num_rt>max_rt:
				max_rt = num_rt
				max_item = self.__id2entity[h]
		return max_rt, min_rt, max_item, min_item, h_rt_distribution

	def train_loss(self, learning_rate, margin, reg_weight):
		# a training triple (h,r,t)
		# input1: h_neighbor: [_, 2]
		# input2: relation: [1]
		# input3: tail: [1]
		with tf.device('/gpu'):
			## input placeholder
			input_h_neighbor = tf.placeholder(tf.int32, [None, 2])
			input_r = tf.placeholder(tf.int32, [1])
			input_t_pos = tf.placeholder(tf.int32, [1])
			input_t_neg = tf.placeholder(tf.int32, [1])

			## normalize embedding
			norm_embedding_entities = tf.nn.l2_normalize(self.__embedding_entities, dim=1)
			norm_embedding_relations = tf.nn.l2_normalize(self.__embedding_relations, dim=1)


			## input embeddings
			embedding_h_neighbor_r = tf.nn.embedding_lookup(norm_embedding_relations, input_h_neighbor[:, 0])
			embedding_h_neighbor_t = tf.nn.embedding_lookup(norm_embedding_entities, input_h_neighbor[:, 1])
			embedding_r = tf.nn.embedding_lookup(norm_embedding_relations, input_r)
			embedding_t_pos = tf.nn.embedding_lookup(norm_embedding_entities, input_t_pos)
			embedding_t_neg = tf.nn.embedding_lookup(norm_embedding_entities, input_t_neg)

			## computation graph
			# [_, dim]
			embedding_h_neighbor = embedding_h_neighbor_t - embedding_h_neighbor_r
			# [dim]
			embedding_h_rt_pos = embedding_t_pos - embedding_r
			embedding_h_rt_neg = embedding_t_neg - embedding_r
			# [1]
			attention_sum_pos = tf.reduce_sum(embedding_h_neighbor * embedding_h_rt_pos)
			attention_sum_neg = tf.reduce_sum(embedding_h_neighbor * embedding_h_rt_neg)

			# [_,1]
			attention_rt_pos = tf.expand_dims(tf.reduce_sum(embedding_h_neighbor * embedding_h_rt_pos, 1)/attention_sum_pos,
											  1)
			attention_rt_neg = tf.expand_dims(tf.reduce_sum(embedding_h_neighbor * embedding_h_rt_neg, 1) / attention_sum_neg,
											  1)
			# [dim]
			embedding_h_pos = tf.reduce_sum(attention_rt_pos * embedding_h_neighbor, 0)
			embedding_h_neg = tf.reduce_sum(attention_rt_neg * embedding_h_neighbor, 0)
			score_pos = tf.sqrt(tf.reduce_sum(tf.square(embedding_h_pos + embedding_r - embedding_t_pos)))
			score_neg = tf.sqrt(tf.reduce_sum(tf.square(embedding_h_neg + embedding_r - embedding_t_neg)))

			## triple loss
			loss_triple = tf.reduce_mean(tf.maximum(0., score_pos + margin - score_neg))
			## regularizer loss
			loss_reg = reg_weight * (tf.reduce_sum(tf.abs(self.__embedding_entities)) + tf.reduce_sum(tf.abs(self.__embedding_relations)))

			loss = loss_triple + loss_reg

			optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate)

			grads = optimizer.compute_gradients(loss, self.__variables)
			op_train = optimizer.apply_gradients(grads)

		return input_h_neighbor, input_r, input_t_pos, input_t_neg,\
				op_train, loss

	def train_loss_batch(self, num_neighbor, learning_rate, margin):
		reg_weight = tf.constant(0.0)
		with tf.device('/gpu'):
			## input placeholder
			input_h_neighbor = tf.placeholder(tf.int32, [None, num_neighbor,2])
			input_r = tf.placeholder(tf.int32, [None, 1])
			input_t_pos = tf.placeholder(tf.int32, [None, 1])
			input_t_neg = tf.placeholder(tf.int32, [None, 1])
			mask = tf.placeholder(tf.float32, [None, num_neighbor])

			## normalize embedding
			norm_embedding_entities = tf.nn.l2_normalize(self.__embedding_entities, dim=1)
			norm_embedding_relations = tf.nn.l2_normalize(self.__embedding_relations, dim=1)

			## input embeddings
			#[batchsize, num_neighbor, dim]
			embedding_h_neighbor_r = tf.nn.embedding_lookup(norm_embedding_relations, input_h_neighbor[:, :, 0])
			embedding_h_neighbor_t = tf.nn.embedding_lookup(norm_embedding_entities, input_h_neighbor[:, :, 1])
			# [batchsize, dim]
			embedding_r = tf.nn.embedding_lookup(norm_embedding_relations, input_r[:, 0])
			# [batchsize, dim]
			embedding_t_pos = tf.nn.embedding_lookup(norm_embedding_entities, input_t_pos[:, 0])
			embedding_t_neg = tf.nn.embedding_lookup(norm_embedding_entities, input_t_neg[:, 0])


			## computation graph
			# [batchsize, num_neighbor, dim]
			embedding_h_neighbor = embedding_h_neighbor_t - embedding_h_neighbor_r

			# [batchsize, 1, dim]
			embedding_h_rt_pos_expand = tf.expand_dims(embedding_t_pos - embedding_r,1)
			embedding_h_rt_neg_expand = tf.expand_dims(embedding_t_neg - embedding_r,1)

			# expand_dim([batchsize, num_neighbor, dim] * [batchsize, 1, dim],2)
			# = [batchsize, num_neighbor,1]
			attention_rt_pos = tf.expand_dims(tf.nn.softmax(tf.reduce_sum(embedding_h_neighbor * embedding_h_rt_pos_expand, 2)*mask, 1), 2)
			attention_rt_neg = tf.expand_dims(tf.nn.softmax(tf.reduce_sum(embedding_h_neighbor * embedding_h_rt_neg_expand, 2)*mask, 1), 2)

			# matmul([batchsize, num_neighbor, dim].T * [batchsize, num_neighbor, 1])-> [batchsize, dim, 1]
			# reduce[batchsize, dim, 1] -> [batchsize, dim]
			embedding_h_pos = tf.reduce_sum(tf.matmul(embedding_h_neighbor, attention_rt_pos, transpose_a=True),2)
			embedding_h_neg = tf.reduce_sum(tf.matmul(embedding_h_neighbor, attention_rt_neg, transpose_a=True),2)

			# [batchsize]
			score_pos = tf.sqrt(tf.reduce_sum(tf.square(embedding_h_pos + embedding_r - embedding_t_pos), 1))
			score_neg = tf.sqrt(tf.reduce_sum(tf.square(embedding_h_neg + embedding_r - embedding_t_neg), 1))

			#loss = tf.reduce_mean(tf.maximum(0., score_pos + margin - score_neg))

			# triple loss
			loss_triple = tf.reduce_mean(tf.maximum(0., score_pos + margin - score_neg))

			## regularizer loss
			loss_reg = reg_weight * (
			tf.reduce_sum(tf.abs(self.__embedding_entities)) + tf.reduce_sum(tf.abs(self.__embedding_relations)))

			loss = loss_triple + loss_reg

			optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
			#optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
			#optimizer = tf.train.AdagradOptimizer(learning_rate=learning_rate)

			grads = optimizer.compute_gradients(loss, self.__variables)
			op_train = optimizer.apply_gradients(grads)

		return input_h_neighbor, input_r, input_t_pos, input_t_neg, mask,\
				op_train, loss

	def test_loss(self):
		# input: current test triple (h,r,t)
		with tf.device('/gpu'):
			input_h_neighbor = tf.placeholder(tf.int32, [None, 2])
			input_r = tf.placeholder(tf.int32, [1])
			input_t = tf.placeholder(tf.int32, [1])

			## normalize embeddings
			norm_embedding_entities = tf.nn.l2_normalize(self.__embedding_entities, dim=1)
			norm_embedding_relations = tf.nn.l2_normalize(self.__embedding_relations, dim=1)

			## input embeddings
			embedding_h_neighbor_r = tf.nn.embedding_lookup(norm_embedding_relations, input_h_neighbor[:, 0])
			embedding_h_neighbor_t = tf.nn.embedding_lookup(norm_embedding_entities, input_h_neighbor[:, 1])
			embedding_r = tf.nn.embedding_lookup(norm_embedding_relations, input_r)
			embedding_t = tf.nn.embedding_lookup(norm_embedding_entities, input_t)
			embedding_scene_org = tf.nn.embedding_lookup(norm_embedding_entities, self.__scenes)


			## computation graph
			# [num_neighbor, dim]
			embedding_h_neighbor = embedding_h_neighbor_t - embedding_h_neighbor_r

			# [num_scene, dim]
			embedding_scene = embedding_scene_org - embedding_r

			# [num_scene, dim]*[num_neighbor, dim]
			# -> [num_scene, num_nighbor]
			attention = tf.nn.softmax(tf.matmul(embedding_scene, embedding_h_neighbor, transpose_b=True),1)

			# [num_scene, dim]
			embedding_h = tf.matmul(attention, embedding_h_neighbor)

			# [num_scene]
			score = tf.sqrt(tf.reduce_sum(tf.square(embedding_h + embedding_r - embedding_scene),1))

		return input_h_neighbor, input_r, input_t, score, attention


	def prepare(self, input_triple_id=None, purpose=None):
		input_triple = self.__triples_train[input_triple_id]
		h_pos,r_pos,t_pos = input_triple
		h_neighbor = self.__h_rt[h_pos]
		## generate negative triples
		if purpose == 'train':
			t_neg = t_pos
			while t_neg == t_pos:
				t_neg = np.random.randint(self.__num_scene)
			return [h_pos], h_neighbor, [r_pos], [t_pos], [t_neg]
		if purpose == 'test':
			t = self.__scenes.index(t_pos)
			return [h_pos], h_neighbor, [r_pos], [t]

	def prepare_train(self, input_triple_ids=None):
		# a batch of training triple: [batchsize, 3]
		h_pos = []
		r_pos = []
		t_pos = []
		#h_neighbor = []
		t_neg = []
		max_neighbor = 0
		mask = np.zeros([len(input_triple_ids), 200], dtype=float)
		h_neighbor = np.zeros([len(input_triple_ids), 200, 2])
		for i in range(len(input_triple_ids)):
			triple_id = input_triple_ids[i]
			h, r, t= self.__triples_train[triple_id]
			h_pos.append(h)
			r_pos.append(r)
			t_pos.append(t)
			#h_neighbor.append(self.__h_rt[h])
			num_h_neighbor = len(self.__h_rt[h])
			h_neighbor[i][0:num_h_neighbor] = np.asarray(self.__h_rt[h])

			# record the maximum neighbors to decide which computation graph to use
			# generate mask for each  train triple
			mask[i][0:num_h_neighbor] = np.ones([num_h_neighbor])
			if num_h_neighbor>max_neighbor: max_neighbor = num_h_neighbor

			# generate negative triples for each positive triple
			t_n = t_pos
			while t_n == t_pos:
				t_n = np.random.randint(self.__num_scene)
			t_neg.append(t_n)

		# decide which computation graph to use
		num_neighbor = min(math.ceil(max_neighbor/10)*10, 100)
		num_neighbor = 160

		# output with proper size for each output
		mask_out = mask[:,:num_neighbor]
		h_neighbor_out = h_neighbor[:,:num_neighbor]
		h_pos_out = np.asarray(h_pos).reshape([-1, 1])
		r_pos_out = np.asarray(r_pos).reshape([-1, 1])
		t_pos_out = np.asarray(t_pos).reshape([-1, 1])
		t_neg_out = np.asarray(t_neg).reshape([-1, 1])

		return h_pos_out, h_neighbor_out, r_pos_out, t_pos_out, t_neg_out, mask_out, num_neighbor

	def prepare_test(self, input_triple_id=None):
		input_triple = self.__triples_test[input_triple_id]
		h_pos, r_pos, t_pos = input_triple
		h_neighbor = self.__h_rt[h_pos]
		t = self.__scenes.index(t_pos)
		return [h_pos], h_neighbor, [r_pos], [t],[t_pos]
