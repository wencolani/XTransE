#!/usr/bin/env python
# -*- coding: utf-8 -*-

## our proposed model
import os
import timeit

import math
import tensorflow as tf
import numpy as np


from model_embedding import Embedding

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def train(model, iter_max, learning_rate, margin, session, test_per,
		  iter_start=None, batch_size = 2048):
	input_h_neighbor_100, input_r_100, input_t_pos_100, input_t_neg_100, mask_100, op_train_100, loss_100 \
		= model.train_loss_batch(num_neighbor=160, learning_rate=learning_rate, margin=margin)

	# test computation graph
	input_h_neighbor_test, input_r_test, input_t_test, score_test, attention_test = model.test_loss()
	# init variables
	tf.initialize_all_variables().run(session=session)

	if iter_start == None: iter_start = 0
	for iter in range(iter_start, iter_max):
		loss_sum = 0
		num_train = model.num_train
		num_test = model.num_test
		num_batch = math.ceil(num_train / batch_size)
		start_time = timeit.default_timer()
		train_ids = np.arange(num_train)
		np.random.shuffle(train_ids)
		start_ids = 0
		for i in range(num_batch):
			end_ids = min(start_ids+batch_size, num_train)
			h_pos, h_neighbor, r_pos, t_pos, t_neg, mask, num_neighbor \
				= model.prepare_train(input_triple_ids=train_ids[start_ids:end_ids])

			feed_dict_train = {input_h_neighbor_100:h_neighbor,
						 input_r_100:r_pos,
						 input_t_pos_100: t_pos,
						 input_t_neg_100: t_neg,
						 mask_100: mask}
			_, l = session.run([op_train_100, loss_100], feed_dict=feed_dict_train)


			loss_sum += l
			# INFO
			#print('loss_batch:', l)
			print('ITER: %d[%d/%d] --loss:%.2f --loss_sum:%2f'%(iter, i, num_batch, l, loss_sum), end='\r')
			start_ids = end_ids
		end_time = timeit.default_timer()
		#print('\n')
		print('ITER: %d 	--loss: %.2f	--time: %.2f'%(iter, loss_sum, end_time-start_time))

		if iter%test_per == 0:
			# explanation_out.write('ITER:'+str(iter))
			# explanation_out.write('\n')
			rank_list = []
			rank_relu_list = []
			for j in range(num_test):
				h, h_neighbor, r, t, t_pos = model.prepare_test(input_triple_id=j)
				feed_dict_test = {input_h_neighbor_test:h_neighbor,
								  input_r_test: r,
								  input_t_test: t}
				test_score, test_attention = session.run([score_test, attention_test],
								  feed_dict=feed_dict_test)
				rank = np.where(np.argsort(test_score) == t[0])[0][0]
				rank_list.append(rank)
				#rank_relu_list.append(rank_relu)
				print('[TEST][ITER:%d][%d/%d]'%(iter, j, num_test), end='\r')
			MR = np.mean(np.asarray(rank_list))

			AC = np.mean(np.asarray(np.asarray(rank_list)<1, dtype=float))
			print('[TEST][ITER:%d]	--MR: %.3f	--AC:%.3f'%(iter, MR, AC))


if __name__ == '__main__':
	f_entity2id = '../data/entity2id.txt'
	f_relation2id = '../data/relation2id.txt'
	f_train = '../data/train.txt'
	f_test = '../data/test.txt'
	TRAIN = True
	TEST = False
	iter_max = 30
	iter_min = 0
	learning_rate = 0.001
	margin = 2.0
	test_per = 1

	dimension = 100

	model = Embedding(dir_entity2id=f_entity2id,
				dir_relation2id=f_relation2id,
				dir_train=f_train,
				dir_test=f_test,
				dimension=dimension)
	max_rt, min_rt, max_item, min_item,h_rt_distribution = model.check_h_rt()
	print('max_rt, min_rt', max_rt, min_rt,max_item, min_item)
	print('h_rt_distribition:', h_rt_distribution)

	config = tf.ConfigProto()
	config.gpu_options.allow_growth = False
	config.log_device_placement = False
	config.allow_soft_placement = True
	config.gpu_options.per_process_gpu_memory_fraction = 0.48

	session = sess = tf.Session(config=config)
	if TRAIN: train(model, iter_max, learning_rate, margin, session, test_per=test_per)
	# if TEST: test(model)
