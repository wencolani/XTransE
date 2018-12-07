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
	# train computation graphs with different neighbors
	#input_h_neighbor_train, input_r_train, input_t_pos_train, input_t_neg_train, op_train, loss \
	#	= model.train_loss(learning_rate=learning_rate, margin=margin)

	'''
	input_h_neighbor_10, input_r_10, input_t_pos_10, input_t_neg_10, mask_10, op_train_10, loss_10 \
		= model.train_loss_batch(num_neighbor=10, learning_rate=learning_rate, margin=margin)
	input_h_neighbor_20, input_r_20, input_t_pos_20, input_t_neg_20, mask_20, op_train_20, loss_20 \
		= model.train_loss_batch(num_neighbor=20, learning_rate=learning_rate, margin=margin)
	input_h_neighbor_30, input_r_30, input_t_pos_30, input_t_neg_30, mask_30, op_train_30, loss_30 \
		= model.train_loss_batch(num_neighbor=30, learning_rate=learning_rate, margin=margin)
	input_h_neighbor_40, input_r_40, input_t_pos_40, input_t_neg_40, mask_40, op_train_40, loss_40 \
		= model.train_loss_batch(num_neighbor=40, learning_rate=learning_rate, margin=margin)
	input_h_neighbor_50, input_r_50, input_t_pos_50, input_t_neg_50, mask_50, op_train_50, loss_50 \
		= model.train_loss_batch(num_neighbor=50, learning_rate=learning_rate, margin=margin)
	input_h_neighbor_60, input_r_60, input_t_pos_60, input_t_neg_60, mask_60, op_train_60, loss_60 \
		= model.train_loss_batch(num_neighbor=60, learning_rate=learning_rate, margin=margin)
	input_h_neighbor_70, input_r_70, input_t_pos_70, input_t_neg_70, mask_70, op_train_70, loss_70 \
		= model.train_loss_batch(num_neighbor=70, learning_rate=learning_rate, margin=margin)
	input_h_neighbor_80, input_r_80, input_t_pos_80, input_t_neg_80, mask_80, op_train_80, loss_80 \
		= model.train_loss_batch(num_neighbor=80, learning_rate=learning_rate, margin=margin)
	input_h_neighbor_90, input_r_90, input_t_pos_90, input_t_neg_90, mask_90, op_train_90, loss_90 \
		= model.train_loss_batch(num_neighbor=90, learning_rate=learning_rate, margin=margin)
	'''
	input_h_neighbor_100, input_r_100, input_t_pos_100, input_t_neg_100, mask_100, op_train_100, loss_100 \
		= model.train_loss_batch(num_neighbor=160, learning_rate=learning_rate, margin=margin)

	# test computation graph
	input_h_neighbor_test, input_r_test, input_t_test, score_test, attention_test = model.test_loss()
	# init variables
	tf.initialize_all_variables().run(session=session)
	explanation_out = open('./output/explanations_tmp.txt', 'w', encoding='utf-8')

	if iter_start == None: iter_start = 0
	for iter in range(iter_start, iter_max):
		loss_sum = 0
		num_train = model.num_train
		num_test = model.num_test
		#num_train = 10000
		#num_test = 10000
		num_batch = math.ceil(num_train / batch_size)
		#print('number_batch', num_batch)
		start_time = timeit.default_timer()
		train_ids = np.arange(num_train)
		np.random.shuffle(train_ids)
		#print('train_ids,', train_ids)
		#print('train_ids[0:10]',train_ids[0:10])
		start_ids = 0
		for i in range(num_batch):
			end_ids = min(start_ids+batch_size, num_train)
			#print('start/end_ids:', start_ids, end_ids)
			#h_pos, h_neighbor, r_pos, t_pos, t_neg = model.prepare(input_triple_id=i, purpose='train')
			h_pos, h_neighbor, r_pos, t_pos, t_neg, mask, num_neighbor \
				= model.prepare_train(input_triple_ids=train_ids[start_ids:end_ids])
			#input_h_neighbor_train, input_r_train, input_t_pos_train, input_t_neg_train \
			#	= select_computation_graph(num_neighbor)

			'''
			if num_neighbor == 10:
				feed_dict_train = {input_h_neighbor_10: h_neighbor,
								   input_r_10: r_pos,
								   input_t_pos_10: t_pos,
								   input_t_neg_10: t_neg,
								   mask_10: mask}
				_, l = session.run([op_train_10, loss_10], feed_dict=feed_dict_train)
			elif num_neighbor == 20:
				feed_dict_train = {input_h_neighbor_20: h_neighbor,
								   input_r_20: r_pos,
								   input_t_pos_20: t_pos,
								   input_t_neg_20: t_neg,
								   mask_20: mask}
				_, l = session.run([op_train_20, loss_20], feed_dict=feed_dict_train)
			elif num_neighbor == 30:
				feed_dict_train = {input_h_neighbor_30: h_neighbor,
								   input_r_30: r_pos,
								   input_t_pos_30: t_pos,
								   input_t_neg_30: t_neg,
								   mask_30: mask}
				_, l = session.run([op_train_30, loss_30], feed_dict=feed_dict_train)
			elif num_neighbor == 40:
				feed_dict_train = {input_h_neighbor_40: h_neighbor,
								   input_r_40: r_pos,
								   input_t_pos_40: t_pos,
								   input_t_neg_40: t_neg,
								   mask_40: mask}
				_, l = session.run([op_train_40, loss_40], feed_dict=feed_dict_train)
			elif num_neighbor == 50:
				feed_dict_train = {input_h_neighbor_50: h_neighbor,
								   input_r_50: r_pos,
								   input_t_pos_50: t_pos,
								   input_t_neg_50: t_neg,
								   mask_50: mask}
				_, l = session.run([op_train_50, loss_50], feed_dict=feed_dict_train)
			elif num_neighbor == 60:
				feed_dict_train = {input_h_neighbor_60: h_neighbor,
								   input_r_60: r_pos,
								   input_t_pos_60: t_pos,
								   input_t_neg_60: t_neg,
								   mask_60: mask}
				_, l = session.run([op_train_60, loss_60], feed_dict=feed_dict_train)
			elif num_neighbor == 70:
				feed_dict_train = {input_h_neighbor_70: h_neighbor,
								   input_r_70: r_pos,
								   input_t_pos_70: t_pos,
								   input_t_neg_70: t_neg,
								   mask_70: mask}
				_, l = session.run([op_train_70, loss_70], feed_dict=feed_dict_train)
			elif num_neighbor == 80:
				feed_dict_train = {input_h_neighbor_80: h_neighbor,
								   input_r_80: r_pos,
								   input_t_pos_80: t_pos,
								   input_t_neg_80: t_neg,
								   mask_80: mask}
				_, l = session.run([op_train_80, loss_80], feed_dict=feed_dict_train)
			elif num_neighbor == 90:
				feed_dict_train = {input_h_neighbor_90: h_neighbor,
								   input_r_90: r_pos,
								   input_t_pos_90: t_pos,
								   input_t_neg_90: t_neg,
								   mask_90: mask}
				_, l = session.run([op_train_90, loss_90], feed_dict=feed_dict_train)

			else:
				feed_dict_train = {input_h_neighbor_100: h_neighbor,
								   input_r_100: r_pos,
								   input_t_pos_100: t_pos,
								   input_t_neg_100: t_neg,
								   mask_100: mask}
				_, l = session.run([op_train_100, loss_100], feed_dict=feed_dict_train)
			'''



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
			explanation_out.write('ITER:'+str(iter))
			explanation_out.write('\n')
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

				#if j < 0:
				if j%10==0 and rank>=0:
					explanation_out.write(str(test_score[0]))
					explanation_out.write('\n')
					#print('test_score:', test_score[0])

					explanation_out.write(str(test_attention.shape))
					explanation_out.write('\n')
					#print('test_attention shape:', test_attention.shape)

					explanation_out.write(str(h) + ',' + str(r) + ', ' + str(t_pos) + '\n')
					explanation_out.write(str(model.id2entity[h[0]]) + ', ' + str(model.id2relation[r[0]]) + ', ' + str(
						model.id2entity[t_pos[0]]) + '\n')

					#print('test_attention:', test_attention[0])

					# sort attention in descend order
					attention_sort = np.argsort(-test_score)
					for ii in range(5):
						explanation_out.write(str(model.id2relation[model.h_rt[h[0]][attention_sort[ii]][0]]) + '：' + str( model.id2entity[model.h_rt[h[0]][attention_sort[ii]][1]]))
						explanation_out.write('\n')
					explanation_out.write('================================'+'\n')
				rank_list.append(rank)
				#rank_relu_list.append(rank_relu)
				print('[TEST][ITER:%d][%d/%d]'%(iter, j, num_test), end='\r')
			MR = np.mean(np.asarray(rank_list))
			#MR_relu = np.mean(np.asarray(rank_relu_list))

			AC = np.mean(np.asarray(np.asarray(rank_list)<1, dtype=float))
			#AC_relu = np.mean(np.asarray(np.asarray(rank_relu_list) < 1, dtype=float))
			print('[TEST][ITER:%d]	--MR: %.3f	--AC:%.3f'%(iter, MR, AC))
	explanation_out.close()

def test(model):
	None



if __name__ == '__main__':
	f_entity2id = './data/entity2id.txt'
	f_relation2id = './data/relation2id.txt'
	f_train = './data/train.txt'
	f_test = './data/test.txt'
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
	#if TEST: test(model)