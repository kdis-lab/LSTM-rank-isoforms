import numpy
import sys
import copy
import pandas
import os

def fase_1(source):
	"""
	Requieres GeneID and IsoID.
	
	Example the first row of values must begins with '?|100130426,uc011lsn.1	... values'
	"""

	read = open(source)
	lines = read.readlines()
	result = []

	curr = ['GeneID', 'IsoID']
	curr.extend(lines[0].replace('\n', '').split('\t'))
	result.append(','.join(curr))

	for i in range(1, len(lines)):
		result.append(','.join(lines[i].replace('\n', '').split('\t')))

	return result

def fase_2(dataframe, ends, top_gen, rankings):
	N = [col for col in dataframe.columns if col.endswith(ends)]
	N = ['GeneID', 'IsoID'] + N
	
	frame = pandas.concat([dataframe[i] for i in N],axis=1, keys=N)
	
	rank_total_gen, rank_total_iso = rankings[ends]

	samples = frame.columns[2:]

	rows = []
	for s in samples:
		# the dataframe is copied so that new columns can be inserted later
		sample = pandas.concat([frame['GeneID'], frame['IsoID'],frame[s]], 
							   axis=1, keys=['GeneID', 'IsoID', s])

		samplegen = sample.groupby(sample['GeneID'])
		gensum = samplegen.sum()
		rank_sum_gen = numpy.asarray([gensum[s][i] for i in sample['GeneID']])

		genvar = samplegen.var()
		rank_var_gen = numpy.asarray([genvar[s][i] for i in sample['GeneID']])

		# put columns
		sample['total_gen'] = pandas.Series(rank_total_gen, index=sample.index)
		sample['sum_gen'] = pandas.Series(rank_sum_gen, index=sample.index)
		sample['var_gen'] = pandas.Series(rank_var_gen, index=sample.index)

		# replace NaN variance for 0
		sample.loc[sample['var_gen'].isna(), 'var_gen'] = 0
		sample['total_iso'] = pandas.Series(rank_total_iso, index=sample.index)

		sample.sort_values(['sum_gen','total_gen','var_gen',s,'total_iso'], 
						   ascending=[False,False,False,False,False], inplace=True)

		# top genes
		sel_genes = sample['GeneID'].groupby(sample['GeneID'], sort=False)
		sel_genes = [i[1].index.values for i in sel_genes]

		if len(sel_genes) > top_gen:
			count = len(sel_genes) - top_gen

			sample_1 = dropgen(sample,s,[j for i in sel_genes[-count:] 
											for j in i])

			sample_2 = dropgen(sample,s,[j for i in sel_genes[:count] 
											for j in i])
		else:
			sample_1 = dropgen(sample,s,[])
			sample_2 = dropgen(sample,s,[])

		sample_2 = sample_2.iloc[::-1]

		rows.append({
			'sample': s, 
			'v1': [i[0] for i in sample_1.groupby('IsoID', sort=False)], 
			'v2': [i[0] for i in sample_2.groupby('IsoID', sort=False)]
		})

	return rows

def ranking_classes(dataframe, ends):
	N = [col for col in dataframe.columns if col.endswith(ends)]
	N = ['GeneID', 'IsoID'] + N
	frame = pandas.concat([dataframe[i] for i in N],axis=1, keys=N)
	
	# genes overall ranking
	genes = frame.groupby(frame['GeneID'])
	sum_genes = genes.sum()
	ranking_genes = {i: 0 for i in frame['GeneID']}

	for col in frame.columns[2:]:
		curr_gen = sum_genes[col].sort_values()
		for i in range(len(curr_gen.index.values)):
			ranking_genes[curr_gen.index.values[i]] = ranking_genes[curr_gen.index.values[i]] + i
	rank_total_gen = numpy.asarray([ranking_genes[i] for i in frame['GeneID']])
	
	# isoforms overall ranking
	ranking_isoformas = {i: 0 for i in frame['IsoID']}
	for i in genes:
		curr_isos = i[1]
		for name_sample in frame.columns[2:]:
			curr_gen = curr_isos[['IsoID',name_sample]].sort_values(name_sample)
			for j in range(len(curr_gen.index.values)):
				ranking_isoformas[curr_gen['IsoID'][curr_gen.index.values[j]]] = ranking_isoformas[curr_gen['IsoID'][curr_gen.index.values[j]]] + j 
	rank_total_iso = [ranking_isoformas[i] for i in frame['IsoID']]
	
	return rank_total_gen, rank_total_iso


def dropgen(frame,value,drop):
	data = frame.copy(deep=True)
	data.drop(drop, inplace=True)
	return data

def iscero(data):
	for i in data:
		if float(i) != 0:
			return False
	return True


def fase_0(source, top_gen, classes=('N','T')):
	data = fase_1(source)
	data = [x.split(',') for x in data]

	print('isoforms', len(data) - 1)

	count_iso_cero = [i for i in data[1:] if iscero(i[2:])]
	print('isoforms 0', len(count_iso_cero))

	data = data[:1] + [i for i in data[1:] if not iscero(i[2:])]
	print('isoforms to try', len(data) - 1)
	data =  [numpy.asarray(i) for i in data]
	data = numpy.asarray(data)

	dataframe = pandas.DataFrame(data=data[1:], columns=data[:1][0])
	for col in dataframe.columns[:2]:
		dataframe[col] = dataframe[col].astype(str)
	for col in dataframe.columns[2:]:
		dataframe[col] = pandas.to_numeric(dataframe[col])

	isocount = dataframe[['GeneID', 'IsoID']].groupby('GeneID').count().sort_values('IsoID', ascending=False)

	rankings = {}
	for i in classes:
		if i not in rankings:
			print('ranking', i)
			rankings[i] = ranking_classes(dataframe,i)
	print('end rankings')
	fase2classes = [fase_2(dataframe,i,top_gen,rankings) for i in classes]
	v1 = []
	v2 = []
	for classi in fase2classes:
		v1.extend(list(map(lambda i: {'sample': i['sample'], 'seq': i['v1']}, classi)))
		v2.extend(list(map(lambda i: {'sample': i['sample'], 'seq': i['v2']}, classi)))

	return v1,v2

def indices_iso(data):
	index = {}
	for sample in data:
		for iso in sample['seq']:
			if iso not in index:
				index[iso] = len(index) + 1
	return index

def indices_classes(data):
	index = {}
	for sample in data:
		if sample['sample'][-1] not in index:
			index[sample['sample'][-1]] = len(index)
	return index

def f_csv(data, destination):
	index = indices_iso(data)
	file = open(destination, 'w+')
	file.write('sample,sequence,class\n')
	lines = '\n'.join([
		','.join([
					sample['sample'],
					'-'.join([str(index[iso]) for iso in sample['seq']]),
					sample['sample'][-1] 
				 ]) 
		for sample in data])
	file.write(lines)
	file.close()

def to_numpy(data):
	index = indices_iso(data)
	classes = indices_classes(data)
	print('classes', classes)
	x = [[index[iso] for iso in sample['seq']] for sample in data]
	x = numpy.asarray([numpy.asarray(i) for i in x])
	y = numpy.asarray([classes[sample['sample'][-1]] for sample in data])
	return x,y


def get_data(filepath, top_gen, classes=('N','T')):
	xnumpy1 = '-'.join([filepath, str(top_gen), 'X-v1.npy'])
	xnumpy2 = '-'.join([filepath, str(top_gen), 'X-v2.npy'])
	ynumpy = '-'.join([filepath, str(top_gen), 'Y.npy'])
	if os.path.exists(xnumpy1):
		x1 = numpy.load(xnumpy1)
		x2 = numpy.load(xnumpy2)
		y = numpy.load(ynumpy)
	else:
		if top_gen == 0:
			import sys
			v1, v2 = fase_0(filepath, sys.maxsize, classes)
		else:
			v1, v2 = fase_0(filepath, top_gen, classes)

		# easy seq
		v1csv = '-'.join([filepath, str(top_gen), 'v1.csv'])
		if not os.path.exists(v1csv):
			f_csv(v1, v1csv)
		v2csv = '-'.join([filepath, str(top_gen), 'v2.csv'])
		if not os.path.exists(v2csv):
			f_csv(v2, v2csv)

		# numpy
		x1,y = to_numpy(v1)
		x2,y = to_numpy(v2)
		
		numpy.save(xnumpy1, x1)
		numpy.save(xnumpy2, x2)
		numpy.save(ynumpy, y)

	return x1,x2,y

def mean_isoforms(source):
	data = fase_1(source)
	data = [x.split(',') for x in data]
	# isoforms
	print('isoforms', len(data) - 1)
	count_iso_cero = [i for i in data[1:] if iscero(i[2:])]
	print('isoforms 0', len(count_iso_cero))

	data = data[:1] + [i for i in data[1:] if not iscero(i[2:])]
	print('isoforms to try', len(data) - 1)
	data =  [numpy.asarray(i) for i in data]
	data = numpy.asarray(data)
	
	dataframe = pandas.DataFrame(data=data[1:], columns=data[0])
	for col in dataframe.columns:
		dataframe[col] = dataframe[col].astype(str)

	isocount = dataframe[['GeneID', 'IsoID']].groupby('GeneID').count().sort_values('IsoID', ascending=False)
	isoforms_genes_mean = isocount.mean()
	print("mean isoforms", isoforms_genes_mean)
	print("genes", isocount.shape[0])

def sum_genes_samples(source):
	data = fase_1(source)
	data = [x.split(',') for x in data]

	print('isoforms', len(data) - 1)

	count_iso_cero = [i for i in data[1:] if iscero(i[2:])]
	print('isoforms 0', len(count_iso_cero))

	data = data[:1] + [i for i in data[1:] if not iscero(i[2:])]
	print('isoforms to try', len(data) - 1)
	data =  [numpy.asarray(i) for i in data]
	data = numpy.asarray(data)

	dataframe = pandas.DataFrame(data=data[1:], columns=data[:1][0])
	for col in dataframe.columns[:2]:
		dataframe[col] = dataframe[col].astype(str)
	for col in dataframe.columns[2:]:
		dataframe[col] = pandas.to_numeric(dataframe[col])

	genes = dataframe.drop(['IsoID'],axis=1)
	genes = genes.groupby('GeneID').sum()
	return genes

def sum_genes_samples_group(dirsource):
	import os
	from os import listdir
	from os.path import isfile, join
	files = [f for f in listdir(dirsource) if isfile(join(dirsource, f)) & 
			 (f.endswith('.txt') | f.endswith('.txt_gen'))]
	for i in files:
		genes = sum_genes_samples(join(dirsource, i))
		genes.to_csv(join(dirsource, i + '.genes'))
		print('complete', join(dirsource, i))