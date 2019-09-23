#-*- coding:utf-8 -*-
import numpy as np
import os,sys,cv2,time,shutil
import tensorflow as tf
from tensorflow.python.platform import gfile
import keras
from PIL import Image,ImageDraw

import bbox_blur as bbox_util


def load_imgpil(imgpil,image_size):
	load_images=[]
	image=imgpil.resize(image_size)
	image=keras.preprocessing.image.img_to_array(image)
	image /= 255
	load_images.append(image)
	return np.asarray(load_images)
	
	
class harmony_protect():
	def __init__(self,pb_file_path='weights/inception_sp_0.9924_0.09_partialmodel.pb',mobile=False,USE_GPU=False):
		if not os.path.isfile(pb_file_path):
			print('file {} does not exist!'.format(pb_file_path))
			sys.exit()
		
		if USE_GPU==True:	
			os.environ["CUDA_VISIBLE_DEVICES"] = "0"
			config = tf.ConfigProto(allow_soft_placement=True)  
			config.gpu_options.allow_growth = True 
		else:
			os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
			config = tf.ConfigProto(allow_soft_placement=True)  
			
		g=tf.Graph()
		self.sess = tf.Session(graph=g,config=config) 
		with self.sess.as_default():
			with self.sess.graph.as_default():
				with gfile.FastGFile(pb_file_path, 'rb') as f:
					graph_def = tf.GraphDef()
					graph_def.ParseFromString(f.read())
					self.sess.graph.as_default()
					tf.import_graph_def(graph_def, name='')
				self.sess.run(tf.global_variables_initializer()) 
		
		
				self.input_img = self.sess.graph.get_tensor_by_name('input_1:0')    
				if mobile==False:
					self.conv_base_output=self.sess.graph.get_tensor_by_name('mixed10/concat:0')
					self.image_size=(299,299)
				else:
					self.conv_base_output=self.sess.graph.get_tensor_by_name('out_relu/Relu6:0')
					self.image_size=(224,224)
				print(self.input_img.shape,self.conv_base_output.shape)

		
	def classify(self, loaded_images,with_hp=False):
		if with_hp==False:
			raise('partial model does not support predict')
		else:
			heatmaps = self.sess.run(self.conv_base_output , feed_dict={self.input_img: loaded_images})
			if self.image_size==(299,299):
				heatmaps_avg=np.mean(heatmaps,axis=3).reshape((8,8)) 
			else:
				heatmaps_avg=np.mean(heatmaps,axis=3).reshape((7,7)) 
				
			return [],heatmaps_avg
	
	def classify_imgpil(self,imgpil,with_hp=False):
		w,h=imgpil.size
		loaded_image=load_imgpil(imgpil,self.image_size)
		if with_hp==True:
			ret,heatmaps=self.classify(loaded_image,True)
			heatmaps=cv2.resize(heatmaps,(w,h),interpolation=cv2.INTER_CUBIC)
			return ret,heatmaps
		ret,_=self.classify(loaded_image)
		return ret,_
		
	def general_harmony(self,imgpil,weight1=1.0,weight2=0):
		ret,hp=self.classify_imgpil(imgpil,True)
		bbox,heatmaps_index,max_value,avg_value=bbox_util.analyze_box(hp,weight1,weight2)
		imgblur=bbox_util.img_blur_2(imgpil,hp,heatmaps_index,max_value)
		return imgblur
	

def file_name(file_dir):   
	L=[] 
	for root, dirs, files in os.walk(file_dir):  
		for file in files:  
			if os.path.splitext(file)[1] == '.jpg':  
				L.append(os.path.join(root, file))  
	return L 

def test(testdir='testdata/'):
	import pylab as plt
	from matplotlib.pyplot import savefig
	harmony=harmony_protect()
	files=file_name(testdir)
	for i,file in enumerate(files):
		print(i,file)
		img=Image.open(file).convert('RGB')
		ret,hp=harmony.classify_imgpil(img,True)
		imgblur_general=harmony.general_harmony(img)
		
		w,h=img.size
		ratio=w/h
		plt.figure(figsize=(9,4/ratio))
		plt.subplot(1,3,1)
		plt.imshow(img)
		plt.title('Origin',fontsize=8)
		plt.axis('off')
		plt.subplot(1,3,2)
		plt.imshow(img)
		plt.imshow(hp,alpha=.75)
		plt.title('Activation',fontsize=8)
		plt.axis('off')	
		plt.subplot(1,3,3)
		plt.imshow(imgblur_general)
		plt.title('general',fontsize=8)
		plt.axis('off')	
		plt.gca().xaxis.set_major_locator(plt.NullLocator())
		plt.gca().yaxis.set_major_locator(plt.NullLocator())
		plt.subplots_adjust(top =1, bottom = 0, right =1, left = 0, hspace = 0.05, wspace = 0.05)
		plt.margins(0,0)
		plt.savefig('results/'+str(i)+'_anti_deepnude.jpg',dpi=160)
		plt.close()

if __name__=="__main__":
	test()

	


