# anti-deepnude
基于NSFW的一些开源工作的启发，自行开发了精度更高的色情识别模型，并将模型用于色情图像的自动打码。  
technology for good，use deep learning model to mosaic nude model precisely automaticly

# Motivation
 [DeepNude](https://github.com/stacklikemind/deepnude_official) project is very popular. it's fancy but can be used evil.  
 As I'm working on recognizing nude/porn/vulgar image project, I developed several deep models which have achieve pretty good precise.   
Meanwhile, these model can be used to mosaic nude picture due to the activation feature maps can concentrate on the nude region.   
For Technology good, I developed a harmony algorithm to anti-deepnude.

本项目利用训练效果很好的色情图像分类模型的激活特征图，进行自动的色情打码。  
相比于已有开源的[open_nsfw](https://github.com/yahoo/open_nsfw)、[nsfw_model](https://github.com/GantMan/nsfw_model)、[nudenet](https://github.com/bedapudi6788/NudeNet)，本项目训练的模型分类精度更好（接近98%），但目前项目中的预训练模型只公开模型的前面一部分层的参数，最后几层用于分类的层暂时没有公开。
 
# How to mosaic with different region area
   modify two parameters: *weight1* and  *weight2*  
```python
def general_harmony(self,imgpil,weight1=1.0,weight2=0):
	ret,hp=self.classify_imgpil(imgpil,True)
	bbox,heatmaps_index,max_value,avg_value=bbox_util.analyze_box(hp,weight1,weight2)
	imgblur=bbox_util.img_blur_2(imgpil,hp,heatmaps_index,max_value)
	return imgblur
```
 
#Some mosaic Example
![example1](https://github.com/1093842024/anti-deepnude/blob/master/results/0_anti_deepnude.jpg)

![example3](https://github.com/1093842024/anti-deepnude/blob/master/results/2_anti_deepnude.jpg)

# TODO
combine humandet model human-keypoints model and human-segmentation model can get more accuracy results!
