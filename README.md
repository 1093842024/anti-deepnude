# anti-deepnude
technology for good，use deep learning model to mosaic nude model precisely automaticly

# Motivation
  several months ago,[DeepNude](https://github.com/stacklikemind/deepnude_official) project is very popular. it's fancy but can be used evil.
  As I'm working on recognizing nude/porn/vulgar image project, I developed several deep models which have achieve pretty good precise. 
  Meanwhile, these model can be used to mosaic nude picture due to the activation feature maps can concentrate on the nude region. For Technology good, I developed a harmony algorithm to anti-deepnude.
 
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
