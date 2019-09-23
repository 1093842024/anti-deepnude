import numpy as np
from PIL import ImageFilter,ImageDraw,ImageEnhance,ImageFont,Image
 
 
def analyze_box(new_hp,f1=1.0,f2=0.0):
	avg_value=np.mean(new_hp)
	max_value=np.max(new_hp)
	decision_value=avg_value*f1+max_value*f2
	heatmaps_avg=np.where(new_hp>decision_value)  #(x_index_ndarray,y_index_ndarray)
	xmin,xmax=np.min(heatmaps_avg[1]),np.max(heatmaps_avg[1])
	ymin,ymax=np.min(heatmaps_avg[0]),np.max(heatmaps_avg[0])
	return [xmin,ymin,xmax,ymax],heatmaps_avg,max_value,avg_value


def img_blur_2(img,heatmap,heatmaps_index,max_value):
	img_w,img_h=img.size
	img_np=np.array(img)
	img_npsample=img_np.copy()
	h,w,c=img_np.shape[0],img_np.shape[1],img_np.shape[2]
	totalnum=len(heatmaps_index[0])
	process_step=int(max(img_w,img_h)/50)#max(2,int(w*h/20000))
	rangexy=int(process_step/2)
	print(totalnum,process_step,rangexy,img_w,img_h)
	for index in range(len(heatmaps_index[0])):
		y,x=heatmaps_index[0][index],heatmaps_index[1][index]
		if x%process_step!=0 or y%process_step!=0:
			continue
		x_st,x_end=max(1,x-rangexy),min(x+rangexy+1,w-1)
		y_st,y_end=max(1,y-rangexy),min(y+rangexy+1,h-1)
		value=heatmap[y][x]
		step=int(value/max_value)*process_step*3+process_step*3
		x_min,x_max=max(1,x-step),min(x+step,w-1)
		y_min,y_max=max(1,y-step),min(y+step,h-1)
		for i in range(c):
			img_np[y_st:y_end,x_st:x_end,i]=np.mean(img_npsample[y_min:y_max,x_min:x_max,i])
	img_blur=Image.fromarray(img_np.astype(np.uint8))
	return img_blur





