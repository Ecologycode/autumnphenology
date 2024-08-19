#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#  code.py
#  
#  Copyright 2024 haoyin <haoyin@YINHAO>
#  
#  This program is free software; you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation; either version 2 of the License, or
#  (at your option) any later version.
#  
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#  
#  You should have received a copy of the GNU General Public License
#  along with this program; if not, write to the Free Software
#  Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston,
#  MA 02110-1301, USA.
#  
#  

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split,cross_val_predict
from datetime import datetime, timedelta
import lightgbm as lgb
from sklearn.ensemble import RandomForestRegressor
import shap
import matplotlib.pyplot as plt
import matplotlib
import matplotlib as mpl
import mpl_scatter_density
from matplotlib.colors import LinearSegmentedColormap
from sklearn.metrics import mean_squared_error,r2_score,mean_absolute_error,explained_variance_score
from catboost import CatBoostRegressor
from netCDF4 import Dataset
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.path as mpath
from sklearn.linear_model import LinearRegression
import datetime as dt
import glob
import scipy.stats as stats
import scipy.optimize as opt
import statsmodels.stats.stattools as st
import pingouin as pg
from scipy.stats import linregress
from datetime import datetime, timedelta
from netCDF4 import Dataset
from matplotlib import cm
import cartopy.crs as ccrs
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
import numpy as np
import matplotlib.path as mpath
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import random
from scipy.stats import norm
from scipy.optimize import curve_fit
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
from statsmodels.formula.api import ols
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error,mean_absolute_error
from statsmodels.api import OLS
#from bayes_opt import BayesianOptimization

def figure1():
	data=Dataset('./data/1.nc')
	lat1=np.array(data['lat'])
	lon1=np.array(data['lon'])
	data1=np.array(data['data'])
	for i in range(len(lat1)):
		for j in range(len(lon1)):
			data1[i,j]=np.nan
				
	data2=pd.read_csv(r'./data/eos_trend.csv')
	eos=np.array(data2.pop('mean'))
	lat2=np.array(data2['lat'])
	lon2=np.array(data2['lon'])
	for i in range(len(lat2)):
		a=np.where(lat1==lat2[i])[0][0]
		b=np.where(lon1==lon2[i])[0][0]
		data1[a,b]=eos[i]
		
	projections = [ccrs.AzimuthalEquidistant(central_longitude=0.0, central_latitude=90.0)]
	
	fig = plt.figure()
	ax = fig.add_subplot(1, 1, 1, projection=projections[0])
	ax.coastlines()
	ax.add_feature(cfeature.BORDERS, linestyle='-', linewidth=0.5, edgecolor='black')
	theta = np.linspace(0, 2*np.pi, 100)
	center, radius = [0.5, 0.5], 0.5
	verts = np.vstack([np.sin(theta), np.cos(theta)]).T
	circle = mpath.Path(verts * radius + center)
	ax.set_boundary(circle, transform=ax.transAxes)
	cs=ax.pcolormesh(lon1,lat1,data1,vmin=-10,vmax=10,cmap=colormap_res11(),transform=ccrs.PlateCarree())	
	ax.set_extent([-180, 180,20, 90], crs=ccrs.PlateCarree())  # Adjust the extent as needed
	ax.set_title('Difference',fontdict={'family':'arial','weight':'normal','size':22,})
	
	fig.subplots_adjust()
	l=0.19
	b=0.075
	w=0.655
	h=0.0155
	rect=[l,b,w,h]
	cbar_ax=fig.add_axes(rect)
	cb=fig.colorbar(cs,cax=cbar_ax,extend='both',orientation='horizontal')
	cb.set_ticks(np.arange(-10,15,5))
	cb.ax.tick_params(labelsize=22)
	cb.set_label('(Days)',fontdict={'family':'arial','weight':'normal','size':22,})
	plt.savefig('./figure/mean.png',bbox_inches='tight',dpi=300)
	
	data=Dataset('./data/1.nc')
	lat1=np.array(data['lat'])
	lon1=np.array(data['lon'])
	data1=np.array(data['data'])
	for i in range(len(lat1)):
		for j in range(len(lon1)):
			data1[i,j]=np.nan
				
	data2=pd.read_csv(r'./data/eos_trend.csv')
	eos=np.array(data2.pop('std'))
	lat2=np.array(data2['lat'])
	lon2=np.array(data2['lon'])
	for i in range(len(lat2)):
		a=np.where(lat1==lat2[i])[0][0]
		b=np.where(lon1==lon2[i])[0][0]
		data1[a,b]=eos[i]

	projections = [ccrs.AzimuthalEquidistant(central_longitude=0.0, central_latitude=90.0)]
	
	fig = plt.figure()
	ax = fig.add_subplot(1, 1, 1, projection=projections[0])
	ax.coastlines()
	ax.add_feature(cfeature.BORDERS, linestyle='-', linewidth=0.5, edgecolor='black')
	theta = np.linspace(0, 2*np.pi, 100)
	center, radius = [0.5, 0.5], 0.5
	verts = np.vstack([np.sin(theta), np.cos(theta)]).T
	circle = mpath.Path(verts * radius + center)
	ax.set_boundary(circle, transform=ax.transAxes)
	cs=ax.pcolormesh(lon1,lat1,data1,vmin=0,vmax=10,cmap=colormap_res11(),transform=ccrs.PlateCarree())	
	ax.set_extent([-180, 180,20, 90], crs=ccrs.PlateCarree())  # Adjust the extent as needed
	ax.set_title('Std',fontdict={'family':'arial','weight':'normal','size':22,})
	
	fig.subplots_adjust()
	l=0.19
	b=0.075
	w=0.655
	h=0.0155
	rect=[l,b,w,h]
	cbar_ax=fig.add_axes(rect)
	cb=fig.colorbar(cs,cax=cbar_ax,extend='both',orientation='horizontal')
	cb.set_ticks(np.arange(0,12,2))
	cb.ax.tick_params(labelsize=22)
	cb.set_label('(Days)',fontdict={'family':'arial','weight':'normal','size':22,})
	plt.savefig('./figure/std.png',bbox_inches='tight',dpi=300)

	type1=['All','ENF','DBF','MF','OS','WS','SA','GL']
	data=pd.read_csv(r'./data/eos_trend.csv')
	lc_type1=[0,1,4,5,7,8,9,10]
	
	slope=[]
	slope_std=[]
	std=[]
	std1=[]
	mean=[]
	mean1=[]
	
	slope.append(np.array(data.mean())[5])
	slope_std.append(np.array(data.std())[5])
	std.append(np.array(data.mean())[4])
	std1.append(np.array(data.std())[4])
	mean.append(np.array(data.mean())[6])
	mean1.append(np.array(data.std())[6])
	
	for i in range(1,len(lc_type1)):
		data1=data[data['type']==lc_type1[i]]
		slope.append(np.array(data1.mean())[5])
		slope_std.append(np.array(data1.std())[5])
		std.append(np.array(data1.mean())[4])
		std1.append(np.array(data1.std())[4])
		mean.append(np.array(data1.mean())[6])
		mean1.append(np.array(data1.std())[6])
	data1=pd.DataFrame()
	data1['type']=lc_type1
	data1['slope']=slope
	data1['slope_std']=slope_std
	data1['std']=std
	data1['std1']=std1
	data1['mean']=mean
	data1['mean1']=mean1
	
	fig1=plt.figure()
	ax=fig1.add_subplot(1,1,1)
	ax.bar(np.arange(1,9,1),slope,yerr=slope_std,color=['black','#05450a','#78d203','#009900','#dcd159','#dade48','#fbff13','#b6ff05'])
	ax.set_yticks(np.arange(-6,9,3))
	ax.set_yticklabels(np.arange(-6,9,3), fontdict={'family':'arial','weight':'normal','size':22,})
	ax.set_ylabel('Difference (Days)', fontdict={'family':'arial','weight':'normal','size':22,})
	ax.set_xticks(np.arange(1,9,1))
	ax.set_xticklabels(type1, fontdict={'family':'arial','weight':'normal','size':18,})
	plt.gcf().set_size_inches(6,6)
	plt.savefig('./figure/difference_bar.png',bbox_inches='tight',dpi=300)
	
	fig1=plt.figure()
	ax=fig1.add_subplot(1,1,1)
	ax.bar(np.arange(1,9,1),std,yerr=std1,color=['black','#05450a','#78d203','#009900','#dcd159','#dade48','#fbff13','#b6ff05'])
	ax.set_yticks(np.arange(0,20,5))
	ax.set_yticklabels(np.arange(0,20,5), fontdict={'family':'arial','weight':'normal','size':22,})
	ax.set_ylabel('Std (Days)', fontdict={'family':'arial','weight':'normal','size':22,})
	ax.set_xticks(np.arange(1,9,1))
	ax.set_xticklabels(type1, fontdict={'family':'arial','weight':'normal','size':18,})
	plt.gcf().set_size_inches(6,6)
	plt.savefig('./figure/std_bar.png',bbox_inches='tight',dpi=300)

def lgb_evaluate(max_depth, learning_rate, n_estimators, num_leaves, subsample):
    params = {
        'objective': 'regression',
        'metric': 'rmse',
        'max_depth': int(max_depth),
        'learning_rate': learning_rate,
        'n_estimators': int(n_estimators),
        'num_leaves': num_leaves,
        'subsample': subsample,
        'verbose': -1
    }
    model = lgb.LGBMRegressor(**params)
    
    x_train, x_test, y_train, y_test = train_test_split(data11, eos, test_size=0.1, random_state=42)
    model.fit(x_train, y_train)
    predictions = model.predict(x_test)
    rmse = mean_squared_error(y_test, predictions, squared=False)
    
    return -rmse  # Negating because Bayesian Optimization minimizes the objective
    
def train():
	data=pd.read_csv(r'./data/train_daylength_new.csv')
	data=data.dropna()
	data=data[data['lat']>20]
	data1=data.groupby('LC_Type1').mean()
	data1=data1.reset_index()
	lc_type1=np.array(data1['LC_Type1'])
	lc_type1=[1,4,5,7,8,9,10]
	for i in range(len(lc_type1)):
		data1=data[data['LC_Type1']==lc_type1[i]]
		data1=data1[data1['eos']>=180]
		eos=data1.pop('eos')
		
		data11=data1[['sos',
		'aod_summer', 'aod_autumn', 'alan_summer','alan_autumn',
		't_summer', 't_autumn', 'pdsi_summer', 'pdsi_autumn',
		'soil_summer','soil_autumn', 'pr_summer', 'pr_autumn',
		'srad_summer', 'srad_autumn','daylength_summer']]

		optimizer = BayesianOptimization(
			f=lgb_evaluate,
			pbounds={
				'max_depth': (2, 16),
				'learning_rate': (0.1, 0.5),
				'n_estimators': (2000, 20000),
				'num_leaves': (2000,20000),
				'subsample': (0.75, 1.15)
			},
			random_state=42,
			verbose=2
		)
		
		optimizer.maximize(init_points=5, n_iter=15)

		best_params = optimizer.max['params']
		best_params['max_depth'] = int(best_params['max_depth'])
		best_params['n_estimators'] = int(best_params['n_estimators'])
		print(lc_type1[i],best_params)
		model = lgb.LGBMRegressor(**best_params)
		model.fit(data11, eos)
		explainer=shap.TreeExplainer(model)
		shap_values=explainer.shap_values(data11)

		columns=['sos',
		'aod_summer', 'aod_autumn', 'alan_summer','alan_autumn',
		't_summer', 't_autumn', 'pdsi_summer', 'pdsi_autumn',
		'soil_summer','soil_autumn', 'pr_summer', 'pr_autumn',
		'srad_summer', 'srad_autumn','daylength_summer']
		shap_values=pd.DataFrame(shap_values,columns=columns)
		shap_values['year']=np.array(data1['year'])
		shap_values['lat']=np.array(data1['lat'])
		shap_values['lon']=np.array(data1['lon'])
		shap_values.to_csv('./data/shap_values_'+str(int(lc_type1[i]))+'.csv')
		predict = cross_val_predict(model, data11, eos, cv=5)
		rmse=np.sqrt(mean_squared_error(predict,eos))
		mae=mean_absolute_error(predict,eos)
		data11=data1[['year','lat','lon','sos',
		'aod_summer', 'aod_autumn', 'alan_summer','alan_autumn',
		't_summer', 't_autumn', 'pdsi_summer', 'pdsi_autumn',
		'soil_summer','soil_autumn', 'pr_summer', 'pr_autumn',
		'srad_summer', 'srad_autumn','daylength_summer','daylength_autumn']]
		data11['eos']=eos
		data11['predict']=predict
		data11.to_csv('./data/validation_'+str(int(lc_type1[i]))+'.csv')

def ttest():
	lc_type=[1,4,5,7,8,9,10]
	type1=['ENF','DBF','MF','OS','WS','SA','GL']
	data=pd.read_csv(r'./data/eos_trend.csv')
	data1 = {
		'ENF': data[data['type']==1]['mean'],
		'DBF': data[data['type']==4]['mean'],
		'MF': data[data['type']==5]['mean'],
		'OS': data[data['type']==7]['mean'],
		'WS': data[data['type']==8]['mean'],
		'SA': data[data['type']==9]['mean'],
		'GL': data[data['type']==10]['mean'],
	}
	
	groups = list(data1.keys())
	print(groups)
	p_values = np.zeros((len(groups), len(groups)))
	t_values = np.zeros((len(groups), len(groups)))
	for i in range(len(groups)):
		for j in range(len(groups)):
			if i != j:
				t, p = ttest_ind(data1[groups[i]], data1[groups[j]], equal_var=False)
				p_values[i, j] = p
				t_values[i,j]=t

	plt.figure(figsize=(8, 6))
	sns.heatmap(p_values, annot=True, cmap='viridis', xticklabels=groups, yticklabels=groups, vmin=0, vmax=1)
	plt.title('P-Values Heatmap for Two-Tail T-Tests')
	plt.figure(figsize=(8, 6))
	sns.heatmap(t_values, annot=True, cmap='viridis', xticklabels=groups, yticklabels=groups, vmin=0, vmax=1)
	plt.title('T-Values Heatmap for Two-Tail T-Tests')
	plt.show()
	
def figure2a():
	cat1=[1,4,5,7,8,9,10]
	lat1=[]
	lon1=[]
	c=[]
	for j in range(len(cat1)):
		data=pd.read_csv('/Users/mengl/Documents/OneDrive - Vanderbilt/autumnphenology/train_map/train_new/type/shap_values_'+str(cat1[j])+'.csv')
		data=data.groupby(['lat','lon']).mean()
		data=data.reset_index()
		lat=np.array(data['lat'])
		lon=np.array(data['lon'])
		#cat=[]
		#c=[]
		columns=np.array(data.columns)
		for i in range(len(data)):
			a=np.array(np.abs(data.iloc[i][3:-1]))
			b=np.where(np.max(a)==a)[0][0]
			c.append(columns[b+3])
		lat1=np.append(lat1,lat)
		lon1=np.append(lon1,lon)
	data=pd.DataFrame()
	data['lat']=lat1
	data['lon']=lon1
	data['type']=c
	
	projections = [ccrs.AzimuthalEquidistant(central_longitude=0.0, central_latitude=90.0)]

	fig = plt.figure()
	ax = fig.add_subplot(1, 1, 1, projection=projections[0])
	extents = [-180, 180, 20, 90]
	ax.set_extent(extents, crs=ccrs.PlateCarree())
	ax.coastlines()
	ax.add_feature(cfeature.BORDERS, linestyle='-', linewidth=1, edgecolor='black')
	
	theta = np.linspace(0, 2*np.pi, 100)
	center, radius = [0.5, 0.5], 0.5
	verts = np.vstack([np.sin(theta), np.cos(theta)]).T
	circle = mpath.Path(verts * radius + center)
	ax.set_boundary(circle, transform=ax.transAxes)
	ax.add_feature(cfeature.LAND, zorder=0, edgecolor='black',color='floralwhite')

	lat=np.array(data['lat'])
	lon=np.array(data['lon'])	
	type=np.array(data['type'])
	
	type11=[]
	for i in range(len(type)):
		if type[i]!='t_autumn' and type[i]!='sos' and type[i]!='daylength_summer' and type[i]!='pr_summer':
			type11.append('other')
		elif type[i]=='t_autumn':
			type11.append('t_autumn')
		elif type[i]=='sos':
			type11.append('sos')
		elif type[i]=='daylength_summer':
			type11.append('daylength_summer')
		elif type[i]=='pr_summer':
			type11.append('pr_summer')
	type=np.array(type11)
	type1=np.unique(type)
	
	a=[]
	color = ['wheat', 'orange', 'grey', 'yellow', 'brown', 'cyan', 'lightgreen', 'blue',
			  'deeppink', 'lightyellow', 'olive', 'red', 'lightgreen', 'forestgreen', 'green', 'black']
	#exit()
	for i in range(len(type1)):
		b=np.where(type1[i]==type)[0]
		lon1=lon[b]
		lat1=lat[b]
		if type1[i]!='t_autumn' and type1[i]!='sos' and type1[i]!='daylength_summer' and type1[i]!='pr_summer':
			cs=ax.scatter(lon1,lat1,label='Other',s=800,marker='s',color='black',transform=ccrs.PlateCarree())
		elif type1[i]=='t_autumn':
			cs=ax.scatter(lon1,lat1,label='T$_{aut}$',s=800,marker='s',color='red',transform=ccrs.PlateCarree())
		elif type1[i]=='sos':
			cs=ax.scatter(lon1,lat1,label='SOS',s=800,marker='s',color='green',transform=ccrs.PlateCarree())
		elif type1[i]=='daylength_summer':
			cs=ax.scatter(lon1,lat1,label='Dayl$_{sum}$',s=800,marker='s',color='blue',transform=ccrs.PlateCarree())
		elif type1[i]=='pr_summer':
			cs=ax.scatter(lon1,lat1,label='Pr$_{sum}$',s=800,marker='o',color='yellow',transform=ccrs.PlateCarree())
	ax.legend(loc='upper center',bbox_to_anchor=(0.1, 0.35, 0.8, 0.8),ncol=5,prop={'family':'arial','weight':'normal','size':22,},frameon=False)
	plt.show()

def figure2b():
	type111=['All','ENF','DBF','MF','OS','WS','SA','GL']
	colors=['#14517C','#2F7FC1','#D76364','#96C37D','#F3D266']
	data=pd.read_csv('./data/shap_values_summary.csv')
	type11=np.array(data['type1'])
	a=0
	b=0
	c=0
	d=0
	e=0
	for j in range(len(type11)):
		if type11[j]==1:
			a=a+1
		elif type11[j]==2:
			b=b+1
		elif type11[j]==3:
			c=c+1
		elif type11[j]==4:
			d=d+1
		else:
			e=e+1
	a=a/len(type11)*100
	b=b/len(type11)*100
	c=c/len(type11)*100
	d=d/len(type11)*100
	e=e/len(type11)*100	
	
	print('all','ds:',a,'pr:',b,'ta:',c,'sos:',d,'other:',e)
	#exit()
	fig=plt.figure()
	ax=fig.add_subplot(1,1,1)	
	ax.bar(1,a,color='#14517C',width=0.5,label='D$_{s}$')
	ax.bar(1,b,bottom=a,color='#2F7FC1',width=0.5,label='Pr$_{s}$')
	ax.bar(1,c,bottom=a+b,color='#D76364',width=0.5,label='T$_{a}$')
	ax.bar(1,d,bottom=a+b+c,color='#96C37D',width=0.5,label='SOS')
	ax.bar(1,e,bottom=a+b+c+d,color='#F3D266',width=0.5,label='Other')
	
	data=pd.read_csv('./data/shap_values_summary.csv')
	type1=np.array(data['lc_type'])
	cat1=[1,4,5,7,8,9,10]
	for i in range(len(cat1)):
		data1=data[type1==cat1[i]]
		a=0
		b=0
		c=0
		d=0
		e=0
		type11=np.array(data1['type1'])
		for j in range(len(type11)):
			if type11[j]==1:
				a=a+1
			elif type11[j]==2:
				b=b+1
			elif type11[j]==3:
				c=c+1
			elif type11[j]==4:
				d=d+1
			else:
				e=e+1
		a=a/len(type11)*100
		b=b/len(type11)*100
		c=c/len(type11)*100
		d=d/len(type11)*100
		e=e/len(type11)*100
		print(type1[i+1],'ds:',a,'pr:',b,'ta:',c,'sos:',d,'other:',e)
		ax.bar(i+2,a,color='#14517C',width=0.5)
		ax.bar(i+2,b,bottom=a,color='#2F7FC1',width=0.5)
		ax.bar(i+2,c,bottom=a+b,color='#D76364',width=0.5)
		ax.bar(i+2,d,bottom=a+b+c,color='#96C37D',width=0.5)
		ax.bar(i+2,e,bottom=a+b+c+d,color='#F3D266',width=0.5)
	ax.set_yticks(np.arange(0,120,20))
	ax.set_yticklabels(np.arange(0,120,20), fontdict={'family':'arial','weight':'normal','size':26,})
	ax.set_ylabel('(%)', fontdict={'family':'arial','weight':'normal','size':26,})
	ax.set_xticks(np.arange(1,9,1))
	ax.set_xticklabels(type111, fontdict={'family':'arial','weight':'normal','size':26,})
	ax.set_ylim(0,120)
	plt.gcf().set_size_inches(9,10)
	plt.show()
	
def figure3():
	lc_type1=[1,4,5,7,8,9,10]
	type1=['ENF','DBF','MF','OS','WS','SA','GL']
	data1=pd.DataFrame()
	for i in range(len(lc_type1)):
		data=pd.read_csv('./data/shap_values_'+str(lc_type1[i])+'.csv')
		data1=pd.concat([data1,data])
	col=['sos','aod_summer', 'aod_autumn', 'alan_summer','alan_autumn','t_summer', 't_autumn', 'pdsi_summer', 'pdsi_autumn','soil_summer','soil_autumn', 'pr_summer', 'pr_autumn','srad_summer', 'srad_autumn','daylength_summer']
	data=data1.groupby(['lat','lon']).mean()
	data=data.reset_index()
	columns=np.array(data.columns)
	columns=columns[3:-1]
	c=0
	d=[]
	f=[]
	for j in range(len(columns)):
		a=np.array(data[columns[j]])
		a=np.abs(a)
		b=np.mean(a)
		e=np.std(a)
		c=c+b
		d.append(b)
		f.append(e)
	d=d/c*100
	f=f/c*100
	data1=pd.DataFrame()
	data1['columns']=columns
	data1['value']=d
	data1['std']=f
	data1=data1.sort_values('value',ascending=False)
	value=np.array(data1['value'])[0:4]
	columns=np.array(data1['columns'])[0:4]
	std=np.array(data1['std'])[0:4]
	
	columns1=[]
	color1=[]
	cdict=['#14517C','#2F7FC1','#D76364','#96C37D','#F3D266'] #Ds Prs Ta SOS Other
	for j in range(len(columns)):
		if columns[j]=='t_autumn':
			columns1.append('T$_{a}$')
			color1.append('#D76364')
		elif columns[j]=='sos':
			columns1.append('SOS')
			color1.append('#96C37D')
		elif columns[j]=='srad_summer':
			columns1.append('Srad$_{s}$')
			color1.append('#F3D266')
		elif columns[j]=='daylength_summer':
			columns1.append('D$_{s}$')
			color1.append('#14517C')
		elif columns[j]=='daylength_autumn':
			columns1.append('D$_{a}$')
			color1.append('#F3D266')
		elif columns[j]=='t_summer':
			columns1.append('T$_{s}$')
			color1.append('#F3D266')
		elif columns[j]=='srad_autumn':
			columns1.append('Srad$_{a}$')
			color1.append('#F3D266')
		elif columns[j]=='pr_summer':
			columns1.append('Pr$_{s}$')
			color1.append('#2F7FC1')
		elif columns[j]=='pr_autumn':
			columns1.append('Pr$_{a}$')
			color1.append('#F3D266')
		elif columns[j]=='pdsi_summer':
			columns1.append('P$_{s}$')
			color1.append('#F3D266')
		elif columns[j]=='pdsi_autumn':
			columns1.append('P$_{a}$')
			color1.append('#F3D266')
		elif columns[j]=='aod_summer':
			columns1.append('AOD$_{s}$')
			color1.append('#F3D266')
		elif columns[j]=='aod_autumn':
			columns1.append('AOD$_{a}$')
			color1.append('#F3D266')
		elif columns[j]=='alan_summer':
			columns1.append('ALAN$_{s}$')
			color1.append('#F3D266')
		elif columns[j]=='alan_autumn':
			columns1.append('ALAN$_{a}$')
			color1.append('#F3D266')
		elif columns[j]=='soil_summer':
			columns1.append('Soil$_{s}$')
			color1.append('#F3D266')
		elif columns[j]=='soil_autumn':
			columns1.append('Soil$_{a}$')
			color1.append('#F3D266')
	fig1=plt.figure()
	ax=fig1.add_subplot(1,1,1)
	ax.bar(columns1,value,yerr=std,color=color1)
	ax.set_xticks(np.arange(0,4,1))
	ax.set_xticklabels(columns1, fontdict={'family':'arial','weight':'normal','size':22,})	
	ax.set_yticks(np.arange(0,120,20))
	ax.set_yticklabels(np.arange(0,120,20), fontdict={'family':'arial','weight':'normal','size':22,})	
	ax.set_ylabel('(%)', fontdict={'family':'arial','weight':'normal','size':22,})	
	ax.set_title('All',fontdict={'family':'arial','weight':'normal','size':26,})	
	plt.gcf().set_size_inches(6,6.5)

	for i in range(len(lc_type1)):
		data=pd.read_csv('./data/shap_values_'+str(lc_type1[i])+'.csv')
		col=['sos','aod_summer', 'aod_autumn', 'alan_summer','alan_autumn','t_summer', 't_autumn', 'pdsi_summer', 'pdsi_autumn','soil_summer','soil_autumn', 'pr_summer', 'pr_autumn','srad_summer', 'srad_autumn','daylength_summer']
		data[col]=data[col].apply(lambda x:x.abs(),axis=1)
		data=data.groupby(['lat','lon']).mean()
		data=data.reset_index()
		columns=np.array(data.columns)
		columns=columns[3:-1]
		c=0
		d=[]
		f=[]
		for j in range(len(columns)):
			a=np.array(data[columns[j]])
			a=np.abs(a)
			b=np.mean(a)
			e=np.std(a)
			c=c+b
			d.append(b)
			f.append(e)
		d=d/c*100
		f=f/c*100
		data1=pd.DataFrame()
		data1['columns']=columns
		data1['value']=d
		data1['std']=f
		data1=data1.sort_values('value',ascending=False)
		value=np.array(data1['value'])[0:4]
		columns=np.array(data1['columns'])[0:4]
		std=np.array(data1['std'])[0:4]
		
		columns1=[]
		color1=[]
		cdict=['#14517C','#2F7FC1','#D76364','#96C37D','#F3D266']
		for j in range(len(columns)):
			if columns[j]=='t_autumn':
				columns1.append('T$_{a}$')
				color1.append('#D76364')
			elif columns[j]=='sos':
				columns1.append('SOS')
				color1.append('#96C37D')
			elif columns[j]=='srad_summer':
				columns1.append('Srad$_{s}$')
				color1.append('#F3D266')
			elif columns[j]=='daylength_summer':
				columns1.append('D$_{s}$')
				color1.append('#14517C')
			elif columns[j]=='daylength_autumn':
				columns1.append('D$_{a}$')
				color1.append('#F3D266')
			elif columns[j]=='t_summer':
				columns1.append('T$_{s}$')
				color1.append('#F3D266')
			elif columns[j]=='srad_autumn':
				columns1.append('Srad$_{a}$')
				color1.append('#F3D266')
			elif columns[j]=='pr_summer':
				columns1.append('Pr$_{s}$')
				color1.append('#2F7FC1')
			elif columns[j]=='pr_autumn':
				columns1.append('Pr$_{a}$')
				color1.append('#F3D266')
			elif columns[j]=='pdsi_summer':
				columns1.append('P$_{s}$')
				color1.append('#F3D266')
			elif columns[j]=='pdsi_autumn':
				columns1.append('P$_{a}$')
				color1.append('#F3D266')
			elif columns[j]=='aod_summer':
				columns1.append('AOD$_{s}$')
				color1.append('#F3D266')
			elif columns[j]=='aod_autumn':
				columns1.append('AOD$_{a}$')
				color1.append('#F3D266')
			elif columns[j]=='alan_summer':
				columns1.append('ALAN$_{s}$')
				color1.append('#F3D266')
			elif columns[j]=='alan_autumn':
				columns1.append('ALAN$_{a}$')
				color1.append('#F3D266')
			elif columns[j]=='soil_summer':
				columns1.append('Soil$_{s}$')
				color1.append('#F3D266')
			elif columns[j]=='soil_autumn':
				columns1.append('Soil$_{a}$')
				color1.append('#F3D266')

		fig1=plt.figure()
		ax=fig1.add_subplot(1,1,1)
		ax.bar(columns1,value,yerr=std,color=color1)
		ax.set_xticks(np.arange(0,4,1))
		ax.set_xticklabels(columns1, fontdict={'family':'arial','weight':'normal','size':22,})	
		ax.set_yticks(np.arange(0,120,20))
		ax.set_yticklabels(np.arange(0,120,20), fontdict={'family':'arial','weight':'normal','size':22,})		
		ax.set_ylabel('(%)', fontdict={'family':'arial','weight':'normal','size':22,})	
		ax.set_title(type1[i],fontdict={'family':'arial','weight':'normal','size':26,})	
		plt.gcf().set_size_inches(6,6.5)
	plt.show()

def figure4a():
	cat=['sos_mean1','temp_mean1','pr_mean1','daylength_mean1']
	for i in range(len(cat)):
		data=Dataset('./data/1.nc')
		lat1=np.array(data['lat'])
		lon1=np.array(data['lon'])
		data1=np.array(data['data'])
		for j in range(len(lat1)):
			for k in range(len(lon1)):
				data1[j,k]=np.nan
					
		data2=pd.read_csv(r'./data/shap_trend.csv')
		eos=np.array(data2.pop(cat[i]))
		lat2=np.array(data2['lat'])
		lon2=np.array(data2['lon'])
		for j in range(len(lat2)):
			a=np.where(lat1==lat2[j])[0][0]
			b=np.where(lon1==lon2[j])[0][0]
			data1[a,b]=eos[j]

		projections = [
			ccrs.AzimuthalEquidistant(central_longitude=0.0, central_latitude=90.0),
		]
		
		fig = plt.figure()
		ax = fig.add_subplot(1, 1, 1, projection=projections[0])
		ax.coastlines()
		ax.add_feature(cfeature.BORDERS, linestyle='-', linewidth=0.5, edgecolor='black')
		theta = np.linspace(0, 2*np.pi, 100)
		center, radius = [0.5, 0.5], 0.5
		verts = np.vstack([np.sin(theta), np.cos(theta)]).T
		circle = mpath.Path(verts * radius + center)
		ax.set_boundary(circle, transform=ax.transAxes)
		if i==0:
			cs=ax.pcolormesh(lon1,lat1,data1,vmin=-10,vmax=10,cmap=colormap_res11(),transform=ccrs.PlateCarree())	
		elif i==1:
			cs=ax.pcolormesh(lon1,lat1,data1,vmin=-10,vmax=10,cmap=colormap_res11(),transform=ccrs.PlateCarree())
		else:
			cs=ax.pcolormesh(lon1,lat1,data1,vmin=-10,vmax=10,cmap=colormap_res11(),transform=ccrs.PlateCarree())	
		ax.set_extent([-180, 180,20, 90], crs=ccrs.PlateCarree())  # Adjust the extent as needed	
		
		fig.subplots_adjust()
		l=0.19
		b=0.075
		w=0.655
		h=0.0155
		rect=[l,b,w,h]
		cbar_ax=fig.add_axes(rect)
		cb=fig.colorbar(cs,cax=cbar_ax,extend='both',orientation='horizontal')
		if i==0:
			cb.set_ticks(np.arange(-10,15,5))
		elif i==1:
			cb.set_ticks(np.arange(-10,15,5))
		else:
			cb.set_ticks(np.arange(-10,15,5))
		cb.ax.tick_params(labelsize=22)
		cb.set_label('(Days)',fontdict={'family':'arial','weight':'normal','size':22,})
		plt.savefig('./figure/'+str(cat[i])+'_shap_mean.png',bbox_inches='tight',dpi=300)
	
	cat=['sos_mean','temp_mean','pr_mean','daylength_mean']
	for i in range(len(cat)):
		data=Dataset('./data/1.nc')
		lat1=np.array(data['lat'])
		lon1=np.array(data['lon'])
		data1=np.array(data['data'])
		for j in range(len(lat1)):
			for k in range(len(lon1)):
				data1[j,k]=np.nan
					
		data2=pd.read_csv(r'./data/shap_trend.csv')
		eos=np.array(data2.pop(cat[i]))
		lat2=np.array(data2['lat'])
		lon2=np.array(data2['lon'])
		for j in range(len(lat2)):
			a=np.where(lat1==lat2[j])[0][0]
			b=np.where(lon1==lon2[j])[0][0]
			data1[a,b]=eos[j]

		projections = [
			ccrs.AzimuthalEquidistant(central_longitude=0.0, central_latitude=90.0),
		]
		
		fig = plt.figure()
		ax = fig.add_subplot(1, 1, 1, projection=projections[0])
		ax.coastlines()
		ax.add_feature(cfeature.BORDERS, linestyle='-', linewidth=0.5, edgecolor='black')
		theta = np.linspace(0, 2*np.pi, 100)
		center, radius = [0.5, 0.5], 0.5
		verts = np.vstack([np.sin(theta), np.cos(theta)]).T
		circle = mpath.Path(verts * radius + center)
		ax.set_boundary(circle, transform=ax.transAxes)
		if i==0:
			cs=ax.pcolormesh(lon1,lat1,data1,vmin=-10,vmax=10,cmap=colormap_res11(),transform=ccrs.PlateCarree())	
		elif i==1:
			cs=ax.pcolormesh(lon1,lat1,data1,vmin=-10,vmax=10,cmap=colormap_res11(),transform=ccrs.PlateCarree())
		else:
			cs=ax.pcolormesh(lon1,lat1,data1,vmin=-10,vmax=10,cmap=colormap_res11(),transform=ccrs.PlateCarree())	
		ax.set_extent([-180, 180,20, 90], crs=ccrs.PlateCarree())  # Adjust the extent as needed	
		
		fig.subplots_adjust()
		l=0.19
		b=0.075
		w=0.655
		h=0.0155
		rect=[l,b,w,h]
		cbar_ax=fig.add_axes(rect)
		cb=fig.colorbar(cs,cax=cbar_ax,extend='both',orientation='horizontal')
		if i==0:
			cb.set_ticks(np.arange(-10,15,5))
		elif i==1:
			cb.set_ticks(np.arange(-10,15,5))
		else:
			cb.set_ticks(np.arange(-10,15,5))
		cb.ax.tick_params(labelsize=22)
		cb.set_label('(Days)',fontdict={'family':'arial','weight':'normal','size':22,})
		plt.savefig('./figure/'+str(cat[i])+'_shap.png',bbox_inches='tight',dpi=300)
	
	cat=['sos_std','temp_std','pr_std','daylength_std']
	for i in range(len(cat)):
		data=Dataset('./data/1.nc')
		lat1=np.array(data['lat'])
		lon1=np.array(data['lon'])
		data1=np.array(data['data'])
		for j in range(len(lat1)):
			for k in range(len(lon1)):
				data1[j,k]=np.nan
					
		data2=pd.read_csv(r'./data/shap_trend.csv')
		eos=np.array(data2.pop(cat[i]))
		lat2=np.array(data2['lat'])
		lon2=np.array(data2['lon'])
		for j in range(len(lat2)):
			a=np.where(lat1==lat2[j])[0][0]
			b=np.where(lon1==lon2[j])[0][0]
			data1[a,b]=eos[j]
			
		projections = [
			ccrs.AzimuthalEquidistant(central_longitude=0.0, central_latitude=90.0),
		]
		
		fig = plt.figure()
		ax = fig.add_subplot(1, 1, 1, projection=projections[0])
		ax.coastlines()
		ax.add_feature(cfeature.BORDERS, linestyle='-', linewidth=0.5, edgecolor='black')

		theta = np.linspace(0, 2*np.pi, 100)
		center, radius = [0.5, 0.5], 0.5
		verts = np.vstack([np.sin(theta), np.cos(theta)]).T
		circle = mpath.Path(verts * radius + center)
		ax.set_boundary(circle, transform=ax.transAxes)
		if i==0:
			cs=ax.pcolormesh(lon1,lat1,data1,vmin=0,vmax=5,cmap=colormap_res11(),transform=ccrs.PlateCarree())	
		elif i==1:
			cs=ax.pcolormesh(lon1,lat1,data1,vmin=0,vmax=5,cmap=colormap_res11(),transform=ccrs.PlateCarree())	
		else:
			cs=ax.pcolormesh(lon1,lat1,data1,vmin=0,vmax=5,cmap=colormap_res11(),transform=ccrs.PlateCarree())	
		ax.set_extent([-180, 180,20, 90], crs=ccrs.PlateCarree())  # Adjust the extent as needed	
		
		fig.subplots_adjust()
		l=0.19
		b=0.075
		w=0.655
		h=0.0155
		rect=[l,b,w,h]
		cbar_ax=fig.add_axes(rect)
		cb=fig.colorbar(cs,cax=cbar_ax,extend='both',orientation='horizontal')
		if i==0:
			cb.set_ticks(np.arange(0,6,1))
		elif i==1:
			cb.set_ticks(np.arange(0,6,1))
		else:
			cb.set_ticks(np.arange(0,6,1))
		cb.ax.tick_params(labelsize=22)
		cb.set_label('(Days)',fontdict={'family':'arial','weight':'normal','size':22,})
		plt.savefig('./data/'+str(cat[i])+'_shap_std.png',bbox_inches='tight',dpi=300)

def figure4b():
	type1=['All','ENF','DBF','MF','OS','WS','SA','GL']
	cat=['daylength','pr','temp','sos']
	data=pd.read_csv(r'./data/shap_trend.csv')
	lc_type1=[0,1,4,5,7,8,9,10]
	for i in range(len(cat)):
		slope=[]
		slope_std=[]
		std=[]
		std1=[]
		mean=[]
		mean_std=[]
		for j in range(len(lc_type1)):
			if j!=0:
				data1=data[data['lc_type']==lc_type1[j]]
			else:
				data1=data
			slope.append(np.mean(data1[cat[i]+'_mean']))
			slope_std.append(np.std(data1[cat[i]+'_mean']))
			mean.append(np.mean(data1[cat[i]+'_mean1']))
			mean_std.append(np.std(data1[cat[i]+'_mean1']))
			std.append(np.mean(data1[cat[i]+'_std']))
			std1.append(np.std(data1[cat[i]+'_std']))
		data1=pd.DataFrame()
		data1['type']=lc_type1
		data1['typ1e']=type1
		data1['mean']=mean
		data1['mean_std']=mean_std
		data1['slope']=slope
		data1['slope_std']=slope_std
		data1['std']=std
		data1['std1']=std1

		fig1=plt.figure()
		ax=fig1.add_subplot(1,1,1)
		ax.bar(np.arange(1,9,1),mean,yerr=mean_std,color=['black','#05450a','#78d203','#009900','#dcd159','#dade48','#fbff13','#b6ff05'])
		if i==0:
			ax.set_yticks(np.arange(0,3,1))
			ax.set_yticklabels(np.arange(0,3,1), fontdict={'family':'arial','weight':'normal','size':22,})
		if i==1:
			ax.set_yticks(np.arange(0,5,1))
			ax.set_yticklabels(np.arange(0,5,1), fontdict={'family':'arial','weight':'normal','size':22,})
		if i==2:
			ax.set_yticks(np.arange(0,4,1))
			ax.set_yticklabels(np.arange(0,4,1), fontdict={'family':'arial','weight':'normal','size':22,})
		if i==3:
			ax.set_yticks(np.arange(0,12,4))
			ax.set_yticklabels(np.arange(0,12,4), fontdict={'family':'arial','weight':'normal','size':22,})
		ax.set_xticks(np.arange(1,9,1))
		ax.set_xticklabels(type1, fontdict={'family':'arial','weight':'normal','size':20,})
		ax.set_ylabel('Std (Days)', fontdict={'family':'arial','weight':'normal','size':22,})

		fig1=plt.figure()
		ax=fig1.add_subplot(1,1,1)
		ax.bar(np.arange(1,9,1),slope,yerr=slope_std,color=['black','#05450a','#78d203','#009900','#dcd159','#dade48','#fbff13','#b6ff05'])
		if i==0:
			ax.set_yticks(np.arange(-2,3,1))
			ax.set_yticklabels(np.arange(-2,3,1), fontdict={'family':'arial','weight':'normal','size':22,})
		if i==1:
			ax.set_yticks(np.arange(-2,3,1))
			ax.set_yticklabels(np.arange(-2,3,1), fontdict={'family':'arial','weight':'normal','size':22,})
		if i==2:
			ax.set_yticks(np.arange(-3,4,1))
			ax.set_yticklabels(np.arange(-3,4,1), fontdict={'family':'arial','weight':'normal','size':22,})
		if i==3:
			ax.set_yticks(np.arange(-3,4,1))
			ax.set_yticklabels(np.arange(-3,4,1), fontdict={'family':'arial','weight':'normal','size':22,})
		ax.set_xticks(np.arange(1,9,1))
		ax.set_xticklabels(type1, fontdict={'family':'arial','weight':'normal','size':20,})
		ax.set_ylabel('Difference (Days)', fontdict={'family':'arial','weight':'normal','size':22,})

		fig1=plt.figure()
		ax=fig1.add_subplot(1,1,1)
		ax.bar(np.arange(1,9,1),std,yerr=std1,color=['black','#05450a','#78d203','#009900','#dcd159','#dade48','#fbff13','#b6ff05'])
		if i==0:
			ax.set_yticks(np.arange(0,3,1))
			ax.set_yticklabels(np.arange(0,3,1), fontdict={'family':'arial','weight':'normal','size':22,})
		if i==1:
			ax.set_yticks(np.arange(0,5,1))
			ax.set_yticklabels(np.arange(0,5,1), fontdict={'family':'arial','weight':'normal','size':22,})
		if i==2:
			ax.set_yticks(np.arange(0,4,1))
			ax.set_yticklabels(np.arange(0,4,1), fontdict={'family':'arial','weight':'normal','size':22,})
		if i==3:
			ax.set_yticks(np.arange(0,12,4))
			ax.set_yticklabels(np.arange(0,12,4), fontdict={'family':'arial','weight':'normal','size':22,})
		ax.set_xticks(np.arange(1,9,1))
		ax.set_xticklabels(type1, fontdict={'family':'arial','weight':'normal','size':20,})
		ax.set_ylabel('Std (Days)', fontdict={'family':'arial','weight':'normal','size':22,})
		plt.show()

def figure5a():
	type1=['daylength_summer','pr_summer','t_autumn','sos']
	type2=['D$_{s}$ (Hours)','Pr$_{s}$ (mm)','T$_{a}$ ($^{\circ}$C)','SOS (DOY)']
	type3=['Evergreen Needleaf Forests','Deciduous Broadleaf Forests','Mixed Forests','Open Shrublands','Woody Savannas','Savannas','Grasslands']
	colors=['#14517C','#2F7FC1','#D76364','#96C37D',]
	cdict=['#14517C','#2F7FC1','#D76364','#96C37D','#F3D266'] #Ds Prs Ta SOS Other
	lat1=[]
	lon1=[]
	c=[]
	data1=pd.DataFrame()
	for j in range(len(type1)):
		data1=pd.read_csv(r'./data/shap_bin/all/'+type1[j]+'_shap.csv')
		sos11=np.array(data1['sos'])
		sos22=np.array(data1['shap'])
		sos33=np.array(data1['shap_std'])
		
		a=[]
		for k in range(len(sos11)):
			if k!=len(sos11)-1:
				a.append(str(sos11[k])+'~'+str(sos11[k+1]))
			else:
				a.append('>'+str(sos11[k]))
		
		fig=plt.figure()
		ax=fig.add_subplot(1,1,1)	
		ax.errorbar(sos11,sos22,sos33,fmt='o-',mfc=colors[j],capsize=10,elinewidth=3,capthick=3,markersize=12,linewidth=3,color=colors[j])
		ax.axhline(0,color='black',linestyle='--')
		ax.set_xlabel(type2[j],fontdict={'family':'arial','weight':'normal','size':22,})   
		ax.set_ylabel('$\Delta$'+'EOS (Days)',fontdict={'family':'arial','weight':'normal','size':22,})
		ax.tick_params(axis='y', labelsize=22) 
		ax.tick_params(axis='x', labelsize=22) 
		plt.savefig('./figure/'+type1[j]+'_shap.png', bbox_inches='tight',dpi = 300)

def figure5b():
	cat1=[1,4,5,7,8,9,10]
	type1=['daylength_summer','pr_summer','t_autumn','sos']
	type2=['D$_{s}$ (Hours)','Pr$_{s}$ (mm)','T$_{a}$ ($^{\circ}$C)','SOS (DOY)']
	type3=['Evergreen Needleaf Forests','Deciduous Broadleaf Forests','Mixed Forests','Open Shrublands','Woody Savannas','Savannas','Grasslands']
	colors=['#14517C','#2F7FC1','#D76364','#96C37D',]
	cdict=['#14517C','#2F7FC1','#D76364','#96C37D','#F3D266'] #Ds Prs Ta SOS Other
	lat1=[]
	lon1=[]
	c=[]
	data1=pd.DataFrame()
	for i in range(len(cat1)):
		for j in range(len(type1)):
			data1=pd.read_csv(r'./data/shap_bin/'+str(cat1[i])+'_'+type1[j]+'.csv')
			sos11=np.array(data1['sos'])
			sos22=np.array(data1['shap'])
			sos33=np.array(data1['shap_std'])
			
			a=[]
			for k in range(len(sos11)):
				if k!=len(sos11)-1:
					a.append(str(sos11[k])+'~'+str(sos11[k+1]))
				else:
					a.append('>'+str(sos11[k]))
			
			fig=plt.figure()
			ax=fig.add_subplot(1,1,1)	
			ax.errorbar(sos11,sos22,sos33,fmt='o-',mfc=colors[j],capsize=10,elinewidth=5,capthick=5,markersize=12,linewidth=5,color=colors[j])
			ax.axhline(0,color='black',linestyle='--')
			ax.set_xlabel(type2[j],fontdict={'family':'arial','weight':'normal','size':22,})   
			ax.set_ylabel('$\Delta$'+'EOS (Days)',fontdict={'family':'arial','weight':'normal','size':22,})
			ax.tick_params(axis='y', labelsize=22) 
			ax.tick_params(axis='x', labelsize=22) 
			plt.savefig('./figure/'+str(cat1[i])+'_'+type1[j]+'.png', bbox_inches='tight',dpi = 300)

def lctype():
	data=Dataset('./data/1.nc')
	lat1=np.array(data['lat'])
	print(lat1)
	lon1=np.array(data['lon'])
	data1=np.array(data['data'])
	for i in range(len(lat1)):
		for j in range(len(lon1)):
			data1[i,j]=np.nan
							
	type=[1,4,5,7,8,9,10]
	type1=['ENF','DBF','MF','OS','WS','SA','GL']
	data=pd.read_csv(r'./data/train_daylength_new.csv')
	data=data.dropna()
	data=data[data['lat']>20]
	data=data[data['eos']>=180]
	lat=np.array(data['lat'])
	lon=np.array(data['lon'])
	lc_type=np.array(data['LC_Type1'])
	data=data[(lc_type==1)|(lc_type==4)|(lc_type==5)|(lc_type==7)|(lc_type==8)|(lc_type==9)|(lc_type==10)]
	lc_type=np.array(data['LC_Type1'])
	lc_type1=np.array(data['LC_Type1'])
	for i in range(len(type)):
		lc_type1[lc_type==type[i]]=i+1
	
	projections = [ccrs.AzimuthalEquidistant(central_longitude=0.0, central_latitude=90.0)]

	fig = plt.figure()
	ax = fig.add_subplot(1, 1, 1, projection=projections[0])
	extents = [-180, 180, 20, 90]
	ax.set_extent(extents, crs=ccrs.PlateCarree())
	ax.coastlines()
	theta = np.linspace(0, 2*np.pi, 100)
	center, radius = [0.5, 0.5], 0.5
	verts = np.vstack([np.sin(theta), np.cos(theta)]).T
	circle = mpath.Path(verts * radius + center)
	ax.set_boundary(circle, transform=ax.transAxes)
	ax.add_feature(cfeature.LAND, zorder=0, edgecolor='black',color='floralwhite')

	lat2=np.array(data['lat'])
	lon2=np.array(data['lon'])
	for k in range(len(lat2)):
		a=np.where(lat1==lat2[k])[0][0]
		b=np.where(lon1==lon2[k])[0][0]
		data1[a,b]=lc_type1[k]
	cs=ax.pcolormesh(lon1,lat1,data1,vmin=1,vmax=7,cmap=colormap_landcover(),transform=ccrs.PlateCarree())	
	ax.set_extent([-180, 180,20, 90], crs=ccrs.PlateCarree())

	#fig.subplots_adjust()
	#l=0.19
	#b=0.075
	#w=0.655
	#h=0.0155
	#rect=[l,b,w,h]
	#cbar_ax=fig.add_axes(rect)
	#cb=fig.colorbar(cs,cax=cbar_ax,extend='both',orientation='horizontal')
	#cb.set_ticks(np.arange(1,8,1))
	#cb.ax.tick_params(labelsize=12)
	#cb.set_label('[%]',fontdict={'family':'arial','weight':'normal','size':12,})
	#ax.set_title(columns[i],fontdict={'family':'arial','weight':'normal','size':32,})
	#print('/Users/mengl/Documents/OneDrive - Vanderbilt/autumnphenology/train_map/train_new/shap/cat/'++str(cat1[j])+'_'+columns[i]+'.png')
	#plt.savefig('/Users/mengl/Documents/OneDrive - Vanderbilt/autumnphenology/train_map/train_new/shap/cat/'+str(cat1[j])+'_'+columns[i]+'.png',bbox_inches='tight',dpi=300)
	plt.show()
							
def main(args):
	#figure1()
	#ttest()
	#train()
	#figure2a()
	#figure2b()
	#figure3()
	figure4a()
	figure4b()
	#figure5()
	#lctype()
	return 0

if __name__ == '__main__':
    import sys
    sys.exit(main(sys.argv))
